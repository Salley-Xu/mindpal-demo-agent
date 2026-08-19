import logging
import json
import time
import asyncio
import re
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

from config import config
from conversation_manager import conversation_manager
from urgent_detector import urgent_detector, urgent_logger
from emotion_analyzer import emotion_analyzer
from risk_evaluator import risk_evaluator
from risk_memory import RiskMemoryReader, RiskMemoryWriter
from recommend_gate import recommend_gate
from output_safety_checker import output_safety_checker
from recommendation_trace import TraceEvent, write_trace
from rejection_detector import detect_rejection
from models import (
    AgentRunRequest,
    AgentRunResponse,
    AgentStep,
    AgentToolCall,
    ChatResponse,
    ContentItem,
    EmotionState,
    RiskState,
    SessionSummary,
    RecommendationDecision,
)
from agent_tools import (
    TOOL_DEFINITIONS,
    KnowledgeBaseTool,
    UserProfileTool,
    MoodTrackingTool,
    ContentRecommendTool,
    KnowledgeBaseQuery
)
from agent_prompts import STRATEGY_PROMPTS, SYSTEM_PROMPT_TEMPLATE
from knowledge_store import knowledge_store, KnowledgeQuery as KBQuery
from risk_levels import (
    LEVEL_0,
    LEVEL_2,
    is_emergency_risk,
    is_non_low_risk,
    normalize_risk_level,
    risk_level_band,
    risk_level_index,
    risk_level_label,
)

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Refactored Agent Orchestrator using LLM Function Calling (ReAct Loop).
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.API_BASE_URL
        )
        self.model = config.CHAT_MODEL
        
        # Strategy Prompts (from agent_prompts)
        self.strategy_prompts = STRATEGY_PROMPTS
        
        # Tool mapping
        self.available_tools = {
            "search_knowledge_base": self._tool_search_knowledge_base,
            "get_user_profile": self._tool_get_user_profile,
            "update_user_profile": self._tool_update_user_profile,
            "log_mood_event": self._tool_log_mood_event,
            "get_recent_mood_trend": self._tool_get_recent_mood_trend,
            "recommend_content": self._tool_recommend_content
        }

    async def _execute_single_tool(
        self,
        tool_call,
        conversation_summary,
        request_text,
        user_id,
        session_id,
        recommendation_decision: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ):
        function_name = tool_call.function.name
        arguments_str = tool_call.function.arguments
        tool_call_id = tool_call.id
        
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
             return {
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": "Error: Invalid JSON arguments",
                "tool_step_info": AgentToolCall(
                    name=function_name,
                    input={"raw": arguments_str},
                    output_summary={"result": "Error: Invalid JSON arguments"},
                    success=False,
                    error_message="Invalid JSON arguments"
                )
            }

        # Prepare context for tools that need it
        if function_name == "recommend_content":
            decision = recommendation_decision or {}
            if decision.get("recommend_type") != "hard":
                skip_reason = decision.get("recommend_type", "none")
                return {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": json.dumps({"skipped": True, "reason": f"recommend_gate_{skip_reason}"}, ensure_ascii=False),
                    "tool_step_info": AgentToolCall(
                        name=function_name,
                        input={"raw": arguments_str},
                        output_summary={"result": f"Skipped by recommend gate: {skip_reason}"},
                        success=True,
                        error_message=None,
                    ),
                }
            arguments["conversation_summary"] = conversation_summary
            if "user_input" not in arguments:
                arguments["user_input"] = request_text
            if "current_emotion" not in arguments:
                arguments["current_emotion"] = conversation_summary.get("primary_emotion", "中性")
            if "user_profile" not in arguments:
                arguments["user_profile"] = user_profile or {}

        # Inject common context (user_id, session_id) if missing
        if "user_id" not in arguments:
            arguments["user_id"] = user_id
        if "session_id" not in arguments:
            arguments["session_id"] = session_id

        tool_result_str = ""
        tool_success = True
        error_msg = None

        try:
            if function_name in self.available_tools:
                result = await self.available_tools[function_name](**arguments)
                tool_result_str = json.dumps(result, ensure_ascii=False, default=str)
            else:
                tool_result_str = f"Error: Tool {function_name} not found"
                tool_success = False
                error_msg = "Tool not found"
        except Exception as e:
            tool_result_str = f"Error executing {function_name}: {str(e)}"
            tool_success = False
            error_msg = str(e)
            logger.error(f"Tool execution failed: {e}")

        return {
            "tool_call_id": tool_call_id,
            "name": function_name,
            "content": tool_result_str,
            "tool_step_info": AgentToolCall(
                name=function_name,
                input=arguments,
                output_summary={"result": tool_result_str[:200] + "..." if len(tool_result_str) > 200 else tool_result_str},
                success=tool_success,
                error_message=error_msg
            )
        }

    async def run_agent(self, request: AgentRunRequest) -> AgentRunResponse:
        run_id = str(uuid.uuid4())
        steps: List[AgentStep] = []

        # Initialize default values for error handling
        current_emotion = "中性"
        context_emotion = "中性"
        confidence = 0.5
        urgent_issue = None
        final_response_text = ""
        final_recommendations = []
        final_rationale = ""
        conversation_summary = self._default_session_summary()
        updated_conversation_summary = None
        recommendation_decision = None
        user_profile = {}
        messages: List[Any] = []

        try:
            # 1. Load Context & Session
            step_start = datetime.now(timezone.utc)
            session = await conversation_manager.get_or_create_session_async(
                request.user_id, request.session_id
            )
            conversation_summary = await conversation_manager.get_conversation_summary_async(
                request.user_id, request.session_id
            )
            
            # 1.1 Emotion Analysis
            try:
                current_emotion, context_emotion, confidence = await emotion_analyzer.analyze_with_context_async(
                    request.text, conversation_summary
                )
                # Update summary for prompt context
                conversation_summary['primary_emotion'] = current_emotion
            except Exception as e:
                logger.error(f"Emotion analysis failed: {e}")
            
            long_term_risk_level = None
            historical_high_risk_count = 0
            try:
                profile_model = await UserProfileTool.get_profile(request.user_id)
                user_profile = profile_model.model_dump() if profile_model else {}
                long_term_risk_level = normalize_risk_level(user_profile.get("risk_level"))
                user_profile["risk_level"] = long_term_risk_level
            except Exception as e:
                logger.warning(f"加载用户画像失败，使用空画像继续: {e}")
                user_profile = {}

            try:
                recent_mood_events = await MoodTrackingTool.get_recent_trend(
                    request.user_id,
                    limit=20,
                )
                historical_high_risk_count = sum(
                    1
                    for event in recent_mood_events
                    if risk_level_band(getattr(event, "risk_level", LEVEL_0)) == "high"
                )
            except Exception as e:
                logger.warning(f"加载历史风险事件失败，按无历史高风险继续: {e}")

            # 1.2 Risk Assessment (Pre-check)
            preliminary_emotion_state = emotion_analyzer.build_emotion_state_payload(
                text=request.text,
                current_emotion=current_emotion,
                context_emotion=context_emotion,
                confidence=confidence,
                conversation_summary=conversation_summary,
            )
            # v6.0: 加载长期风险基线
            risk_baseline = "low"
            try:
                memory = await RiskMemoryReader().get_baseline(request.user_id)
                risk_baseline = memory.get("baseline", "low")
            except Exception:
                pass

            urgent_issue = risk_evaluator.evaluate(
                text=request.text,
                emotion_state=preliminary_emotion_state,
                conversation_summary=conversation_summary,
                long_term_risk_level=long_term_risk_level,
                historical_high_risk_count=historical_high_risk_count,
                risk_baseline=risk_baseline,
            )
            # ── Trace: 开始计时 ──
            _trace_start = datetime.now(timezone.utc)

            recommendation_decision = recommend_gate.decide(
                emotion_state=preliminary_emotion_state,
                risk_state=urgent_issue,
                conversation_summary=conversation_summary,
                user_profile=user_profile,
            )

            # ── Trace: 门控阶段 ──
            _risk_level_for_trace = str(urgent_issue.get("level", "level_0")) if urgent_issue else "level_0"
            _recommend_trace = TraceEvent(
                request_id=getattr(request, "request_id", ""),
                user_id=request.user_id,
                session_id=request.session_id,
                turn_id=conversation_summary.get("turn_count", 0),
                gate_inputs={
                    "emotion_intensity": float(preliminary_emotion_state.get("emotion_intensity", 0) or 0),
                    "risk_score": {"level_0": 0.0, "level_1": 0.4, "level_2": 0.75, "level_3": 1.0}.get(
                        _risk_level_for_trace, 0.0
                    ),
                    "intent_score": {"sharing": 0.0, "seeking_relief": 0.45, "planning": 0.55, "seeking_help": 0.75}.get(
                        preliminary_emotion_state.get("user_intent", "sharing"), 0.0
                    ),
                    "trend_score": 0.35 if preliminary_emotion_state.get("negative_trend") else 0.0,
                    "preference_score": 0.0,
                },
                emotion_state={
                    "current_emotion": preliminary_emotion_state.get("current_emotion", ""),
                    "emotion_intensity": preliminary_emotion_state.get("emotion_intensity", 0),
                    "user_intent": preliminary_emotion_state.get("user_intent", ""),
                    "negative_trend": preliminary_emotion_state.get("negative_trend", False),
                },
                risk_state={
                    "level": str(urgent_issue.get("level", "level_0")) if urgent_issue else "level_0",
                    "risk_score": urgent_issue.get("risk_score", 0) if urgent_issue else 0,
                },
                conversation_summary={
                    "turn_count": conversation_summary.get("turn_count", 0),
                    "stage": conversation_summary.get("conversation_stage", "initial"),
                    "risk_expressions": conversation_summary.get("risk_expressions", False),
                },
                user_profile={
                    "risk_level": user_profile.get("risk_level", "low"),
                    "preferred_types": user_profile.get("preferred_types", [])[:3],
                },
                gate_output={
                    "should_recommend": recommendation_decision.get("should_recommend", False),
                    "recommend_type": recommendation_decision.get("recommend_type", "none"),
                    "score": recommendation_decision.get("score", 0.0),
                    "threshold": recommendation_decision.get("threshold", 0.0),
                    "reason_codes": recommendation_decision.get("reason_codes", []),
                    "cooldown_remaining": recommendation_decision.get("cooldown_remaining", 0),
                },
            )
            steps.append(AgentStep(
                name="Initialization",
                description="Loaded session and performed risk assessment",
                started_at=step_start,
                finished_at=datetime.now(timezone.utc),
                tool_calls=[]
            ))

            if urgent_issue and is_emergency_risk(urgent_issue.get("level")):
                safety_step_start = datetime.now(timezone.utc)
                final_response_text = await urgent_detector.generate_crisis_response_async(
                    user_input=request.text,
                    urgent_issue=urgent_issue,
                    conversation_summary=conversation_summary,
                )
                steps.append(
                    AgentStep(
                        name="SafetyResponse",
                        description="High-risk input routed to dedicated safety response",
                        started_at=safety_step_start,
                        finished_at=datetime.now(timezone.utc),
                        tool_calls=[],
                    )
                )
            elif self._is_third_party_crisis_help_request(urgent_issue):
                support_step_start = datetime.now(timezone.utc)
                final_response_text = await urgent_detector.generate_third_party_support_response_async(
                    user_input=request.text,
                    urgent_issue=urgent_issue,
                    conversation_summary=conversation_summary,
                )
                recommendation_decision = {
                    "should_recommend": False,
                    "recommend_type": "third_party_support",
                    "score": 0.0,
                    "threshold": 0.0,
                    "reason_codes": ["third_party_crisis_support_route"],
                    "cooldown_remaining": 0,
                }
                steps.append(
                    AgentStep(
                        name="ThirdPartyCrisisSupport",
                        description="Third-party crisis help request routed to dedicated support guidance",
                        started_at=support_step_start,
                        finished_at=datetime.now(timezone.utc),
                        tool_calls=[],
                    )
                )
            elif urgent_issue and normalize_risk_level(urgent_issue.get("level")) == LEVEL_2:
                support_step_start = datetime.now(timezone.utc)
                final_response_text = await urgent_detector.generate_crisis_response_async(
                    user_input=request.text,
                    urgent_issue=urgent_issue,
                    conversation_summary=conversation_summary,
                )
                steps.append(
                    AgentStep(
                        name="HighRiskSupport",
                        description="Level-2 input routed to dedicated high-risk support mode",
                        started_at=support_step_start,
                        finished_at=datetime.now(timezone.utc),
                        tool_calls=[],
                    )
                )
            else:

                # 2. Build Messages
                messages = await self._build_initial_messages(
                    request.text, 
                    session.get("history", []), 
                    conversation_summary,
                    urgent_issue,
                    user_id=request.user_id,
                    emotion_state=preliminary_emotion_state,
                    user_profile=user_profile,
                    recommendation_decision=recommendation_decision,
                )
                
                # 3. Agent Loop
                loop_step = AgentStep(
                    name="AgentLoop",
                    description="LLM reasoning and tool execution loop",
                    started_at=datetime.now(timezone.utc),
                    finished_at=datetime.now(timezone.utc),
                    tool_calls=[]
                )
                
                max_turns = 5
                turn = 0
                
                while turn < max_turns:
                    turn += 1
                    try:
                        # Call LLM
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=TOOL_DEFINITIONS,
                            tool_choice="auto",
                            temperature=0.7
                        )
                        
                        message = response.choices[0].message
                        # 统一消息格式为 dict（避免 SDK 对象与 dict 混用）
                        if hasattr(message, 'model_dump'):
                            message_dict = message.model_dump()
                        else:
                            message_dict = {"role": "assistant", "content": message.content}
                        messages.append(message_dict)

                        # Check for tool calls（使用 SDK 对象执行工具，保持 _execute_single_tool 兼容）
                        if message.tool_calls:
                            # Parallel tool execution
                            tasks = [
                                self._execute_single_tool(
                                    tc,
                                    conversation_summary,
                                    request.text,
                                    request.user_id,
                                    request.session_id,
                                    recommendation_decision=recommendation_decision,
                                    user_profile=user_profile,
                                )
                                for tc in message.tool_calls
                            ]
                            results = await asyncio.gather(*tasks)
                            
                            for res in results:
                                # Record tool call in step
                                loop_step.tool_calls.append(res["tool_step_info"])
                                
                                # Add tool result to messages
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": res["tool_call_id"],
                                    "name": res["name"],
                                    "content": res["content"]
                                })
                        
                        else:
                            # No tool calls, this is the final response
                            final_response_text = message.content
                            break
                            
                    except Exception as e:
                        logger.error(f"Agent loop error: {e}", exc_info=True)
                        final_response_text = "抱歉，我现在遇到了一些技术问题，请稍后再试。"
                        break

                loop_step.finished_at = datetime.now(timezone.utc)
                steps.append(loop_step)

            # 4. Save & Post-processing
            # Save to conversation history using add_interaction (which handles stats, persistence, etc.)
            conversation_manager.add_interaction(
                user_id=request.user_id,
                session_id=request.session_id,
                user_input=request.text,
                emotion=current_emotion,
                context_emotion=context_emotion,
                confidence=confidence,
                ai_response=final_response_text,
                emotion_state=preliminary_emotion_state,
                risk_state=urgent_issue,
            )
            updated_conversation_summary = await conversation_manager.get_conversation_summary_async(
                request.user_id, request.session_id
            )

            # v4.5: 拒绝推荐检测（影响下一轮门控决策）
            rejection = detect_rejection(request.text, _recommend_trace.recommendation_ids)
            if rejection["has_rejected"]:
                updated_conversation_summary["has_rejected_recommendation"] = True
                _recommend_trace.gate_inputs["has_rejected_recommendation"] = 1

            # 4.1 Urgent Case Logging
            if urgent_issue and is_non_low_risk(urgent_issue.get("level")):
                interaction_data = {
                    'user_id': request.user_id,
                    'session_id': request.session_id,
                    'urgent_issue': urgent_issue,
                    'user_input': request.text,
                    'emotion': current_emotion,
                    'ai_response': final_response_text
                }
                # Fire and forget (async)
                asyncio.create_task(urgent_logger.log_interaction_async(interaction_data))

            # Extract recommendations
            final_recommendations, final_rationale = self._extract_recommendations(messages)
            if recommendation_decision and recommendation_decision.get("recommend_type") != "hard":
                final_recommendations = []
                final_rationale = ""
            elif recommendation_decision and recommendation_decision.get("recommend_type") == "hard" and not final_recommendations:
                final_recommendations, final_rationale, _ = await ContentRecommendTool.recommend(
                    user_input=request.text,
                    current_emotion=current_emotion,
                    conversation_summary=updated_conversation_summary or conversation_summary,
                    user_profile=user_profile,
                    limit=2,
                )
            # ── Trace: 推荐结果阶段 ──
            has_rerank = getattr(content_recommender, "enable_ai_rerank", False)
            _recommend_trace.recommendation_ids = [item.id for item in (final_recommendations or [])]
            _recommend_trace.recommendation_scores = [
                float(item.retrieval_metadata.get("final_score", 0)) if item.retrieval_metadata and hasattr(item, "retrieval_metadata") else 0.0
                for item in (final_recommendations or [])
            ]
            _recommend_trace.rerank_used = has_rerank
            _recommend_trace.rerank_success = has_rerank
            _recommend_trace.candidate_count = len(final_recommendations) if final_recommendations else 0

            safety_check_started = datetime.now(timezone.utc)
            safety_review = output_safety_checker.review(
                response_text=final_response_text,
                risk_state=urgent_issue,
                recommendation_decision=recommendation_decision,
                recommendations=final_recommendations,
                recommendation_rationale=final_rationale,
            )
            final_response_text = safety_review["response_text"]
            final_recommendations = safety_review["recommendations"]
            final_rationale = safety_review["recommendation_rationale"]
            recommendation_decision = safety_review["recommendation_decision"]
            steps.append(
                AgentStep(
                    name="OutputSafetyCheck",
                    description=(
                        "Applied output safety fallback rules"
                        if safety_review["triggered_rules"]
                        else "Validated final response against output safety rules"
                    ),
                    started_at=safety_check_started,
                    finished_at=datetime.now(timezone.utc),
                    tool_calls=[],
                )
            )

            # ── Trace: 安全与持久化阶段 ──
            _recommend_trace.safety_overridden = bool(safety_review.get("triggered_rules", False))
            _recommend_trace.safety_before = {
                "should_recommend": recommendation_decision.get("should_recommend", False) if recommendation_decision else False,
            }
            _recommend_trace.safety_after = {
                "should_recommend": recommendation_decision.get("should_recommend", False) if recommendation_decision else False,
            }
            _recommend_trace.persisted = bool(
                final_recommendations and recommendation_decision and recommendation_decision.get("should_recommend")
            )

            if final_recommendations and recommendation_decision and recommendation_decision.get("should_recommend"):
                conversation_manager.mark_recommendation(
                    request.user_id,
                    request.session_id,
                    recommendation_decision.get("recommend_type", "soft"),
                    [item.id for item in final_recommendations],
                    trace_data={
                        "gate_score": recommendation_decision.get("score"),
                        "gate_threshold": recommendation_decision.get("threshold"),
                        "reason_codes": recommendation_decision.get("reason_codes", []),
                        "cooldown_remaining": recommendation_decision.get("cooldown_remaining", 0),
                        "safety_overridden": bool(safety_review.get("triggered_rules", False)),
                    },
                )

            # ── Trace: 最终写入 ──
            _recommend_trace.latency_ms = int((datetime.now(timezone.utc) - _trace_start).total_seconds() * 1000)
            write_trace(_recommend_trace)

            # v6.0: 写入长期风险记忆
            try:
                await RiskMemoryWriter().update(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    risk_state=urgent_issue,
                    conversation_summary=updated_conversation_summary or conversation_summary or {},
                    emotion_state=preliminary_emotion_state,
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Critical error in run_agent: {e}", exc_info=True)
            if not final_response_text:
                final_response_text = "抱歉，系统暂时无法处理您的请求。"

        final_summary = updated_conversation_summary or conversation_summary or self._default_session_summary()
        emotion_state = self._build_emotion_state(
            request_text=request.text,
            current_emotion=current_emotion,
            context_emotion=context_emotion,
            confidence=confidence,
            conversation_summary=final_summary,
        )
        risk_state = self._build_risk_state(urgent_issue)
        session_summary = self._build_session_summary(final_summary, current_emotion)

        chat_response = ChatResponse(
            response=final_response_text,
            emotion_state=emotion_state,
            risk_state=risk_state,
            session_summary=session_summary,
            recommendation_decision=RecommendationDecision(**(recommendation_decision or {})),
            emotion_summary=self._build_legacy_emotion_summary(emotion_state, session_summary),
            urgent_issue=risk_state.model_dump(),
            recommendations=final_recommendations if final_recommendations else None,
            recommendation_rationale=final_rationale if final_rationale else None
        )

        return AgentRunResponse(
            run_id=run_id,
            chat=chat_response,
            steps=steps if request.return_steps else None
        )

    def _extract_recommendations(self, messages: List[Any]) -> tuple[List[ContentItem], str]:
        """Extract structured recommendations from tool outputs in message history."""
        final_recommendations = []
        final_rationale = ""
        
        for msg in reversed(messages):
            # Handle both dict and object message formats
            if isinstance(msg, dict):
                role = msg.get("role")
                name = msg.get("name")
                content = msg.get("content")
            else:
                role = getattr(msg, "role", None)
                # Helper for object (ChatCompletionMessage)
                # But typically tool OUTPUTS are appended as dicts in this code
                name = None 
                content = getattr(msg, "content", None)
            
            if role == "tool" and name == "recommend_content":
                try:
                    data = json.loads(content)
                    if "recommendations" in data:
                         for item in data["recommendations"]:
                             final_recommendations.append(ContentItem(**item))
                    if "rationale" in data:
                        final_rationale = data["rationale"]
                    break
                except Exception as e:
                    logger.warning(f"Failed to parse recommendation output: {e}")
        
        return final_recommendations, final_rationale

    def _default_session_summary(self) -> Dict[str, Any]:
        return {
            "conversation_stage": "initial",
            "primary_emotion": "中性",
            "emotion_trend": "new",
            "key_concerns": [],
            "turn_count": 0,
        }

    def _build_emotion_state(
        self,
        request_text: str,
        current_emotion: str,
        context_emotion: str,
        confidence: float,
        conversation_summary: Dict[str, Any],
    ) -> EmotionState:
        payload = emotion_analyzer.build_emotion_state_payload(
            text=request_text,
            current_emotion=current_emotion,
            context_emotion=context_emotion,
            confidence=confidence,
            conversation_summary=conversation_summary,
        )
        return EmotionState(**payload)

    def _build_risk_state(self, urgent_issue: Optional[Dict[str, Any]]) -> RiskState:
        issue = urgent_issue or {
            "level": LEVEL_0,
            "message": "",
            "suggestions": [],
            "triggers": [],
            "risk_score": 0.0,
        }
        canonical_level = normalize_risk_level(issue.get("level", LEVEL_0))
        return RiskState(
            level=canonical_level,
            legacy_level=issue.get("legacy_level", risk_level_band(canonical_level)),
            level_index=issue.get("level_index", risk_level_index(canonical_level)),
            level_label=issue.get("level_label", risk_level_label(canonical_level)),
            message=issue.get("message", ""),
            suggestions=issue.get("suggestions", []),
            triggers=issue.get("triggers", []),
            risk_score=issue.get("risk_score", 0.0),
            raw_score=issue.get("raw_score", issue.get("risk_score", 0.0)),
            risk_dimensions=issue.get("risk_dimensions", {}),
            risk_evidence=issue.get("risk_evidence", {}),
            escalation_reasons=issue.get("escalation_reasons", []),
            risk_context=issue.get("risk_context", {}),
        )

    def _is_third_party_crisis_help_request(self, urgent_issue: Optional[Dict[str, Any]]) -> bool:
        context = (urgent_issue or {}).get("risk_context", {})
        return (
            context.get("subject") == "third_party"
            and context.get("is_help_request") is True
        )

    def _build_session_summary(
        self, conversation_summary: Dict[str, Any], current_emotion: str
    ) -> SessionSummary:
        return SessionSummary(
            conversation_stage=conversation_summary.get("conversation_stage", "initial"),
            key_concerns=conversation_summary.get("key_concerns", []),
            turn_count=conversation_summary.get("turn_count", 0),
            emotion_trend=conversation_summary.get("emotion_trend"),
            primary_emotion=conversation_summary.get("primary_emotion", current_emotion),
        )

    def _build_legacy_emotion_summary(
        self, emotion_state: EmotionState, session_summary: SessionSummary
    ) -> Dict[str, Any]:
        return {
            "current_emotion": emotion_state.current_emotion,
            "emotion_type": emotion_state.emotion_type,
            "context_emotion": emotion_state.context_emotion,
            "emotion_intensity": emotion_state.emotion_intensity,
            "stress_source": emotion_state.stress_source,
            "user_intent": emotion_state.user_intent,
            "negative_trend": emotion_state.negative_trend,
            "confidence": emotion_state.confidence,
            "conversation_stage": session_summary.conversation_stage,
            "key_concerns": session_summary.key_concerns,
            "turn_count": session_summary.turn_count,
            "emotion_trend": session_summary.emotion_trend,
            "primary_emotion": session_summary.primary_emotion,
        }

    # --- Tool Wrappers ---

    async def _tool_search_knowledge_base(self, query: str, limit: int = 3, **kwargs):
        res = await KnowledgeBaseTool.search(KnowledgeBaseQuery(query=query, limit=limit))
        return [doc.model_dump() for doc in res.documents]

    async def _tool_get_user_profile(self, user_id: str, **kwargs):
        profile = await UserProfileTool.get_profile(user_id)
        return profile.model_dump() if profile else None

    async def _tool_update_user_profile(self, user_id: str, **kwargs):
        # kwargs match upsert_profile args
        profile = await UserProfileTool.upsert_profile(user_id, **kwargs)
        return profile.model_dump()

    async def _tool_log_mood_event(self, user_id: str, session_id: str, emotion: str, text_snippet: str, source: str = "agent", **kwargs):
        event = await MoodTrackingTool.log_event(user_id, session_id, emotion, source, text_snippet)
        return event.model_dump()

    async def _tool_get_recent_mood_trend(self, user_id: str, limit: int = 10, **kwargs):
        events = await MoodTrackingTool.get_recent_trend(user_id, limit=limit)
        return [event.model_dump() for event in events]

    async def _tool_recommend_content(self, user_input: str, current_emotion: str, conversation_summary: Dict, limit: int = 2, **kwargs):
        items, rationale, scores = await ContentRecommendTool.recommend(
            user_input,
            current_emotion,
            conversation_summary,
            user_profile=kwargs.get("user_profile"),
            limit=limit,
        )
        # Serialize ContentItems
        items_dict = [item.model_dump() for item in items]
        return {
            "recommendations": items_dict,
            "rationale": rationale,
            "match_scores": scores
        }

    async def _inject_knowledge_context(
        self,
        user_input: str,
        emotion_state: Optional[Dict[str, Any]] = None,
        risk_level: str = "low",
    ) -> str:
        """根据用户输入 + 情绪 + 风险等级，检索相关知识并构建注入文本"""
        try:
            if not knowledge_store.is_loaded:
                knowledge_store.load_corpus()

            if not knowledge_store.is_loaded:
                return ""

            # 构建查询
            emotion = emotion_state.get("current_emotion", "") if emotion_state else ""
            query_text = f"{user_input} {emotion}".strip()
            if not query_text:
                return ""

            # 检索知识
            result = knowledge_store.search(KBQuery(
                query=query_text,
                top_k=3,
            ))
            if not result.chunks:
                return ""

            # 构建注入文本
            lines = ["下面是一些与当前情境相关的心理学专业知识（参考来源已标注）："]
            for c in result.chunks:
                ref = c.metadata.get("source_reference", "")
                ref_str = f" —— {ref}" if ref else ""
                content_preview = c.content[:200].replace("\n", " ").strip()
                lines.append(f"- [{c.title}]{ref_str}\n  {content_preview}...")
            lines.append("（注意：这些知识供你参考，请根据用户具体情况进行适配，不要直接复述内容。）")

            return "\n\n" + "\n\n".join(lines)
        except Exception as e:
            logger.warning(f"知识注入失败: {e}")
            return ""

    async def _build_initial_messages(
        self,
        text: str,
        history: List[Dict],
        summary: Dict,
        urgent_issue: Dict,
        user_id: Optional[str] = None,
        emotion_state: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        recommendation_decision: Optional[Dict[str, Any]] = None,
    ):
        # System Prompt
        summary = summary or {}
        urgent_issue = urgent_issue or {}
        emotion_state = emotion_state or {}
        user_profile = user_profile or {}
        stage = summary.get('conversation_stage', 'initial')
        strategy_guidance = self.strategy_prompts.get(stage, self.strategy_prompts['initial'])
        
        # Sanitize key_concerns
        key_concerns = summary.get('key_concerns', [])
        if isinstance(key_concerns, list):
            key_concerns_str = ', '.join([str(k).replace('{', '{{').replace('}', '}}') for k in key_concerns])
        else:
            key_concerns_str = str(key_concerns).replace('{', '{{').replace('}', '}}')

        risk_level = urgent_issue.get('level', 'low')

        # 转义 strategy_guidance 中的花括号，防止 .format() 崩溃
        strategy_guidance = strategy_guidance.replace('{', '{{').replace('}', '}}')

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            stage=stage,
            key_concerns=key_concerns_str,
            risk_level=risk_level,
            strategy_guidance=strategy_guidance
        )

        # Phase 3a: 使用 MemoryContextBuilder（含 BM25 检索 + token budget 注入）
        try:
            from memory_context_builder import memory_context_builder as _mcb
            _memory_ctx = await _mcb.build(
                user_id=user_id,
                user_input=text,
                emotion_state=emotion_state,
                risk_state=urgent_issue,
                conversation_summary=summary,
            )
            relevant_memory_context = f"（记忆注入: {_memory_ctx.used_tokens}/{int(config.MAX_CONTEXT_TOKENS * config.MEMORY_INJECTION_BUDGET_RATIO)} tokens, {len(_memory_ctx.included_memory_ids)} 条注入）\n"
            if _memory_ctx.text:
                relevant_memory_context += _memory_ctx.text
        except Exception as e:
            logger.warning("MemoryContextBuilder 失败，回退旧路径: %s", e)
            relevant_memory_context = await self._build_relevant_memory_context(
                text=text,
                user_id=user_id,
                summary=summary,
                emotion_state=emotion_state,
                user_profile=user_profile,
            )

        prompt_sections = [
            ("会话摘要", self._format_session_summary_for_prompt(summary)),
            ("压缩上下文", self._safe_prompt_text(summary.get("compressed_context"), "暂无压缩上下文")),
            ("用户长期画像", self._format_user_profile_for_prompt(user_profile)),
            ("相关长期记忆", relevant_memory_context),
            ("当前情绪状态", self._format_emotion_state_for_prompt(emotion_state)),
            ("当前风险状态", self._format_risk_state_for_prompt(urgent_issue)),
        ]

        # Phase 3b: 知识库注入（专业知识增强 RAG）
        try:
            _knowledge_ctx = await self._inject_knowledge_context(
                user_input=text,
                emotion_state=emotion_state,
                risk_level=urgent_issue.get("level", "low"),
            )
            if _knowledge_ctx:
                prompt_sections.append(("专业知识参考", _knowledge_ctx))
        except Exception as e:
            logger.debug(f"知识注入跳过: {e}")

        for section_title, section_content in prompt_sections:
            system_prompt += f"\n\n【{section_title}】\n{section_content}"

        decision = recommendation_decision or {}
        recommend_type = decision.get("recommend_type", "none")
        system_prompt += (
            f"\n推荐门控结果：should_recommend={decision.get('should_recommend', False)}"
            f", recommend_type={recommend_type}, reason_codes={','.join(decision.get('reason_codes', []))}"
        )
        if recommend_type == "hard":
            system_prompt += "\n当前允许在合适时机调用 recommend_content，给出明确、可执行的工具或内容。"
        elif recommend_type == "soft":
            system_prompt += "\n当前仅允许在自然语言回复中给出软建议，不要调用 recommend_content。"
        else:
            system_prompt += "\n当前不应触发推荐内容，请专注于支持性对话。"
        
        messages = [{"role": "system", "content": system_prompt}]
        
        history_window = 6 if summary.get("compressed_context") else 10

        # Add history (keep a smaller raw window when compressed summary exists)
        for h in history[-history_window:]:
            # Adapter for conversation_manager's history format
            if "user_input" in h:
                messages.append({"role": "user", "content": h["user_input"]})
            if "ai_response" in h and h["ai_response"]:
                messages.append({"role": "assistant", "content": h["ai_response"]})
            
            # Fallback for standard OpenAI format (if mixed)
            if "role" in h and "content" in h:
                 messages.append({"role": h["role"], "content": h["content"]})
            
        # Add current input
        messages.append({"role": "user", "content": text})
        
        return messages

    async def _build_relevant_memory_context(
        self,
        text: str,
        user_id: Optional[str],
        summary: Dict[str, Any],
        emotion_state: Dict[str, Any],
        user_profile: Dict[str, Any],
    ) -> str:
        focus_terms = self._collect_memory_focus_terms(text, summary, emotion_state)
        lines: List[str] = []

        if focus_terms:
            lines.append(
                f"- 记忆检索焦点: {self._safe_prompt_text(', '.join(focus_terms[:6]), '无')}"
            )

        relevant_sources = [
            source
            for source in (user_profile.get("main_stress_sources", []) or [])
            if self._memory_term_matches(source, focus_terms)
        ]
        if not relevant_sources:
            relevant_sources = (user_profile.get("main_stress_sources", []) or [])[:2]
        if relevant_sources:
            lines.append(
                f"- 相关长期压力源: {self._safe_prompt_text(', '.join(relevant_sources[:3]), '无')}"
            )

        support_preferences = []
        preferred_support_style = user_profile.get("preferred_support_style")
        avoid_style = user_profile.get("avoid_style", []) or []
        if preferred_support_style:
            support_preferences.append(f"偏好 {preferred_support_style}")
        if avoid_style:
            support_preferences.append(f"避免 {', '.join(avoid_style[:3])}")
        if support_preferences:
            lines.append(
                f"- 稳定支持偏好: {self._safe_prompt_text('；'.join(support_preferences), '无')}"
            )

        feedback_summary = self._format_recommendation_feedback_memory(
            user_profile.get("recommendation_feedback", {}) or {}
        )
        if feedback_summary:
            lines.append(f"- 历史推荐反馈: {feedback_summary}")

        recent_events = []
        if user_id:
            try:
                recent_events = await MoodTrackingTool.get_recent_trend(user_id, limit=12)
            except Exception as e:
                logger.warning(f"加载近期情绪事件失败，跳过相关记忆检索: {e}")

        relevant_events = self._select_relevant_mood_events(
            recent_events,
            focus_terms=focus_terms,
            emotion_state=emotion_state,
            summary=summary,
        )
        for index, event in enumerate(relevant_events, start=1):
            lines.append(f"- 相似经历{index}: {self._format_mood_event_for_memory(event)}")

        return "\n".join(lines) if lines else "暂无明显相关的长期记忆"

    def _collect_memory_focus_terms(
        self,
        text: str,
        summary: Dict[str, Any],
        emotion_state: Dict[str, Any],
    ) -> List[str]:
        concern_aliases = {
            "academic": ["学业", "求职", "面试"],
            "relationship": ["关系", "人际", "朋友", "伴侣"],
            "future": ["未来", "规划", "求职"],
            "self": ["自我评价", "能力", "自信"],
        }
        raw_terms: List[str] = []
        raw_terms.extend(self._split_memory_terms(summary.get("current_topic")))
        raw_terms.extend(self._split_memory_terms(summary.get("stress_sources", [])))
        raw_terms.extend(self._split_memory_terms(emotion_state.get("stress_source")))
        for concern in summary.get("key_concerns", []) or []:
            raw_terms.extend(concern_aliases.get(concern, []))
        if text and len(text) <= 24:
            raw_terms.extend(self._split_memory_terms(text))

        unique_terms: List[str] = []
        seen = set()
        for term in raw_terms:
            normalized = self._normalize_memory_term(term)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_terms.append(term.strip())
        return unique_terms[:8]

    def _split_memory_terms(self, value: Any) -> List[str]:
        if value in (None, "", [], {}):
            return []
        if isinstance(value, list):
            terms: List[str] = []
            for item in value:
                terms.extend(self._split_memory_terms(item))
            return terms

        text = str(value).strip()
        if not text:
            return []
        parts = [part.strip() for part in re.split(r"[，,、/；;\s]+|与|和", text) if part.strip()]
        candidates = [text]
        candidates.extend(parts)
        return candidates

    def _normalize_memory_term(self, term: Any) -> str:
        if term in (None, ""):
            return ""
        normalized = str(term).strip().lower()
        generic_terms = {"日常交流", "初始对话", "initial", "exploring", "deepening", "resolving"}
        if normalized in generic_terms or len(normalized) < 2:
            return ""
        return normalized

    def _memory_term_matches(self, text: Any, focus_terms: List[str]) -> bool:
        if text in (None, "") or not focus_terms:
            return False
        normalized_text = self._normalize_memory_term(text)
        if not normalized_text:
            return False
        return any(
            normalized_term in normalized_text or normalized_text in normalized_term
            for normalized_term in (self._normalize_memory_term(term) for term in focus_terms)
            if normalized_term
        )

    def _format_recommendation_feedback_memory(self, feedback_map: Dict[str, str]) -> str:
        accepted = [
            content_id
            for content_id, feedback in feedback_map.items()
            if feedback in {"accepted", "preferred", "helpful"}
        ]
        rejected = [
            content_id
            for content_id, feedback in feedback_map.items()
            if feedback in {"rejected", "not_helpful", "avoid"}
        ]
        parts = []
        if accepted:
            parts.append(f"更可能接受 {', '.join(accepted[-2:])}")
        if rejected:
            parts.append(f"明确拒绝 {', '.join(rejected[-2:])}")
        return self._safe_prompt_text("；".join(parts), "")

    def _select_relevant_mood_events(
        self,
        events: List[Any],
        focus_terms: List[str],
        emotion_state: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[Any]:
        target_emotions = {
            str(value).strip()
            for value in [
                emotion_state.get("current_emotion"),
                emotion_state.get("emotion_type"),
                summary.get("primary_emotion"),
            ]
            if value
        }
        scored_events = []
        for event in events:
            score = self._score_mood_event_relevance(event, focus_terms, target_emotions)
            if score > 0:
                scored_events.append((score, getattr(event, "created_at", datetime.min), event))

        scored_events.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = []
        seen_signatures = set()
        for _, _, event in scored_events:
            signature = (
                getattr(event, "stress_source", None),
                getattr(event, "event_summary", None),
                getattr(event, "text_snippet", None),
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            selected.append(event)
            if len(selected) >= 3:
                break
        return selected

    def _score_mood_event_relevance(
        self,
        event: Any,
        focus_terms: List[str],
        target_emotions: set[str],
    ) -> int:
        score = 0
        stress_source = getattr(event, "stress_source", None)
        if self._memory_term_matches(stress_source, focus_terms):
            score += 3

        searchable_text = " ".join(
            str(value)
            for value in [
                getattr(event, "event_summary", None),
                getattr(event, "text_snippet", None),
                getattr(event, "user_intent", None),
            ]
            if value
        )
        if self._memory_term_matches(searchable_text, focus_terms):
            score += 2

        event_emotions = {
            str(value).strip()
            for value in [getattr(event, "emotion", None), getattr(event, "emotion_type", None)]
            if value
        }
        if target_emotions.intersection(event_emotions):
            score += 1

        if is_non_low_risk(getattr(event, "risk_level", LEVEL_0)):
            score += 1
        return score

    def _format_mood_event_for_memory(self, event: Any) -> str:
        emotion = getattr(event, "emotion", None) or getattr(event, "emotion_type", "中性")
        stress_source = getattr(event, "stress_source", None)
        risk_level = normalize_risk_level(getattr(event, "risk_level", LEVEL_0))
        summary_text = getattr(event, "event_summary", None) or getattr(event, "text_snippet", "") or "无摘要"
        summary_text = summary_text[:80]
        parts = [f"情绪={emotion}"]
        if stress_source:
            parts.append(f"主题={stress_source}")
        if risk_level != LEVEL_0:
            parts.append(f"风险={risk_level}")
        return f"{'，'.join(parts)}；{self._safe_prompt_text(summary_text, '无摘要')}"

    def _safe_prompt_text(self, value: Any, fallback: str = "无") -> str:
        if value in (None, "", [], {}):
            return fallback
        return str(value).replace("{", "{{").replace("}", "}}")

    def _format_session_summary_for_prompt(self, summary: Dict[str, Any]) -> str:
        recent_intents = summary.get("recent_intents", []) or []
        stress_sources = summary.get("stress_sources", []) or []
        accepted = summary.get("accepted_recommendations", []) or []
        rejected = summary.get("rejected_recommendations", []) or []
        lines = [
            f"- 对话阶段: {self._safe_prompt_text(summary.get('conversation_stage'), 'initial')}",
            f"- 主要情绪: {self._safe_prompt_text(summary.get('primary_emotion'), '中性')}",
            f"- 情绪趋势: {self._safe_prompt_text(summary.get('emotion_trend'), 'new')}",
            f"- 关键关切: {self._safe_prompt_text(', '.join(summary.get('key_concerns', []) or []), '无')}",
            f"- 当前主题: {self._safe_prompt_text(summary.get('current_topic'), '日常交流')}",
            f"- 最近意图: {self._safe_prompt_text(', '.join(recent_intents), '无')}",
            f"- 压力来源: {self._safe_prompt_text(', '.join(stress_sources), '无')}",
            f"- 已接受推荐: {self._safe_prompt_text(', '.join(accepted), '无')}",
            f"- 已拒绝推荐: {self._safe_prompt_text(', '.join(rejected), '无')}",
        ]
        return "\n".join(lines)

    def _format_user_profile_for_prompt(self, user_profile: Dict[str, Any]) -> str:
        if not user_profile:
            return "暂无长期画像信息"
        lines = [
            f"- 长期风险等级: {self._safe_prompt_text(user_profile.get('risk_level'), LEVEL_0)}",
            f"- 偏好支持风格: {self._safe_prompt_text(user_profile.get('preferred_support_style'), '未知')}",
            f"- 避免风格: {self._safe_prompt_text(', '.join(user_profile.get('avoid_style', []) or []), '无')}",
            f"- 主要压力来源: {self._safe_prompt_text(', '.join(user_profile.get('main_stress_sources', []) or []), '无')}",
            f"- 推荐反馈画像: {self._safe_prompt_text(json.dumps(user_profile.get('recommendation_feedback', {}) or {}, ensure_ascii=False), '{}')}",
        ]
        return "\n".join(lines)

    def _format_emotion_state_for_prompt(self, emotion_state: Dict[str, Any]) -> str:
        if not emotion_state:
            return "暂无结构化情绪状态"
        lines = [
            f"- 当前情绪: {self._safe_prompt_text(emotion_state.get('current_emotion'), '中性')}",
            f"- 情绪类型: {self._safe_prompt_text(emotion_state.get('emotion_type'), 'neutral')}",
            f"- 上下文情绪: {self._safe_prompt_text(emotion_state.get('context_emotion'), '无')}",
            f"- 情绪强度: {self._safe_prompt_text(emotion_state.get('emotion_intensity'), '0.5')}",
            f"- 用户意图: {self._safe_prompt_text(emotion_state.get('user_intent'), 'sharing')}",
            f"- 压力来源: {self._safe_prompt_text(emotion_state.get('stress_source'), '无')}",
            f"- 负面趋势: {self._safe_prompt_text(emotion_state.get('negative_trend'), 'False')}",
        ]
        return "\n".join(lines)

    def _format_risk_state_for_prompt(self, risk_state: Dict[str, Any]) -> str:
        if not risk_state:
            return "暂无风险状态"
        canonical_level = normalize_risk_level(risk_state.get("level", LEVEL_0))
        lines = [
            f"- 风险等级: {self._safe_prompt_text(canonical_level, LEVEL_0)}",
            f"- 兼容等级: {self._safe_prompt_text(risk_state.get('legacy_level', risk_level_band(canonical_level)), 'low')}",
            f"- 风险提示: {self._safe_prompt_text(risk_state.get('message'), '无')}",
            f"- 触发信号: {self._safe_prompt_text(', '.join(risk_state.get('triggers', []) or []), '无')}",
            f"- 风险分数: {self._safe_prompt_text(risk_state.get('risk_score'), '0.0')}",
        ]
        return "\n".join(lines)

agent_orchestrator = AgentOrchestrator()
