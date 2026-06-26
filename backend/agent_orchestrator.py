import logging
import json
import time
import asyncio
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

from config import config
from conversation_manager import conversation_manager
from urgent_detector import urgent_detector, urgent_logger
from emotion_analyzer import emotion_analyzer
from models import (
    AgentRunRequest,
    AgentRunResponse,
    AgentStep,
    AgentToolCall,
    ChatResponse,
    ContentItem
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

    async def _execute_single_tool(self, tool_call, conversation_summary, request_text, user_id, session_id):
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
            arguments["conversation_summary"] = conversation_summary
            if "user_input" not in arguments:
                arguments["user_input"] = request_text
            if "current_emotion" not in arguments:
                arguments["current_emotion"] = conversation_summary.get("primary_emotion", "中性")

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
        start_time = time.time()
        
        # Initialize default values for error handling
        current_emotion = "中性"
        context_emotion = "中性"
        confidence = 0.5
        urgent_issue = None
        final_response_text = ""
        final_recommendations = []
        final_rationale = ""

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
            
            # 1.2 Risk Assessment (Pre-check)
            urgent_issue = urgent_detector.detect(request.text, current_emotion)
            
            steps.append(AgentStep(
                name="Initialization",
                description="Loaded session and performed risk assessment",
                started_at=step_start,
                finished_at=datetime.now(timezone.utc),
                tool_calls=[]
            ))

            # 2. Build Messages
            messages = await self._build_initial_messages(
                request.text, 
                session.get("history", []), 
                conversation_summary,
                urgent_issue
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
                    messages.append(message)
                    
                    # Check for tool calls
                    if message.tool_calls:
                        # Parallel tool execution
                        tasks = [
                            self._execute_single_tool(tc, conversation_summary, request.text, request.user_id, request.session_id)
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
                ai_response=final_response_text
            )

            # 4.1 Urgent Case Logging
            if urgent_issue and urgent_issue.get('level') != 'normal':
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

        except Exception as e:
            logger.error(f"Critical error in run_agent: {e}", exc_info=True)
            if not final_response_text:
                final_response_text = "抱歉，系统暂时无法处理您的请求。"

        chat_response = ChatResponse(
            response=final_response_text,
            urgent_issue=urgent_issue,
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
            user_input, current_emotion, conversation_summary, limit=limit
        )
        # Serialize ContentItems
        items_dict = [item.model_dump() for item in items]
        return {
            "recommendations": items_dict,
            "rationale": rationale,
            "match_scores": scores
        }

    async def _build_initial_messages(self, text: str, history: List[Dict], summary: Dict, urgent_issue: Dict):
        # System Prompt
        stage = summary.get('conversation_stage', 'initial')
        strategy_guidance = self.strategy_prompts.get(stage, self.strategy_prompts['initial'])
        
        # Sanitize key_concerns
        key_concerns = summary.get('key_concerns', [])
        if isinstance(key_concerns, list):
            key_concerns_str = ', '.join([str(k).replace('{', '{{').replace('}', '}}') for k in key_concerns])
        else:
            key_concerns_str = str(key_concerns).replace('{', '{{').replace('}', '}}')

        risk_level = urgent_issue.get('level', 'normal')
        
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            stage=stage,
            key_concerns=key_concerns_str,
            risk_level=risk_level,
            strategy_guidance=strategy_guidance
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history (last 10 turns to save context window)
        for h in history[-10:]:
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

agent_orchestrator = AgentOrchestrator()
