# -*- coding: utf-8 -*-
"""
AgentState Schema v1（Phase 2 Task 2.2）。

设计原则（对齐执行文档 §3）：
  1. State ≠ Action —— 禁止 should_recommend / primary_action / tool_actions 等 Policy 输出
  2. State ≠ Raw History —— 只保存结构化摘要信号 + 必要引用 ID
  3. 区分生命周期：Turn / Session / Cross-session Reference / Derived
  4. 每字段有唯一 source / owner / update rule（见 docs/agent_state_update_rules.md）

来源：
  - Intent:     Phase 1.5 Intent Service（冻结）
  - Emotion:    emotion_analyzer（包装，不修改）
  - Risk:       risk_evaluator（包装，不修改）
  - Conversation: conversation_manager（消费结构化结果）
  - Profile:    UserProfileTool
  - Recommendation: conversation_manager history
  - Memory:     memory subsystem 信号
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class IdentityState(BaseModel):
    """身份标识。不保存真实姓名等敏感原始资料。"""
    user_id: str = ""
    session_id: str = ""


class TurnState(BaseModel):
    """Turn 级状态（只保留 short context，不保存完整 history）。"""
    turn_index: int = 0
    current_text: str = ""
    previous_user_text: Optional[str] = None
    previous_assistant_text: Optional[str] = None


class IntentState(BaseModel):
    """直接消费 Phase 1.5 冻结的 IntentResult。"""
    labels: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    label_scores: dict = Field(default_factory=dict)
    is_open_set: bool = False
    source: str = ""                       # classifier | llm | rule
    fallback_reason: Optional[str] = None
    context_used: int = 0


class EmotionState(BaseModel):
    """情绪状态（包装 emotion_analyzer 输出，不合并语义）。"""
    current_emotion: str = "中性"
    confidence: float = 0.0
    intensity: Optional[float] = None
    context_emotion: Optional[str] = None
    trend: Optional[str] = None
    stress_source: Optional[str] = None
    # legacy 4 类表达意图（与 Intent 10 类语义不同，保留兼容引用）
    legacy_user_intent: Optional[str] = None


class RiskState(BaseModel):
    """风险状态（包装 risk_evaluator 输出，Phase 2 不修复 FPR/FNR）。"""
    level: int = 0
    confidence: Optional[float] = None
    score: Optional[float] = None
    trend: Optional[str] = None
    persistence: int = 0                   # 连续满足 L2+ 的 turn 数（派生）
    subject: Optional[str] = None          # self | third_party | discussion
    is_discussion: bool = False
    is_third_party: bool = False
    safe_denial: bool = False
    evidence: List[str] = Field(default_factory=list)
    escalation_reasons: List[str] = Field(default_factory=list)
    baseline: Optional[str] = None
    historical_high_risk_count: int = 0
    recent_risk_levels: List[int] = Field(default_factory=list)  # 滚动窗口（5），由 updater 维护


class ConversationState(BaseModel):
    """会话状态（消费 conversation_manager 结构化结果，非真值源）。"""
    stage: str = "initial"
    turn_count: int = 0
    current_topic: Optional[str] = None
    key_concerns: List[str] = Field(default_factory=list)
    recent_intents: List[str] = Field(default_factory=list)
    emotion_trend: Optional[str] = None
    summary_version: Optional[str] = None


class UserContextState(BaseModel):
    """长期画像摘要（本轮决策真正需要的部分，非全量 profile）。"""
    preferred_support_style: Optional[str] = None
    avoid_styles: List[str] = Field(default_factory=list)
    main_stress_sources: List[str] = Field(default_factory=list)
    profile_risk_level: Optional[str] = None


class RecommendationState(BaseModel):
    """推荐历史状态（非推荐决策）。"""
    recent_recommendation_ids: List[str] = Field(default_factory=list)
    last_recommendation_turn: Optional[int] = None
    turns_since_last_recommendation: Optional[int] = None
    recent_feedback: List[str] = Field(default_factory=list)
    rejection_detected: bool = False


class MemorySignalState(BaseModel):
    """Memory 信号（不执行 Memory 2.0，只记录可供 Policy 使用的信号）。"""
    explicit_memory_reference: bool = False
    personalization_relevant: bool = False
    available_memory_count: int = 0
    last_retrieved_memory_ids: List[str] = Field(default_factory=list)


class DerivedState(BaseModel):
    """确定性派生信号（不涉及 Policy 决策）。"""
    help_seeking: bool = False
    resource_seeking: bool = False
    information_seeking: bool = False
    high_risk_intent_signal: bool = False
    memory_reference_signal: bool = False
    intent_uncertain: bool = False
    negative_emotion: bool = False


class StateMeta(BaseModel):
    """状态元信息。"""
    created_at: str = ""
    updated_at: str = ""
    trace_id: str = ""
    state_version: int = 0
    source_versions: dict = Field(default_factory=dict)


class AgentState(BaseModel):
    """统一 Agent 状态 v1。"""
    schema_version: str = "1.0"

    identity: IdentityState = Field(default_factory=IdentityState)
    turn: TurnState = Field(default_factory=TurnState)

    intent: IntentState = Field(default_factory=IntentState)
    emotion: EmotionState = Field(default_factory=EmotionState)
    risk: RiskState = Field(default_factory=RiskState)

    conversation: ConversationState = Field(default_factory=ConversationState)
    user_context: UserContextState = Field(default_factory=UserContextState)
    recommendation: RecommendationState = Field(default_factory=RecommendationState)
    memory: MemorySignalState = Field(default_factory=MemorySignalState)

    derived: DerivedState = Field(default_factory=DerivedState)
    meta: StateMeta = Field(default_factory=StateMeta)

    def to_debug_dict(self) -> dict:
        """最小 State Trace（不含敏感原文）。"""
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.model_dump(),
            "turn": {"turn_index": self.turn.turn_index,
                     "current_text_preview": self.turn.current_text[:50]},
            "intent": {"labels": self.intent.labels,
                       "confidence": self.intent.confidence,
                       "is_open_set": self.intent.is_open_set,
                       "source": self.intent.source,
                       "fallback_reason": self.intent.fallback_reason},
            "emotion": {"current_emotion": self.emotion.current_emotion,
                        "intensity": self.emotion.intensity,
                        "trend": self.emotion.trend},
            "risk": {"level": self.risk.level, "trend": self.risk.trend,
                     "persistence": self.risk.persistence,
                     "is_discussion": self.risk.is_discussion,
                     "is_third_party": self.risk.is_third_party,
                     "safe_denial": self.risk.safe_denial},
            "conversation": {"stage": self.conversation.stage,
                             "turn_count": self.conversation.turn_count,
                             "key_concerns": self.conversation.key_concerns},
            "user_context": {"preferred_support_style": self.user_context.preferred_support_style,
                             "profile_risk_level": self.user_context.profile_risk_level},
            "recommendation": {"recent_ids": self.recommendation.recent_recommendation_ids[-5:],
                               "rejection_detected": self.recommendation.rejection_detected},
            "memory": {"explicit_memory_reference": self.memory.explicit_memory_reference,
                       "available_count": self.memory.available_memory_count},
            "derived": self.derived.model_dump(),
            "meta": {"trace_id": self.meta.trace_id,
                     "state_version": self.meta.state_version,
                     "source_versions": self.meta.source_versions},
        }
