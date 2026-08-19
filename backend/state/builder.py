# -*- coding: utf-8 -*-
"""
StateBuilder（Phase 2 Task 2.3 §18）。

职责：字段归一化 / 结构转换 / Derived State / 默认值 / 版本信息。
不负责：业务决策 / Tool 调用 / LLM 调用 / Recommendation / Safety Routing。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from state.adapters import (
    ConversationAdapter,
    EmotionAdapter,
    IntentAdapter,
    ProfileAdapter,
    RecommendationAdapter,
    RiskAdapter,
)
from state.schema import AgentState, DerivedState, IdentityState, StateMeta, TurnState

NEGATIVE_EMOTIONS = {"焦虑", "抑郁", "愤怒", "压力", "无助", "孤独", "疲惫", "恐慌", "绝望", "悲伤"}
UNCERTAINTY_CONF_THRESHOLD = 0.55  # 低置信即视为 uncertain


def _derive(intent_state, emotion_state) -> DerivedState:
    labels = set(intent_state.labels)
    return DerivedState(
        help_seeking="explicit_help_request" in labels,
        resource_seeking="resource_request" in labels,
        information_seeking="information_request" in labels,
        high_risk_intent_signal="high_risk_expression" in labels,
        memory_reference_signal="memory_reference" in labels,
        intent_uncertain=intent_state.is_open_set or intent_state.confidence < UNCERTAINTY_CONF_THRESHOLD,
        negative_emotion=emotion_state.current_emotion in NEGATIVE_EMOTIONS,
    )


def build_agent_state(
    *,
    user_id: str,
    session_id: str,
    request_text: str,
    turn_index: int,
    intent_result: Optional[Dict] = None,
    emotion_state: Optional[Dict] = None,
    urgent_issue: Optional[Dict] = None,
    conversation_summary: Optional[Dict] = None,
    user_profile: Optional[Dict] = None,
    risk_baseline: Optional[str] = None,
    historical_high_risk_count: int = 0,
    previous_user_text: Optional[str] = None,
    previous_assistant_text: Optional[str] = None,
    trace_id: Optional[str] = None,
    source_versions: Optional[Dict] = None,
    memory_signal: Optional[Dict] = None,
) -> AgentState:
    """构建一轮的完整 AgentState。"""
    summary = conversation_summary or {}

    intent_state = IntentAdapter.to_state(intent_result or {})
    emotion = EmotionAdapter.to_state(emotion_state or {})
    risk = RiskAdapter.to_state(urgent_issue or {})
    risk.baseline = risk_baseline
    risk.historical_high_risk_count = historical_high_risk_count
    conversation = ConversationAdapter.to_state(summary)
    profile = ProfileAdapter.to_state(user_profile or {})
    recommendation = RecommendationAdapter.to_state(summary)

    memory_signal = memory_signal or {}
    from state.schema import MemorySignalState
    memory = MemorySignalState(
        explicit_memory_reference=intent_state is not None and "memory_reference" in intent_state.labels,
        personalization_relevant=bool(memory_signal.get("personalization_relevant", False)),
        available_memory_count=int(memory_signal.get("available_memory_count", 0) or 0),
        last_retrieved_memory_ids=list(memory_signal.get("last_retrieved_memory_ids", []) or []),
    )

    derived = _derive(intent_state, emotion)

    now = datetime.now(timezone.utc).isoformat()
    meta = StateMeta(
        created_at=now,
        updated_at=now,
        trace_id=trace_id or str(uuid.uuid4()),
        state_version=summary.get("state_version", turn_index),
        source_versions=source_versions or {},
    )

    return AgentState(
        identity=IdentityState(user_id=user_id, session_id=session_id),
        turn=TurnState(
            turn_index=turn_index,
            current_text=request_text,
            previous_user_text=previous_user_text,
            previous_assistant_text=previous_assistant_text,
        ),
        intent=intent_state,
        emotion=emotion,
        risk=risk,
        conversation=conversation,
        user_context=profile,
        recommendation=recommendation,
        memory=memory,
        derived=derived,
        meta=meta,
    )
