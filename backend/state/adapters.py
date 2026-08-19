# -*- coding: utf-8 -*-
"""
Legacy 输出 → AgentState Schema 适配器（Phase 2 Task 2.3 §19）。

不强制旧模块改输出 Schema；由 Adapter 做字段归一化与结构转换。
纯确定性转换，无 LLM / 无业务决策。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from state.schema import (
    ConversationState,
    EmotionState,
    IntentState,
    MemorySignalState,
    RecommendationState,
    RiskState,
    UserContextState,
)

# 归一化风险等级别名 -> 整数
RISK_LEVEL_MAP = {"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3,
                  "low": 0, "medium": 2, "high": 3, "none": 0}


def _risk_int(level) -> int:
    if isinstance(level, int):
        return max(0, min(int(level), 3))
    return RISK_LEVEL_MAP.get(str(level).lower(), 0)


class IntentAdapter:
    """IntentResult → IntentState（Phase 1.5 冻结接口）。"""

    @staticmethod
    def to_state(intent_result: Dict) -> IntentState:
        return IntentState(
            labels=list(intent_result.get("labels", []) or []),
            confidence=float(intent_result.get("confidence", 0.0) or 0.0),
            label_scores=dict(intent_result.get("label_scores", {}) or {}),
            is_open_set=bool(intent_result.get("is_open_set", False)),
            source=str(intent_result.get("source", "")),
            fallback_reason=intent_result.get("fallback_reason"),
            context_used=int(intent_result.get("context_used", 0) or 0),
        )


class EmotionAdapter:
    """emotion_analyzer 输出 → EmotionState。"""

    @staticmethod
    def to_state(emotion_state: Dict) -> EmotionState:
        emotion_state = emotion_state or {}
        return EmotionState(
            current_emotion=str(emotion_state.get("current_emotion", "中性")),
            confidence=float(emotion_state.get("confidence", 0.0) or 0.0),
            intensity=_to_opt_float(emotion_state.get("emotion_intensity")),
            context_emotion=emotion_state.get("context_emotion"),
            trend=emotion_state.get("emotion_trend"),
            stress_source=emotion_state.get("stress_source"),
            legacy_user_intent=emotion_state.get("user_intent"),
        )


class RiskAdapter:
    """urgent_issue dict → RiskState。"""

    @staticmethod
    def to_state(urgent_issue: Dict) -> RiskState:
        urgent_issue = urgent_issue or {}
        ctx = urgent_issue.get("risk_context", {}) or {}
        return RiskState(
            level=_risk_int(urgent_issue.get("level", 0)),
            confidence=_to_opt_float(urgent_issue.get("confidence")),
            score=_to_opt_float(urgent_issue.get("risk_score")),
            trend=urgent_issue.get("risk_trend"),
            subject=ctx.get("subject"),
            is_discussion=bool(ctx.get("is_discussion_context", False)),
            is_third_party=bool(ctx.get("is_third_party_risk", False)),
            safe_denial=bool(ctx.get("is_safe_denial", False)),
            evidence=list(urgent_issue.get("triggers", []) or []),
            escalation_reasons=list(urgent_issue.get("escalation_reasons", []) or []),
            baseline=None,
            historical_high_risk_count=0,
        )


class ConversationAdapter:
    """conversation_manager summary → ConversationState。"""

    @staticmethod
    def to_state(summary: Dict) -> ConversationState:
        summary = summary or {}
        return ConversationState(
            stage=str(summary.get("conversation_stage", "initial")),
            turn_count=int(summary.get("turn_count", 0) or 0),
            current_topic=summary.get("current_topic"),
            key_concerns=list(summary.get("key_concerns", []) or []),
            recent_intents=list(summary.get("recent_intents", []) or []),
            emotion_trend=summary.get("emotion_trend"),
            summary_version=str(summary.get("summary_version", "") or None) or None,
        )


class ProfileAdapter:
    """user_profile dict → UserContextState。"""

    @staticmethod
    def to_state(user_profile: Dict) -> UserContextState:
        user_profile = user_profile or {}
        return UserContextState(
            preferred_support_style=user_profile.get("preferred_support_style"),
            avoid_styles=list(user_profile.get("avoid_style", []) or []),
            main_stress_sources=list(user_profile.get("main_stress_sources", []) or []),
            profile_risk_level=str(user_profile.get("risk_level", "") or None) or None,
        )


class RecommendationAdapter:
    """conversation summary 推荐历史 → RecommendationState。"""

    @staticmethod
    def to_state(summary: Dict) -> RecommendationState:
        summary = summary or {}
        accepted = list(summary.get("accepted_recommendations", []) or [])
        rejected = list(summary.get("rejected_recommendations", []) or [])
        recent_ids = accepted + rejected
        recent_feedback = summary.get("recommendation_feedback_map", {})
        feedback_values = list(recent_feedback.values()) if isinstance(recent_feedback, dict) else []
        return RecommendationState(
            recent_recommendation_ids=recent_ids[-10:],
            last_recommendation_turn=summary.get("last_recommendation_turn"),
            turns_since_last_recommendation=summary.get("turns_since_last_recommendation"),
            recent_feedback=feedback_values[-10:],
            rejection_detected=bool(summary.get("has_rejected_recommendation", False)),
        )


def _to_opt_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
