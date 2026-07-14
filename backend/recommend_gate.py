from typing import Any, Dict, List, Optional

from config import config
from risk_levels import (
    LEVEL_0,
    LEVEL_1,
    LEVEL_2,
    LEVEL_3,
    is_emergency_risk,
    is_high_support_risk,
    normalize_risk_level,
)


class RecommendGate:
    """推荐门控：决定当前轮是否推荐，以及采用软推荐还是硬推荐。

    v4.5 校准变更:
    - soft_threshold 从 0.22 降至 0.20
    - 新增 help_intent_hard_threshold: seeking_help 时更低硬门槛
    - 冷却惩罚区分 hard/soft，值可配置
    - preference_score 检测 has_rejected_recommendation
    """

    def __init__(self):
        # 权重与阈值从 config 读取，可通过 .env 覆盖调参。
        self.weights = {
            "emotion_intensity": config.RECOMMEND_GATE_EMOTION_WEIGHT,
            "risk_score": config.RECOMMEND_GATE_RISK_WEIGHT,
            "intent_score": config.RECOMMEND_GATE_INTENT_WEIGHT,
            "trend_score": config.RECOMMEND_GATE_TREND_WEIGHT,
            "preference_score": config.RECOMMEND_GATE_PREFERENCE_WEIGHT,
        }
        self.hard_threshold = config.RECOMMEND_GATE_HARD_THRESHOLD
        self.help_intent_hard_threshold = config.RECOMMEND_GATE_HELP_INTENT_HARD_THRESHOLD
        self.soft_threshold = config.RECOMMEND_GATE_SOFT_THRESHOLD
        # 冷却惩罚（按推荐类型区分，值可配置）
        self._cooldown_soft_d1 = config.RECOMMEND_GATE_COOLDOWN_SOFT_D1
        self._cooldown_soft_d2 = config.RECOMMEND_GATE_COOLDOWN_SOFT_D2
        self._cooldown_hard_d1 = config.RECOMMEND_GATE_COOLDOWN_HARD_D1
        self._cooldown_hard_d2 = config.RECOMMEND_GATE_COOLDOWN_HARD_D2

    def decide(
        self,
        emotion_state: Optional[Dict[str, Any]],
        risk_state: Optional[Dict[str, Any]],
        conversation_summary: Optional[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        emotion_state = emotion_state or {}
        risk_state = risk_state or {}
        conversation_summary = conversation_summary or {}
        user_profile = user_profile or {}

        reason_codes: List[str] = []
        emotion_intensity = float(emotion_state.get("emotion_intensity", 0.0) or 0.0)
        risk_level = normalize_risk_level(risk_state.get("level", LEVEL_0))
        user_intent = emotion_state.get("user_intent", "sharing")
        negative_trend = bool(emotion_state.get("negative_trend", False))

        risk_score = {
            LEVEL_0: 0.0,
            LEVEL_1: 0.4,
            LEVEL_2: 0.75,
            LEVEL_3: 1.0,
        }.get(risk_level, 0.0)
        intent_score = {
            "sharing": 0.0,
            "seeking_relief": 0.45,
            "planning": 0.55,
            "seeking_help": 0.75,
        }.get(user_intent, 0.0)
        trend_score = 0.35 if negative_trend else 0.0
        preference_score = self._calculate_preference_score(emotion_state, user_profile, conversation_summary)
        cooldown_penalty, cooldown_remaining = self._calculate_cooldown_penalty(conversation_summary)

        # 检查拒绝推荐信号
        if conversation_summary.get("has_rejected_recommendation"):
            reason_codes.append("user_rejected_recommendation")

        # 检查跨轮次去重
        recent_item_ids = conversation_summary.get("recent_recommendation_item_ids", []) or []
        if recent_item_ids:
            reason_codes.append("has_recent_recommendations")

        recommend_score = (
            self.weights["emotion_intensity"] * emotion_intensity
            + self.weights["risk_score"] * risk_score
            + self.weights["intent_score"] * intent_score
            + self.weights["trend_score"] * trend_score
            + self.weights["preference_score"] * preference_score
            - cooldown_penalty
        )

        if is_emergency_risk(risk_level):
            return {
                "should_recommend": False,
                "recommend_type": "none",
                "score": round(recommend_score, 2),
                "threshold": self.soft_threshold,
                "reason_codes": ["high_risk_safety_route", "level_3_safety_route"],
                "cooldown_remaining": cooldown_remaining,
            }
        if risk_level == LEVEL_2:
            return {
                "should_recommend": False,
                "recommend_type": "safety_only",
                "score": round(recommend_score, 2),
                "threshold": self.soft_threshold,
                "reason_codes": ["level_2_support_route", "ordinary_recommendation_disabled"],
                "cooldown_remaining": cooldown_remaining,
            }

        if emotion_intensity >= 0.75:
            reason_codes.append("high_emotion_intensity")
        if intent_score >= 0.55:
            reason_codes.append("explicit_help_seeking")
        if negative_trend:
            reason_codes.append("negative_trend")
        if cooldown_penalty > 0:
            reason_codes.append("cooldown_penalty")
        if preference_score > 0:
            reason_codes.append("preference_match")

        recommend_type = "none"
        should_recommend = False
        threshold = self.soft_threshold

        # v4.5: 通用硬阈值条件 + 求助意图专用低阈值
        if recommend_score >= self.hard_threshold and not is_high_support_risk(risk_level) and (
            user_intent in {"seeking_help", "planning"} or emotion_intensity >= 0.8 or negative_trend
        ):
            should_recommend = True
            recommend_type = "hard"
            threshold = self.hard_threshold
            reason_codes.append("hard_recommendation")
        elif (user_intent == "seeking_help"
              and recommend_score >= self.help_intent_hard_threshold
              and not is_high_support_risk(risk_level)):
            # v4.5 新增: seeking_help 二级硬阈值（默认 0.38）
            should_recommend = True
            recommend_type = "hard"
            threshold = self.help_intent_hard_threshold
            reason_codes.append("hard_recommendation")
            reason_codes.append("help_intent_hard")
        elif recommend_score >= self.soft_threshold:
            should_recommend = True
            recommend_type = "soft"
            reason_codes.append("soft_recommendation")

        return {
            "should_recommend": should_recommend,
            "recommend_type": recommend_type,
            "score": round(recommend_score, 2),
            "threshold": threshold,
            "reason_codes": sorted(set(reason_codes)),
            "cooldown_remaining": cooldown_remaining,
        }

    def _calculate_preference_score(
        self,
        emotion_state: Dict[str, Any],
        user_profile: Dict[str, Any],
        conversation_summary: Dict[str, Any],
    ) -> float:
        score = 0.0
        support_style = user_profile.get("preferred_support_style")
        if support_style in {"direct_actionable", "直接、具体、有行动建议"}:
            score += 0.45

        stress_source = emotion_state.get("stress_source")
        main_sources = user_profile.get("main_stress_sources", []) or []
        if stress_source and stress_source in main_sources:
            score += 0.35

        # 基于 content_id 的拒绝（通过 /content/feedback API）
        rejected = set(conversation_summary.get("rejected_recommendations", []) or [])
        if rejected:
            score -= 0.25

        # v4.5: 基于对话文本的拒绝检测（用户说"不用了""不需要"等）
        if conversation_summary.get("has_rejected_recommendation"):
            score -= 0.25

        return max(-0.3, min(score, 1.0))

    def _calculate_cooldown_penalty(
        self,
        conversation_summary: Dict[str, Any],
        recommend_type: str = "soft",
    ) -> tuple[float, int]:
        """冷却惩罚，按推荐类型区分 hard/soft。

        冷却系数从 config 读取，可 .env 覆盖。
        """
        recent_turns = conversation_summary.get("recent_recommendation_turns", []) or []
        current_turn = int(conversation_summary.get("turn_count", 0) or 0)
        if not recent_turns:
            return 0.0, 0

        last_turn = max(int(turn) for turn in recent_turns if isinstance(turn, (int, float)))
        distance = current_turn - last_turn

        if recommend_type == "hard":
            d1 = self._cooldown_hard_d1
            d2 = self._cooldown_hard_d2
        else:
            d1 = self._cooldown_soft_d1
            d2 = self._cooldown_soft_d2

        if distance <= 1:
            return d1, max(0, 2 - distance)
        if distance == 2:
            return d2, 0
        return 0.0, 0


recommend_gate = RecommendGate()
