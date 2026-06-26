from typing import Any, Dict, List, Optional


class RecommendGate:
    """推荐门控：决定当前轮是否推荐，以及采用软推荐还是硬推荐。"""

    def __init__(self):
        # 归一化后的权重总和为 1，便于解释分数区间并稳定阈值语义。
        self.weights = {
            "emotion_intensity": 0.30,
            "risk_score": 0.23,
            "intent_score": 0.23,
            "trend_score": 0.14,
            "preference_score": 0.10,
        }
        self.hard_threshold = 0.58
        self.soft_threshold = 0.22

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
        risk_level = risk_state.get("level", "low")
        user_intent = emotion_state.get("user_intent", "sharing")
        negative_trend = bool(emotion_state.get("negative_trend", False))

        risk_score = {"low": 0.0, "medium": 0.65, "high": 1.0}.get(risk_level, 0.0)
        intent_score = {
            "sharing": 0.0,
            "seeking_relief": 0.45,
            "planning": 0.55,
            "seeking_help": 0.75,
        }.get(user_intent, 0.0)
        trend_score = 0.35 if negative_trend else 0.0
        preference_score = self._calculate_preference_score(emotion_state, user_profile, conversation_summary)
        cooldown_penalty, cooldown_remaining = self._calculate_cooldown_penalty(conversation_summary)

        recommend_score = (
            self.weights["emotion_intensity"] * emotion_intensity
            + self.weights["risk_score"] * risk_score
            + self.weights["intent_score"] * intent_score
            + self.weights["trend_score"] * trend_score
            + self.weights["preference_score"] * preference_score
            - cooldown_penalty
        )

        if risk_level == "high":
            return {
                "should_recommend": False,
                "recommend_type": "none",
                "score": round(recommend_score, 2),
                "threshold": self.soft_threshold,
                "reason_codes": ["high_risk_safety_route"],
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

        if recommend_score >= self.hard_threshold and (
            user_intent in {"seeking_help", "planning"} or emotion_intensity >= 0.8 or negative_trend
        ):
            should_recommend = True
            recommend_type = "hard"
            threshold = self.hard_threshold
            reason_codes.append("hard_recommendation")
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

        rejected = set(conversation_summary.get("rejected_recommendations", []) or [])
        if rejected:
            score -= 0.25

        return max(-0.3, min(score, 1.0))

    def _calculate_cooldown_penalty(self, conversation_summary: Dict[str, Any]) -> tuple[float, int]:
        recent_turns = conversation_summary.get("recent_recommendation_turns", []) or []
        current_turn = int(conversation_summary.get("turn_count", 0) or 0)
        if not recent_turns:
            return 0.0, 0

        last_turn = max(int(turn) for turn in recent_turns if isinstance(turn, (int, float)))
        distance = current_turn - last_turn
        if distance <= 1:
            return 0.35, max(0, 2 - distance)
        if distance == 2:
            return 0.18, 0
        return 0.0, 0


recommend_gate = RecommendGate()
