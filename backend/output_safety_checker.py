import re
from typing import Any, Dict, List, Optional, Sequence

from risk_levels import LEVEL_2, LEVEL_3, normalize_risk_level, risk_level_index


class OutputSafetyChecker:
    """最终回复出站前的规则化安全检查器。"""

    def __init__(self) -> None:
        self.invalidating_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r"你想太多了",
                r"别想太多",
                r"没什么大不了",
                r"这不算什么",
                r"别矫情",
                r"你就是太敏感",
            ]
        ]
        self.guarantee_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r"一定不会有事",
                r"保证没问题",
                r"肯定会没事",
                r"绝对不会出事",
            ]
        ]
        self.isolation_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r"一个人待着",
                r"自己待着",
                r"别告诉别人",
                r"先别联系别人",
                r"自己消化",
                r"别找别人",
            ]
        ]
        self.dangerous_detail_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r"具体步骤",
                r"详细步骤",
                r"怎么自杀",
                r"怎么自残",
                r"药量",
                r"剂量",
                r"用刀",
                r"上吊",
                r"烧炭",
                r"跳楼",
                r"割腕",
            ]
        ]
        self.ordinary_recommendation_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r"推荐.*文章",
                r"推荐.*课程",
                r"推荐.*运动",
                r"推荐.*冥想",
                r"看一篇文章",
                r"做个运动计划",
                r"先去跑步",
                r"先去冥想",
            ]
        ]

    def review(
        self,
        response_text: Optional[str],
        risk_state: Optional[Dict[str, Any]] = None,
        recommendation_decision: Optional[Dict[str, Any]] = None,
        recommendations: Optional[Sequence[Any]] = None,
        recommendation_rationale: str = "",
    ) -> Dict[str, Any]:
        canonical_level = normalize_risk_level((risk_state or {}).get("level"))
        text = (response_text or "").strip()
        triggered_rules: List[str] = []
        used_fallback = False
        safe_recommendations = list(recommendations or [])
        safe_rationale = recommendation_rationale or ""
        safe_decision = dict(recommendation_decision or {})

        if not text:
            triggered_rules.append("empty_response")
            text = self._fallback_response(canonical_level)
            used_fallback = True

        if self._matches_any(text, self.dangerous_detail_patterns):
            triggered_rules.append("dangerous_details")
        if self._matches_any(text, self.invalidating_patterns):
            triggered_rules.append("invalidating_language")
        if self._matches_any(text, self.guarantee_patterns):
            triggered_rules.append("irresponsible_guarantee")
        if self._matches_any(text, self.isolation_patterns):
            triggered_rules.append("encourage_isolation")
        if risk_level_index(canonical_level) >= 2 and self._matches_any(text, self.ordinary_recommendation_patterns):
            triggered_rules.append("ordinary_recommendation_in_high_risk")

        if triggered_rules and triggered_rules != ["empty_response"]:
            text = self._fallback_response(canonical_level)
            used_fallback = True

        if canonical_level in {LEVEL_2, LEVEL_3}:
            if safe_recommendations or safe_rationale:
                triggered_rules.append("high_risk_recommendations_blocked")
            safe_recommendations = []
            safe_rationale = ""
            safe_decision["should_recommend"] = False
            safe_decision["recommend_type"] = "none" if canonical_level == LEVEL_3 else "safety_only"

        return {
            "response_text": text,
            "recommendations": safe_recommendations,
            "recommendation_rationale": safe_rationale,
            "recommendation_decision": safe_decision,
            "triggered_rules": triggered_rules,
            "used_fallback": used_fallback,
        }

    def _matches_any(self, text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
        return any(pattern.search(text) for pattern in patterns)

    def _fallback_response(self, canonical_level: str) -> str:
        if canonical_level == LEVEL_3:
            return (
                "我很在意你现在的安全。请先立刻联系身边能到场陪你的家人、朋友、老师或同事，"
                "并尽快联系当地紧急服务或心理危机热线；如果身边有药物、刀具、绳索等危险物品，"
                "请先把它们移开并离开危险环境。你可以直接告诉我：现在你身边有没有人。"
            )
        if canonical_level == LEVEL_2:
            return (
                "听起来你现在已经很难独自扛住了。先把重点放在安全上：尽量不要一个人待着，"
                "联系一位可信任的人陪你，并暂时远离可能伤害自己的物品；如果可以，也尽快联系学校、医院或当地心理支持资源。"
            )
        return (
            "我在认真听你说。我们先不急着下结论，也先不责怪你自己。"
            "如果你愿意，可以先告诉我现在最难受、最想立刻缓解的那一部分是什么。"
        )


output_safety_checker = OutputSafetyChecker()
