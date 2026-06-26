import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class RiskEvaluator:
    """结构化风险评估器，输出 low / medium / high 三层风险。"""

    def __init__(self):
        self.high_risk_keywords = [
            "自杀",
            "不想活了",
            "结束生命",
            "想死",
            "死掉",
            "跳楼",
            "割腕",
            "服毒",
            "上吊",
            "烧炭",
            "活着没意义",
        ]
        self.medium_risk_keywords = [
            "活不下去",
            "撑不住",
            "崩溃",
            "绝望",
            "想消失",
            "好累",
            "没有希望",
            "想放弃",
            "没人理解",
            "被抛弃",
            "我是不是没用",
        ]
        self.self_negation_keywords = [
            "我很差",
            "我不行",
            "我没用",
            "都是我的错",
            "我什么都做不好",
        ]

    def evaluate(
        self,
        text: str,
        emotion_state: Optional[Dict[str, Any]] = None,
        conversation_summary: Optional[Dict[str, Any]] = None,
        long_term_risk_level: Optional[str] = None,
        historical_high_risk_count: int = 0,
    ) -> Dict[str, Any]:
        summary = conversation_summary or {}
        state = emotion_state or {}
        triggers = self._collect_triggers(text)
        score = self._calculate_risk_score(
            text=text,
            triggers=triggers,
            emotion_state=state,
            conversation_summary=summary,
            long_term_risk_level=long_term_risk_level,
            historical_high_risk_count=historical_high_risk_count,
        )
        level = self._map_level(score, triggers)

        response = {
            "level": level,
            "message": self._build_message(level),
            "suggestions": self._build_suggestions(level),
            "triggers": triggers,
            "risk_score": round(score, 2),
        }
        logger.info("风险评估完成: level=%s score=%.2f triggers=%s", level, score, triggers)
        return response

    def _collect_triggers(self, text: str) -> List[str]:
        found: List[str] = []
        for keyword in self.high_risk_keywords + self.medium_risk_keywords + self.self_negation_keywords:
            if keyword in text and keyword not in found:
                found.append(keyword)
        return found

    def _calculate_risk_score(
        self,
        text: str,
        triggers: List[str],
        emotion_state: Dict[str, Any],
        conversation_summary: Dict[str, Any],
        long_term_risk_level: Optional[str],
        historical_high_risk_count: int,
    ) -> float:
        score = 0.5

        high_hits = [item for item in triggers if item in self.high_risk_keywords]
        medium_hits = [item for item in triggers if item in self.medium_risk_keywords]
        self_negation_hits = [item for item in triggers if item in self.self_negation_keywords]

        score += len(high_hits) * 4.0
        score += len(medium_hits) * 1.6
        score += len(self_negation_hits) * 1.2

        emotion_type = emotion_state.get("emotion_type")
        emotion_intensity = emotion_state.get("emotion_intensity", 0.0)
        negative_trend = emotion_state.get("negative_trend", False)

        if emotion_type in {"sadness", "helplessness", "anxiety", "stress"}:
            score += 1.0
        if emotion_intensity >= 0.8:
            score += 1.8
        elif emotion_intensity >= 0.65:
            score += 1.0

        if negative_trend:
            score += 1.5

        trend = conversation_summary.get("emotion_trend")
        if trend == "escalating":
            score += 1.0
        elif trend == "consistent" and emotion_type in {"sadness", "helplessness", "anxiety"}:
            score += 0.6

        if long_term_risk_level == "high":
            score += 1.5
        elif long_term_risk_level == "medium":
            score += 0.8

        score += min(historical_high_risk_count, 3) * 0.5

        if any(token in text for token in ["立刻", "马上", "现在", "今晚"]):
            score += 0.8

        return min(score, 10.0)

    def _map_level(self, score: float, triggers: List[str]) -> str:
        if any(item in self.high_risk_keywords for item in triggers):
            return "high"
        if score >= 5.0:
            return "medium"
        return "low"

    def _build_message(self, level: str) -> str:
        messages = {
            "high": "检测到高风险信号，请优先稳定情绪并尽快联系可信任的人或专业支持。",
            "medium": "检测到持续负面或较强烈情绪，需要加强安抚并提供具体支持建议。",
            "low": "",
        }
        return messages[level]

    def _build_suggestions(self, level: str) -> List[str]:
        if level == "high":
            return [
                "立即联系信任的家人、朋友或辅导员",
                "尽快联系当地心理援助热线或医院急诊",
                "暂时远离可能伤害自己的环境或物品",
            ]
        if level == "medium":
            return [
                "先做一次缓慢深呼吸，给身体一个暂停",
                "把最强烈的感受告诉信任的人",
                "如果状态持续恶化，尽快联系专业支持",
            ]
        return []


risk_evaluator = RiskEvaluator()
