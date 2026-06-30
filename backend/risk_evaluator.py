import logging
from typing import Any, Dict, List, Optional

from risk_levels import (
    LEVEL_0,
    LEVEL_1,
    LEVEL_2,
    LEVEL_3,
    normalize_risk_level,
    risk_level_band,
    risk_level_index,
    risk_level_label,
)


logger = logging.getLogger(__name__)


class RiskEvaluator:
    """结构化风险评估器，输出四级风险语义并保留旧等级兼容字段。"""

    def __init__(self):
        self.dimension_names = [
            "risk_ideation",
            "action_intent",
            "plan_specificity",
            "time_urgency",
            "means_access",
            "self_control",
            "protective_factors",
        ]
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
        self.passive_ideation_keywords = [
            "想消失",
            "不想再醒来",
            "不想醒来",
            "如果我消失了",
            "明天之后就都结束了",
            "活着没意义",
            "没有希望",
            "想放弃",
        ]
        self.action_intent_keywords = [
            "准备去做",
            "准备这么做",
            "我会做傻事",
            "做出什么事",
            "结束生命",
            "伤害自己",
            "伤害别人",
            "已经准备好了",
            "我已经准备好了",
        ]
        self.plan_keywords = [
            "跳楼",
            "割腕",
            "服毒",
            "上吊",
            "烧炭",
            "遗书",
            "天台",
            "刀",
            "药",
            "绳",
            "安排好了",
            "准备好了",
        ]
        self.time_urgent_keywords = ["现在", "马上", "立刻", "已经"]
        self.time_near_keywords = ["今晚", "今天", "这几天", "明天"]
        self.means_keywords = ["刀", "药", "绳", "天台", "阳台", "独处", "一个人", "没人"]
        self.self_control_keywords = ["控制不住", "停不下来", "失控", "我怕自己会做出什么事", "控制不了"]
        self.supportive_keywords = ["想慢慢调整", "想做个计划", "想试试看", "愿意求助", "想找人聊", "寻求帮助"]
        self.help_seeking_intents = {"seeking_help", "planning"}
        self.dimension_weights = {
            "risk_ideation": 0.23,
            "action_intent": 0.20,
            "plan_specificity": 0.15,
            "time_urgency": 0.14,
            "means_access": 0.10,
            "self_control": 0.12,
            "protective_factors": 0.06,
        }

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
        context = self._analyze_context(text)
        triggers = self._collect_triggers(text)
        dimensions = self._score_dimensions(
            text=text,
            triggers=triggers,
            emotion_state=state,
            conversation_summary=summary,
            context=context,
        )
        raw_score = self._calculate_raw_score(
            dimensions=dimensions,
            emotion_state=state,
            conversation_summary=summary,
            long_term_risk_level=long_term_risk_level,
            historical_high_risk_count=historical_high_risk_count,
        )
        level, escalation_reasons = self._determine_level(
            text=text,
            triggers=triggers,
            dimensions=dimensions,
            raw_score=raw_score,
            context=context,
        )
        level, escalation_reasons = self._apply_risk_inertia(
            level=level,
            dimensions=dimensions,
            emotion_state=state,
            conversation_summary=summary,
            reasons=escalation_reasons,
        )
        level, escalation_reasons = self._apply_context_guardrails(
            level=level,
            dimensions=dimensions,
            context=context,
            reasons=escalation_reasons,
        )
        legacy_level = risk_level_band(level)
        risk_evidence = self._build_risk_evidence(dimensions)
        calibrated_score = self._calibrate_display_score(level, raw_score)

        response = {
            "level": level,
            "legacy_level": legacy_level,
            "level_index": risk_level_index(level),
            "level_label": risk_level_label(level),
            "message": self._build_message(level),
            "suggestions": self._build_suggestions(level),
            "triggers": triggers,
            "risk_score": round(calibrated_score, 2),
            "raw_score": round(raw_score, 2),
            "risk_dimensions": {
                name: item["score"] for name, item in dimensions.items()
            },
            "risk_evidence": risk_evidence,
            "escalation_reasons": escalation_reasons,
            "recent_risk_levels": [
                normalize_risk_level(item)
                for item in summary.get("recent_risk_levels", [])
            ],
            "risk_context": context,
        }
        logger.info(
            "风险评估完成: level=%s raw_score=%.2f display_score=%.2f triggers=%s escalation=%s",
            level,
            raw_score,
            calibrated_score,
            triggers,
            escalation_reasons,
        )
        return response

    def _collect_triggers(self, text: str) -> List[str]:
        found: List[str] = []
        for keyword in self.high_risk_keywords + self.medium_risk_keywords + self.self_negation_keywords:
            if keyword in text and keyword not in found:
                found.append(keyword)
        return found

    def _score_dimensions(
        self,
        text: str,
        triggers: List[str],
        emotion_state: Dict[str, Any],
        conversation_summary: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        text = text or ""
        dimensions = {
            name: {"score": 0, "evidence": []}
            for name in self.dimension_names
        }

        explicit_ideation_hits = self._matched_keywords(text, self.high_risk_keywords)
        passive_ideation_hits = self._matched_keywords(text, self.passive_ideation_keywords)
        medium_distress_hits = self._matched_keywords(text, self.medium_risk_keywords)
        negation_hits = self._matched_keywords(text, self.self_negation_keywords)
        action_hits = self._matched_keywords(text, self.action_intent_keywords)
        plan_hits = self._matched_keywords(text, self.plan_keywords)
        urgent_time_hits = self._matched_keywords(text, self.time_urgent_keywords)
        near_time_hits = self._matched_keywords(text, self.time_near_keywords)
        means_hits = self._matched_keywords(text, self.means_keywords)
        self_control_hits = self._matched_keywords(text, self.self_control_keywords)
        support_hits = self._matched_keywords(text, self.supportive_keywords)

        if context["is_discussion_context"]:
            dimensions["risk_ideation"]["score"] = 0
            dimensions["risk_ideation"]["evidence"].append("识别为影视/新闻/论文等讨论语境")
        elif context["is_safe_denial"]:
            dimensions["risk_ideation"]["score"] = 0
            dimensions["risk_ideation"]["evidence"].append("显式否认当前自伤/自杀意图")
        elif context["is_third_party_risk"]:
            dimensions["risk_ideation"]["score"] = 1 if context["is_help_request"] else 0
            dimensions["risk_ideation"]["evidence"].append("风险主体更像第三方而非用户本人")
        elif explicit_ideation_hits:
            dimensions["risk_ideation"]["score"] = 3
            dimensions["risk_ideation"]["evidence"].extend(explicit_ideation_hits)
        elif passive_ideation_hits:
            dimensions["risk_ideation"]["score"] = 2
            dimensions["risk_ideation"]["evidence"].extend(passive_ideation_hits)
        elif medium_distress_hits or negation_hits:
            dimensions["risk_ideation"]["score"] = 1
            dimensions["risk_ideation"]["evidence"].extend((medium_distress_hits or [])[:2] + (negation_hits or [])[:1])

        if context["is_discussion_context"] or context["is_third_party_risk"]:
            dimensions["action_intent"]["score"] = 0
            if context["is_third_party_risk"]:
                dimensions["action_intent"]["evidence"].append("行动风险来自第三方转述，不直接视为用户本人行动倾向")
        elif any(token in text for token in ["现在就", "马上就", "今晚就", "已经准备"]):
            dimensions["action_intent"]["score"] = 3
            dimensions["action_intent"]["evidence"].append("存在明确近期行动表达")
        elif action_hits:
            dimensions["action_intent"]["score"] = 2
            dimensions["action_intent"]["evidence"].extend(action_hits[:2])
        elif any(token in text for token in ["撑不住", "怕自己会做出什么事", "做傻事"]):
            dimensions["action_intent"]["score"] = 1
            dimensions["action_intent"]["evidence"].append("存在模糊行动倾向或担心失控")

        if context["is_discussion_context"] or context["is_third_party_risk"]:
            dimensions["plan_specificity"]["score"] = 0
            if context["is_discussion_context"]:
                dimensions["plan_specificity"]["evidence"].append("计划信息位于讨论语境，不直接计为用户计划")
        elif any(token in text for token in ["安排好了", "遗书"]) or len(plan_hits) >= 2:
            dimensions["plan_specificity"]["score"] = 3
            dimensions["plan_specificity"]["evidence"].extend(plan_hits[:3] or ["存在明确准备/安排表达"])
        elif plan_hits:
            dimensions["plan_specificity"]["score"] = 2
            dimensions["plan_specificity"]["evidence"].extend(plan_hits[:2])
        elif any(token in text for token in ["怎么做", "这样做"]):
            dimensions["plan_specificity"]["score"] = 1
            dimensions["plan_specificity"]["evidence"].append("提到模糊的行为方式")

        if context["is_discussion_context"] or context["is_third_party_risk"]:
            dimensions["time_urgency"]["score"] = 0
            if context["is_third_party_risk"] and context["is_help_request"]:
                dimensions["time_urgency"]["evidence"].append("第三方危机求助需谨慎跟进，但不直接视为用户立即风险")
        elif urgent_time_hits:
            dimensions["time_urgency"]["score"] = 3
            dimensions["time_urgency"]["evidence"].extend(urgent_time_hits[:2])
        elif near_time_hits:
            dimensions["time_urgency"]["score"] = 2
            dimensions["time_urgency"]["evidence"].extend(near_time_hits[:2])
        elif conversation_summary.get("emotion_trend") == "escalating":
            dimensions["time_urgency"]["score"] = 1
            dimensions["time_urgency"]["evidence"].append("近期风险趋势上升")

        if context["is_discussion_context"] or context["is_third_party_risk"]:
            dimensions["means_access"]["score"] = 0
        elif any(token in text for token in ["手边", "就在旁边", "一个人待着"]) and means_hits:
            dimensions["means_access"]["score"] = 3
            dimensions["means_access"]["evidence"].extend(means_hits[:3])
        elif means_hits or plan_hits:
            dimensions["means_access"]["score"] = 2
            dimensions["means_access"]["evidence"].extend((means_hits or plan_hits)[:2])
        elif any(token in text for token in ["一个人", "没人", "独处"]):
            dimensions["means_access"]["score"] = 1
            dimensions["means_access"]["evidence"].append("可能处于较孤立或独处状态")

        if context["is_discussion_context"]:
            dimensions["self_control"]["score"] = 0
        elif any(token in text for token in ["控制不住", "已经失控", "停不下来"]):
            dimensions["self_control"]["score"] = 3
            dimensions["self_control"]["evidence"].append("表达已经无法控制自己")
        elif self_control_hits:
            dimensions["self_control"]["score"] = 2
            dimensions["self_control"]["evidence"].extend(self_control_hits[:2])
        elif any(token in text for token in ["崩溃", "撑不住"]):
            dimensions["self_control"]["score"] = 1
            dimensions["self_control"]["evidence"].append("情绪接近崩溃，但仍可交流")

        user_intent = emotion_state.get("user_intent")
        if context["is_third_party_risk"] and context["is_help_request"]:
            dimensions["protective_factors"]["score"] = 0
            dimensions["protective_factors"]["evidence"].append("用户正在主动为第三方风险寻求帮助")
        elif context["is_discussion_context"]:
            dimensions["protective_factors"]["score"] = 0
            dimensions["protective_factors"]["evidence"].append("讨论语境不视为当前自我风险")
        elif support_hits or user_intent in self.help_seeking_intents:
            dimensions["protective_factors"]["score"] = 0
            dimensions["protective_factors"]["evidence"].append("仍存在求助或调整意愿")
        elif any(token in text for token in ["没人帮我", "不想求助", "谁也别管我", "一个人待着"]):
            dimensions["protective_factors"]["score"] = 3
            dimensions["protective_factors"]["evidence"].append("保护性因素很弱或拒绝求助")
        elif conversation_summary.get("turn_count", 0) >= 3:
            dimensions["protective_factors"]["score"] = 2
            dimensions["protective_factors"]["evidence"].append("持续负面表达但未出现明确求助")
        else:
            dimensions["protective_factors"]["score"] = 1
            dimensions["protective_factors"]["evidence"].append("保护性因素有限")

        return dimensions

    def _calculate_raw_score(
        self,
        dimensions: Dict[str, Dict[str, Any]],
        emotion_state: Dict[str, Any],
        conversation_summary: Dict[str, Any],
        long_term_risk_level: Optional[str],
        historical_high_risk_count: int,
    ) -> float:
        score = 0.5
        normalized_long_term_level = normalize_risk_level(long_term_risk_level)
        weighted_dimension_score = 0.0
        for name, weight in self.dimension_weights.items():
            weighted_dimension_score += dimensions[name]["score"] * weight
        score += (weighted_dimension_score / 3.0) * 8.8

        emotion_type = emotion_state.get("emotion_type")
        emotion_intensity = emotion_state.get("emotion_intensity", 0.0)
        negative_trend = emotion_state.get("negative_trend", False)

        if emotion_type in {"sadness", "helplessness", "anxiety", "stress"}:
            score += 0.55
        if emotion_intensity >= 0.8:
            score += 0.95
        elif emotion_intensity >= 0.65:
            score += 0.5

        if negative_trend:
            score += 0.6

        trend = conversation_summary.get("emotion_trend")
        if trend == "escalating":
            score += 0.35
        elif trend == "consistent" and emotion_type in {"sadness", "helplessness", "anxiety"}:
            score += 0.2

        if normalized_long_term_level == LEVEL_3:
            score += 1.6
        elif normalized_long_term_level == LEVEL_2:
            score += 1.0
        elif normalized_long_term_level == LEVEL_1:
            score += 0.5

        score += min(historical_high_risk_count, 3) * 0.5

        return min(score, 10.0)

    def _determine_level(
        self,
        text: str,
        triggers: List[str],
        dimensions: Dict[str, Dict[str, Any]],
        raw_score: float,
        context: Dict[str, Any],
    ) -> tuple[str, List[str]]:
        reasons: List[str] = []
        has_safe_denial = context["is_safe_denial"]
        ideation = dimensions["risk_ideation"]["score"]
        action = dimensions["action_intent"]["score"]
        plan = dimensions["plan_specificity"]["score"]
        urgency = dimensions["time_urgency"]["score"]
        means = dimensions["means_access"]["score"]
        self_control = dimensions["self_control"]["score"]

        if (
            not has_safe_denial
            and not context["is_third_party_risk"]
            and not context["is_discussion_context"]
            and any(item in self.high_risk_keywords for item in triggers)
        ):
            reasons.append("explicit_high_risk_keywords")
            return LEVEL_3, reasons
        if self_control >= 3:
            reasons.append("hard_rule:self_control_loss")
            return LEVEL_3, reasons
        if action >= 3 and urgency >= 2:
            reasons.append("hard_rule:action_with_near_term_urgency")
            return LEVEL_3, reasons
        if action >= 2 and plan >= 2:
            reasons.append("hard_rule:action_with_plan")
            return LEVEL_3, reasons
        if plan >= 2 and urgency >= 3:
            reasons.append("hard_rule:plan_with_immediate_urgency")
            return LEVEL_3, reasons
        if ideation >= 2 and urgency >= 3:
            reasons.append("hard_rule:ideation_with_immediate_urgency")
            return LEVEL_3, reasons

        level = LEVEL_0
        if raw_score >= 4.8:
            level = LEVEL_2
            reasons.append("score_threshold:level_2")
        elif raw_score >= 2.8:
            level = LEVEL_1
            reasons.append("score_threshold:level_1")

        if action >= 2:
            if level != LEVEL_2:
                reasons.append("hard_rule:action_intent_floor_level_2")
            level = max(level, LEVEL_2, key=risk_level_index)
        if plan >= 2 and means >= 2:
            if level != LEVEL_2:
                reasons.append("hard_rule:plan_and_means_floor_level_2")
            level = max(level, LEVEL_2, key=risk_level_index)
        if ideation >= 3 and raw_score >= 5.5:
            if level != LEVEL_2:
                reasons.append("hard_rule:explicit_ideation_floor_level_2")
            level = max(level, LEVEL_2, key=risk_level_index)
        # 这里直接读取文本外的情绪强度不方便，改用基础分近似不足，因此由调用侧在 score 中体现；
        # 如果已经表现为明显崩溃 + 自控变弱，也至少应进入 Level 2。
        if ideation >= 1 and self_control >= 1 and raw_score >= 3.5:
            if level != LEVEL_2:
                reasons.append("hard_rule:distress_with_dysregulation_floor_level_2")
            level = max(level, LEVEL_2, key=risk_level_index)

        if not reasons:
            reasons.append("score_threshold:level_0")
        return level, reasons

    def _apply_context_guardrails(
        self,
        level: str,
        dimensions: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        reasons: List[str],
    ) -> tuple[str, List[str]]:
        adjusted_level = level
        adjusted_reasons = list(reasons)

        if context["is_discussion_context"]:
            adjusted_level = LEVEL_0
            adjusted_reasons.append("context_guard:discussion_context")
        elif context["is_third_party_risk"]:
            adjusted_level = LEVEL_1 if context["is_help_request"] else LEVEL_0
            adjusted_reasons.append("context_guard:third_party_context")

        if adjusted_level == LEVEL_0 and dimensions["risk_ideation"]["score"] > 0 and context["is_safe_denial"]:
            adjusted_reasons.append("context_guard:safe_denial")

        return adjusted_level, adjusted_reasons

    def _apply_risk_inertia(
        self,
        level: str,
        dimensions: Dict[str, Dict[str, Any]],
        emotion_state: Dict[str, Any],
        conversation_summary: Dict[str, Any],
        reasons: List[str],
    ) -> tuple[str, List[str]]:
        recent_levels = [
            normalize_risk_level(item)
            for item in (conversation_summary.get("recent_risk_levels", []) or [])
        ]
        if not recent_levels:
            return level, reasons

        latest_level = recent_levels[-1]
        max_recent_level = max(recent_levels, key=risk_level_index)
        stabilized = self._has_stabilizing_signals(
            dimensions=dimensions,
            emotion_state=emotion_state,
            conversation_summary=conversation_summary,
        )
        adjusted_level = level
        adjusted_reasons = list(reasons)

        if latest_level == LEVEL_3 and risk_level_index(level) < risk_level_index(LEVEL_2):
            adjusted_level = LEVEL_2
            adjusted_reasons.append("risk_inertia:recent_level_3_floor_level_2")
        elif max_recent_level == LEVEL_3 and risk_level_index(level) == risk_level_index(LEVEL_0):
            adjusted_level = LEVEL_1 if stabilized else LEVEL_2
            adjusted_reasons.append("risk_inertia:recent_level_3_decay_guard")
        elif latest_level == LEVEL_2 and risk_level_index(level) < risk_level_index(LEVEL_1) and not stabilized:
            adjusted_level = LEVEL_1
            adjusted_reasons.append("risk_inertia:recent_level_2_floor_level_1")
        elif recent_levels.count(LEVEL_2) >= 2 and risk_level_index(level) < risk_level_index(LEVEL_1):
            adjusted_level = LEVEL_1
            adjusted_reasons.append("risk_inertia:repeated_level_2_floor_level_1")

        return adjusted_level, adjusted_reasons

    def _has_stabilizing_signals(
        self,
        dimensions: Dict[str, Dict[str, Any]],
        emotion_state: Dict[str, Any],
        conversation_summary: Dict[str, Any],
    ) -> bool:
        low_risk_signal = (
            dimensions["risk_ideation"]["score"] == 0
            and dimensions["action_intent"]["score"] == 0
            and dimensions["plan_specificity"]["score"] == 0
            and dimensions["time_urgency"]["score"] == 0
            and dimensions["self_control"]["score"] == 0
        )
        has_protection = dimensions["protective_factors"]["score"] == 0
        user_intent = emotion_state.get("user_intent")
        emotion_trend = conversation_summary.get("emotion_trend")
        return (
            low_risk_signal
            and has_protection
            and user_intent in self.help_seeking_intents
            and emotion_trend in {"improving", "calming", "stable"}
        )

    def _build_risk_evidence(
        self,
        dimensions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        return {
            name: item["evidence"]
            for name, item in dimensions.items()
            if item["evidence"]
        }

    def _matched_keywords(self, text: str, keywords: List[str]) -> List[str]:
        found: List[str] = []
        for keyword in keywords:
            if keyword in text and keyword not in found:
                found.append(keyword)
        return found

    def _analyze_context(self, text: str) -> Dict[str, Any]:
        text = text or ""
        discussion_markers = [
            "电影里",
            "电影中",
            "主角",
            "剧情里",
            "新闻里",
            "新闻中",
            "报道里",
            "论文里",
            "论文中",
            "研究里",
            "案例里",
            "书里",
            "作品里",
        ]
        third_party_markers = [
            "我朋友",
            "朋友说",
            "朋友最近",
            "他不想活了",
            "她不想活了",
            "我同学",
            "同学说",
            "室友说",
            "我室友",
            "我家人",
            "我妈",
            "我爸",
            "别人说",
            "有人说",
        ]
        help_request_markers = ["怎么办", "怎么帮", "该怎么做", "我该怎么做", "怎么帮助", "要不要报警"]
        safe_denials = [
            "没有想自杀",
            "没有想伤害自己",
            "没有想伤害别人",
            "不会自杀",
            "不会伤害自己",
            "我没有想死",
        ]

        is_discussion_context = any(marker in text for marker in discussion_markers)
        is_third_party_risk = any(marker in text for marker in third_party_markers)
        is_help_request = any(marker in text for marker in help_request_markers)
        is_safe_denial = any(token in text for token in safe_denials)

        subject = "self"
        if is_discussion_context:
            subject = "discussion"
        elif is_third_party_risk:
            subject = "third_party"

        return {
            "subject": subject,
            "is_discussion_context": is_discussion_context,
            "is_third_party_risk": is_third_party_risk,
            "is_help_request": is_help_request,
            "is_safe_denial": is_safe_denial,
        }

    def _contains_safe_denial(self, text: str) -> bool:
        safe_denials = [
            "没有想自杀",
            "没有想伤害自己",
            "没有想伤害别人",
            "不会自杀",
            "不会伤害自己",
            "我没有想死",
        ]
        return any(token in text for token in safe_denials)

    def _calibrate_display_score(self, level: str, raw_score: float) -> float:
        if level == LEVEL_3:
            return max(raw_score, 8.0)
        if level == LEVEL_2:
            return max(raw_score, 5.0)
        if level == LEVEL_1:
            return max(raw_score, 2.8)
        return raw_score

    def _build_message(self, level: str) -> str:
        messages = {
            LEVEL_3: "检测到紧急风险信号，请优先保证当前安全并立即联系可信任的人或专业支持。",
            LEVEL_2: "检测到高风险倾向，请先停止普通问题分析，优先确认当前安全并寻求现实支持。",
            LEVEL_1: "检测到持续负面或较强烈情绪，需要加强安抚并提供具体支持建议。",
            LEVEL_0: "",
        }
        return messages[level]

    def _build_suggestions(self, level: str) -> List[str]:
        if level == LEVEL_3:
            return [
                "立即联系信任的家人、朋友或辅导员",
                "尽快联系当地心理援助热线或医院急诊",
                "暂时远离可能伤害自己的环境或物品",
            ]
        if level == LEVEL_2:
            return [
                "先确认自己现在是否处于安全环境",
                "联系身边可信任的人，不要独自承受",
                "如果风险感持续上升，尽快联系专业支持",
            ]
        if level == LEVEL_1:
            return [
                "先做一次缓慢深呼吸，给身体一个暂停",
                "把最强烈的感受告诉信任的人",
                "如果状态持续恶化，尽快联系专业支持",
            ]
        return []


risk_evaluator = RiskEvaluator()
