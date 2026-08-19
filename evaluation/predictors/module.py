# -*- coding: utf-8 -*-
"""
ModulePredictor — 直接调用当前系统真实决策模块的 Baseline 预测器。

测量的对象（全部离线、确定性）：
  - 情绪     : BERT 情绪模型（bert_emotion_predictor）或规则近似
  - 意图     : 复刻 emotion_analyzer._detect_user_intent 的当前关键词规则（4 类）
  - 风险     : risk_evaluator（BERT v4.2 + SessionRiskAggregator 状态机）或规则近似
  - 推荐门控 : recommend_gate.decide（纯规则加权）
  - 动作     : 由风险 + 门控 + 意图推导（current_routing 单独记录当前确定性路由）

注意：
  - 当前系统没有独立 intent 分类器，intent 预测是对 user_intent 关键词规则的复刻。
  - 当前系统没有 AgentAction / memory_needed 字段，action 由确定性信号推导。
  - 多轮 case 逐轮累加 recent_risk_levels 到 summary，模拟会话级聚合。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.benchmark_schema import BenchmarkCase
from evaluation.predictors.base import BasePredictor, PredictedOutcome

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ---------------------------------------------------------------------------
# 情绪标签映射
# ---------------------------------------------------------------------------

# 规范英文标签 -> 系统中文标签（用于 emotion_state payload）
CANON_TO_CHINESE = {
    "neutral": "中性", "happy": "快乐", "anxiety": "焦虑", "sadness": "抑郁",
    "anger": "愤怒", "stress": "压力", "fatigue": "疲惫", "panic": "恐慌",
    "fear": "害怕", "hopelessness": "绝望", "grief": "悲伤", "guilt": "愧疚",
    "shame": "羞耻", "loneliness": "孤独", "relief": "释然", "hope": "希望",
    "calm": "平静", "gratitude": "感激",
}

NEGATIVE_EMOTIONS = {"anxiety", "sadness", "anger", "stress", "fatigue",
                     "panic", "fear", "hopelessness", "grief", "guilt",
                     "shame", "loneliness"}

# BERT 5 类（英文） -> 规范标签
BERT_EMOTION_TO_CANON = {
    "neutral": "neutral", "positive": "happy", "anxiety": "anxiety",
    "sadness": "sadness", "anger": "anger",
}

# 规则情绪关键词（优先级从高到低）
RULE_EMOTION_KEYWORDS = [
    ("hopelessness", ["想死", "自杀", "活不下去", "没意义", "绝望", "不想活"]),
    ("anger", ["生气", "愤怒", "气死", "烦死了", "暴躁", "恼火"]),
    ("panic", ["惊恐", "失控", "崩溃", "喘不过气", "发抖"]),
    ("anxiety", ["焦虑", "紧张", "心慌", "担心", "害怕", "睡不着", "失眠"]),
    ("sadness", ["抑郁", "想哭", "低落", "难过", "伤心", "失落", "没意思"]),
    ("stress", ["压力", "加班", "累", "喘不过气", "撑不住"]),
    ("fatigue", ["疲惫", "没精神", "很累", "不想动"]),
    ("happy", ["开心", "高兴", "太好", "不错", "棒"]),
]

# 当前系统 _detect_user_intent 的关键词规则（原样复刻）
USER_INTENT_RULES = {
    "seeking_help": ["怎么办", "怎么做", "帮帮我", "我该怎么办", "有什么办法"],
    "planning": ["计划", "安排", "拆分", "步骤", "怎么开始"],
    "seeking_relief": ["缓解", "放松", "平静", "睡着", "减轻"],
    "sharing": [],
}

# 当前系统 _detect_stress_source 的关键词规则（原样复刻）
STRESS_SOURCE_RULES = {
    "学业/求职压力": ["考试", "论文", "学习", "毕业", "面试", "求职", "找工作", "实习"],
    "人际关系压力": ["朋友", "室友", "对象", "男朋友", "女朋友", "关系", "沟通", "家人"],
    "未来规划压力": ["未来", "方向", "规划", "迷茫", "不知道做什么"],
    "自我评价压力": ["自己不行", "怀疑自己", "没用", "做不好", "不够好"],
    "睡眠/身体压力": ["睡不着", "失眠", "很累", "身体", "头疼", "没精神"],
}

# 高风险关键词近似（用于 rule 风险通道；BERT 通道用真实模型）
HIGH_RISK_KEYWORDS_L3 = ["想死", "自杀", "跳楼", "活不下去", "不想活", "伤害自己", "结束生命", "离开这个世界"]
HIGH_RISK_KEYWORDS_L2 = ["撑不住", "快崩溃", "绝望", "控制不住", "没有意义"]


def detect_user_intent(text: str) -> str:
    """复刻当前系统 _detect_user_intent。"""
    for intent, keywords in USER_INTENT_RULES.items():
        if keywords and any(k in text for k in keywords):
            return intent
    return "sharing"


def detect_stress_source(text: str) -> Optional[str]:
    """复刻当前系统 _detect_stress_source。"""
    for source, keywords in STRESS_SOURCE_RULES.items():
        if any(k in text for k in keywords):
            return source
    return None


def rule_emotion(text: str) -> str:
    """规则情绪分类（近似 BERT 情绪）。"""
    for label, keywords in RULE_EMOTION_KEYWORDS:
        if any(k in text for k in keywords):
            return label
    return "neutral"


def rule_intensity(emotion: str, text: str) -> float:
    """估计情绪强度（近似系统 LLM/BERT 启发式）。"""
    base = 0.5
    if emotion in {"anxiety", "sadness", "anger", "stress", "panic", "hopelessness"}:
        base += 0.15
    if any(k in text for k in HIGH_RISK_KEYWORDS_L3 + HIGH_RISK_KEYWORDS_L2):
        base += 0.15
    return round(min(1.0, base), 2)


class ModulePredictor(BasePredictor):
    """
    基于当前系统真实模块的预测器。

    参数：
      emotion_channel : "bert" | "rule"
      risk_channel    : "bert" | "rule"
    """

    name = "module"

    def __init__(self, emotion_channel: str = "bert", risk_channel: str = "bert"):
        self.emotion_channel = emotion_channel
        self.risk_channel = risk_channel
        self._emotion_predictor = None
        self._risk_evaluator = None
        self._gate = None

    # ---- 懒加载 ----

    def _get_emotion_predictor(self):
        if self._emotion_predictor is None:
            from bert_emotion_predictor import BertEmotionPredictor
            from config import config
            self._emotion_predictor = BertEmotionPredictor(
                model_path=config.EMOTION_MODEL_PATH,
                device=config.EMOTION_DEVICE,
                confidence_threshold=config.EMOTION_CONFIDENCE_THRESHOLD,
            )
            logger.info("[ModulePredictor] BERT 情绪模型已加载")
        return self._emotion_predictor

    def _get_risk_evaluator(self):
        if self._risk_evaluator is None:
            from risk_evaluator import risk_evaluator
            self._risk_evaluator = risk_evaluator
            logger.info("[ModulePredictor] risk_evaluator 就绪")
        return self._risk_evaluator

    def _get_gate(self):
        if self._gate is None:
            from recommend_gate import recommend_gate
            self._gate = recommend_gate
        return self._gate

    # ---- 感知通道 ----

    def _predict_emotion(self, text: str) -> str:
        if self.emotion_channel == "bert":
            pred = self._get_emotion_predictor()
            label, conf = pred.predict(text)
            canon = BERT_EMOTION_TO_CANON.get(label, "neutral")
            self._last_emotion_conf = conf
            return canon
        return rule_emotion(text)

    def _predict_risk(self, text: str, emotion_payload: Dict, summary: Dict) -> Dict:
        if self.risk_channel == "rule":
            return self._rule_risk(text)
        evaluator = self._get_risk_evaluator()
        return evaluator.evaluate(
            text=text,
            emotion_state=emotion_payload,
            conversation_summary=summary,
            long_term_risk_level="level_0",
            historical_high_risk_count=0,
            risk_baseline="low",
        )

    def _rule_risk(self, text: str) -> Dict:
        if any(k in text for k in HIGH_RISK_KEYWORDS_L3):
            return {"level": "level_3", "risk_score": 0.9, "risk_trend": "new",
                    "risk_context": {}, "session_aggregation": {}}
        if any(k in text for k in HIGH_RISK_KEYWORDS_L2):
            return {"level": "level_2", "risk_score": 0.75, "risk_trend": "new",
                    "risk_context": {}, "session_aggregation": {}}
        emotion = rule_emotion(text)
        if emotion in NEGATIVE_EMOTIONS:
            return {"level": "level_1", "risk_score": 0.4, "risk_trend": "new",
                    "risk_context": {}, "session_aggregation": {}}
        return {"level": "level_0", "risk_score": 0.0, "risk_trend": "new",
                "risk_context": {}, "session_aggregation": {}}

    # ---- 动作推导 ----

    def _is_third_party_crisis(self, urgent: Dict) -> bool:
        ctx = (urgent or {}).get("risk_context", {})
        return ctx.get("subject") == "third_party" and ctx.get("is_help_request") is True

    def _map_recommendation(self, urgent: Dict, decision: Dict, risk_level: int) -> str:
        if risk_level >= 3:
            return "none"
        if risk_level == 2 or self._is_third_party_crisis(urgent):
            return "safety_only"
        return decision.get("recommend_type", "none")

    def _derive_primary_action(self, urgent: Dict, intent: List[str], risk_level: int) -> str:
        """由确定性信号推导的 Primary Action（Phase 3 应形式化此逻辑）。"""
        if self._is_third_party_crisis(urgent):
            return "safety_intervention"
        if risk_level >= 2:
            return "safety_intervention"
        if "information_request" in intent:
            return "information_response"
        if not intent:
            return "ask_clarification"
        return "continue_chat"

    def _derive_tool_actions(self, urgent: Dict, intent: List[str], decision: Dict,
                             risk_level: int, memory_needed: bool) -> List[str]:
        """由确定性信号推导的 Tool Actions（可多选）。"""
        tools: List[str] = []
        if risk_level >= 2 or self._is_third_party_crisis(urgent):
            return tools
        if "resource_request" in intent and decision.get("recommend_type") == "hard":
            tools.append("recommend_resource")
        if memory_needed or "memory_reference" in intent:
            if "retrieve_memory" not in tools:
                tools.append("retrieve_memory")
        if "information_request" in intent and "retrieve_knowledge" not in tools:
            tools.append("retrieve_knowledge")
        return tools

    def _current_routing_action(self, urgent: Dict, risk_level: int) -> str:
        """当前系统的确定性路由（仅安全分级 + ReAct，无显式 action 层）。"""
        if self._is_third_party_crisis(urgent):
            return "safety_intervention"
        if risk_level >= 2:
            return "safety_intervention"
        return "continue_chat"

    def _derive_safety_target(self, urgent: Dict, risk_level: int) -> str:
        if self._is_third_party_crisis(urgent):
            return "third_party"
        if risk_level >= 2:
            return "self"
        return "none"

    # ---- 主入口 ----

    def predict(self, case: BenchmarkCase) -> PredictedOutcome:
        # 取最后一个 user turn 作为当前输入；多轮时逐轮累加风险到 summary
        user_turns = [t.content for t in case.conversation if t.role.value == "user"]
        last_text = user_turns[-1]

        # 会话摘要（多轮模拟：累积 recent_risk_levels / turn_count）
        summary: Dict = {"conversation_stage": "initial", "turn_count": 0,
                         "recent_risk_levels": [], "recent_intents": []}
        last_urgent: Dict = {}

        for i, text in enumerate(user_turns):
            summary["turn_count"] = i + 1
            # 情绪感知（仅对每轮更新 primary_emotion，最终轮作为预测）
            canon = self._predict_emotion(text)
            summary["primary_emotion"] = CANON_TO_CHINESE.get(canon, "中性")
            user_intent = detect_user_intent(text)
            summary["recent_intents"].append(user_intent)
            stress_source = detect_stress_source(text)
            negative = canon in NEGATIVE_EMOTIONS

            emotion_payload = {
                "current_emotion": CANON_TO_CHINESE.get(canon, "中性"),
                "emotion_type": canon,
                "context_emotion": "中性",
                "emotion_intensity": rule_intensity(canon, text),
                "stress_source": stress_source,
                "user_intent": user_intent,
                "negative_trend": negative,
                "confidence": 0.7,
                "emotion_trend": "new",
            }

            urgent = self._predict_risk(text, emotion_payload, summary)
            # 累积本轮风险等级供下一轮聚合
            level = str(urgent.get("level", "level_0"))
            summary["recent_risk_levels"].append(level)
            last_urgent = urgent

        # ---- 最终预测使用最后一轮的状态 ----
        final_emotion = self._predict_emotion(last_text)
        final_user_intent = detect_user_intent(last_text)
        final_intensity = rule_intensity(final_emotion, last_text)
        final_stress = detect_stress_source(last_text)
        final_negative = final_emotion in NEGATIVE_EMOTIONS

        final_emotion_payload = {
            "current_emotion": CANON_TO_CHINESE.get(final_emotion, "中性"),
            "emotion_type": final_emotion,
            "context_emotion": "中性",
            "emotion_intensity": final_intensity,
            "stress_source": final_stress,
            "user_intent": final_user_intent,
            "negative_trend": final_negative,
            "confidence": getattr(self, "_last_emotion_conf", 0.7),
            "emotion_trend": "new",
        }

        # 风险（用累积后的 summary 再算最终轮，得到会话级聚合结果）
        final_urgent = self._predict_risk(last_text, final_emotion_payload, summary)
        risk_level_str = str(final_urgent.get("level", "level_0"))
        risk_level = self._parse_level(risk_level_str)

        # 推荐门控
        gate = self._get_gate()
        decision = gate.decide(
            emotion_state=final_emotion_payload,
            risk_state=final_urgent,
            conversation_summary=summary,
            user_profile={},
        ) if self.risk_channel == "bert" else self._rule_gate(final_emotion_payload, final_urgent, summary)

        # 意图 -> 本 Schema 的 Intent Taxonomy（对 user_intent 关键词规则的映射）
        intent = self._map_intent(final_user_intent, final_emotion)

        # 记忆/知识（当前系统正常路由每轮都尝试检索；安全路由跳过）
        routed_safety = risk_level >= 2 or self._is_third_party_crisis(final_urgent)
        memory_needed = not routed_safety
        retrieve_knowledge = not routed_safety

        recommendation_action = self._map_recommendation(final_urgent, decision, risk_level)
        primary_action = self._derive_primary_action(final_urgent, intent, risk_level)
        tool_actions = self._derive_tool_actions(final_urgent, intent, decision, risk_level, memory_needed)
        safety_target = self._derive_safety_target(final_urgent, risk_level)
        current_routing = self._current_routing_action(final_urgent, risk_level)

        return PredictedOutcome(
            intent=intent,
            intent_confidence=0.6,
            emotion=final_emotion,
            emotion_intensity=final_intensity,
            risk_level=risk_level,
            risk_trend=self._parse_trend(final_urgent.get("risk_trend", "new")),
            memory_needed=memory_needed,
            recommendation_action=recommendation_action,
            primary_action=primary_action,
            tool_actions=tool_actions,
            safety_target=safety_target,
            extra={
                "user_intent": final_user_intent,
                "stress_source": final_stress,
                "gate": decision,
                "risk_context": final_urgent.get("risk_context", {}),
                "session_aggregation": final_urgent.get("session_aggregation", {}),
                "current_routing_action": current_routing,
                "raw_risk": {k: v for k, v in final_urgent.items() if k in
                             {"level", "risk_score", "risk_trend", "escalation_reasons"}},
            },
        )

    # ---- 辅助 ----

    def _map_intent(self, user_intent: str, emotion: str) -> List[str]:
        if user_intent == "seeking_help":
            return ["explicit_help_request"]
        if user_intent == "planning":
            return ["explicit_help_request"]
        if user_intent == "seeking_relief":
            return ["explicit_help_request", "resource_request"]
        # sharing：按情绪归为情绪表达或闲聊
        if emotion in NEGATIVE_EMOTIONS:
            return ["emotional_expression"]
        return ["casual_chat"]

    @staticmethod
    def _parse_level(level: str) -> int:
        m = {"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}
        return m.get(str(level).lower(), 0)

    @staticmethod
    def _parse_trend(trend: str) -> str:
        t = str(trend or "").lower()
        return t if t in {"rising", "stable", "falling", "fluctuating", "new"} else "new"

    def _rule_gate(self, emotion_payload: Dict, urgent: Dict, summary: Dict) -> Dict:
        """规则近似 recommend_gate（仅当 risk_channel=rule 时使用）。"""
        level = self._parse_level(str(urgent.get("level", "level_0")))
        if level >= 3:
            return {"should_recommend": False, "recommend_type": "none", "score": 0.0,
                    "threshold": 0.0, "reason_codes": ["risk_l3_block"], "cooldown_remaining": 0}
        if level == 2:
            return {"should_recommend": False, "recommend_type": "safety_only", "score": 0.0,
                    "threshold": 0.0, "reason_codes": ["risk_l2_safety"], "cooldown_remaining": 0}
        intensity = emotion_payload.get("emotion_intensity", 0.5)
        score = 0.30 * intensity
        if emotion_payload.get("user_intent") == "seeking_help":
            score += 0.23 * 0.75
        elif emotion_payload.get("user_intent") == "seeking_relief":
            score += 0.23 * 0.45
        if emotion_payload.get("negative_trend"):
            score += 0.14 * 0.35
        if score >= 0.58:
            rec = "hard"
        elif score >= 0.20:
            rec = "soft"
        else:
            rec = "none"
        return {"should_recommend": rec in {"hard", "soft"}, "recommend_type": rec,
                "score": round(score, 3), "threshold": 0.2, "reason_codes": [], "cooldown_remaining": 0}
