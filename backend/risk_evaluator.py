"""
风险评估器 — 基于 BERT v4.2 多任务模型，完全替代旧的规则系统。

保留 _analyze_context() 方法用于第三方危机检测（纯 regex，不依赖规则评分）。
其余 keyword 列表、7 维度打分、风险惯性计算已全部移除。
"""

import logging
import re
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
from session_risk_aggregator import SessionRiskAggregator

logger = logging.getLogger(__name__)


class RiskEvaluator:
    """风险评估器 — BERT 单轮预测 + 会话级聚合（v5.0）。"""

    def __init__(self):
        self._predictor = None
        self._session_aggregator = SessionRiskAggregator()

    # ---------------------------------------------------------------
    # 延迟初始化 BERT 预测器
    # ---------------------------------------------------------------
    def _get_predictor(self):
        if self._predictor is None:
            try:
                from bert_risk_predictor import BertRiskPredictor
                from config import config

                self._predictor = BertRiskPredictor(
                    model_path=config.BERT_MODEL_PATH,
                    device=config.BERT_DEVICE,
                    binary_threshold=config.BERT_BINARY_THRESHOLD,
                )
                logger.info(
                    "BERT 风险预测器已初始化: path=%s device=%s threshold=%.2f",
                    config.BERT_MODEL_PATH, config.BERT_DEVICE, config.BERT_BINARY_THRESHOLD,
                )
            except Exception as e:
                logger.error("BERT 风险预测器初始化失败，使用安全降级: %s", e)
                self._predictor = _FallbackPredictor()
        return self._predictor

    # ---------------------------------------------------------------
    # 公开接口
    # ---------------------------------------------------------------
    def evaluate(
        self,
        text: str,
        emotion_state: Optional[Dict[str, Any]] = None,
        conversation_summary: Optional[Dict[str, Any]] = None,
        long_term_risk_level: Optional[str] = None,
        historical_high_risk_count: int = 0,
        risk_baseline: str = "low",
    ) -> Dict[str, Any]:
        """
        对用户输入进行风险评级。

        返回字典兼容旧版本字段结构，供下游 consume（recommend_gate、
        output_safety_checker、conversation_manager 等）。
        """
        summary = conversation_summary or {}
        predictor = self._get_predictor()
        bert_result = predictor.predict(text)

        utterance_level = bert_result["level"]

        # 会话级聚合（v5.0 + v6.0 Safety Gate）
        session_result = self._session_aggregator.aggregate(
            utterance_level=utterance_level,
            utterance_binary_prob=bert_result["binary_probability"],
            utterance_rule_hit=bert_result["rule_matched"],
            text=text,
            conversation_summary=summary,
            baseline=risk_baseline,
        )

        level = session_result["session_level"]
        level_idx = risk_level_index(level)

        # 上下文分析（保留用于第三方危机路由检测）
        context = self._analyze_context(text)

        # 拼接 escalation_reasons（BERT 来源 + 会话聚合规则）
        escalation_reasons = [f"bert_{bert_result['fusion_source']}"]

        # ================================================================
        # 边界安全后处理：用上下文分析结果覆盖 level
        # 修复评测报告 "边界安全 acc=0.409" 问题——之前检测正确但未生效
        # ================================================================
        _LEVELS = [LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3]
        verb_level = utterance_level  # BERT 原始单轮预测（未聚合前）

        # 规则 A: 讨论语境（电影/新闻/书）且非规则兜底命中 → Level 0
        if context.get("is_discussion_context") and not bert_result.get("rule_matched"):
            level = LEVEL_0
            escalation_reasons.append("context:discussion_floor_0")

        # 规则 B: 第三方语境且非求助意图 → Level 0
        elif context.get("is_third_party_risk") and not context.get("is_help_request"):
            level = LEVEL_0
            escalation_reasons.append("context:third_party_floor_0")

        # 规则 C: 安全否认 → 在不低于 BERT 原始等级的前提下降一级
        elif context.get("is_safe_denial"):
            cur_idx = risk_level_index(level)
            utt_idx = risk_level_index(verb_level)
            if cur_idx > utt_idx and cur_idx > 0:
                lowered = _LEVELS[cur_idx - 1]
                if risk_level_index(lowered) >= utt_idx:
                    level = lowered
                    escalation_reasons.append("context:safe_denial_deescalate")

        level_idx = risk_level_index(level)
        raw_level_map = {0: LEVEL_0, 1: LEVEL_1, 2: LEVEL_2, 3: LEVEL_3}
        raw_4 = raw_level_map.get(bert_result["level_4_prediction"], LEVEL_0)
        if raw_4 != utterance_level:
            escalation_reasons.append(f"bert_fused_from_{raw_4}")
        escalation_reasons.extend(session_result["session_aggregation"]["active_rules"])

        # 计算 risk_score（基于 BERT 二分类概率和会话级等级索引）
        risk_score = max(
            bert_result["binary_probability"] * 10.0,
            level_idx * 2.5,
            min(bert_result["binary_probability"] * level_idx * 3.0 + 1.0, 10.0),
        )

        return {
            "level": level,
            "legacy_level": risk_level_band(level),
            "level_index": level_idx,
            "level_label": risk_level_label(level),
            "message": self._build_message(level),
            "suggestions": self._build_suggestions(level),
            "risk_score": round(min(risk_score, 10.0), 2),
            "raw_score": round(bert_result["binary_probability"] * 10.0, 2),
            "risk_evidence": {
                "bert": {
                    "fusion_source": bert_result["fusion_source"],
                    "level_4_prediction": bert_result["level_4_prediction"],
                    "binary_probability": bert_result["binary_probability"],
                    "class_probabilities": bert_result["class_probabilities"],
                    "rule_matched": bert_result["rule_matched"],
                },
            },
            "risk_trend": session_result["risk_trend"],
            "session_aggregation": session_result["session_aggregation"],
            "escalation_reasons": escalation_reasons,
            "recent_risk_levels": [
                normalize_risk_level(item)
                for item in (summary.get("recent_risk_levels", []) or [])
            ],
            "risk_context": context,
        }

    # ---------------------------------------------------------------
    # 上下文分析（仅用于第三方/讨论语境识别，下游路由需要）
    # ---------------------------------------------------------------
    def _analyze_context(self, text: str) -> Dict[str, Any]:
        """识别文本的主体归属：讨论、第三方、安全否认。"""
        text = text or ""
        discussion_markers = [
            "电影里", "电影中", "主角", "剧情里",
            "新闻里", "新闻中", "报道里", "论文里", "论文中",
            "研究里", "案例里", "书里", "作品里",
        ]
        third_party_markers = [
            "我朋友", "朋友说", "朋友最近",
            "他不想活了", "她不想活了",
            "我同学", "同学说", "室友说", "我室友",
            "我家人", "我妈", "我爸",
            "别人说", "有人说",
        ]
        help_request_markers = [
            "怎么办", "怎么帮", "该怎么做", "我该怎么做",
            "怎么帮助", "要不要报警",
        ]
        safe_denials = [
            "没有想自杀", "没有想伤害自己", "没有想伤害别人",
            "不会自杀", "不会伤害自己", "我没有想死",
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

    # ---------------------------------------------------------------
    # 辅助：消息 & 建议
    # ---------------------------------------------------------------
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


class _FallbackPredictor:
    """BERT 模型不可用时的安全降级预测器，始终返回 LEVEL_0。"""

    def predict(self, text: str) -> Dict:
        return {
            "level": LEVEL_0,
            "level_4_prediction": 0,
            "binary_probability": 0.0,
            "class_probabilities": [0.0, 0.0, 0.0, 0.0],
            "fusion_source": "fallback",
            "rule_matched": False,
        }


# 全局单例
risk_evaluator = RiskEvaluator()
