# -*- coding: utf-8 -*-
"""
Risk Pipeline Ablation — 逐层拆解完整风险管线，定位错误来源。

管线层级（对齐 Phase 0.5 §6 / §7）：
  Raw BERT → Rule Override → Context Rules → Session Aggregator → Final Risk

Variant：
  A  raw_bert      仅 4 分类 argmax（无 rule / context / aggregator）
  B  rule_override  raw + 高危正则兜底
  C  context_rules  B + discussion/third_party/safe_denial 上下文规则
  D  full_pipeline  C + SessionRiskAggregator（= 当前生产 risk_evaluator）
  E  rule_only      纯关键词规则 baseline（对照）

每个 case 输出结构化 debug 字段（对齐 Phase 0.5 §9）。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.benchmark_schema import BenchmarkCase
from evaluation.predictors.base import PredictedOutcome

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# 与 ModulePredictor 一致的关键词近似（rule_only 变体）
HIGH_RISK_KEYWORDS_L3 = ["想死", "自杀", "跳楼", "活不下去", "不想活", "伤害自己", "结束生命", "离开这个世界"]
HIGH_RISK_KEYWORDS_L2 = ["撑不住", "快崩溃", "绝望", "控制不住", "没有意义"]
NEGATIVE_EMOTIONS = {"anxiety", "sadness", "anger", "stress", "fatigue",
                     "panic", "fear", "hopelessness", "grief", "guilt",
                     "shame", "loneliness"}


def _rule_keyword_emotion(text: str) -> str:
    if any(k in text for k in HIGH_RISK_KEYWORDS_L3):
        return "hopelessness"
    if any(k in text for k in HIGH_RISK_KEYWORDS_L2):
        return "stress"
    for w in ["焦虑", "紧张", "心慌", "担心", "害怕", "睡不着"]:
        if w in text:
            return "anxiety"
    for w in ["抑郁", "想哭", "低落", "难过", "伤心", "没意思"]:
        if w in text:
            return "sadness"
    for w in ["生气", "愤怒", "烦", "暴躁"]:
        if w in text:
            return "anger"
    return "neutral"


class RiskAblationPredictor:
    """逐层拆解 Risk Pipeline 的预测器。"""

    variants = ["raw_bert", "rule_override", "context_rules", "full_pipeline", "rule_only"]
    name = "risk_ablation"

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        from config import config
        self.model_path = model_path or config.BERT_MODEL_PATH
        self.device = device
        self._predictor = None
        self._risk_evaluator = None

    def _get_predictor(self):
        if self._predictor is None:
            from bert_risk_predictor import BertRiskPredictor
            self._predictor = BertRiskPredictor(model_path=self.model_path, device=self.device)
            logger.info("[RiskAblation] BERT 风险模型已加载")
        return self._predictor

    def _get_risk_evaluator(self):
        if self._risk_evaluator is None:
            from risk_evaluator import risk_evaluator
            self._risk_evaluator = risk_evaluator
        return self._risk_evaluator

    # ------------------------------------------------------------------
    # 各层
    # ------------------------------------------------------------------

    def raw_bert(self, text: str) -> Dict:
        """Variant A: 仅原始 BERT 4 分类（绕过 rule override）。"""
        import torch
        p = self._get_predictor()
        rule_hit = self._match_rule(text)
        try:
            enc = p.tokenizer(
                text, truncation=True, padding="max_length",
                max_length=p.max_length, return_tensors="pt",
            )
            enc = {k: v.to(p.device) for k, v in enc.items()}
            with torch.no_grad():
                outputs = p.model(enc["input_ids"], enc["attention_mask"])

            if p._is_coral:
                from bert_risk_predictor import _coral_to_level
                coral_probs = torch.sigmoid(outputs).cpu().squeeze().tolist()
                if isinstance(coral_probs[0], list):
                    coral_probs = coral_probs[0]
                level = _coral_to_level(coral_probs, threshold=0.28)
                level_map = {"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}
                raw_label = level_map.get(level, 0)
                binary_prob = coral_probs[1] if len(coral_probs) > 1 else 0.0
                probs_4 = [round(x, 4) for x in (coral_probs + [0.0])[:4]]
                logits_4 = []
            else:
                logits_4, logits_2 = outputs
                probs_4 = torch.softmax(logits_4, dim=-1).cpu().squeeze().tolist()
                raw_label = int(torch.argmax(logits_4, dim=-1).cpu().item())
                binary_prob = float(torch.softmax(logits_2, dim=-1)[0, 1].cpu().item())
                logits_4 = [round(float(x), 4) for x in logits_4.cpu().squeeze().tolist()]

            return {
                "logits_4": logits_4,
                "probs_4": [round(float(x), 4) for x in probs_4],
                "raw_label": raw_label,
                "raw_confidence": round(float(probs_4[raw_label]), 4),
                "binary_prob": round(float(binary_prob), 4),
                "rule_matched": rule_hit,
            }
        except Exception as e:  # noqa: BLE001
            logger.error("Raw BERT 推理失败: %s", e)
            return {"logits_4": [], "probs_4": [0, 0, 0, 0], "raw_label": 0,
                    "raw_confidence": 0.0, "binary_prob": 0.0, "rule_matched": rule_hit}

    def _match_rule(self, text: str) -> bool:
        from bert_risk_predictor import rule_match
        return rule_match(text)

    def rule_override_level(self, raw: Dict) -> str:
        """Variant B: raw + rule override。"""
        if raw["rule_matched"]:
            return "level_3"
        return ["level_0", "level_1", "level_2", "level_3"][raw["raw_label"]]

    def context_level(self, text: str, raw: Dict, rule_level: str) -> str:
        """Variant C: B + context rules（无 aggregator）。"""
        evaluator = self._get_risk_evaluator()
        ctx = evaluator._analyze_context(text)
        level = rule_level
        if ctx.get("is_discussion_context") and not raw["rule_matched"]:
            level = "level_0"
        elif ctx.get("is_third_party_risk") and not ctx.get("is_help_request"):
            level = "level_0"
        elif ctx.get("is_safe_denial"):
            order = ["level_0", "level_1", "level_2", "level_3"]
            cur = order.index(level) if level in order else 0
            utt = raw["raw_label"]
            if cur > utt and cur > 0:
                lower = order[cur - 1]
                if order.index(lower) >= utt:
                    level = lower
        return level

    def full_pipeline_level(self, text: str, summary: Dict) -> Dict:
        """Variant D: 完整生产管线（risk_evaluator.evaluate）。"""
        evaluator = self._get_risk_evaluator()
        return evaluator.evaluate(
            text=text,
            emotion_state=None,
            conversation_summary=summary,
            long_term_risk_level="level_0",
            historical_high_risk_count=0,
            risk_baseline="low",
        )

    def rule_only_level(self, text: str) -> str:
        """Variant E: 纯关键词规则。"""
        if any(k in text for k in HIGH_RISK_KEYWORDS_L3):
            return "level_3"
        if any(k in text for k in HIGH_RISK_KEYWORDS_L2):
            return "level_2"
        if _rule_keyword_emotion(text) in NEGATIVE_EMOTIONS:
            return "level_1"
        return "level_0"

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def analyze(self, case: BenchmarkCase) -> Dict:
        """分析一条 case，输出结构化 debug 记录 + 各 variant 等级。"""
        user_turns = [t.content for t in case.conversation if t.role.value == "user"]
        summary: Dict = {"turn_count": 0, "recent_risk_levels": []}

        # 多轮：逐轮用 full_pipeline 累积 recent_risk_levels
        for i, text in enumerate(user_turns[:-1]):
            summary["turn_count"] = i + 1
            r = self.full_pipeline_level(text, summary)
            summary["recent_risk_levels"].append(str(r.get("level", "level_0")))

        text = user_turns[-1]
        summary["turn_count"] = len(user_turns)

        raw = self.raw_bert(text)
        rule_level = self.rule_override_level(raw)
        context_level = self.context_level(text, raw, rule_level)
        full = self.full_pipeline_level(text, summary)
        rule_only = self.rule_only_level(text)

        # 各 variant 等级（映射到 0-3）
        _order = ["level_0", "level_1", "level_2", "level_3"]
        variants = {
            "raw_bert": _order.index(rule_level) if not raw["rule_matched"] else 3,
            "rule_override": _order.index(rule_level),
            "context_rules": _order.index(context_level),
            "full_pipeline": _order.index(str(full.get("level", "level_0"))),
            "rule_only": _order.index(rule_only),
        }

        record = {
            "case_id": case.case_id,
            "text": text,
            "expected": case.expected.risk_level.value,
            "raw_bert": {
                "label": raw["raw_label"],
                "confidence": raw["raw_confidence"],
                "probs": raw["probs_4"],
                "binary_prob": raw["binary_prob"],
            },
            "rule_override": {"triggered": raw["rule_matched"], "rule": "high_risk_patterns" if raw["rule_matched"] else None},
            "context": {
                "subject": (full.get("risk_context") or {}).get("subject", "self"),
                "is_discussion": (full.get("risk_context") or {}).get("is_discussion_context", False),
                "is_third_party": (full.get("risk_context") or {}).get("is_third_party_risk", False),
                "safe_denial": (full.get("risk_context") or {}).get("is_safe_denial", False),
            },
            "aggregator": {
                "input_levels": [str(x) for x in summary["recent_risk_levels"]],
                "output_level": str(full.get("level", "level_0")),
                "trend": full.get("risk_trend", "new"),
                "active_rules": (full.get("session_aggregation") or {}).get("active_rules", []),
            },
            "final_level": _order.index(str(full.get("level", "level_0"))),
            "variants": variants,
        }
        return record


def build_prediction_outcome(record: Dict) -> PredictedOutcome:
    """把 ablation debug 记录转成 PredictedOutcome（供统一评测 runner 复用，可选）。"""
    return PredictedOutcome(
        risk_level=record["variants"]["full_pipeline"],
        extra={"risk_ablation": record},
    )
