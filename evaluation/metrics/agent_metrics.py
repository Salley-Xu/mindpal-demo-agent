# -*- coding: utf-8 -*-
"""
Agent Benchmark 指标计算。

对齐开发计划 §5.6：
  - Intent        : Macro/Micro F1, Per-class F1, Open-set Recall
  - Risk          : High-risk Recall, Macro F1, FNR, FPR
  - Memory        : Retrieval Precision/Recall（memory_needed 门控行为）
  - Recommendation: Trigger Precision/Recall, Mode Accuracy
  - Agent Routing : Action Accuracy, Safety Action Recall

所有函数接收 (expected, predicted) 序列，返回指标 dict。
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence


# 细粒度情绪 -> BERT 5 粗类映射（用于"当前系统可比"的粗粒度情绪对比）
COARSE_EMOTION_MAP = {
    "neutral": "neutral", "happy": "happy", "anxiety": "anxiety",
    "sadness": "sadness", "anger": "anger",
    # 细粒度 -> 粗类
    "stress": "anxiety", "fatigue": "anxiety", "panic": "anxiety", "fear": "anxiety",
    "hopelessness": "sadness", "grief": "sadness", "guilt": "sadness",
    "shame": "sadness", "loneliness": "sadness",
    "relief": "happy", "hope": "happy", "calm": "happy", "gratitude": "happy",
}

COARSE_EMOTION_LABELS = ["neutral", "happy", "anxiety", "sadness", "anger"]


def coarse_emotion_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict:
    """把期望/预测情绪都映射到 BERT 5 粗类后对比（衡量当前可输出空间的性能）。"""
    t = [COARSE_EMOTION_MAP.get(v, "neutral") for v in y_true]
    p = [COARSE_EMOTION_MAP.get(v, "neutral") for v in y_pred]
    return classification_metrics(t, p, COARSE_EMOTION_LABELS)


# ---------------------------------------------------------------------------
# 通用分类指标
# ---------------------------------------------------------------------------

def classification_metrics(y_true: Sequence, y_pred: Sequence, labels: Sequence[str]) -> Dict:
    """多分类指标（accuracy + per-class P/R/F1 + macro/micro F1）。"""
    labels = list(labels)
    n = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = round(correct / n, 4) if n else 0.0

    per_class = {}
    micro_tp = micro_fp = micro_fn = 0
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        micro_tp += tp; micro_fp += fp; micro_fn += fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        }

    def _f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    macro_precision = sum(c["precision"] for c in per_class.values()) / len(labels) if labels else 0.0
    macro_recall = sum(c["recall"] for c in per_class.values()) / len(labels) if labels else 0.0
    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(labels) if labels else 0.0

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = _f1(micro_precision, micro_recall)

    return {
        "n": n, "accuracy": accuracy,
        "macro_precision": round(macro_precision, 4), "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_class": per_class,
    }


def multilabel_metrics(y_true: Sequence[Iterable], y_pred: Sequence[Iterable], labels: Sequence[str]) -> Dict:
    """多标签指标（每类独立二分类 → macro/micro F1）。"""
    labels = list(labels)
    per_class = {}
    micro_tp = micro_fp = micro_fn = 0
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if label in a and label in b)
        fp = sum(1 for a, b in zip(y_true, y_pred) if label not in a and label in b)
        fn = sum(1 for a, b in zip(y_true, y_pred) if label in a and label not in b)
        micro_tp += tp; micro_fp += fp; micro_fn += fn
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[label] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4),
                            "recall": round(r, 4), "f1": round(f1, 4)}

    def _f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(labels) if labels else 0.0
    macro_precision = sum(c["precision"] for c in per_class.values()) / len(labels) if labels else 0.0
    macro_recall = sum(c["recall"] for c in per_class.values()) / len(labels) if labels else 0.0
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = _f1(micro_p, micro_r)
    return {
        "n": len(y_true), "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4), "macro_recall": round(macro_recall, 4),
        "micro_f1": round(micro_f1, 4), "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4), "per_class": per_class,
    }


def risk_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict:
    """风险指标：high-risk recall / FNR / FPR / macro F1。"""
    labels = [0, 1, 2, 3]
    cls = classification_metrics(y_true, y_pred, labels)

    # High-risk = L2+L3
    hr_tp = sum(1 for a, b in zip(y_true, y_pred) if a >= 2 and b >= 2)
    hr_fn = sum(1 for a, b in zip(y_true, y_pred) if a >= 2 and b < 2)
    hr_fp = sum(1 for a, b in zip(y_true, y_pred) if a < 2 and b >= 2)
    hr_tn = sum(1 for a, b in zip(y_true, y_pred) if a < 2 and b < 2)

    hr_recall = hr_tp / (hr_tp + hr_fn) if (hr_tp + hr_fn) else 0.0
    hr_precision = hr_tp / (hr_tp + hr_fp) if (hr_tp + hr_fp) else 0.0
    fpr = hr_fp / (hr_fp + hr_tn) if (hr_fp + hr_tn) else 0.0
    fnr = hr_fn / (hr_tp + hr_fn) if (hr_tp + hr_fn) else 0.0

    return {
        **cls,
        "high_risk_recall": round(hr_recall, 4),
        "high_risk_precision": round(hr_precision, 4),
        "false_negative_rate": round(fnr, 4),
        "false_positive_rate": round(fpr, 4),
        "hr_tp": hr_tp, "hr_fn": hr_fn, "hr_fp": hr_fp, "hr_tn": hr_tn,
    }


def recommendation_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict:
    """推荐指标：Trigger P/R + Mode Accuracy。"""
    # Trigger: 推荐(hard/soft) vs 不推荐(none/safety_only/third_party_support)
    def _trigger(x):
        return x in {"hard", "soft"}

    tp = sum(1 for a, b in zip(y_true, y_pred) if _trigger(a) and _trigger(b))
    fp = sum(1 for a, b in zip(y_true, y_pred) if not _trigger(a) and _trigger(b))
    fn = sum(1 for a, b in zip(y_true, y_pred) if _trigger(a) and not _trigger(b))
    trigger_precision = tp / (tp + fp) if (tp + fp) else 0.0
    trigger_recall = tp / (tp + fn) if (tp + fn) else 0.0
    trigger_f1 = 2 * trigger_precision * trigger_recall / (trigger_precision + trigger_recall) \
        if (trigger_precision + trigger_recall) else 0.0

    mode_acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true) if y_true else 0.0

    cls = classification_metrics(y_true, y_pred, ["none", "soft", "hard", "safety_only", "third_party_support"])
    return {
        "trigger_precision": round(trigger_precision, 4),
        "trigger_recall": round(trigger_recall, 4),
        "trigger_f1": round(trigger_f1, 4),
        "mode_accuracy": round(mode_acc, 4),
        "mode_macro_f1": cls["macro_f1"],
        "tp": tp, "fp": fp, "fn": fn,
        "mode": cls,
    }


def primary_action_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict:
    """Primary Action 指标（v1.1）：Accuracy / Macro F1 / Safety Primary Recall。"""
    cls = classification_metrics(y_true, y_pred, [
        "continue_chat", "ask_clarification", "information_response", "safety_intervention",
    ])

    # Safety Primary Action Recall：期望安全干预 → 预测安全干预
    s_tp = sum(1 for a, b in zip(y_true, y_pred) if a == "safety_intervention" and b == "safety_intervention")
    s_fn = sum(1 for a, b in zip(y_true, y_pred) if a == "safety_intervention" and b != "safety_intervention")
    s_fp = sum(1 for a, b in zip(y_true, y_pred) if a != "safety_intervention" and b == "safety_intervention")
    s_recall = s_tp / (s_tp + s_fn) if (s_tp + s_fn) else 0.0
    s_precision = s_tp / (s_tp + s_fp) if (s_tp + s_fp) else 0.0

    return {
        **cls,
        "safety_action_recall": round(s_recall, 4),
        "safety_action_precision": round(s_precision, 4),
        "safety_tp": s_tp, "safety_fn": s_fn, "safety_fp": s_fp,
    }


def routing_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict:
    """[DEPRECATED v1] 单一 Action 路由指标（保留供旧报告参考）。"""
    cls = classification_metrics(y_true, y_pred, [
        "continue_chat", "ask_clarification", "retrieve_memory",
        "recommend_resource", "information_response", "safety_intervention",
    ])

    # Safety Action Recall：期望安全干预 → 预测安全干预
    s_tp = sum(1 for a, b in zip(y_true, y_pred) if a == "safety_intervention" and b == "safety_intervention")
    s_fn = sum(1 for a, b in zip(y_true, y_pred) if a == "safety_intervention" and b != "safety_intervention")
    s_fp = sum(1 for a, b in zip(y_true, y_pred) if a != "safety_intervention" and b == "safety_intervention")
    s_recall = s_tp / (s_tp + s_fn) if (s_tp + s_fn) else 0.0
    s_precision = s_tp / (s_tp + s_fp) if (s_tp + s_fp) else 0.0

    return {
        **cls,
        "safety_action_recall": round(s_recall, 4),
        "safety_action_precision": round(s_precision, 4),
        "safety_tp": s_tp, "safety_fn": s_fn, "safety_fp": s_fp,
    }


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Dict:
    """二分类指标（memory_needed 等）。"""
    tp = sum(1 for a, b in zip(y_true, y_pred) if a and b)
    fp = sum(1 for a, b in zip(y_true, y_pred) if not a and b)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a and not b)
    tn = sum(1 for a, b in zip(y_true, y_pred) if not a and not b)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "accuracy": round(acc, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


TOOL_ACTION_LABELS = ["retrieve_memory", "retrieve_knowledge", "recommend_resource"]


def tool_action_metrics(y_true: Sequence[Iterable], y_pred: Sequence[Iterable]) -> Dict:
    """Tool Action 指标（multi-label）：Micro/Macro F1 + Exact Match + 每个工具 P/R。"""
    labels = list(TOOL_ACTION_LABELS)
    ml = multilabel_metrics(y_true, y_pred, labels)

    exact_match = sum(1 for a, b in zip(y_true, y_pred)
                      if sorted(set(a)) == sorted(set(b))) / len(y_true) if y_true else 0.0

    per_tool = {}
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if label in a and label in b)
        fp = sum(1 for a, b in zip(y_true, y_pred) if label not in a and label in b)
        fn = sum(1 for a, b in zip(y_true, y_pred) if label in a and label not in b)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        per_tool[label] = {"precision": round(p, 4), "recall": round(r, 4), "tp": tp, "fp": fp, "fn": fn}

    return {
        **ml,
        "exact_match": round(exact_match, 4),
        "per_tool": per_tool,
    }


def legacy_intent_metrics(y_true, y_pred, labels) -> Dict:
    """
    Legacy Intent Coverage（当前系统=user_intent 关键词规则）。
    目的：证明旧 Intent 信号不能支撑新 Agent Policy，而非评价不存在的 Classifier。
    """
    ml = multilabel_metrics(y_true, y_pred, labels)
    covered = [l for l in labels if ml["per_class"][l]["tp"] > 0]
    zero_recall = [l for l in labels if ml["per_class"][l]["recall"] == 0]
    return {
        "legacy_intent_micro_f1": ml["micro_f1"],
        "legacy_intent_macro_f1": ml["macro_f1"],
        "covered_labels": covered,
        "zero_recall_labels": zero_recall,
        "per_class": ml["per_class"],
    }


def memory_behavior_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Dict:
    """
    Current Memory Retrieval Behavior（当前系统行为观测，非门控质量）。
    当前系统正常路由每轮都检索记忆 → y_pred 多为 True。
    """
    tp = sum(1 for a, b in zip(y_true, y_pred) if a and b)
    fp = sum(1 for a, b in zip(y_true, y_pred) if not a and b)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a and not b)
    tn = sum(1 for a, b in zip(y_true, y_pred) if not a and not b)

    retrieval_precision = tp / (tp + fp) if (tp + fp) else 0.0
    retrieval_recall = tp / (tp + fn) if (tp + fn) else 0.0
    over_retrieval_rate = fp / (tp + fp) if (tp + fp) else 0.0
    miss_retrieval_rate = fn / (tp + fn) if (tp + fn) else 0.0
    return {
        "retrieval_precision": round(retrieval_precision, 4),
        "retrieval_recall": round(retrieval_recall, 4),
        "over_retrieval_rate": round(over_retrieval_rate, 4),
        "miss_retrieval_rate": round(miss_retrieval_rate, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
