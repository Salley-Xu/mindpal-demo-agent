# -*- coding: utf-8 -*-
"""
Policy Metrics（Phase 3 §18）。

统一 Policy 评测指标：Primary / Safety / Tool / Recommendation / Exact / LLM Rate。
"""
from __future__ import annotations

from typing import Dict, List

PRIMARY_LABELS = ["continue_chat", "ask_clarification", "information_response", "safety_intervention"]
TOOL_LABELS = ["retrieve_memory", "retrieve_knowledge", "recommend_resource"]
SAFETY_LABELS = ["none", "self", "third_party"]
REC_LABELS = ["none", "soft", "hard", "safety_only"]


def classification_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    """多分类指标（accuracy + macro F1）。"""
    n = len(y_true)
    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n if n else 0
    per_class_f1 = {}
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        per_class_f1[label] = round(f1, 4)
    macro_f1 = sum(per_class_f1.values()) / len(labels) if labels else 0
    return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "per_class_f1": per_class_f1}


def tool_action_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
    """多标签 tool 指标（micro F1 + exact match）。"""
    # 展平
    tp = fp = fn = 0
    exact = 0
    for a, b in zip(y_true, y_pred):
        sa, sb = set(a), set(b)
        tp += len(sa & sb)
        fp += len(sb - sa)
        fn += len(sa - sb)
        if sa == sb:
            exact += 1
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    micro_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    # macro F1（per-tool）
    macro = []
    for tool in TOOL_LABELS:
        t = sum(1 for a in y_true if tool in a)
        p = sum(1 for a, b in zip(y_true, y_pred) if tool in a and tool in b)
        pp = sum(1 for b in y_pred if tool in b)
        pr = p / pp if pp else 0
        rc = p / t if t else 0
        macro.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0)
    macro_f1 = sum(macro) / len(macro) if macro else 0
    return {"micro_f1": round(micro_f1, 4), "macro_f1": round(macro_f1, 4),
            "exact_match": round(exact / len(y_true), 4) if y_true else 0,
            "precision": round(prec, 4), "recall": round(rec, 4)}


def policy_metrics(rows: List[Dict]) -> Dict:
    """
    从 rows（每行含 expected/predicted 的 4 字段）计算完整 Policy 指标。
    rows[].expected = {primary_action, tool_actions, safety_target, recommendation_mode}
    """
    pa_t = [r["expected"]["primary_action"] for r in rows]
    pa_p = [r["predicted"]["primary_action"] for r in rows]
    st_t = [r["expected"]["safety_target"] for r in rows]
    st_p = [r["predicted"]["safety_target"] for r in rows]
    rm_t = [r["expected"]["recommendation_mode"] for r in rows]
    rm_p = [r["predicted"]["recommendation_mode"] for r in rows]
    ta_t = [r["expected"]["tool_actions"] for r in rows]
    ta_p = [r["predicted"]["tool_actions"] for r in rows]

    pa = classification_metrics(pa_t, pa_p, PRIMARY_LABELS)
    st = classification_metrics(st_t, st_p, SAFETY_LABELS)
    rm = classification_metrics(rm_t, rm_p, REC_LABELS)
    ta = tool_action_metrics(ta_t, ta_p)
    exact = sum(1 for r in rows if r["expected"] == r["predicted"]) / len(rows)

    s_tp = sum(1 for r in rows if r["expected"]["primary_action"] == "safety_intervention"
               and r["predicted"]["primary_action"] == "safety_intervention")
    s_total = sum(1 for r in rows if r["expected"]["primary_action"] == "safety_intervention")
    return {
        "primary_action_accuracy": pa["accuracy"],
        "primary_action_macro_f1": pa["macro_f1"],
        "safety_action_recall": round(s_tp / s_total, 4) if s_total else 0.0,
        "safety_target_accuracy": st["accuracy"],
        "tool_action_micro_f1": ta["micro_f1"],
        "tool_action_macro_f1": ta["macro_f1"],
        "tool_action_exact_match": ta["exact_match"],
        "recommendation_mode_accuracy": rm["accuracy"],
        "policy_exact_match": round(exact, 4),
    }
