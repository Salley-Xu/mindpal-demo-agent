# -*- coding: utf-8 -*-
"""Intent 评测指标（Phase 1 §20 / §33）。"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def intent_metrics(y_true: Sequence[Iterable], y_pred: Sequence[Iterable],
                   labels: Sequence[str]) -> Dict:
    """多标签指标：Macro/Micro F1 + Exact Match + Hamming Loss + Per-label。"""
    labels = list(labels)
    n = len(y_true)

    per_class = {}
    micro_tp = micro_fp = micro_fn = 0
    hamming_denom = 0
    hamming_err = 0
    exact_match = 0
    total_pred = 0

    for a, b in zip(y_true, y_pred):
        sa, sb = set(a), set(b)
        total_pred += len(sb)
        # hamming
        for lab in labels:
            if (lab in sa) != (lab in sb):
                hamming_err += 1
            hamming_denom += 1
        if sa == sb:
            exact_match += 1

    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if lab in a and lab in b)
        fp = sum(1 for a, b in zip(y_true, y_pred) if lab not in a and lab in b)
        fn = sum(1 for a, b in zip(y_true, y_pred) if lab in a and lab not in b)
        micro_tp += tp; micro_fp += fp; micro_fn += fn
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[lab] = {"precision": round(p, 4), "recall": round(r, 4),
                          "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}

    def _f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(labels) if labels else 0.0
    macro_precision = sum(c["precision"] for c in per_class.values()) / len(labels) if labels else 0.0
    macro_recall = sum(c["recall"] for c in per_class.values()) / len(labels) if labels else 0.0
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0

    return {
        "n": n,
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "micro_f1": round(_f1(micro_p, micro_r), 4),
        "exact_match": round(exact_match / n, 4) if n else 0.0,
        "hamming_loss": round(hamming_err / hamming_denom, 4) if hamming_denom else 0.0,
        "avg_pred_labels": round(total_pred / n, 3) if n else 0.0,
        "per_class": per_class,
    }
