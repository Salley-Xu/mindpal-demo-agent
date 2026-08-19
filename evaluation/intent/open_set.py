# -*- coding: utf-8 -*-
"""
Open-set Detection（Phase 1 Task 1.9）。

策略：max positive score < threshold → is_open_set = true（第一版，§30）。
指标：OOD Recall / Precision / In-domain False Reject / AUROC。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def open_set_decision(label_scores: Dict[str, float], threshold: float) -> bool:
    """max positive score < threshold → open-set。"""
    if not label_scores:
        return True
    return max(label_scores.values()) < threshold


class MaxScoreOpenSet:
    """基于 max positive score 的 open-set 检测器。"""

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def is_open_set(self, label_scores: Dict[str, float]) -> bool:
        return open_set_decision(label_scores, self.threshold)


def ood_metrics(in_domain_scores: List[float], ood_scores: List[float],
                threshold: float) -> Dict:
    """
    in_domain_scores: 域内样本的 max-positive score
    ood_scores:       OOD 样本的 max-positive score
    """
    # OOD 判定：score < threshold → open
    ood_recall = sum(1 for s in ood_scores if s < threshold) / len(ood_scores) if ood_scores else 0.0
    # In-domain false reject：域内被误判为 open
    id_fr = sum(1 for s in in_domain_scores if s < threshold) / len(in_domain_scores) if in_domain_scores else 0.0

    # AUROC：in-domain 为正类（期望 score 更高）→ P(in-domain score > OOD score)
    if ood_scores and in_domain_scores:
        all_scores = np.concatenate([np.array(in_domain_scores), np.array(ood_scores)])
        all_labels = np.concatenate([np.ones(len(in_domain_scores)), np.zeros(len(ood_scores))])
        order = np.argsort(all_scores)  # 按 score 升序（高分 = 高 rank）
        sorted_labels = all_labels[order]
        pos = sorted_labels.sum()
        if pos == 0 or pos == len(sorted_labels):
            auroc = 1.0
        else:
            ranks = np.where(sorted_labels == 1)[0] + 1
            auroc = (ranks.sum() - pos * (pos + 1) / 2) / (pos * (len(sorted_labels) - pos))
    else:
        auroc = 0.5

    ood_precision = sum(1 for s in ood_scores if s < threshold) / \
        (sum(1 for s in ood_scores if s < threshold) + sum(1 for s in in_domain_scores if s < threshold)) \
        if (sum(1 for s in ood_scores if s < threshold) + sum(1 for s in in_domain_scores if s < threshold)) else 0.0

    return {
        "threshold": threshold,
        "ood_recall": round(ood_recall, 4),
        "ood_precision": round(ood_precision, 4),
        "in_domain_false_reject": round(id_fr, 4),
        "auroc": round(float(auroc), 4),
        "n_in_domain": len(in_domain_scores),
        "n_ood": len(ood_scores),
    }
