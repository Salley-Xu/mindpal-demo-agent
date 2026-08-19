# -*- coding: utf-8 -*-
"""
Intent 置信度校准（Phase 1 Task 1.8 §27）。

- Temperature Scaling（单温度，优化 NLL）
- Per-label Threshold 优化（在 dev 上最大化 Macro F1）
- 指标：ECE / Brier Score
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


def ece_score(probs, labels, n_bins: int = 10) -> float:
    """Expected Calibration Error（多标签按置信度分箱）。"""
    confs = []
    accs = []
    for p, l in zip(probs, labels):
        for i, pc in enumerate(p):
            confs.append(pc)
            accs.append(1.0 if l[i] == 1 else 0.0)
    confs = np.array(confs)
    accs = np.array(accs)
    if len(confs) == 0:
        return 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        conf = confs[mask].mean()
        acc = accs[mask].mean()
        ece += (mask.sum() / len(confs)) * abs(conf - acc)
    return round(float(ece), 4)


def brier_score(probs, labels) -> float:
    """多标签 Brier Score（越小越好）。"""
    probs = np.array(probs)
    labels = np.array(labels)
    if probs.size == 0:
        return 0.0
    return round(float(((probs - labels) ** 2).mean()), 4)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, iters: int = 100,
                    lr: float = 0.01) -> float:
    """温度缩放：优化多标签 NLL。logits: (n, C), labels: (n, C) ∈ {0,1}。"""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    T = 1.0
    for _ in range(iters):
        scaled = logits / T
        # BCE NLL（多标签）
        probs = 1.0 / (1.0 + np.exp(-scaled))
        eps = 1e-9
        nll = -np.mean(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps))
        grad = np.mean((probs - labels) * (-scaled / T))
        T -= lr * grad
        T = max(T, 0.1)
    return round(float(T), 4)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64) / T))


def optimize_per_label_thresholds(probs: np.ndarray, labels: np.ndarray,
                                  metric: str = "macro_f1") -> List[float]:
    """在 dev 上逐标签网格搜索阈值，最大化 Macro F1。"""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    n_classes = probs.shape[1]
    best_thresholds = []
    for c in range(n_classes):
        best_f1, best_t = -1, 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            pred = (probs[:, c] >= t).astype(int)
            tp = ((pred == 1) & (labels[:, c] == 1)).sum()
            fp = ((pred == 1) & (labels[:, c] == 0)).sum()
            fn = ((pred == 0) & (labels[:, c] == 1)).sum()
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds.append(round(float(best_t), 3))
    return best_thresholds


def reliability_plot_data(probs, labels, n_bins: int = 10) -> List[Dict]:
    """Reliability Plot 数据点。"""
    confs, accs = [], []
    for p, l in zip(probs, labels):
        for i, pc in enumerate(p):
            confs.append(pc)
            accs.append(1.0 if l[i] == 1 else 0.0)
    confs = np.array(confs)
    accs = np.array(accs)
    bins = []
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        bins.append({"bin": f"[{lo:.1f},{hi:.1f})", "confidence": round(float(confs[mask].mean()), 3),
                     "accuracy": round(float(accs[mask].mean()), 3), "n": int(mask.sum())})
    return bins
