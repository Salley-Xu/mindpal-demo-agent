"""
eval_metrics.py — 共享指标函数库

提供情绪分类、风险分类、强度回归等评测所需的所有指标函数。
纯 Python + numpy，无外部 ML 框架依赖。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ============================================================
# 分类指标
# ============================================================

def classification_report(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict:
    """
    返回每类的 precision/recall/f1/support + macro avg + accuracy

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        labels: 标签集合（默认从数据自动推导）

    Returns:
        {
            "accuracy": float,
            "macro_avg": {"precision": float, "recall": float, "f1": float},
            "per_class": {
                "label": {"precision": float, "recall": float, "f1": float, "support": int},
                ...
            }
        }
    """
    labels = labels or sorted(set(y_true + y_pred))
    result: Dict = {"per_class": {}, "macro_avg": {}, "accuracy": None}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    result["accuracy"] = round(correct / len(y_true), 4) if y_true else 0.0

    precisions, recalls, f1s = [], [], []

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result["per_class"][label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    result["macro_avg"] = {
        "precision": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
    }

    return result


def per_class_details(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict:
    """返回每类的 TP/FP/FN 计数（用于调试分析）"""
    labels = labels or sorted(set(y_true + y_pred))
    details = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        details[label] = {"tp": tp, "fp": fp, "fn": fn}
    return details


def confusion_matrix_data(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict:
    """
    返回混淆矩阵数据（非矩阵，而是逐条记录，方便排查）

    Returns:
        {"errors": [{"true": ..., "pred": ...}, ...], "matrix": {...}}
    """
    errors = []
    for t, p in zip(y_true, y_pred):
        if t != p:
            errors.append({"true": t, "pred": p})
    matrix = {}
    for t_label in labels:
        matrix[t_label] = {}
        for p_label in labels:
            matrix[t_label][p_label] = sum(
                1 for t, p in zip(y_true, y_pred) if t == t_label and p == p_label
            )
    return {"errors": errors, "matrix": matrix}


# ============================================================
# 置信度校准
# ============================================================

def ece_score(
    y_correct: List[bool],
    y_confidence: List[float],
    n_bins: int = 10,
) -> Dict:
    """
    Expected Calibration Error

    将置信度分为 n_bins 个桶，计算 |accuracy - avg_confidence| 的加权平均。

    Args:
        y_correct: 每个预测是否正确
        y_confidence: 每个预测的置信度
        n_bins: 分桶数

    Returns:
        {"ece": float, "n_bins": int, "bins": {bin_i: {"count": int, "accuracy": float, "avg_confidence": float}}}
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = {
        i: {"count": 0, "correct": 0, "conf_sum": 0.0}
        for i in range(n_bins)
    }

    for correct, conf in zip(y_correct, y_confidence):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_data[bin_idx]["count"] += 1
        bin_data[bin_idx]["correct"] += int(correct)
        bin_data[bin_idx]["conf_sum"] += conf

    ece = 0.0
    bin_metrics = {}
    for i, data in bin_data.items():
        n = data["count"]
        if n == 0:
            continue
        acc = data["correct"] / n
        conf = data["conf_sum"] / n
        ece += (n / len(y_correct)) * abs(acc - conf)
        bin_metrics[f"bin_{i}"] = {
            "count": n,
            "accuracy": round(acc, 4),
            "avg_confidence": round(conf, 4),
        }

    return {
        "ece": round(ece, 4),
        "n_bins": n_bins,
        "bins": bin_metrics,
    }


# ============================================================
# 强度回归指标
# ============================================================

def intensity_metrics(
    gold: List[float],
    pred: List[float],
) -> Dict:
    """
    强度回归指标

    Args:
        gold: 真实强度值列表
        pred: 预测强度值列表

    Returns:
        {"mae": float, "rmse": float, "within_01": float, "max_error": float}
    """
    errors = [abs(g - p) for g, p in zip(gold, pred)]
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean([e**2 for e in errors])))
    within_01 = sum(1 for e in errors if e <= 0.1) / len(errors) if errors else 0.0
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "within_01": round(within_01, 4),
        "max_error": round(max(errors), 4) if errors else 0.0,
    }


# ============================================================
# 有序分类指标
# ============================================================

def quadratic_weighted_kappa(
    y_true: List[int],
    y_pred: List[int],
    n_classes: int = 4,
) -> float:
    """
    Quadratic Weighted Kappa

    有序分类指标，越级惩罚比相邻错误更大。
    使用 sklearn 实现，回退到纯 numpy 实现。

    Args:
        y_true: 真实等级（整数）
        y_pred: 预测等级（整数）
        n_classes: 等级数
    """
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except ImportError:
        pass

    # 纯 numpy 实现
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1.0

    # 期望矩阵
    row_sum = O.sum(axis=1)
    col_sum = O.sum(axis=0)
    total = O.sum()
    E = np.outer(row_sum, col_sum) / total

    # 权重矩阵
    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = (i - j) ** 2 / (n_classes - 1) ** 2

    kappa = 1.0 - (W * O).sum() / (W * E).sum()
    return round(float(kappa), 4)


# ============================================================
# 等级召回/精确率
# ============================================================

def recall_at_level(
    y_true: List[str],
    y_pred: List[str],
    target: str = "level_3",
) -> Dict:
    """
    特定等级的召回率、精确率、F1

    Args:
        y_true: 真实等级列表
        y_pred: 预测等级列表
        target: 目标等级
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == target and p == target)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == target and p != target)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != target and p == target)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "support": tp + fn,
    }


# ============================================================
# 辅助
# ============================================================

def print_metrics_table(title: str, metrics: Dict, indent: int = 0):
    """打印指标表格（终端友好）"""
    prefix = " " * indent
    print(f"\n{prefix}[{title}]")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{prefix}  {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"{prefix}    {k}: {v:.4f}")
                else:
                    print(f"{prefix}    {k}: {v}")
        elif isinstance(value, float):
            print(f"{prefix}  {key}: {value:.4f}")
        else:
            print(f"{prefix}  {key}: {value}")
