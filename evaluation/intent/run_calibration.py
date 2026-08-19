# -*- coding: utf-8 -*-
"""
Threshold + Calibration（Phase 1 Task 1.8 §25-27）。

比较：
  A. global threshold = 0.5
  B. optimized global threshold（dev 上搜索）
  C. per-label threshold（dev 上逐标签搜索）
Calibration：Raw Sigmoid vs Temperature Scaling，报告 ECE / Brier。

用法：
  PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe evaluation/intent/run_calibration.py --data-suffix v1 --model-dir models/intent/best_model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from data.intent.intent_schema import INTENT_LABELS, IntentData  # noqa: E402
from evaluation.intent.calibration import (  # noqa: E402
    apply_temperature, brier_score, ece_score, fit_temperature,
    optimize_per_label_thresholds,
)
from evaluation.intent.metrics import intent_metrics  # noqa: E402
from evaluation.intent.small_model import SmallModelPredictor  # noqa: E402


def load_items(path: Path):
    return [IntentData.model_validate(json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()]


def get_logits_matrix(predictor, items) -> np.ndarray:
    return np.array([predictor.get_logits(d.text) for d in items])


def labels_matrix(items) -> np.ndarray:
    m = np.zeros((len(items), len(INTENT_LABELS)))
    for i, d in enumerate(items):
        for l in d.labels:
            m[i, LABEL2ID[l.value]] = 1.0
    return m


LABEL2ID = {lab: i for i, lab in enumerate(INTENT_LABELS)}


def eval_with_thresholds(probs: np.ndarray, gold: np.ndarray, thresholds) -> dict:
    y_pred = [[INTENT_LABELS[c] for c in range(len(INTENT_LABELS)) if probs[i, c] >= thresholds[c]]
              for i in range(probs.shape[0])]
    y_true = [[INTENT_LABELS[c] for c in range(len(INTENT_LABELS)) if gold[i, c] == 1.0]
              for i in range(gold.shape[0])]
    return intent_metrics(y_true, y_pred, INTENT_LABELS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/intent/best_model")
    parser.add_argument("--data-suffix", default="v1")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "intent"
    sfx = args.data_suffix
    dev_items = load_items(data_dir / f"intent_dev_{sfx}.jsonl")
    test_items = load_items(data_dir / f"intent_test_{sfx}.jsonl")

    predictor = SmallModelPredictor(model_dir=str(PROJECT_ROOT / args.model_dir))
    dev_logits = get_logits_matrix(predictor, dev_items)
    test_logits = get_logits_matrix(predictor, test_items)
    dev_gold = labels_matrix(dev_items)
    test_gold = labels_matrix(test_items)

    dev_probs = 1 / (1 + np.exp(-dev_logits))
    test_probs = 1 / (1 + np.exp(-test_logits))

    # A. global 0.5
    thr_a = [0.5] * len(INTENT_LABELS)
    m_a_dev = eval_with_thresholds(dev_probs, dev_gold, thr_a)
    m_a_test = eval_with_thresholds(test_probs, test_gold, thr_a)

    # B. optimized global threshold（dev 搜索，最大化 macro f1）
    best_g, best_mf1 = 0.5, -1
    for t in np.arange(0.2, 0.9, 0.05):
        thr = [float(t)] * len(INTENT_LABELS)
        m = eval_with_thresholds(dev_probs, dev_gold, thr)
        if m["macro_f1"] > best_mf1:
            best_mf1, best_g = m["macro_f1"], t
    thr_b = [float(best_g)] * len(INTENT_LABELS)
    m_b_dev = eval_with_thresholds(dev_probs, dev_gold, thr_b)
    m_b_test = eval_with_thresholds(test_probs, test_gold, thr_b)

    # C. per-label thresholds（dev 逐标签搜索）
    thr_c = optimize_per_label_thresholds(dev_probs, dev_gold)
    m_c_dev = eval_with_thresholds(dev_probs, dev_gold, thr_c)
    m_c_test = eval_with_thresholds(test_probs, test_gold, thr_c)

    # Calibration：Temperature Scaling
    T = fit_temperature(dev_logits, dev_gold)
    cal_dev_probs = apply_temperature(dev_logits, T)
    cal_test_probs = apply_temperature(test_logits, T)

    ece_raw_dev = ece_score(dev_probs.tolist(), dev_gold.tolist())
    ece_cal_dev = ece_score(cal_dev_probs.tolist(), dev_gold.tolist())
    brier_raw_dev = brier_score(dev_probs, dev_gold)
    brier_cal_dev = brier_score(cal_dev_probs, dev_gold)

    result = {
        "meta": {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "model": args.model_dir, "data_suffix": sfx,
                 "n_dev": len(dev_items), "n_test": len(test_items)},
        "threshold_A_global_05": {"dev": m_a_dev, "test": m_a_test},
        "threshold_B_optimized_global": {"threshold": round(best_g, 3), "dev": m_b_dev, "test": m_b_test},
        "threshold_C_per_label": {"thresholds": thr_c, "dev": m_c_dev, "test": m_c_test},
        "calibration": {
            "temperature": T,
            "ece_raw_dev": ece_raw_dev, "ece_cal_dev": ece_cal_dev,
            "brier_raw_dev": brier_raw_dev, "brier_cal_dev": brier_cal_dev,
        },
    }

    reports_dir = PROJECT_ROOT / "evaluation" / "intent" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"calibrated_classifier_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] 校准报告 -> {path.name}")
    print(f"  A  global0.5:      dev_macro={m_a_dev['macro_f1']} test_macro={m_a_test['macro_f1']}")
    print(f"  B  opt_global({best_g:.2f}): dev_macro={m_b_dev['macro_f1']} test_macro={m_b_test['macro_f1']}")
    print(f"  C  per-label({thr_c}): dev_macro={m_c_dev['macro_f1']} test_macro={m_c_test['macro_f1']}")
    print(f"  Calibration: T={T} ECE {ece_raw_dev}->{ece_cal_dev} Brier {brier_raw_dev}->{brier_cal_dev}")


if __name__ == "__main__":
    main()
