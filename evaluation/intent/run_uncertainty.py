# -*- coding: utf-8 -*-
"""
Intent Uncertainty 检测对比（Phase 1.5 Task 1.5.4 §8.5-8.6）。

策略比较：
  A. max calibrated score < threshold
  B. prediction entropy > threshold
  C. top1 - top2 margin < threshold

指标：Uncertain Recall / Precision / In-domain False Reject / AUROC。
目标：Uncertain Recall >= 0.75，In-domain False Reject <= 0.10。

用法：
  PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe evaluation/intent/run_uncertainty.py \
    --model-dir models/intent/best_model --data-suffix v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from data.intent.intent_schema import IntentData  # noqa: E402
from evaluation.intent.small_model import SmallModelPredictor  # noqa: E402


def load_items(path: Path):
    return [IntentData.model_validate(json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()]


def entropy(probs):
    s = 0.0
    for p in probs:
        if p > 0:
            s -= p * math.log(p)
    return s


def strategies(scores: dict) -> dict:
    """从 label_scores 计算三种 uncertainty 信号。"""
    probs = list(scores.values()) if scores else []
    max_score = max(probs) if probs else 0.0
    # 熵（归一化到 0-1）
    h = entropy(probs) / math.log(max(len(probs), 2)) if probs else 1.0
    # top1 - top2 margin
    sorted_p = sorted(probs, reverse=True)
    margin = sorted_p[0] - sorted_p[1] if len(sorted_p) >= 2 else sorted_p[0] if sorted_p else 0.0
    return {"max_score": max_score, "entropy": h, "margin": margin}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/intent/best_model")
    parser.add_argument("--data-suffix", default="v1")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "intent"
    sfx = args.data_suffix
    dev_items = load_items(data_dir / f"intent_dev_{sfx}.jsonl")
    unc_items = load_items(data_dir / "intent_uncertainty_v1.jsonl")

    predictor = SmallModelPredictor(model_dir=str(PROJECT_ROOT / args.model_dir))

    def collect(items):
        sigs = []
        for d in items:
            r = predictor.predict(d.text)
            sigs.append(strategies(r["label_scores"]))
        return sigs

    in_sigs = collect(dev_items)
    unc_sigs = collect(unc_items)

    # 三策略扫描
    results = {}
    for name, key, direction in [("max_score", "max_score", "low"),
                                 ("entropy", "entropy", "high"),
                                 ("margin", "margin", "low")]:
        in_vals = [s[key] for s in in_sigs]
        unc_vals = [s[key] for s in unc_sigs]
        rows = []
        for thr in np.arange(0.1, 0.9, 0.05):
            if direction == "low":
                unc_recall = sum(1 for v in unc_vals if v < thr) / len(unc_vals) if unc_vals else 0
                id_fr = sum(1 for v in in_vals if v < thr) / len(in_vals) if in_vals else 0
            else:
                unc_recall = sum(1 for v in unc_vals if v > thr) / len(unc_vals) if unc_vals else 0
                id_fr = sum(1 for v in in_vals if v > thr) / len(in_vals) if in_vals else 0
            rows.append({"threshold": round(float(thr), 2), "uncertain_recall": round(unc_recall, 4),
                         "in_domain_false_reject": round(id_fr, 4)})
        # 选 Uncertain Recall >= 0.75 时 FR 最低
        feasible = [r for r in rows if r["uncertain_recall"] >= 0.75]
        best = min(feasible, key=lambda r: r["in_domain_false_reject"]) if feasible else rows[-1]
        results[name] = {"scan": rows, "best": best}

    reports_dir = PROJECT_ROOT / "evaluation" / "intent" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"intent_uncertainty_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "model": args.model_dir, "n_in": len(in_sigs), "n_unc": len(unc_sigs)},
                   "strategies": results}, f, ensure_ascii=False, indent=2)

    print(f"[OK] uncertainty 报告 -> {path.name}")
    print(f"  {'策略':<12} {'最佳阈值':>8} {'Uncertain Recall':>16} {'In-domain FR':>12}")
    for name, r in results.items():
        b = r["best"]
        print(f"  {name:<12} {b['threshold']:>8.2f} {b['uncertain_recall']:>16.3f} {b['in_domain_false_reject']:>12.3f}")


if __name__ == "__main__":
    main()
