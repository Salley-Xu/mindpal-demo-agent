# -*- coding: utf-8 -*-
"""
Open-set Detection 评测（Phase 1 Task 1.9 §28-30）。

策略：max positive score < threshold → is_open_set = true。
指标：OOD Recall / Precision / In-domain False Reject / AUROC。

用法：
  PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe evaluation/intent/run_open_set.py \
    --model-dir models/intent/best_model --data-suffix v1
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

from data.intent.intent_schema import IntentData  # noqa: E402
from evaluation.intent.open_set import ood_metrics  # noqa: E402
from evaluation.intent.small_model import SmallModelPredictor  # noqa: E402


def load_items(path: Path):
    return [IntentData.model_validate(json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/intent/best_model")
    parser.add_argument("--data-suffix", default="v1")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "intent"
    sfx = args.data_suffix
    dev_items = load_items(data_dir / f"intent_dev_{sfx}.jsonl")
    ood_items = load_items(data_dir / "intent_ood_v1.jsonl")

    predictor = SmallModelPredictor(model_dir=str(PROJECT_ROOT / args.model_dir))

    def max_scores(items):
        scores = []
        for d in items:
            r = predictor.predict(d.text)
            ms = max(r["label_scores"].values()) if r["label_scores"] else 0.0
            scores.append(ms)
        return scores

    in_scores = max_scores(dev_items)
    ood_scores = max_scores(ood_items)

    # 多阈值扫描
    results = []
    for thr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        results.append(ood_metrics(in_scores, ood_scores, thr))

    # 选最优：OOD Recall >= 0.80 时 False Reject 最低
    feasible = [r for r in results if r["ood_recall"] >= 0.80]
    best = min(feasible, key=lambda r: r["in_domain_false_reject"]) if feasible else results[-1]

    payload = {
        "meta": {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "model": args.model_dir, "n_in_domain": len(in_scores), "n_ood": len(ood_scores)},
        "threshold_scan": results,
        "recommended": best,
    }
    reports_dir = PROJECT_ROOT / "evaluation" / "intent" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"open_set_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] open-set 报告 -> {path.name}")
    print(f"  {'thr':>5} {'OOD_Recall':>10} {'OOD_Prec':>9} {'ID_FR':>6} {'AUROC':>6}")
    for r in results:
        print(f"  {r['threshold']:>5.2f} {r['ood_recall']:>10.3f} {r['ood_precision']:>9.3f} "
              f"{r['in_domain_false_reject']:>6.3f} {r['auroc']:>6.3f}")
    print(f"  >> 推荐阈值 {best['threshold']}: OOD Recall={best['ood_recall']} ID_FR={best['in_domain_false_reject']}")


if __name__ == "__main__":
    main()
