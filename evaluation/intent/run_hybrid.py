# -*- coding: utf-8 -*-
"""
Hybrid Intent 评测（Phase 1 Task 1.10/1.11 §31-34）。

实验矩阵（§33）：
  C  Small Model（阈值 0.5）
  D  Small Model + Calibration（per-label threshold）
  E  Small Model + LLM Fallback（calibrated + open-set + fallback）

输出：Macro/Micro F1 / Exact Match / OOD Recall / LLM Call Rate / Avg Latency

用法：
  PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe evaluation/intent/run_hybrid.py \
    --model-dir models/intent/best_model --data-suffix v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.intent.intent_schema import INTENT_LABELS, IntentData  # noqa: E402
from evaluation.intent.hybrid import HybridIntentPredictor  # noqa: E402
from evaluation.intent.metrics import intent_metrics  # noqa: E402
from evaluation.intent.open_set import MaxScoreOpenSet  # noqa: E402
from evaluation.intent.small_model import SmallModelPredictor  # noqa: E402


def load_items(path: Path):
    return [IntentData.model_validate(json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()]


# 从 run_calibration 输出得到的 per-label thresholds（INTENT_LABELS 顺序）
DEFAULT_PER_LABEL_THRESHOLDS = [0.25, 0.55, 0.4, 0.2, 0.35, 0.4, 0.45, 0.25, 0.35, 0.25]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/intent/best_model")
    parser.add_argument("--data-suffix", default="v1")
    parser.add_argument("--open-set-threshold", type=float, default=0.35)
    parser.add_argument("--fallback-threshold", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=0, help="限制评测样本数（LLM fallback 加速）")
    parser.add_argument("--enable-llm", action="store_true", help="启用 LLM fallback")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "intent"
    sfx = args.data_suffix
    test_items = load_items(data_dir / f"intent_test_{sfx}.jsonl")
    ood_items = load_items(data_dir / "intent_ood_v1.jsonl")
    if args.limit:
        test_items = test_items[:args.limit]

    # C: Small Model（0.5 阈值）
    clf_plain = SmallModelPredictor(model_dir=str(PROJECT_ROOT / args.model_dir))

    # D: Small Model + Calibration（per-label threshold）
    clf_cal = SmallModelPredictor(model_dir=str(PROJECT_ROOT / args.model_dir),
                                  thresholds=DEFAULT_PER_LABEL_THRESHOLDS)

    # E: Hybrid（calibrated + open-set + LLM fallback）
    hybrid = HybridIntentPredictor(
        classifier=clf_cal,
        open_set=MaxScoreOpenSet(threshold=args.open_set_threshold),
        fallback_confidence_threshold=args.fallback_threshold,
        open_set_threshold=args.open_set_threshold,
        enable_llm_fallback=args.enable_llm,
    )

    def run(predictor, items):
        preds, times, llm_calls = [], [], 0
        for d in items:
            t0 = time.time()
            r = predictor.predict(d.text)
            times.append(time.time() - t0)
            preds.append(r)
            if r.get("source") == "llm" or r.get("fallback_reason"):
                llm_calls += 1
        y_pred = [p["labels"] for p in preds]
        y_true = [[l.value for l in d.labels] for d in items]
        m = intent_metrics(y_true, y_pred, INTENT_LABELS)
        m["avg_latency_ms"] = round(sum(times) / len(times) * 1000, 2)
        m["llm_call_rate"] = round(llm_calls / len(items), 4)
        return m

    results = {}
    results["C_small_model_05"] = run(clf_plain, test_items)
    results["D_small_calibrated"] = run(clf_cal, test_items)
    if args.enable_llm:
        results["E_hybrid"] = run(hybrid, test_items)
        # OOD 评测（Hybrid 的 open-set 判定）
        ood_preds = [hybrid.predict(d.text) for d in ood_items]
        ood_open = sum(1 for p in ood_preds if p.get("is_open_set") or p.get("fallback_reason"))
        results["E_ood_recall"] = round(ood_open / len(ood_items), 4)
        results["E_ood_llm_calls"] = sum(1 for p in ood_preds if p.get("source") == "llm")

    reports_dir = PROJECT_ROOT / "evaluation" / "intent" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"hybrid_intent_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "model": args.model_dir, "data_suffix": sfx,
                            "n_test": len(test_items), "n_ood": len(ood_items)},
                   "results": results}, f, ensure_ascii=False, indent=2)

    print(f"[OK] hybrid 报告 -> {path.name}")
    print(f"  {'Method':<22} {'MacroF1':>8} {'MicroF1':>8} {'Exact':>7} {'LLMCall':>8} {'Latency':>9}")
    for name, m in results.items():
        if isinstance(m, dict) and "macro_f1" in m:
            print(f"  {name:<22} {m['macro_f1']:>8.3f} {m['micro_f1']:>8.3f} {m['exact_match']:>7.3f} "
                  f"{m['llm_call_rate']:>8.3f} {m['avg_latency_ms']:>8.1f}ms")
    if "E_ood_recall" in results:
        print(f"  E hybrid OOD Recall: {results['E_ood_recall']}  OOD LLM calls: {results['E_ood_llm_calls']}")


if __name__ == "__main__":
    main()
