# -*- coding: utf-8 -*-
"""
Unified Evaluation（Phase 1.5 Task 1.5.5 §9 / §12）。

在同一 Independent Test（intent_test_independent_v1.jsonl）上对比：
  A Legacy Rule / B LLM-only / C Small Model / D Calibrated / E Context-aware / F Hybrid
含 Slice Evaluation（single/multi/context/implicit-high-risk/hard-negative）。

用法：
  PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe evaluation/intent/run_unified.py \
    --llm-limit 150   # LLM-only 抽样规模（控制成本）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
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
from evaluation.intent.predictors import (  # noqa: E402
    LLMPredictor, LegacyRulePredictor,
)
from evaluation.intent.small_model import SmallModelPredictor  # noqa: E402

PER_LABEL_THRESHOLDS = [0.25, 0.55, 0.4, 0.2, 0.35, 0.4, 0.45, 0.25, 0.35, 0.25]


def load_independent():
    path = PROJECT_ROOT / "data" / "intent" / "intent_test_independent_v1.jsonl"
    return [IntentData.model_validate(json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()]


def predict_all(predictor, items, use_context=False, ctx_turns=1):
    preds, times = [], []
    for d in items:
        t0 = time.time()
        if use_context:
            r = predictor.predict(d.text, (d.conversation, ctx_turns))
        else:
            r = predictor.predict(d.text)
        times.append(time.time() - t0)
        preds.append(r)
    return preds, times


def run_method(name, predictor, items, use_context=False, ctx_turns=1):
    preds, times = predict_all(predictor, items, use_context, ctx_turns)
    y_true = [[l.value for l in d.labels] for d in items]
    m = intent_metrics(y_true, [p["labels"] for p in preds], INTENT_LABELS)
    m["avg_latency_ms"] = round(sum(times) / len(times) * 1000, 2) if times else 0
    m["llm_call_rate"] = round(sum(1 for p in preds if p.get("source") == "llm" or p.get("fallback_reason")) / len(preds), 4)
    return m


def slice_metrics(items, preds, name_filter):
    """计算特定 slice 的指标。"""
    subset = [(d, p) for d, p in zip(items, preds) if name_filter(d)]
    if not subset:
        return None
    y_true = [[l.value for l in d.labels] for _, d in []]  # placeholder
    y_true = [[l.value for l in d.labels] for d, _ in subset]
    y_pred = [p["labels"] for _, p in subset]
    m = intent_metrics(y_true, y_pred, INTENT_LABELS)
    m["n"] = len(subset)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-limit", type=int, default=150, help="LLM-only 抽样数")
    parser.add_argument("--run-llm", action="store_true", help="运行 LLM-only（默认跳过省成本）")
    parser.add_argument("--run-hybrid-llm", action="store_true", help="Hybrid 启用 LLM fallback")
    args = parser.parse_args()

    items = load_independent()
    print(f"[INFO] Independent Test: {len(items)} 条")

    # 预测器
    legacy = LegacyRulePredictor()
    clf_plain = SmallModelPredictor(model_dir=str(PROJECT_ROOT / "models/intent/best_model"))
    clf_cal = SmallModelPredictor(model_dir=str(PROJECT_ROOT / "models/intent/best_model"),
                                  thresholds=PER_LABEL_THRESHOLDS)
    clf_ctx = SmallModelPredictor(model_dir=str(PROJECT_ROOT / "models/intent/phase1_5_final_model"),
                                  thresholds=PER_LABEL_THRESHOLDS)

    results = {}
    results["A_legacy"] = run_method("legacy", legacy, items)
    results["C_small_current"] = run_method("small", clf_plain, items)
    results["D_calibrated"] = run_method("cal", clf_cal, items)
    results["E_context"] = run_method("ctx", clf_ctx, items, use_context=True, ctx_turns=1)

    if args.run_llm:
        import random
        rng = random.Random(42)
        llm_items = rng.sample(items, min(args.llm_limit, len(items)))
        llm = LLMPredictor()
        results["B_llm_only"] = run_method("llm", llm, llm_items)
        results["B_llm_only"]["n"] = len(llm_items)

    if args.run_hybrid_llm:
        hybrid = HybridIntentPredictor(
            classifier=clf_ctx,
            open_set=MaxScoreOpenSet(threshold=0.5),
            fallback_confidence_threshold=0.55,
            open_set_threshold=0.5,
            enable_llm_fallback=True,
        )
        results["F_hybrid"] = run_method("hybrid", hybrid, items, use_context=True, ctx_turns=1)

    # E 方法（context）的 slice 评测
    if "E_context" in results:
        e_preds, _ = predict_all(clf_ctx, items, use_context=True, ctx_turns=1)
        slices = {
            "single_label": lambda d: len(d.labels) == 1,
            "multi_label": lambda d: len(d.labels) > 1,
            "context_aware": lambda d: len(d.conversation) > 1,
            "implicit_high_risk": lambda d: d.notes == "hard" and any(l.value == "high_risk_expression" for l in d.labels) and len(d.conversation) == 1,
            "hard_negative": lambda d: (d.notes or "").startswith("否定") or "讨论" in (d.notes or "") or "学术" in (d.notes or ""),
        }
        results["E_slices"] = {k: slice_metrics(items, e_preds, fn) for k, fn in slices.items()}

    # 输出
    reports_dir = PROJECT_ROOT / "evaluation" / "intent" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"independent_unified_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "n": len(items)},
                   "results": results}, f, ensure_ascii=False, indent=2)

    print(f"[OK] unified 报告 -> {path.name}")
    print(f"  {'Method':<18} {'Macro':>7} {'Micro':>7} {'Exact':>7} {'follow_up':>9} {'high_risk_R':>11} {'LLMCall':>8} {'Latency':>8}")
    for name in ["A_legacy", "B_llm_only", "C_small_current", "D_calibrated", "E_context", "F_hybrid"]:
        if name not in results:
            continue
        m = results[name]
        fu = m["per_class"].get("follow_up", {}).get("f1", 0)
        hr = m["per_class"].get("high_risk_expression", {}).get("recall", 0)
        print(f"  {name:<18} {m['macro_f1']:>7.3f} {m['micro_f1']:>7.3f} {m['exact_match']:>7.3f} "
              f"{fu:>9.3f} {hr:>11.3f} {m['llm_call_rate']:>8.3f} {m['avg_latency_ms']:>7.1f}ms")
    if "E_slices" in results:
        print("  --- E context slices ---")
        for k, m in results["E_slices"].items():
            if m:
                print(f"  {k:<20} n={m['n']} Macro={m['macro_f1']} Exact={m['exact_match']}")


if __name__ == "__main__":
    main()
