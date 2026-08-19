# -*- coding: utf-8 -*-
"""
Intent Baseline runner（Phase 1 Task 1.4/1.5/1.6/1.11）。

用法：
  # Legacy Rule
  python evaluation/intent/run_baseline.py --predictor legacy --data data/intent/intent_seed_v1.jsonl --name legacy_rule
  # Rule Classifier
  python evaluation/intent/run_baseline.py --predictor rule --data data/intent/intent_seed_v1.jsonl --name rule_classifier
  # LLM-only
  python evaluation/intent/run_baseline.py --predictor llm --data data/intent/intent_seed_v1.jsonl --name llm_only --limit 200
  # Small Model（训练后）
  python evaluation/intent/run_baseline.py --predictor classifier --data data/intent/intent_test_v1.jsonl --name classifier

输出：evaluation/intent/reports/<name>.<json|md>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.intent.intent_schema import INTENT_LABELS, IntentData  # noqa: E402
from evaluation.intent.metrics import intent_metrics  # noqa: E402


def load_intent_data(path: Path, limit: int = 0):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(IntentData.model_validate(json.loads(line)))
    if limit:
        items = items[:limit]
    return items


def load_benchmark_cases(path: Path, limit: int = 0):
    """从 Agent Benchmark 提取 (text, labels) 列表（Task 1.12 用）。"""
    from evaluation.benchmark_schema import BenchmarkCase
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = BenchmarkCase.model_validate(json.loads(line))
                last_user = [t.content for t in c.conversation if t.role.value == "user"][-1]
                out.append((c.case_id, last_user, [i.value for i in c.expected.intent],
                            [t.model_dump() for t in c.conversation]))
    if limit:
        out = out[:limit]
    return out


def get_predictor(name: str):
    from evaluation.intent.predictors import (
        LLMPredictor, LegacyRulePredictor, RuleClassifierPredictor,
    )
    if name == "legacy":
        return LegacyRulePredictor()
    if name == "rule":
        return RuleClassifierPredictor()
    if name == "llm":
        return LLMPredictor()
    if name == "classifier":
        from evaluation.intent.small_model import SmallModelPredictor
        return SmallModelPredictor()
    raise ValueError(f"未知 predictor: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", choices=["legacy", "rule", "llm", "classifier"], default="legacy")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "intent" / "intent_seed_v1.jsonl")
    parser.add_argument("--benchmark", action="store_true", help="对 Agent Benchmark v1.1 评测")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use-context", action="store_true", help="预测时传入对话上下文")
    args = parser.parse_args()

    if args.benchmark:
        items = load_benchmark_cases(PROJECT_ROOT / "evaluation" / "datasets" / "agent_benchmark_v1_1.jsonl", args.limit)
        y_true = [x[2] for x in items]
        texts = [x[1] for x in items]
        contexts = [x[3] for x in items] if args.use_context else None
        case_ids = [x[0] for x in items]
    else:
        items = load_intent_data(args.data, args.limit)
        y_true = [[l.value for l in d.labels] for d in items]
        texts = [d.text for d in items]
        contexts = [d.conversation for d in items] if args.use_context else None
        case_ids = [d.id for d in items]

    predictor = get_predictor(args.predictor)
    print(f"[INFO] predictor={predictor.name} n={len(items)} use_context={args.use_context}")

    t0 = time.time()
    preds = [predictor.predict(t, contexts[i] if contexts else None) for i, t in enumerate(texts)]
    elapsed = time.time() - t0

    y_pred = [p["labels"] for p in preds]
    metrics = intent_metrics(y_true, y_pred, INTENT_LABELS)
    metrics["avg_latency_ms"] = round(elapsed / len(texts) * 1000, 2) if texts else 0
    metrics["total_seconds"] = round(elapsed, 2)
    metrics["llm_call_rate"] = 1.0 if predictor.name == "llm" else 0.0
    invalid_json = sum(1 for p in preds if p.get("error"))
    metrics["invalid_json"] = invalid_json

    # 输出
    reports_dir = PROJECT_ROOT / "evaluation" / "intent" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.name}_{predictor.name}_{ts}"
    json_path = reports_dir / f"{base}.json"
    md_path = reports_dir / f"{base}.md"

    payload = {
        "meta": {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "predictor": predictor.name, "n": len(texts),
                 "use_context": args.use_context, "dataset": str(args.data),
                 "avg_latency_ms": metrics["avg_latency_ms"], "llm_call_rate": metrics["llm_call_rate"]},
        "metrics": metrics,
        "details": [{"id": case_ids[i], "true": y_true[i], "pred": y_pred[i],
                     "conf": preds[i].get("confidence"), "source": preds[i].get("source")}
                    for i in range(len(texts))],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md = build_md(predictor.name, metrics, args, items)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[OK] {json_path.name}")
    print(f"  Macro F1={metrics['macro_f1']} Micro F1={metrics['micro_f1']} Exact={metrics['exact_match']}")
    print(f"  Latency={metrics['avg_latency_ms']}ms LLM_call_rate={metrics['llm_call_rate']}")
    print(f"  Per-label F1: { {k: v['f1'] for k, v in metrics['per_class'].items()} }")


def build_md(name, m, args, items) -> str:
    L = []
    L.append(f"# Intent Baseline Report — {name}")
    L.append("")
    L.append(f"- 日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- dataset: {args.data.name} (n={m['n']}, context={args.use_context})")
    L.append("")
    L.append("## 指标")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| Macro F1 | {m['macro_f1']} |")
    L.append(f"| Micro F1 | {m['micro_f1']} |")
    L.append(f"| Exact Match | {m['exact_match']} |")
    L.append(f"| Hamming Loss | {m['hamming_loss']} |")
    L.append(f"| Avg Pred Labels | {m['avg_pred_labels']} |")
    L.append(f"| Avg Latency | {m['avg_latency_ms']} ms |")
    L.append(f"| LLM Call Rate | {m.get('llm_call_rate', 0)} |")
    L.append("")
    L.append("## Per-label")
    L.append("")
    L.append("| label | P | R | F1 | tp/fp/fn |")
    L.append("|---|---|---|---|---|")
    for lab, d in m["per_class"].items():
        L.append(f"| {lab} | {d['precision']} | {d['recall']} | {d['f1']} | {d['tp']}/{d['fp']}/{d['fn']} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
