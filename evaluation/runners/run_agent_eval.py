# -*- coding: utf-8 -*-
"""
Agent Benchmark v1.1 评测 runner。

用法：
  # 真实模块（BERT 情绪 + BERT 风险 + 真实门控）— 离线 Baseline
  python evaluation/runners/run_agent_eval.py --predictor module --emotion bert --risk bert --report-name v1_1

  # 纯规则快速模式（不加载 BERT，用于冒烟测试）
  python evaluation/runners/run_agent_eval.py --predictor module --emotion rule --risk rule --report-name v1_1_rule

  # 只看前 N 条
  python evaluation/runners/run_agent_eval.py --limit 20

  # 从已存 JSON 重算指标（不重跑模型）
  python evaluation/runners/run_agent_eval.py --from-json evaluation/reports/baseline_v1_1_*.json

指标口径（对齐 Phase 0.5 §5）：
  - Risk: Macro F1 / High-risk Recall / FPR / FNR
  - Emotion: Coarse 5-class 为主，Fine 为 Coverage
  - Intent: Legacy Intent Coverage（不当作正式模型指标）
  - Recommendation: Trigger P/R / Mode Accuracy（注明受上游信号影响）
  - Primary Action: Accuracy / Macro F1 / Safety Recall
  - Tool Action: Micro/Macro F1 / Exact Match / 每工具 P/R
  - Memory: Current Memory Retrieval Behavior
  - Risk Trend: 多轮 accuracy
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

# 允许无 .env 密钥时仍可导入 backend 模块（ModulePredictor 不调用 LLM）
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark_schema import BenchmarkCase  # noqa: E402
from evaluation.metrics.agent_metrics import (  # noqa: E402
    binary_metrics,
    coarse_emotion_metrics,
    legacy_intent_metrics,
    memory_behavior_metrics,
    multilabel_metrics,
    primary_action_metrics,
    recommendation_metrics,
    risk_metrics,
    tool_action_metrics,
)
from evaluation.predictors.base import PredictedOutcome  # noqa: E402
from evaluation.predictors.module import ModulePredictor  # noqa: E402


def load_cases(path: Path) -> list:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(BenchmarkCase.model_validate(json.loads(line)))
    return cases


def build_predictor(args):
    if args.predictor == "module":
        return ModulePredictor(emotion_channel=args.emotion, risk_channel=args.risk)
    raise ValueError(f"未知 predictor: {args.predictor}")


def compute_all_metrics(cases, preds):
    risk_true = [c.expected.risk_level.value for c in cases]
    risk_pred = [p.risk_level for p in preds]

    emotion_true = [c.expected.emotion for c in cases]
    emotion_pred = [p.emotion for p in preds]
    emotion_labels = sorted({*emotion_true, *emotion_pred})

    intent_true = [[i.value for i in c.expected.intent] for c in cases]
    intent_pred = [p.intent for p in preds]
    intent_labels = sorted({i.value for c in cases for i in c.expected.intent})

    rec_true = [c.expected.recommendation_action.value for c in cases]
    rec_pred = [p.recommendation_action for p in preds]

    primary_true = [c.expected.primary_action.value for c in cases]
    primary_pred = [p.primary_action for p in preds]

    tool_true = [[t.value for t in c.expected.tool_actions] for c in cases]
    tool_pred = [p.tool_actions for p in preds]

    safety_true = [c.expected.safety_target.value for c in cases]
    safety_pred = [p.safety_target for p in preds]

    # 当前确定性路由（extra.current_routing_action，primary 级）对照
    current_routing_pred = [p.extra.get("current_routing_action", "continue_chat") for p in preds]

    memory_true = [c.expected.memory_needed for c in cases]
    memory_pred = [p.memory_needed for p in preds]

    # 风险趋势（仅对 expected.trend != "new" 的 case）
    trend_true = [c.expected.risk_trend.value for c in cases if c.expected.risk_trend.value != "new"]
    trend_pred = [p.risk_trend for c, p in zip(cases, preds) if c.expected.risk_trend.value != "new"]
    trend_acc = (sum(1 for a, b in zip(trend_true, trend_pred) if a == b) / len(trend_true)
                 if trend_true else None)
    trend_acc = round(trend_acc, 4) if trend_acc is not None else None

    metrics = {
        "risk": risk_metrics(risk_true, risk_pred),
        "emotion": _classification(emotion_true, emotion_pred, emotion_labels),
        "emotion_coarse": coarse_emotion_metrics(emotion_true, emotion_pred),
        "legacy_intent": legacy_intent_metrics(intent_true, intent_pred, intent_labels),
        "recommendation": recommendation_metrics(rec_true, rec_pred),
        "primary_action": primary_action_metrics(primary_true, primary_pred),
        "tool_action": tool_action_metrics(tool_true, tool_pred),
        "safety_target": _classification(safety_true, safety_pred, ["none", "self", "third_party"]),
        "routing_current": primary_action_metrics(primary_true, current_routing_pred),
        "memory_behavior": memory_behavior_metrics(memory_true, memory_pred),
        "risk_trend": {"n": len(trend_true), "accuracy": trend_acc},
    }
    return metrics


def _classification(y_true, y_pred, labels):
    from evaluation.metrics.agent_metrics import classification_metrics
    return classification_metrics(y_true, y_pred, labels)


def build_report(cases, preds, metrics, args, elapsed):
    lines = []
    lines.append(f"# Agent Benchmark v1.1 Baseline Report")
    lines.append(f"")
    lines.append(f"- 日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- cases: {len(cases)}")
    lines.append(f"- predictor: {args.predictor} (emotion={args.emotion}, risk={args.risk})")
    lines.append(f"- 耗时: {elapsed:.1f}s")
    lines.append(f"- benchmark: {args.cases.name}")
    lines.append(f"")

    lines.append("## 1. 总览")
    lines.append("")
    lines.append("| 模块 | 指标 | 值 |")
    lines.append("|---|---|---|")
    r = metrics["risk"]
    lines.append(f"| Risk | Macro F1 | {r['macro_f1']} |")
    lines.append(f"| Risk | High-risk Recall (L2+L3) | {r['high_risk_recall']} (tp={r['hr_tp']}, fn={r['hr_fn']}) |")
    lines.append(f"| Risk | False Positive / Negative Rate | {r['false_positive_rate']} / {r['false_negative_rate']} |")
    ec = metrics["emotion_coarse"]
    lines.append(f"| Emotion (coarse 5-class) | Accuracy / Macro F1 | {ec['accuracy']} / {ec['macro_f1']} |")
    li = metrics["legacy_intent"]
    lines.append(f"| Legacy Intent Coverage | Micro F1 / Macro F1 | {li['legacy_intent_micro_f1']} / {li['legacy_intent_macro_f1']} |")
    lines.append(f"| Legacy Intent Coverage | Zero-recall labels | {len(li['zero_recall_labels'])} |")
    rc = metrics["recommendation"]
    lines.append(f"| Recommendation | Trigger Precision / Recall | {rc['trigger_precision']} / {rc['trigger_recall']} |")
    lines.append(f"| Recommendation | Mode Accuracy | {rc['mode_accuracy']} |")
    pa = metrics["primary_action"]
    lines.append(f"| Primary Action | Accuracy / Macro F1 | {pa['accuracy']} / {pa['macro_f1']} |")
    lines.append(f"| Primary Action | Safety Recall | {pa['safety_action_recall']} |")
    ta = metrics["tool_action"]
    lines.append(f"| Tool Action | Micro F1 / Exact Match | {ta['micro_f1']} / {ta['exact_match']} |")
    cur = metrics["routing_current"]
    lines.append(f"| Current Deterministic Routing | Safety Recall | {cur['safety_action_recall']} |")
    mb = metrics["memory_behavior"]
    lines.append(f"| Memory Retrieval Behavior | Precision / Recall | {mb['retrieval_precision']} / {mb['retrieval_recall']} |")
    trend = metrics["risk_trend"]
    lines.append(f"| Risk Trend | Accuracy (multi-turn) | {trend['accuracy']} (n={trend['n']}) |")
    lines.append("")
    lines.append("> 注：Recommendation 指标受上游 Risk / Emotion / Legacy Intent 信号共同影响，不直接等价于 Gate 本身质量。")
    lines.append("")

    lines.append("## 2. 模块明细")
    lines.append("")

    lines.append("### Risk（四级分类）")
    lines.append("")
    lines.append("| level | P | R | F1 | tp/fp/fn |")
    lines.append("|---|---|---|---|---|")
    for lv, d in r["per_class"].items():
        lines.append(f"| L{lv} | {d['precision']} | {d['recall']} | {d['f1']} | {d['tp']}/{d['fp']}/{d['fn']} |")
    lines.append("")

    lines.append("### Emotion Coarse（5 类）")
    lines.append("")
    lines.append("| label | P | R | F1 |")
    lines.append("|---|---|---|---|")
    for lab, d in ec["per_class"].items():
        lines.append(f"| {lab} | {d['precision']} | {d['recall']} | {d['f1']} |")
    lines.append("")

    lines.append("### Legacy Intent Coverage")
    lines.append("")
    lines.append(f"- Zero-recall labels: `{', '.join(li['zero_recall_labels']) if li['zero_recall_labels'] else '无'}`")
    lines.append(f"- Covered labels: `{', '.join(li['covered_labels']) if li['covered_labels'] else '无'}`")
    lines.append("")
    lines.append("| label | P | R | F1 |")
    lines.append("|---|---|---|---|")
    for lab, d in li["per_class"].items():
        lines.append(f"| {lab} | {d['precision']} | {d['recall']} | {d['f1']} |")
    lines.append("")

    lines.append("### Primary Action")
    lines.append("")
    lines.append("| action | P | R | F1 |")
    lines.append("|---|---|---|---|")
    for lab, d in pa["per_class"].items():
        lines.append(f"| {lab} | {d['precision']} | {d['recall']} | {d['f1']} |")
    lines.append("")

    lines.append("### Tool Action")
    lines.append("")
    lines.append("| tool | P | R | tp/fp/fn |")
    lines.append("|---|---|---|---|")
    for lab, d in ta["per_tool"].items():
        lines.append(f"| {lab} | {d['precision']} | {d['recall']} | {d['tp']}/{d['fp']}/{d['fn']} |")
    lines.append("")

    lines.append("### Safety Target")
    lines.append("")
    lines.append("| target | P | R | F1 |")
    lines.append("|---|---|---|---|")
    for lab, d in metrics["safety_target"]["per_class"].items():
        lines.append(f"| {lab} | {d['precision']} | {d['recall']} | {d['f1']} |")
    lines.append("")

    # 失败样本 Top 20
    lines.append("## 3. 失败样本（Top 20）")
    lines.append("")
    lines.append("| case_id | 期望 vs 预测 | 模块 |")
    lines.append("|---|---|---|")
    failures = []
    for c, p in zip(cases, preds):
        exp = c.expected
        text = c.conversation[-1].content[:40]
        if exp.risk_level.value >= 2 and p.risk_level < 2:
            failures.append((c.case_id, f"risk: 期望L{exp.risk_level.value} 预测L{p.risk_level} {text}", "risk-fn"))
        elif exp.primary_action.value == "safety_intervention" and p.primary_action != "safety_intervention":
            failures.append((c.case_id, f"primary: 期望{exp.primary_action.value} 预测{p.primary_action} {text}", "primary-safety"))
        elif exp.recommendation_action.value in {"hard", "soft"} and p.recommendation_action == "none":
            failures.append((c.case_id, f"rec: 期望{exp.recommendation_action.value} 预测none {text}", "rec-miss"))
        elif exp.recommendation_action.value == "none" and p.recommendation_action in {"hard", "soft"}:
            failures.append((c.case_id, f"rec: 期望none 预测{p.recommendation_action} {text}", "rec-over"))
        elif sorted(t.value for t in exp.tool_actions) != sorted(p.tool_actions):
            failures.append((c.case_id, f"tool: 期望{sorted(t.value for t in exp.tool_actions)} 预测{sorted(p.tool_actions)} {text}", "tool-mismatch"))
    for case_id, desc, kind in failures[:20]:
        lines.append(f"| {case_id} | {desc} | {kind} |")
    lines.append("")

    lines.append("## 4. 结论与 Phase 建议")
    lines.append("")
    lines.append("见 docs/baseline_report.md 详细分析。")
    lines.append("")
    return "\n".join(lines)


def recompute_from_json(json_path: Path):
    """从已保存的 baseline JSON 重算指标并重写报告（不重新运行预测器）。"""
    data = json.load(open(json_path, encoding="utf-8"))
    cases = load_cases(PROJECT_ROOT / "evaluation" / "datasets" / "agent_benchmark_v1_1.jsonl")
    by_id = {c.case_id: c for c in cases}

    preds = []
    for d in data["details"]:
        c = by_id.get(d["case_id"])
        if c is None:
            continue
        ex = d.get("extra", {})
        pr = d["predicted"]
        preds.append(PredictedOutcome(
            intent=pr["intent"],
            intent_confidence=0.6,
            emotion=pr["emotion"],
            emotion_intensity=pr.get("emotion_intensity", 0.5),
            risk_level=pr["risk_level"],
            risk_trend=pr.get("risk_trend", "new"),
            memory_needed=pr["memory_needed"],
            recommendation_action=pr["recommendation_action"],
            primary_action=pr["primary_action"],
            tool_actions=pr.get("tool_actions", []),
            safety_target=pr.get("safety_target", "none"),
            extra=ex,
        ))
    cases = [c for c in cases if c.case_id in {d["case_id"] for d in data["details"]}]

    metrics = compute_all_metrics(cases, preds)

    class _Args:
        predictor = data["meta"].get("predictor", "module")
        emotion = data["meta"].get("emotion_channel", "bert")
        risk = data["meta"].get("risk_channel", "bert")
        cases = json_path.parent / "agent_benchmark_v1_1.jsonl"
    args = _Args()

    md_content = build_report(cases, preds, metrics, args, data["meta"].get("elapsed_s", 0))
    md_path = json_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    data["metrics"] = metrics
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已从 {json_path.name} 重算指标并更新报告")
    print(f"  Risk Macro F1: {metrics['risk']['macro_f1']}")
    print(f"  High-risk Recall: {metrics['risk']['high_risk_recall']}")
    print(f"  Emotion coarse Acc: {metrics['emotion_coarse']['accuracy']}")
    print(f"  Primary Action Acc: {metrics['primary_action']['accuracy']}")


def main():
    parser = argparse.ArgumentParser(description="Agent Benchmark v1.1 evaluation runner")
    parser.add_argument("--predictor", choices=["module"], default="module")
    parser.add_argument("--emotion", choices=["bert", "rule"], default="bert")
    parser.add_argument("--risk", choices=["bert", "rule"], default="bert")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "evaluation" / "datasets" / "agent_benchmark_v1_1.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="只看前 N 条（0=全部）")
    parser.add_argument("--report-name", default="agent_benchmark")
    parser.add_argument("--from-json", type=Path, default=None,
                        help="从已保存的 baseline JSON 重算指标与报告（不再运行预测）")
    args = parser.parse_args()

    if args.from_json is not None:
        recompute_from_json(args.from_json)
        return

    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    print(f"[INFO] 加载 {len(cases)} 条 case")

    predictor = build_predictor(args)
    print(f"[INFO] predictor={args.predictor} emotion={args.emotion} risk={args.risk}")

    t0 = time.time()
    preds = [predictor.predict(c) for c in cases]
    elapsed = time.time() - t0
    print(f"[INFO] 预测完成，耗时 {elapsed:.1f}s")

    metrics = compute_all_metrics(cases, preds)

    risk_true_counter = Counter(c.expected.risk_level.value for c in cases)
    risk_pred_counter = Counter(p.risk_level for p in preds)
    print(f"[INFO] 期望风险分布: {dict(sorted(risk_true_counter.items()))}")
    print(f"[INFO] 预测风险分布: {dict(sorted(risk_pred_counter.items()))}")

    reports_dir = PROJECT_ROOT / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"baseline_{args.report_name}_{args.emotion}_{args.risk}_{ts}"
    json_path = reports_dir / f"{report_name}.json"
    md_path = reports_dir / f"{report_name}.md"

    detail = []
    for c, p in zip(cases, preds):
        detail.append({
            "case_id": c.case_id,
            "expected": {
                "risk_level": c.expected.risk_level.value,
                "emotion": c.expected.emotion,
                "intent": [i.value for i in c.expected.intent],
                "recommendation_action": c.expected.recommendation_action.value,
                "primary_action": c.expected.primary_action.value,
                "tool_actions": [t.value for t in c.expected.tool_actions],
                "safety_target": c.expected.safety_target.value,
                "memory_needed": c.expected.memory_needed,
            },
            "predicted": {
                "risk_level": p.risk_level, "emotion": p.emotion, "intent": p.intent,
                "recommendation_action": p.recommendation_action,
                "primary_action": p.primary_action,
                "tool_actions": p.tool_actions,
                "safety_target": p.safety_target,
                "memory_needed": p.memory_needed,
            },
            "extra": {k: v for k, v in p.extra.items() if k != "raw_risk"},
        })

    payload = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cases": len(cases), "predictor": args.predictor,
            "emotion_channel": args.emotion, "risk_channel": args.risk,
            "elapsed_s": round(elapsed, 1),
        },
        "metrics": metrics,
        "details": detail,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md_content = build_report(cases, preds, metrics, args, elapsed)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[OK] 报告已输出:")
    print(f"     {json_path}")
    print(f"     {md_path}")
    print(f"\n关键指标:")
    print(f"  Risk Macro F1: {metrics['risk']['macro_f1']}")
    print(f"  High-risk Recall: {metrics['risk']['high_risk_recall']}")
    print(f"  Emotion coarse Acc: {metrics['emotion_coarse']['accuracy']}")
    print(f"  Legacy Intent Macro F1: {metrics['legacy_intent']['legacy_intent_macro_f1']}")
    print(f"  Recommendation Trigger P/R: {metrics['recommendation']['trigger_precision']}/{metrics['recommendation']['trigger_recall']}")
    print(f"  Primary Action Acc: {metrics['primary_action']['accuracy']}")
    print(f"  Safety Primary Recall: {metrics['primary_action']['safety_action_recall']}")
    print(f"  Tool Action Micro F1: {metrics['tool_action']['micro_f1']}")


if __name__ == "__main__":
    main()
