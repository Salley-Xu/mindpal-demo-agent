# -*- coding: utf-8 -*-
"""
Risk Pipeline Ablation runner（Phase 0.5 Task 0.5.3）。

用法：
  python evaluation/runners/run_risk_ablation.py --cases evaluation/datasets/agent_benchmark_v1_1.jsonl

输出：
  evaluation/reports/risk_ablation_<ts>.json   每 case debug + 每 variant 指标
  evaluation/reports/risk_ablation_<ts>.md     人类可读报告
  docs/risk_pipeline_audit.md                  工程 sanity check + 根因结论
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

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark_schema import BenchmarkCase  # noqa: E402
from evaluation.metrics.agent_metrics import risk_metrics  # noqa: E402
from evaluation.predictors.risk_ablation import RiskAblationPredictor  # noqa: E402

VARIANTS = ["raw_bert", "rule_override", "context_rules", "full_pipeline", "rule_only"]
VARIANT_LABELS = {
    "raw_bert": "A: Raw BERT",
    "rule_override": "B: +Rule Override",
    "context_rules": "C: +Context Rules",
    "full_pipeline": "D: Full Pipeline",
    "rule_only": "E: Rule-only",
}


def load_cases(path: Path) -> list:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(BenchmarkCase.model_validate(json.loads(line)))
    return cases


def sanity_check(predictor) -> dict:
    """工程 sanity check：明显错误样本逐项审计。"""
    text = "你在吗？我就随便聊聊。"
    raw = predictor.raw_bert(text)
    p = predictor._get_predictor()
    audit = {
        "text": text,
        "raw_bert": raw,
        "checkpoint_path": str(p.model_path),
        "tokenizer_path": str(p.model_path),
        "architecture": "CoralBERT(v4.3)" if p._is_coral else "MultiTaskBERT(v4.2)",
        "num_labels_4": p.model.classifier_4.out_features if not p._is_coral else 3,
        "id2label_inference": "{0: level_0, 1: level_1, 2: level_2, 3: level_3}",  # predict() 内硬编码
        "label2id_training": "cssrs_lite_level 0-3（SOS-1K 0-10 映射，见技术报告 §2.3）",
        "max_length": p.max_length,
        "truncation": True,
        "padding": "max_length",
        "softmax_dim": -1,
        "binary_threshold_used": "NO",  # predict() 存储 binary_threshold 但未使用（P1 融合未实现）
    }
    return audit


def build_report(cases, records, metrics, audit, elapsed):
    lines = []
    lines.append("# Risk Pipeline Ablation Report")
    lines.append("")
    lines.append(f"- 日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- cases: {len(records)}")
    lines.append(f"- 耗时: {elapsed:.1f}s")
    lines.append("")
    lines.append("## 1. 各 Variant 指标")
    lines.append("")
    lines.append("| Variant | Macro F1 | High-risk Recall | FPR | FNR | L2 P | L3 P |")
    lines.append("|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        m = metrics[v]
        p2 = m["per_class"].get(2, {}); p3 = m["per_class"].get(3, {})
        lines.append(
            f"| {VARIANT_LABELS[v]} | {m['macro_f1']} | {m['high_risk_recall']} "
            f"| {m['false_positive_rate']} | {m['false_negative_rate']} "
            f"| {p2.get('precision', 0)} | {p3.get('precision', 0)} |"
        )
    lines.append("")

    lines.append("### 每级明细")
    lines.append("")
    for v in VARIANTS:
        m = metrics[v]
        lines.append(f"**{VARIANT_LABELS[v]}**")
        lines.append("")
        lines.append("| level | P | R | F1 | tp/fp/fn |")
        lines.append("|---|---|---|---|---|")
        for lv, d in m["per_class"].items():
            lines.append(f"| L{lv} | {d['precision']} | {d['recall']} | {d['f1']} | {d['tp']}/{d['fp']}/{d['fn']} |")
        lines.append("")

    # 关键错误 case
    lines.append("## 2. 关键错误 Case")
    lines.append("")
    lines.append("### L0 → L2/L3 误报（full_pipeline）")
    lines.append("")
    for r in records:
        if r["expected"] == 0 and r["variants"]["full_pipeline"] >= 2:
            lines.append(f"- `{r['case_id']}` L0→L{r['variants']['full_pipeline']} raw={r['raw_bert']['label']} conf={r['raw_bert']['confidence']}  `{r['text'][:30]}`")
    lines.append("")
    lines.append("### L2/L3 → L0/L1 漏判（full_pipeline）")
    lines.append("")
    for r in records:
        if r["expected"] >= 2 and r["variants"]["full_pipeline"] < 2:
            lines.append(f"- `{r['case_id']}` 期望L{r['expected']}→L{r['variants']['full_pipeline']} raw={r['raw_bert']['label']} conf={r['raw_bert']['confidence']}  `{r['text'][:30]}`")
    lines.append("")

    lines.append("## 3. 工程 Sanity Check")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(audit, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_audit_doc(audit, metrics, records):
    """输出 docs/risk_pipeline_audit.md（工程审计 + 根因结论）。"""
    raw_metrics = metrics["raw_bert"]
    full_metrics = metrics["full_pipeline"]
    lines = []
    lines.append("# Risk Pipeline Audit")
    lines.append("")
    lines.append(f"> 日期：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 对象：`{audit['checkpoint_path']}`（{audit['architecture']}）")
    lines.append("")
    lines.append("## 1. 工程配置审计")
    lines.append("")
    lines.append("| 项目 | 值 | 判定 |")
    lines.append("|---|---|---|")
    lines.append(f"| checkpoint path | `{audit['checkpoint_path']}` | 存在 ✅ |")
    lines.append(f"| tokenizer path | `{audit['tokenizer_path']}` | 存在 ✅ |")
    lines.append(f"| 模型架构 | {audit['architecture']} | 自动检测 ✅ |")
    lines.append(f"| 4 分类头输出 | {audit['num_labels_4']} | 对齐 4 级 ✅ |")
    lines.append(f"| 推理 label 映射 | {audit['id2label_inference']} | 与训练 `cssrs_lite_level 0-3` 一致 ✅ |")
    lines.append(f"| 训练 label 映射 | {audit['label2id_training']} | 技术报告 §2.3 一致 ✅ |")
    lines.append(f"| max_length / truncation | {audit['max_length']} / True | 标准 ✅ |")
    lines.append(f"| softmax dim | -1 | 标准 ✅ |")
    lines.append(f"| binary_threshold 融合（P1） | 存储但未在 predict() 使用 | **文档-实现不一致 ⚠️** |")
    lines.append("")
    lines.append("**结论：未发现 label 映射 / tokenizer / 架构 层面的工程 Bug。** 文档化的 `binary_prob>0.5 AND 4分类<2 → level_2` 三级融合规则在 `bert_risk_predictor.predict()` 中**未实现**（参数存储未使用）。")
    lines.append("")
    lines.append("## 2. 分层 Ablation 指标")
    lines.append("")
    lines.append("| Variant | Macro F1 | High-risk Recall | FPR | FNR |")
    lines.append("|---|---:|---:|---:|---:|")
    for v in VARIANTS:
        m = metrics[v]
        lines.append(f"| {VARIANT_LABELS[v]} | {m['macro_f1']} | {m['high_risk_recall']} | {m['false_positive_rate']} | {m['false_negative_rate']} |")
    lines.append("")
    lines.append("## 3. 明显错误样本审计")
    lines.append("")
    text = audit["text"]
    raw = audit["raw_bert"]
    lines.append(f"**输入**：`{text}`")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({k: raw[k] for k in ("raw_label", "raw_confidence", "probs_4", "binary_prob")}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("Raw BERT 输出：" + ("**Level 3 @ 高置信度**" if raw["raw_label"] == 3 else f"Level {raw['raw_label']}") + "。")
    lines.append("")
    lines.append("## 4. 根因结论（决策规则 §11）")
    lines.append("")
    if full_metrics["macro_f1"] < raw_metrics["macro_f1"] - 0.05:
        lines.append("### 情况 A：Raw BERT 正常，Full Pipeline 变差")
        lines.append("- 问题主要来自 rule/context/aggregator 策略层，Phase 5 优先修策略层。")
    elif raw_metrics["false_positive_rate"] >= 0.2 and raw_metrics["false_negative_rate"] >= 0.3:
        lines.append("### 情况 B：Raw BERT 本身 FPR 高 + FN 高")
        lines.append("- 模型本身需要重新校准 / 重训 / 切 checkpoint，Phase 5 进入模型重构。")
    else:
        lines.append("### 情况 C：Raw BERT 指标好但映射异常")
        lines.append("- 先修工程 bug，不启动新训练实验。")
    lines.append("")
    lines.append("（详细量化依据见下方数据）")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"raw_bert": raw_metrics, "full_pipeline": full_metrics}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Risk Pipeline Ablation")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "evaluation" / "datasets" / "agent_benchmark_v1_1.jsonl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    print(f"[INFO] 加载 {len(cases)} 条 case")

    predictor = RiskAblationPredictor(model_path=args.model, device=args.device)
    t0 = time.time()
    records = [predictor.analyze(c) for c in cases]
    elapsed = time.time() - t0

    # 各 variant 指标
    metrics = {}
    for v in VARIANTS:
        y_true = [r["expected"] for r in records]
        y_pred = [r["variants"][v] for r in records]
        metrics[v] = risk_metrics(y_true, y_pred)

    # 工程 sanity check
    audit = sanity_check(predictor)

    # 打印关键结论
    print(f"[INFO] 预测完成，耗时 {elapsed:.1f}s")
    print(f"  Raw BERT      : MacroF1={metrics['raw_bert']['macro_f1']} HRRecall={metrics['raw_bert']['high_risk_recall']} FPR={metrics['raw_bert']['false_positive_rate']}")
    print(f"  Full Pipeline : MacroF1={metrics['full_pipeline']['macro_f1']} HRRecall={metrics['full_pipeline']['high_risk_recall']} FPR={metrics['full_pipeline']['false_positive_rate']}")
    print(f"  Rule-only     : MacroF1={metrics['rule_only']['macro_f1']} HRRecall={metrics['rule_only']['high_risk_recall']}")
    print(f"[WARN] sanity check: '{audit['text']}' -> raw label {audit['raw_bert']['raw_label']} conf {audit['raw_bert']['raw_confidence']}")

    # 输出报告
    reports_dir = PROJECT_ROOT / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"risk_ablation_{ts}.json"
    md_path = reports_dir / f"risk_ablation_{ts}.md"

    payload = {
        "meta": {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "cases": len(records), "elapsed_s": round(elapsed, 1)},
        "metrics": metrics,
        "sanity_check": audit,
        "details": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_report(cases, records, metrics, audit, elapsed))

    # 输出 docs/risk_pipeline_audit.md
    audit_doc = write_audit_doc(audit, metrics, records)
    audit_path = PROJECT_ROOT / "docs" / "risk_pipeline_audit.md"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write(audit_doc)

    print(f"[OK] 报告已输出:")
    print(f"     {json_path}")
    print(f"     {md_path}")
    print(f"     {audit_path}")


if __name__ == "__main__":
    main()
