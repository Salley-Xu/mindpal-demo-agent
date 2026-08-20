# -*- coding: utf-8 -*-
"""
Phase 5 Task 5.3/5.4：Binary/4-class Fusion + Calibration 阈值扫描。

目的：在 792 条跨源独立 Risk Test 上，找到当前 v4_2 模型能达到的最佳工作点
      （HR Recall vs FPR 权衡），判断是否满足目标（Recall>=0.95, FPR<=0.10），
      以及是否需要 Targeted Expansion 重训。

扫描的决策规则：
  F1  4-class argmax（当前行为）
  F2  binary_prob > t
  F3  P(L3) > t
  F4  P(L2)+P(L3) > t
  F5  argmax + binary 联合门控（argmax>=2 AND binary>t）
  F6  rule 命中（高优先）否则 P(L2)+P(L3) > t

用法：cd project_root && python evaluation/risk/run_fusion_calibration.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(BACKEND_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from evaluation.benchmark_schema import BenchmarkCase  # noqa: E402


def build_test():
    pairs = []
    for path in ["evaluation/datasets/agent_benchmark_v1_1.jsonl",
                 "evaluation/policy/policy_independent_test_v1.jsonl"]:
        for l in open(PROJECT_ROOT / path, encoding="utf-8"):
            if not l.strip():
                continue
            c = BenchmarkCase.model_validate(json.loads(l))
            text = [t.content for t in c.conversation if t.role.value == "user"][-1]
            pairs.append((text, c.expected.risk_level.value))
    return pairs


def predict_all(pairs):
    """加载 v4_2 模型并对全部 case 预测，收集原始概率。"""
    from bert_risk_predictor import BertRiskPredictor
    model = BertRiskPredictor(model_path=str(PROJECT_ROOT / "bert_data/models/v4_2_domain_only_v2/best_model"),
                              device="cpu")
    rows = []
    for text, gold in pairs:
        r = model.predict(text)
        rows.append({
            "gold": gold,
            "pred_4": r["level_4_prediction"],
            "binary": r["binary_probability"],
            "probs": r["class_probabilities"],
            "rule": r["rule_matched"],
            "text": text[:50],
        })
    return rows


def hr(rows, is_high_fn):
    """计算 HR Recall / FPR / Precision。"""
    tp = sum(1 for r in rows if r["gold"] >= 2 and is_high_fn(r))
    fp = sum(1 for r in rows if r["gold"] < 2 and is_high_fn(r))
    fn = sum(1 for r in rows if r["gold"] >= 2 and not is_high_fn(r))
    tn = sum(1 for r in rows if r["gold"] < 2 and not is_high_fn(r))
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    return {"recall": round(recall, 4), "fpr": round(fpr, 4), "prec": round(prec, 4),
            "tp": tp, "fp": fp, "fn": fn}


def main():
    pairs = build_test()
    rows = predict_all(pairs)
    print(f"test: {len(rows)} cases (gold L2/L3: {sum(1 for r in rows if r['gold']>=2)})")

    results = {}

    # F1: 当前 4-class argmax
    results["F1_4class_argmax"] = hr(rows, lambda r: r["pred_4"] >= 2)

    # F2: binary threshold 扫描
    for t in [0.3, 0.5, 0.7, 0.8, 0.9]:
        results[f"F2_binary>{t}"] = hr(rows, lambda r, _t=t: r["binary"] > _t)

    # F3: P(L3) threshold
    for t in [0.1, 0.2, 0.3, 0.5]:
        results[f"F3_P3>{t}"] = hr(rows, lambda r, _t=t: r["probs"][3] > _t)

    # F4: P(L2)+P(L3) threshold
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        results[f"F4_P23>{t}"] = hr(rows, lambda r, _t=t: (r["probs"][2] + r["probs"][3]) > _t)

    # F5: argmax>=2 AND binary>t 联合门控
    for t in [0.3, 0.5, 0.7]:
        results[f"F5_argmax2_and_binary>{t}"] = hr(
            rows, lambda r, _t=t: r["pred_4"] >= 2 and r["binary"] > _t)

    # F6: rule 优先，否则 P23>t
    for t in [0.2, 0.3, 0.4]:
        results[f"F6_rule_else_P23>{t}"] = hr(
            rows, lambda r, _t=t: r["rule"] or (r["probs"][2] + r["probs"][3]) > _t)

    print("\n=== FUSION / CALIBRATION SCAN ===")
    print(f"{'rule':<28} {'Recall':>7} {'FPR':>7} {'Prec':>7} {'TP':>4} {'FP':>4} {'FN':>4}")
    for k, v in results.items():
        print(f"{k:<28} {v['recall']:>7} {v['fpr']:>7} {v['prec']:>7} {v['tp']:>4} {v['fp']:>4} {v['fn']:>4}")

    # 找满足 Recall>=0.95 且 FPR 最小的点；以及 FPR<=0.10 且 Recall 最大的点
    print("\n=== 操作点分析 ===")
    for k, v in results.items():
        if v["recall"] >= 0.95:
            print(f"  Recall>=0.95: {k} (FPR={v['fpr']})")
    for k, v in results.items():
        if v["fpr"] <= 0.10:
            print(f"  FPR<=0.10: {k} (Recall={v['recall']})")

    reports = PROJECT_ROOT / "evaluation" / "risk" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "fusion_calibration_scan.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n": len(rows), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] -> {path.name}")


if __name__ == "__main__":
    main()
