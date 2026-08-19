# -*- coding: utf-8 -*-
"""
Phase 3 Closeout C3.3：Independent Safety Slice 评估。

指标：Safety Recall / Precision / FPR / Target Accuracy / Over-trigger Rate。

状态：
  - oracle：gold 信号 → 验证 Policy 自身安全逻辑（含 S04 Contract A）
  - predicted_rule：ModulePredictor 粗感知 → 验证端到端安全 FPR/FNR（归因感知）

用法：cd project_root && python evaluation/policy/run_safety_slice.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(BACKEND_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from evaluation.benchmark_schema import BenchmarkCase  # noqa: E402
from evaluation.policy.run_policy_eval import build_oracle_state, build_state_from_prediction  # noqa: E402
from policy.safety import safety_policy  # noqa: E402


def _policy_pred(state):
    """确定性规则栈的安全判定（P0 Safety 优先）。"""
    r = safety_policy.decide(state)
    if r is not None:
        return True, r.action_plan.safety_target.value
    return False, "none"


def evaluate(cases, mode, predictor=None):
    rows = []
    for c in cases:
        st = build_oracle_state(c) if mode == "oracle" else build_state_from_prediction(c, predictor.predict(c))
        pred_safety, pred_target = _policy_pred(st)
        gold_safety = c.expected.primary_action.value == "safety_intervention"
        gold_target = c.expected.safety_target.value
        rows.append({
            "case_id": c.case_id, "tags": c.tags,
            "gold_safety": gold_safety, "gold_target": gold_target,
            "pred_safety": pred_safety, "pred_target": pred_target,
            "text": [t.content for t in c.conversation if t.role.value == "user"][-1][:40],
        })

    tp = sum(1 for r in rows if r["gold_safety"] and r["pred_safety"])
    fn = sum(1 for r in rows if r["gold_safety"] and not r["pred_safety"])
    fp = sum(1 for r in rows if not r["gold_safety"] and r["pred_safety"])
    tn = sum(1 for r in rows if not r["gold_safety"] and not r["pred_safety"])
    pos = tp + fn
    neg = fp + tn

    recall = tp / pos if pos else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    fpr = fp / neg if neg else 0.0
    over_trigger = fp / len(rows)

    # Target Accuracy（gold 与 pred 都是 safety 的 case 中 target 一致率）
    both_safety = [r for r in rows if r["gold_safety"] and r["pred_safety"]]
    target_acc = sum(1 for r in both_safety if r["gold_target"] == r["pred_target"]) / len(both_safety) if both_safety else 1.0

    metrics = {
        "n": len(rows), "positive": pos, "negative": neg,
        "safety_recall": round(recall, 4),
        "safety_precision": round(precision, 4),
        "safety_fpr": round(fpr, 4),
        "safety_target_accuracy": round(target_acc, 4),
        "over_trigger_rate": round(over_trigger, 4),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }
    return metrics, rows


def main():
    cases = [BenchmarkCase.model_validate(json.loads(l))
             for l in open(PROJECT_ROOT / "evaluation/policy/policy_safety_independent_v1.jsonl", encoding="utf-8")
             if l.strip()]

    results = {}
    m, rows = evaluate(cases, "oracle")
    results["oracle"] = m
    print("[OK] oracle  Safety Slice:")
    for k, v in m.items():
        print(f"  {k}: {v}")

    from evaluation.predictors.module import ModulePredictor
    p_rule = ModulePredictor(emotion_channel="rule", risk_channel="rule")
    m2, rows2 = evaluate(cases, "predicted", predictor=p_rule)
    results["predicted_rule"] = m2
    print("\n[OK] predicted_rule  Safety Slice:")
    for k, v in m2.items():
        print(f"  {k}: {v}")

    # 保存报告
    reports = PROJECT_ROOT / "evaluation" / "policy" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports / f"safety_slice_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "results": results, "rows_oracle": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] -> {path.name}")


if __name__ == "__main__":
    main()
