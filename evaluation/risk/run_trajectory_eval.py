# -*- coding: utf-8 -*-
"""
Phase 5 Task 5.7：Multi-turn Trajectory 评测。

用 DynamicRiskTracker 处理每条轨迹（gold utterance_level 作为单轮输入），
对比聚合后的 session 轨迹与 expected。

指标：Trajectory Accuracy / Early Detection Rate / Escalation Accuracy /
      Recovery Accuracy / Trend 参考 / Time-to-detect。

用法：cd project_root && python evaluation/risk/run_trajectory_eval.py
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
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from risk.dynamic_state import DynamicRiskTracker  # noqa: E402


def run_traj(traj):
    tracker = DynamicRiskTracker()
    states = []
    for turn in traj["turns"]:
        st = tracker.update(
            level=turn["utterance_level"],
            confidence=0.9,
            text=turn["text"],
            context=turn.get("context", {}),
        )
        states.append(st)
    return [s.level for s in states], states


def main():
    trajs = [json.loads(l) for l in
             open(PROJECT_ROOT / "evaluation/risk/risk_trajectory_benchmark_v1.jsonl", encoding="utf-8")
             if l.strip()]

    exact = 0
    early_ok = 0
    n_early = 0
    rec_ok, n_rec = 0, 0
    esc_ok, n_esc = 0, 0
    detect_times = []

    rows = []
    for t in trajs:
        got, states = run_traj(t)
        exp = t["expected"]
        match = got == exp
        if match:
            exact += 1
        rows.append({"id": t["trajectory_id"], "scenario": t["scenario"],
                     "expected": exp, "got": got, "match": match, "trends": [s.trend for s in states]})

        # Early detection：首个 session>=2 的轮
        exp_detect = t.get("expected_early_detect_turn")
        if exp_detect is not None:
            n_early += 1
            got_detect = next((i for i, l in enumerate(got) if l >= 2), None)
            if got_detect is not None:
                detect_times.append(got_detect)
                if got_detect <= exp_detect:
                    early_ok += 1
            else:
                if exp_detect is None:
                    early_ok += 1

        # Escalation：0→1→2→3 类型
        if t["scenario"] in ("gradual_escalation", "third_party_escalation"):
            n_esc += 1
            if got == exp:
                esc_ok += 1
        # Recovery：3→2→1→0
        if t["scenario"] == "recovery":
            n_rec += 1
            if got == exp:
                rec_ok += 1

    metrics = {
        "n": len(trajs),
        "trajectory_accuracy": round(exact / len(trajs), 4),
        "early_detection_rate": round(early_ok / n_early, 4) if n_early else None,
        "escalation_accuracy": round(esc_ok / n_esc, 4) if n_esc else None,
        "recovery_accuracy": round(rec_ok / n_rec, 4) if n_rec else None,
        "avg_time_to_detect": round(sum(detect_times) / len(detect_times), 2) if detect_times else None,
    }
    print("=== TRAJECTORY EVAL ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()
    for r in rows:
        flag = "✓" if r["match"] else "✗"
        print(f"  {flag} {r['id']} [{r['scenario']}] exp={r['expected']} got={r['got']}")

    reports = PROJECT_ROOT / "evaluation/risk/reports"
    reports.mkdir(parents=True, exist_ok=True)
    with open(reports / "trajectory_eval.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "rows": rows}, f, ensure_ascii=False, indent=2)
    print("\n[OK] -> reports/trajectory_eval.json")


if __name__ == "__main__":
    main()
