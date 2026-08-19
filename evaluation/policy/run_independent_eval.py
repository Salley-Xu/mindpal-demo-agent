# -*- coding: utf-8 -*-
"""
Phase 3 Closeout C3.4：Independent Policy Test 统一评测。

在 policy_independent_test_v1.jsonl（430 cases，新措辞，未参与规则设计）上跑：
  Legacy / Deterministic / Final Hybrid
状态：oracle（Policy 泛化上限）+ predicted_rule（端到端）

Frozen Benchmark v1.1 只做 regression，不在此出现。

用法：cd project_root && python evaluation/policy/run_independent_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(BACKEND_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from evaluation.benchmark_schema import BenchmarkCase  # noqa: E402
from evaluation.policy.run_policy_eval import (  # noqa: E402
    build_oracle_state, build_state_from_prediction, compute_metrics,
)
from policy.ambiguity import ambiguity_detector  # noqa: E402
from policy.hybrid import hybrid_policy  # noqa: E402
from policy.legacy_policy_adapter import legacy_policy_adapter  # noqa: E402
from policy.safety import safety_policy  # noqa: E402


def run(cases, method, states):
    from policy.deterministic import deterministic_policy
    sources = Counter()
    ambiguous = 0
    rows = []
    for c, st in zip(cases, states):
        if method == "legacy":
            result = legacy_policy_adapter.decide(st)
        elif method == "deterministic":
            r = safety_policy.decide(st)
            if r is not None:
                result = r
            else:
                result = deterministic_policy.decide(st).to_policy_result(source="rule")
        else:  # hybrid
            hybrid_policy.config.mode = "hybrid"
            result = hybrid_policy.decide(st, use_llm=False)
            det = deterministic_policy.decide(st)
            if ambiguity_detector.detect(st, det).is_ambiguous:
                ambiguous += 1
        ap = result.action_plan
        exp = c.expected
        rows.append({
            "case_id": c.case_id,
            "expected": {"primary_action": exp.primary_action.value,
                         "tool_actions": sorted(t.value for t in exp.tool_actions),
                         "safety_target": exp.safety_target.value,
                         "recommendation_mode": exp.recommendation_action.value},
            "predicted": {"primary_action": ap.primary_action.value,
                          "tool_actions": sorted(t.value for t in ap.tool_actions),
                          "safety_target": ap.safety_target.value,
                          "recommendation_mode": ap.recommendation_mode.value},
            "tags": c.tags,
        })
    metrics = compute_metrics(rows)
    metrics["llm_fallback_rate"] = round(ambiguous / len(rows), 4) if method == "hybrid" else 0.0
    return metrics, rows


def main():
    cases = [BenchmarkCase.model_validate(json.loads(l))
             for l in open(PROJECT_ROOT / "evaluation/policy/policy_independent_test_v1.jsonl", encoding="utf-8")
             if l.strip()]
    print(f"independent test: {len(cases)} cases")

    states_oracle = [build_oracle_state(c) for c in cases]
    from evaluation.predictors.module import ModulePredictor
    p_rule = ModulePredictor(emotion_channel="rule", risk_channel="rule")
    states_rule = [build_state_from_prediction(c, p_rule.predict(c)) for c in cases]

    results = {}
    for mode, states in [("oracle", states_oracle), ("predicted_rule", states_rule)]:
        for method in ["legacy", "deterministic", "hybrid"]:
            metrics, rows = run(cases, method, states)
            results[f"{method}/{mode}"] = {"metrics": metrics}
            print(f"[OK] {method}/{mode}: primary={metrics['primary_action_accuracy']} "
                  f"safety_recall={metrics['safety_action_recall']} tool={metrics['tool_action_micro_f1']} "
                  f"rec={metrics['recommendation_mode_accuracy']} exact={metrics['policy_exact_match']} "
                  f"llm={metrics['llm_fallback_rate']}")

    reports = PROJECT_ROOT / "evaluation" / "policy" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports / f"independent_eval_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "n": len(cases),
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] -> {path.name}")


if __name__ == "__main__":
    main()
