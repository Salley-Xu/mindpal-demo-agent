# -*- coding: utf-8 -*-
"""
Policy Ablation（Phase 3 §20）。

方法：A Legacy / C Deterministic / E Final Hybrid（Learned 跳过，Deterministic 达标）
状态：oracle（Policy 上限） / predicted-rule（粗感知下界） / predicted-bert（生产感知）

额外报告：
    - LLM fallback rate（= ambiguity gate 判定 ambiguous 的比例，LLM 关闭时的 deterministic_fallback 率）
    - source 分布（safety / deterministic / fallback）

用法：
  cd backend && PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe ../evaluation/policy/run_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
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

METHODS = ["legacy", "deterministic", "hybrid"]


def run_cell(cases, method, states):
    """运行一个 ablation cell（states 为预构建的 AgentState 列表），返回 metrics + 附加统计。"""
    sources = Counter()
    ambiguous = 0
    total = len(cases)
    t0 = time.time()
    rows = []
    for c, st in zip(cases, states):
        if method == "legacy":
            result = legacy_policy_adapter.decide(st)
        elif method == "deterministic":
            from policy.deterministic import deterministic_policy
            r = safety_policy.decide(st)
            if r is not None:
                result = r
                sources["safety"] += 1
            else:
                det = deterministic_policy.decide(st)
                result = det.to_policy_result(source="rule")
                sources["deterministic"] += 1
        else:  # hybrid
            hybrid_policy.config.mode = "hybrid"
            result = hybrid_policy.decide(st, use_llm=False)
            det = __det_for(st)
            amb = ambiguity_detector.detect(st, det)
            if amb.is_ambiguous:
                ambiguous += 1
            if result.source == "rule":
                sources["safety" if "S0" in "".join(result.matched_rules) else "deterministic"] += 1
            else:
                sources[result.source] += 1
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
        })
    elapsed = time.time() - t0
    metrics = compute_metrics(rows)
    metrics["avg_decision_latency_ms"] = round(elapsed / total * 1000, 2)
    metrics["llm_fallback_rate"] = round(ambiguous / total, 4) if method == "hybrid" else 0.0
    metrics["source_distribution"] = dict(sources)
    return metrics, rows


def __det_for(state):
    from policy.deterministic import deterministic_policy
    return deterministic_policy.decide(state)


def main():
    cases = [BenchmarkCase.model_validate(json.loads(l))
             for l in open(PROJECT_ROOT / "evaluation/datasets/agent_benchmark_v1_1.jsonl", encoding="utf-8")
             if l.strip()]
    bert_path = (PROJECT_ROOT / "bert_data/models/v4_2_domain_only_v2/best_model").as_posix()

    # 预构建各状态模式的 AgentState 列表（复用，避免重复预测）
    print("[..] oracle states")
    states_oracle = [build_oracle_state(c) for c in cases]

    from evaluation.predictors.module import ModulePredictor
    print("[..] predicted-rule states")
    p_rule = ModulePredictor(emotion_channel="rule", risk_channel="rule")
    states_rule = [build_state_from_prediction(c, p_rule.predict(c)) for c in cases]

    print("[..] predicted-bert states (模型加载 + 362 次推理，约 2-3 分钟)")
    p_bert = ModulePredictor(emotion_channel="rule", risk_channel="bert")
    states_bert = [build_state_from_prediction(c, p_bert.predict(c)) for c in cases]

    results = {}
    for mode_name, states in [("oracle", states_oracle),
                              ("predicted_rule", states_rule),
                              ("predicted_bert", states_bert)]:
        for m in METHODS:
            metrics, rows = run_cell(cases, m, states)
            results[f"{m}/{mode_name}"] = {"metrics": metrics}
            print(f"[OK] {m}/{mode_name}  primary={metrics['primary_action_accuracy']} "
                  f"safety_recall={metrics['safety_action_recall']} exact={metrics['policy_exact_match']} "
                  f"llm_rate={metrics['llm_fallback_rate']}")

    reports_dir = PROJECT_ROOT / "evaluation" / "policy" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = reports_dir / f"policy_ablation_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "n": len(cases),
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] ablation -> {out.name}")
    print(f"    BERT model path used: {bert_path}")
    return out


if __name__ == "__main__":
    main()
