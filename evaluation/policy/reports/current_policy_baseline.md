# Current Policy Baseline

> 对应 Phase 3 Preflight P3-0.4 · 2026-08-19 23:55:48
> 对象：LegacyPolicyAdapter on Agent Benchmark v1.1（362 cases，emotion=rule/risk=rule）

## 指标

| 指标 | 值 | 目标（Phase 3） |
|---|---:|---:|
| primary_action_accuracy | 0.6547 | ≥0.90 |
| primary_action_macro_f1 | 0.3006 | ≥0.85 |
| safety_action_recall | 0.3134 | ≥0.98 |
| safety_target_accuracy | 0.8315 | ≥0.95 |
| tool_action_micro_f1 | 0.1722 | ≥0.80 |
| tool_action_exact_match | 0.0829 | - |
| recommendation_mode_accuracy | 0.6133 | ≥0.80 |
| policy_exact_match | 0.0193 | ≥0.75 |

## 结论

- Primary Action Acc **0.655** / Safety Recall **0.313** / Policy Exact **0.019**
- 当前 legacy 路由距离 Phase 3 目标差距巨大（Safety Recall 需从 0.31 → 0.98）
- 这正是 Phase 3 要解决的核心问题（当前 routing 由零散 if-else + LLM 决定）