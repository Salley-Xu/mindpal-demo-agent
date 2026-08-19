# Phase 2 Regression Closeout

> 对应 Phase 3 Preflight P3-0.1
> 日期：2026-08-19
> 目的：实际重跑冻结 Agent Benchmark v1.1，证明 Phase 2 是 behavior-preserving refactor。

---

## 1. 设置

- Benchmark：`evaluation/datasets/agent_benchmark_v1_1.jsonl`（362 cases）
- 通道：BERT 情绪 + BERT 风险（与 Phase 0.5 相同配置）
- Before：`evaluation/reports/baseline_v1_1_bert_bert_20260818_233508.json`（Phase 0.5）
- After：`evaluation/reports/baseline_p3_regression_bert_bert_*.json`（本次）

## 2. 全指标 diff（after − before）

| 模块 | 差异 | 状态 |
|---|---|---|
| Risk | 0 | ✅ |
| Emotion | 0 | ✅ |
| Emotion Coarse | 0 | ✅ |
| Legacy Intent | 0 | ✅ |
| Recommendation | 0 | ✅ |
| Primary Action | 0 | ✅ |
| Tool Action | 0 | ✅ |
| Safety Target | 0 | ✅ |
| Memory Behavior | 0 | ✅ |

## 3. 关键指标（after = before）

| 指标 | Before | After | diff |
|---|---:|---:|---:|
| Risk Macro F1 | 0.371 | 0.371 | 0 |
| High-risk Recall | 0.6119 | 0.6119 | 0 |
| Risk FPR / FNR | 0.4373 / 0.3881 | 0.4373 / 0.3881 | 0 |
| Emotion Coarse Acc | 0.5967 | 0.5967 | 0 |
| Legacy Intent Macro | 0.1606 | 0.1606 | 0 |
| Rec Trigger P/R | 0.1724 / 0.3289 | 0.1724 / 0.3289 | 0 |
| Primary Action Acc | 0.5276 | 0.5276 | 0 |
| Safety Recall | 0.6269 | 0.6269 | 0 |
| Tool Micro F1 | 0.1608 | 0.1608 | 0 |
| Memory P/R | 0.1466 / 0.6087 | 0.1466 / 0.6087 | 0 |

## 4. 结论

**after − before = 0（全部模块）** → Phase 2 确认为 **behavior-preserving refactor** ✅

- Phase 2 仅新增 `backend/state/`（旁路模块）+ orchestrator Shadow 钩子（try/except 包裹，不进入评测模块路径）
- Risk / Recommendation / Routing / Memory 生产逻辑零改动
- 允许差异（日志/Trace）未出现；业务指标完全一致

**Phase 3 GO 条件 P3-0.1 通过。**
