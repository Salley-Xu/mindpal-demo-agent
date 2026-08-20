# Replay Design

> 对应 Phase 7 Task 7.6
> 日期：2026-08-20
> 实现：`backend/tracing/replay.py`

---

## 1. 目标

```text
trace → 重建 AgentState → 重跑 Policy / Tool 选择
```

支持：
- same-version replay（复现原始决策）
- new-version counterfactual replay（对比新版本行为）

## 2. 流程

```text
TraceLogger.read(trace_id)
  → ReplayRunner.reconstruct_state(trace)   // trace.perception → AgentState
  → ReplayRunner.replay_policy(trace, mode) // hybrid | legacy
  → 输出 action_plan / source / matched_rules
```

## 3. 用例

| 用途 | 说明 |
|---|---|
| Regression | 冻结 benchmark 上重放，对比版本行为 |
| Bug reproduction | failure case 精确回放定位 |
| Ablation | 同一 trace 跑不同 policy 模式 |
| Offline policy comparison | counterfactual legacy vs new |

## 4. 验证

- 从 `tr_demo_001` trace 重放 → policy 重新决策 safety_intervention ✅
- counterfactual legacy replay 可用 ✅
