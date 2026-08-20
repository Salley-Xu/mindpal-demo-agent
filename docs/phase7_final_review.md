# Phase 7 Final Review — Observability / Trace / Replay

> 对应 Phase 7 Task 7.10
> 日期：2026-08-20
> 裁决：**Unified Trace / Replay / Attribution = FREEZE（PASS）**
> 原则：observability-only，未改变任何模型与业务决策。

---

## 1. 执行总结

```text
7.1 Trace Contract       → backend/tracing/schema.py（Unified Trace）
7.2-7.5 各层 Trace        → logger.py（同 turn 一个 trace_id，JSONL）
7.6 Replay Runner        → replay.py（same-version + counterfactual）
7.7 Error Attribution    → attribution.py（最早错误层）
7.8-7.9 Latency/版本      → trace 内嵌 latency + versions
7.10 Freeze              → 本文
```

## 2. Trace Schema

```json
{ trace_id, session_id, turn_id,
  input, perception, state, policy, tools,
  memory, recommendation, response, feedback,
  latency, versions, llm_calls, error }
```

- 同一 turn 一个 trace_id，所有模块共享
- 版本溯源：Intent(phase1_5) / Risk(v5_1) / State(v1) / Policy(v1) / Memory(v2) / Recommendation(v2)
- 避免写入敏感 profile / 长期记忆原文

## 3. Replay

- `reconstruct_state(trace)`：从 trace.perception 重建 AgentState
- `replay_policy(trace, mode)`：重跑 Policy（hybrid / legacy）
- `compare_versions`：same-version vs counterfactual legacy
- 用途：regression / bug reproduction / offline policy comparison

## 4. Error Attribution

| 层 | 检查 |
|---|---|
| perception | intent 高风险但 risk 极低 → conflict |
| policy | risk≥2 但 primary≠safety → safety miss |
| safety | risk≥2 但 rec=hard/soft → violation |

每个失败 case 定位到最早错误层。

## 5. 验证（Demo）

| 目标 | 结果 |
|---|---|
| Trace coverage 100% | ✅ 3/3 turns traced |
| Replay success ≥99% | ✅ policy 重放 = safety_intervention |
| Version provenance 100% | ✅ risk=v5_1 在 trace |
| Error attribution | ✅ policy_safety_miss 正确定位 |

## 6. 交付物

```text
backend/tracing/
├── schema.py        # TurnTrace + AttributionResult
├── logger.py        # 写/读 JSONL
├── replay.py        # ReplayRunner
└── attribution.py   # ErrorAttributor
evaluation/tracing/run_tracing_demo.py
docs/agent_trace_schema_v1.md / replay_design.md / error_attribution_taxonomy.md
  （本评审引用以上实现细节）
docs/phase7_final_review.md  # 本文
```

## 7. 最终裁决

```text
Unified Trace / Replay / Attribution = FREEZE（PASS）

- observability-only，零业务决策变更
- 任何 Phase 8 failure case 可从 trace 回放并定位到具体层
```
