# Memory Retrieval Gate

> 对应 Phase 4 Task 4.5（最高优先级）
> 日期：2026-08-20
> 实现：`backend/memory_v2/retrieval_gate.py`

---

## 1. 目标

> 从"每轮都检索"变为"需要时才检索"。

## 2. 输入 / 输出

```text
输入：AgentState.intent / memory_reference signal / personalization need / risk / turn
输出：RetrievalDecision { retrieve_memory, retrieval_scope, memory_types, reason }
```

## 3. 决策规则

| 优先级 | 条件 | 决策 |
|---|---|---|
| 1 | intent 含 memory_reference | 检索 long_term（preference/event/coping_strategy） |
| 2 | memory.explicit_memory_reference flag | 检索 long_term |
| 3 | resource/help/follow-up + 非首轮 | 检索 long_term（personalization） |
| 3b | 偏好变化表达（"以前...现在不"） | 检索 preference（conflict 检测） |
| 4 | risk>=2 + 有记忆 | 检索 risk 上下文 |
| 5 | 否则 | 不检索（no_memory_needed） |

## 4. 设计依据（Frozen Benchmark gold）

- memory_reference → 必然检索（40/46）
- information_request 是知识查询 → **不触发**（非个性化）
- casual 首轮 → 不检索

## 5. 结果

| 指标 | 结果 | 目标 |
|---|---:|---:|
| Gate Precision | 1.0 | ≥0.80 ✅ |
| Gate Recall | 0.9333 | ≥0.85 ✅ |
| Over-retrieval | 0.0 | ≤0.15 ✅ |
| 检索量（vs Always Retrieve） | **-61%** | 显著下降 ✅ |
