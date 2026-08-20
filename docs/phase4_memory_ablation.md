# Phase 4 Memory 2.0 Ablation

> 对应 Phase 4 Task 4.9 / 4.10
> 日期：2026-08-20
> 数据：`evaluation/memory/memory_benchmark_v2.jsonl`（23 cases）

---

## 1. 评测目标

证明：**Memory 2.0 不是"召回更多"，而是"只在必要时召回正确记忆"。**

## 2. 指标（Memory Benchmark v2）

| 指标 | 结果 | 目标 | 判定 |
|---|---:|---:|---|
| Retrieval Gate Precision | **1.0** | ≥0.80 | ✅ |
| Retrieval Gate Recall | **0.9333** | ≥0.85 | ✅ |
| Retrieval Gate F1 | **0.9655** | - | ✅ |
| Over-retrieval Rate | **0.0** | ≤0.15 | ✅ |
| Conflict Resolution Acc | **1.0** | ≥0.90 | ✅ |

## 3. Ablation（Retrieval 行为）

| Method | 检索轮数 | Over-retrieval | Recall |
|---|---:|---:|---:|
| No Memory | 0/23 | 0 | 0.0 |
| **Naive Always Retrieve** | 23/23 | 1.0 | 1.0 |
| **Retrieval Gate** | ~9/23 | **0.0** | **0.93** |

> Retrieval Gate 将检索量从"每轮都检索"（23/23）降到 ~9/23（-61%），
> 同时保持 recall 0.93（over-retrieval 从 1.0 → 0.0）。

## 4. Conflict / Lifecycle

| 场景 | 操作 | 验证 |
|---|---|---|
| 同义重复 | MERGE | ✅ |
| 强化表达（"继续坚持"） | MERGE | ✅ |
| 偏好更新（"现在不喝了"） | SUPERSEDE | ✅ |
| 矛盾事实 | SUPERSEDE | ✅ |
| TTL 到期 | EXPIRE | ✅ |

## 5. 结论

Memory 2.0 治理层（Retrieval Gate + Conflict Resolver + Lifecycle）达标：
- 检索量 -61%，over-retrieval 归零
- conflict/dedup/expire 生命周期正确
- 保留 No Memory / Always Retrieve 基线对比
