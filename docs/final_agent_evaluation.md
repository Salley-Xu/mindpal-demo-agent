# Final Agent Evaluation

> 对应 Phase 8 §3
> 日期：2026-08-20
> 最终系统 = Risk v5.1 + Frozen AgentPolicy v1 + Memory 2.0 + Recommendation 2.0

---

## 1. Module-level（独立测试支撑）

| 模块 | 数据集 | 关键指标 |
|---|---|---|
| Intent | Intent Independent Test（535） | Macro F1 **0.844** / high_risk Recall 0.873 |
| Risk v5.1 | 跨源独立（792） | Macro 0.706 / **HR Recall 0.936** / **FPR 0.022** |
| Memory 2.0 | Memory Benchmark v2（23） | Gate F1 0.965 / Over-retrieval 0.0 / Conflict 1.0 |
| Recommendation 2.0 | Rec 场景集 | Trigger 1.0 / Safety violation 0 / Repeat 0 |

## 2. State / Policy（独立测试 + 冻结 benchmark）

| 指标 | 数据集 | 结果 |
|---|---|---|
| State Transition Acc | State Benchmark | **1.0** |
| Primary Action Acc | Independent Policy（oracle） | **0.8837** |
| Safety Recall | Independent Policy（oracle） | **1.0** |
| Tool Micro F1 | Independent Policy（oracle） | **0.995** |
| Policy Exact Match | Independent Policy（oracle） | **0.7767** |

## 3. End-to-end（冻结 Agent Benchmark v1.1，predicted-state）

| 指标 | Baseline | **Full System** | 提升 |
|---|---:|---:|---|
| Primary Action Acc | 0.5249 | **0.7541** | +0.23 |
| Safety Recall | 0.6119 | **0.7761** | +0.16 |
| Safety Target Acc | 0.5663 | **0.9420** | +0.38 |
| Policy Exact Match | 0.0580 | **0.4530** | +0.40 |

> Baseline = legacy policy + v4_2 risk；Full = hybrid policy + v5.1 risk（Policy 冻结未动）。

## 4. 系统级指标（Phase 8 §5）

| 指标 | 值 |
|---|---|
| Safety FPR（模块级） | 0.022 |
| Safety Recall（模块级） | 0.936 |
| Over-retrieval（Memory） | 0.0 |
| Recommendation Safety violation | 0 |
| LLM Fallback Rate | 6.1%（independent oracle） |
| Latency（Risk v5.1 决策） | ~180ms/case |

## 5. 结论

整个 Agent 相比初始版本在 Safety / Policy 决策 / Memory / Recommendation 各层均有独立数据支撑的提升。
