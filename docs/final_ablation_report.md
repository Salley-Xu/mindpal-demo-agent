# Final Ablation Report

> 对应 Phase 8 §4
> 日期：2026-08-20
> 冻结评测：Agent Benchmark v1.1（362，predicted-state）+ Independent Policy Test（430，oracle-state）

---

## 1. 系统级增量 Ablation（端到端，冻结 benchmark predicted）

| 版本 | Primary Acc | Safety Recall | Safety Target | Exact |
|---|---:|---:|---:|---:|
| Baseline Initial System | 0.5249 | 0.6119 | 0.5663 | 0.0580 |
| + AgentPolicy（deterministic） | 0.5249 | 0.6119 | 0.5663 | **0.3453** |
| + Risk 2.0（v5.1） | **0.7541** | **0.7761** | **0.9420** | **0.4530** |

> 解读：
> - **AgentPolicy**：端到端 primary/safety 受感知限制不变，但 **Exact Match 0.058→0.345**（决策质量提升）
> - **Risk 2.0**：Primary +0.23 / Safety Target +0.38 —— **最大系统收益来自 Risk 修复**

## 2. Policy Ablation（Independent Policy Test，oracle-state = 决策上限）

| Method | Primary Acc | Tool Micro F1 | Rec Mode Acc | Exact |
|---|---:|---:|---:|---:|
| Legacy | 0.6558 | 0.3468 | 0.6721 | 0.0721 |
| Deterministic | **0.8837** | **0.995** | **0.9000** | **0.7767** |
| Hybrid（+LLM 兜底） | 0.8837 | 0.995 | 0.9000 | 0.7767 |

## 3. Risk Ablation（跨源独立 792）

| Method | Macro F1 | HR Recall | FPR |
|---|---:|---:|---:|
| rule_only | 0.3548 | 0.2857 | 0.0210 |
| v4_2 | 0.3141 | 0.4762 | 0.4204 |
| v5.1 | **0.7060** | **0.9365** | **0.0225** |

## 4. Memory Ablation（Memory Benchmark v2）

| Method | Over-retrieval | Recall |
|---|---:|---:|
| No Memory | 0 | 0.0 |
| Always Retrieve | 1.0 | 1.0 |
| Retrieval Gate | **0.0** | **0.9333** |

## 5. Recommendation Ablation（Phase 6）

| Method | Safety Violation | Repeat | Neg-feedback Violation |
|---|---:|---:|---:|
| 旧 Gate | - | 高 | 高 |
| **Feature Ranker + Feedback + Cooldown** | **0** | **0** | **0** |

## 6. 结论

- 每层改进都有独立 benchmark 支撑（未使用同一 benchmark 既设计又评测）
- Risk 2.0 是端到端最大杠杆；Policy/Memory/Recommendation 各自专项达标
