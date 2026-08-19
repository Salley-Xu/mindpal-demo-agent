# MindPal Agent 后续主路线图（Phase 3 Closeout → Phase 8）

> 版本：v1.0  
> 日期：2026-08-20  
> 执行顺序依据：以当前系统真实瓶颈优先级排序，而非机械按 Phase 编号执行。

## 1. 当前状态

```text
Phase 0 / 0.5   PASS
Phase 1 / 1.5   PASS
Phase 2         PASS
Phase 3 Core    PASS
```

已冻结：IntentResult、AgentState v1、ActionPlan v1、Policy Contract / Invariants、AgentPolicy 核心架构。

当前关键事实：
1. Phase 3 deterministic policy 的规则设计参考了 Frozen Benchmark v1.1 的 gold 分布，因此当前 oracle 指标属于 design-benchmark performance，不应直接当最终独立指标。
2. Oracle-state 与 predicted-state 差距很大，Primary/Safety 当前主要端到端瓶颈来自 Risk perception。
3. Phase 0.5 已定位 Raw BERT 同时存在高 FPR 与高 FNR。
4. Memory 当前最大结构问题是每轮都检索，缺真正 Retrieval Gate、冲突消解和生命周期治理。
5. Recommendation 受 Risk 污染，必须在 Risk 修复后再重构。
6. 当前缺统一 Trace / Replay / Error Attribution。

## 2. 推荐执行顺序

```text
Phase 3 Closeout
→ Phase 5 Risk 2.0
→ Phase 4 Memory 2.0
→ Phase 6 Recommendation 2.0
→ Phase 7 Observability
→ Phase 8 Final Evaluation
```

## 3. 为什么 Phase 5 在 Phase 4 前

当前 Risk：
```text
Raw BERT Macro F1 ≈ 0.364
High-risk Recall ≈ 0.627
FPR ≈ 0.431
FNR ≈ 0.373
```

而 Phase 3 oracle-state Primary Acc 0.9475，predicted-state 只有约 0.52–0.65；Safety Recall 从 oracle 1.0 降到 0.31–0.61。

所以当前主瓶颈不是 Policy，而是 Risk 信号质量。Risk 同时污染 Safety routing、Recommendation gate、Agent policy 与 risk trend。

## 4. 每阶段核心输出

| 阶段 | 核心输出 |
|---|---|
| Phase 3 Closeout | Independent Policy Test + Safety Slice + Final Policy Metrics |
| Phase 5 | RiskResult v2 + DynamicRiskState + calibrated/fused risk pipeline |
| Phase 4 | Memory 2.0 lifecycle + retrieval gate + conflict resolution |
| Phase 6 | Recommendation Policy + feedback-aware ranking |
| Phase 7 | Unified Agent Trace + Replay + Attribution |
| Phase 8 | Final independent Agent Benchmark + Ablation + Resume metrics |

## 5. 总体原则

- Frozen benchmark 只做 regression，不再拿来设计规则。
- 每个阶段都区分 module-intrinsic 与 upstream-caused error。
- 不为了“高级”强行加入 learned model / LLM。
- 所有新增能力先 offline/shadow eval，再 cutover。
- 每阶段都必须输出 Final Review 和 Freeze 判断。
