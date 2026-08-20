# Phase 5 Policy End-to-end Regression

> 对应 Phase 5 Task 5.8
> 日期：2026-08-20
> 链路：Risk 单轮模型 → SessionRiskAggregator → AgentState → **Frozen AgentPolicy v1** → ActionPlan

---

## 1. 设置

- 评测：Frozen Agent Benchmark v1.1（362 cases），predicted-state（ModulePredictor，risk 通道切换）
- Policy：AgentPolicy v1（FINAL FREEZE，未改动）
- Risk 通道：rule / v4_2_domain_only_v2 / v5.1_tuned

## 2. 主表（v4_2 vs v5.1）

| 指标 | rule | v4_2 | **v5.1** | 提升 |
|---|---:|---:|---:|---|
| Primary Action Acc | 0.6547 | 0.5249 | **0.7541** | +0.23 |
| Primary Macro F1 | 0.3006 | 0.2660 | **0.4216** | +0.16 |
| Safety Action Recall | 0.3134 | 0.6119 | **0.7761** | +0.16 |
| Safety Target Acc | 0.8315 | 0.5663 | **0.9420** | +0.38 |
| Policy Exact Match | 0.3591 | 0.3453 | **0.4530** | +0.10 |
| Recommendation Mode Acc | 0.7569 | 0.7597 | 0.7569 | 持平 |

## 3. 解读

1. **Risk 修复直接转化为系统收益**：Primary Acc +0.23、Safety Recall +0.16、Target Acc +0.38。
   印证路线图判断：Phase 5（Risk）是端到端主瓶颈。
2. **端到端 Safety Recall 0.776 vs 模块级 HR Recall 0.936 的差距**：
   - 模块级（单轮模型）：HR recall 0.936 / FPR 0.022
   - 端到端（全管道）：context floor 规则（discussion 降 0、第三方非求助降 0、safe_denial 降级）
     会正确降级部分"讨论/否认"case，但也在 benchmark 上漏掉少量安全 case
   - 差距 = 上下文规则 + 会话聚合的保守性（安全优先设计）
3. **Policy 本身未改动**（Frozen），收益 100% 来自 Risk v5.1。
4. 剩余端到端差距（Safety recall 0.78）可后续通过：
   - 提高单轮模型 recall（再扩 training）
   - 或放宽 context floor 规则（需权衡 FPR）

## 4. 结论

Risk v2 → Frozen Policy 联动验证通过：**Risk 修复确实转化为系统级 Safety/Routing 提升**。
