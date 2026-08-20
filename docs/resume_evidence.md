# Resume Evidence — MindPal Stateful Adaptive Agent

> 对应 Phase 8 §9
> 日期：2026-08-20
> 只放独立测试支撑、可复现、可解释、不夸大的指标。

---

## 0. 一句话

将心理支持对话系统升级为 **State → Policy → Action 的 Stateful Adaptive Agent**，
通过分层评测与 Targeted 数据驱动改进，把高风险误报率从 42% 降到 2%。

---

## 1. 独立评测体系（Benchmark Engineering）

- **问题**：Agent 系统难以证明"真的变好了"，自建评测集导致过拟合。
- **设计**：多轮冻结 Benchmark v1.1（362）→ 独立 Policy Test（430，全新措辞）→
  Safety Slice（124，10 类安全场景）→ 风险轨迹基准（14）。冻结集只做 regression，
  规则设计一律用独立集。
- **实验**：同一 benchmark 用后不再用于规则调参（design-benchmark vs independent 双轨报告）。
- **指标**：独立测试上 Policy Exact 0.78（oracle），相对 legacy 提升 +10.8×。
- **系统收益**：可复现的评测流水线，杜绝"评测集过拟合"的常见陷阱。

## 2. Hybrid Agent Policy（决策层）

- **问题**：散落的多模块决策（routing/gate/ReAct）无法解释、无法评测、无法回退。
- **设计**：显式分层 `AgentState → P0 Safety → P1 Deterministic → P3 Ambiguity → P4 LLM → ActionPlan`，
  规则全部以 legacy 审计 + benchmark gold 为依据（非拍脑袋）。
- **实验**：Ablation（Legacy vs Deterministic vs Hybrid）+ oracle-state 上限分析。
- **指标**：独立集 oracle Primary 0.884 / Safety Recall 1.0 / Tool F1 0.995 / Exact 0.777。
- **系统收益**：确定性规则全面碾压 legacy；LLM 兜底率仅 1.4%（远低于 20% 预算）。

## 3. Risk 2.0（端到端最大瓶颈修复）

- **问题**：Raw BERT 在对话域严重过度触发（FPR 42%，良性对话被判高风险）。
- **设计**：数据审计（发现 test 同源/hard negatives）→ checkpoint 对比 → fusion/calibration
  证明无法达标 → **Targeted Data Expansion + 两轮重训**（v5.1）。
- **实验**：跨源独立 792 条对话型风险标注评测；多轮 DynamicRiskState 轨迹验证。
- **指标**：**FPR 0.42 → 0.022**，High-risk Recall 0.48 → **0.936**，Macro F1 0.31 → 0.71。
- **系统收益**：端到端 Safety Target Acc 0.57 → 0.94，Primary Acc 0.52 → 0.75。
  **把最大端到端瓶颈（感知层）从根上修复，而非堆模型。**

## 4. Memory 2.0（Retrieval Governance）

- **问题**：每轮都检索记忆（over-retrieval 100%），无 conflict/dedup 生命周期。
- **设计**：Retrieval Gate（需要时才检索）+ Conflict Resolver（MERGE/SUPERSEDE）+ TTL 生命周期。
- **实验**：Memory Benchmark v2，保留 No Memory / Always Retrieve 基线。
- **指标**：检索量 **-61%**，over-retrieval 1.0 → 0.0，Gate F1 0.965，conflict 1.0。
- **系统收益**：证明"不是召回更多，而是只在必要时召回正确记忆"。

## 5. Recommendation 2.0（Feedback 闭环）

- **问题**：推荐被 Risk 污染、feedback 不进入 ranking、无 cooldown。
- **设计**：Feature Ranker（7 特征）+ Feedback 闭环（accept/reject/effective）+ Cooldown + Safety。
- **实验**：场景化 benchmark（trigger/no-rec/rejection/repeat/safety）。
- **指标**：Safety violation 0、repeat 0、neg-feedback violation 0、effective-boost 生效。
- **系统收益**：与 AgentPolicy 解耦（Phase 3 已拆分 recommendation_mode 与 primary_action）。

## 6. Observability（Phase 7）

- **问题**：决策链路不可追踪、错误不可定位。
- **设计**：统一 Trace（同 turn 一个 trace_id，版本溯源）+ Replay（counterfactual）+ Error Attribution。
- **指标**：Trace coverage 100%、replay 成功、归因到具体层。
- **系统收益**：任何 failure case 可回放并定位到最早错误层。

---

## 7. 关键数字（全部独立测试支撑）

| 指标 | Before | After |
|---|---:|---:|
| Risk FPR | 0.42 | **0.022** |
| Risk HR Recall | 0.48 | **0.936** |
| Policy Exact（oracle） | 0.11 | **0.78** |
| Memory over-retrieval | 1.0 | **0.0** |
| Safety Target Acc（e2e） | 0.57 | **0.94** |
| Primary Acc（e2e） | 0.52 | **0.75** |
