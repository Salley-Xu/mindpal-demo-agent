# MindPal Agent 技术文档索引

> 项目：MindPal Pro — AI 心理支持对话系统（Stateful Adaptive Agent 升级）
> 目标：从"多模块功能集合"升级为"可决策、可评测、可解释、可迭代"的 Agent 系统
> 文档按**技术故事线**组织，见 §7 总览。

---

## 1. 项目计划（根目录 `MindPal_Agent_*.md`）

| 文档 | 说明 |
|---|---|
| [MindPal_Agent_开发任务计划_v1.0.md](../MindPal_Agent_开发任务计划_v1.0.md) | **总计划**：9 个 Phase、六周排期、目标架构、验收标准 |
| [MindPal_Agent_Phase0.5_执行文档.md](../MindPal_Agent_Phase0.5_执行文档.md) | Phase 0.5：Benchmark 冻结前校正 |
| [MindPal_Agent_Phase1_Intent_Recognition_执行文档.md](../MindPal_Agent_Phase1_Intent_Recognition_执行文档.md) | Phase 1：Intent Recognition |
| [MindPal_Agent_Phase1.5_执行计划.md](../MindPal_Agent_Phase1.5_执行计划.md) | Phase 1.5：Intent 收尾冻结 |

## 2. 架构与现状审计

| 文档 | 说明 |
|---|---|
| [current_agent_architecture.md](current_agent_architecture.md) | 当前系统架构审计（模块清单、决策链、信号映射） |
| [current_agent_dataflow.md](current_agent_dataflow.md) | 当前系统数据流审计（单轮调用链、持久化、LLM 调用点） |

## 3. 评测体系（Evaluation Foundation）

| 文档 | 说明 |
|---|---|
| [benchmark_schema_v1_1.md](benchmark_schema_v1_1.md) | **Benchmark Schema v1.1（冻结）**：Action 拆分 + 迁移 helper |
| [benchmark_v1_1_data_report.md](benchmark_v1_1_data_report.md) | Benchmark v1.1 数据审计（362 条，查重归零） |
| [baseline_report.md](baseline_report.md) | 系统级 Baseline 报告（含 v1.1 更新主表） |
| [risk_pipeline_audit.md](risk_pipeline_audit.md) | **Risk 根因审计**：Ablation 定位 = Raw BERT 本身 |

## 4. Phase 评审（决策结论）

| 文档 | 结论 |
|---|---|
| [phase0_final_review.md](phase0_final_review.md) | Benchmark v1.1 **冻结 PASS**；Risk 根因 = Raw BERT（情况 B） |
| [phase1_5_final_review.md](phase1_5_final_review.md) | **Intent Layer 正式冻结 PASS（有条件）**（含 Phase 1 汇总） |
| [phase2_final_review.md](phase2_final_review.md) | **AgentState v1 冻结 PASS**（Transition Acc 1.0） |
| [phase3_final_review.md](phase3_final_review.md) | **AgentPolicy v1 冻结 PASS**（Deterministic 全指标达标，Shadow 集成） |
| [phase3_final_independent_review.md](phase3_final_independent_review.md) | **AgentPolicy v1 FINAL FREEZE PASS**（独立测试集泛化验证） |
| [phase5_final_review.md](phase5_final_review.md) | **RiskResult v2 FREEZE PASS**（FPR 0.42→0.02，HR Recall→0.94） |
| [phase4_final_review.md](phase4_final_review.md) | **Memory 2.0 FREEZE PASS**（Over-retrieval 1.0→0.0） |
| [phase6_final_review.md](phase6_final_review.md) | **Recommendation 2.0 FREEZE PASS**（Feedback 闭环 + Cooldown） |
| [phase7_final_review.md](phase7_final_review.md) | **Trace/Replay/Attribution FREEZE PASS** |
| [phase8_final_review.md](phase8_final_review.md) | **全系统 PASS（可交付）** |

## 4.6 Agent Policy 模块（Phase 3）

| 文档 | 说明 |
|---|---|
| [agent_policy_contract_v1.md](agent_policy_contract_v1.md) | **Policy Contract v1（冻结）**：AgentState → PolicyResult → ActionPlan |
| [legacy_policy_audit.md](legacy_policy_audit.md) | **Legacy 审计**：12 条规则 → ActionPlan 映射 |
| [policy_invariants_v1.md](policy_invariants_v1.md) | **Policy Invariants v1**：INV-01~11 + Safety precedence |
| [phase3_policy_design.md](phase3_policy_design.md) | Phase 3 设计：P0 Safety → P1 Deterministic → P3 Ambiguity → P4 LLM |
| [phase3_policy_ablation.md](phase3_policy_ablation.md) | **Ablation**：Legacy vs Deterministic vs Hybrid（oracle + predicted） |
| [phase3_policy_error_analysis.md](phase3_policy_error_analysis.md) | **错误分析**：Policy-intrinsic vs Upstream-caused 归因 |
| [phase3_safety_contract_reconciliation.md](phase3_safety_contract_reconciliation.md) | **合同对齐**：high_risk_intent fallback（S04，方案 A） |
| [phase3_independent_policy_eval.md](phase3_independent_policy_eval.md) | **独立评测**：430 cases 新措辞，全指标 PASS |
| [phase3_independent_safety_eval.md](phase3_independent_safety_eval.md) | **独立 Safety Slice**：124 cases，Recall 1.0 / FPR 0 |

## 4.7 Risk 模块（Phase 5）

| 文档 | 说明 |
|---|---|
| [risk_dataset_audit_v2.md](risk_dataset_audit_v2.md) | **Risk 数据审计**：test 同源 / hard negatives / 覆盖缺口 |
| [phase5_checkpoint_ablation.md](phase5_checkpoint_ablation.md) | **Checkpoint 对比**：v4_2/coral/ft/rule（全高 FPR） |
| [phase5_fusion_calibration.md](phase5_fusion_calibration.md) | **Fusion/Calibration**：既有模型无法达标 → 重训 |
| [phase5_multiturn_risk_eval.md](phase5_multiturn_risk_eval.md) | **DynamicRiskState + 轨迹评测**（全 PASS） |
| [phase5_policy_end_to_end.md](phase5_policy_end_to_end.md) | **Policy 联动**：Risk v5.1 → Frozen Policy 系统收益 |
| [phase5_final_review.md](phase5_final_review.md) | **RiskResult v2 FREEZE PASS** |

## 4.5 AgentState 模块（Phase 2）

| 文档 | 说明 |
|---|---|
| [agent_state_source_audit.md](agent_state_source_audit.md) | 状态来源审计（33 字段 + 语义冲突） |
| [agent_state_schema_v1.md](agent_state_schema_v1.md) | **AgentState Schema v1**（12 子状态，State≠Action） |
| [agent_state_update_rules.md](agent_state_update_rules.md) | 字段级更新规则（REPLACE/ACCUMULATE/ROLLING_WINDOW） |
| [phase2_state_evaluation.md](phase2_state_evaluation.md) | State 一致性评测（Transition Acc 1.0 / Session 100%） |

## 5. Intent 模块（Phase 1 / 1.5）

### 标注与数据
| 文档 | 说明 |
|---|---|
| [intent_annotation_guideline.md](intent_annotation_guideline.md) | **标注指南**：10 类 + 7 组混淆对 + 边界规则 |
| [phase1_5_dataset_report.md](phase1_5_dataset_report.md) | 数据集报告（seed 951 条 + 独立测试 535 + Uncertainty 65） |

### 实验与分析
| 文档 | 说明 |
|---|---|
| [phase1_5_context_ablation.md](phase1_5_context_ablation.md) | **Context-aware Ablation**（+1 turn 最优） |
| [phase1_5_unified_evaluation.md](phase1_5_unified_evaluation.md) | **统一评测**：独立测试 535 条上 6 方法对比 |
| [phase1_5_error_analysis.md](phase1_5_error_analysis.md) | 错误分析（C03/C04/C05） |

## 6. 执行进度

| 文档 | 说明 |
|---|---|
| [execution_progress.md](execution_progress.md) | **五要素进度日志**：每阶段总结（任务/完成/卡点/计划/踩坑） |

## 7. 技术故事线（总览）

```text
模型理解用户           → Intent/Emotion/Risk 感知层（Phase 1/5 完成，Risk v5.1 冻结）
State 表示用户状态      → AgentState v1（Phase 2，冻结）
Policy 决定动作        → AgentPolicy v1（Phase 3，FINAL FREEZE）
Memory 提供长期上下文   → Memory 2.0（Phase 4，冻结）
Recommendation/Safety  → Recommendation 2.0（Phase 6，冻结）
Observability          → Trace/Replay/Attribution（Phase 7，冻结）
Evaluation 判断是否变好 → 多套独立 Benchmark + 评测框架（完成）
```

**核心成果**（截至 2026-08-19）：
- Benchmark v1.1 冻结（362 条），Risk 根因定位（Raw BERT）
- Intent Layer 冻结：Macro 0.844 / high_risk Recall 0.873（独立测试）
- Benchmark Intent Macro：0.138 → **0.757**

## 8. 既有设计文档（Legacy，Phase 0 前）

| 文档 | 说明 |
|---|---|
| EVALUATION_DESIGN.md | 评测设计（旧） |
| 技术文档-长期记忆模块现状.md | 长期记忆现状 |
| 长期记忆模块现状与问题分析.md | 长期记忆问题分析 |
| 长期记忆模块v2.0重构设计方案.md | 记忆 v2.0 设计 |
| 推荐模块技术文档.md / 推荐模块迭代计划.md | 推荐模块 |
| 风险安全模块技术文档.md | 风险安全 |
| 压力管理Agent_情绪识别模块系统设计.md | 情绪识别设计 |
| 压力管理Agent风险安全模块第二三阶段设计文档.md | 风险安全阶段设计 |
| FAISS 向量检索接入方案.md | FAISS 检索方案 |
