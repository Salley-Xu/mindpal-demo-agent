# Phase 3 Final Review — Hybrid Agent Policy v1

> 对应 Phase 3 执行文档 §28
> 日期：2026-08-20
> 裁决：**AgentPolicy v1 = FREEZE（PASS）**

---

## 1. Final Review 必答（§28 14 问）

### 1. Final Policy 架构是什么？

```text
AgentState v1
  → P0 Safety Invariants（safety.py，命中即返回）
  → P1 Deterministic Rules（deterministic.py：primary / tools / rec）
  → P3 Ambiguity Gate（ambiguity.py：规则冲突 / 低置信 / open-set）
  → P4 LLM Fallback（llm_fallback.py，仅 ambiguous，永远不能覆盖 Safety）
  → ActionPlan v1（validator.py 校验 INV-01~11）
```

### 2. Deterministic Policy 单独能达到多少？

**全部达标**：Primary Acc 0.9475，Primary Macro F1 0.8742，Safety Recall 1.0，
Safety Target Acc 1.0，Tool Micro F1 0.8679，Rec Mode Acc 0.8867，Exact 0.7818（oracle-state）。

### 3. Learned Policy 是否真的需要？

**不需要。** Deterministic 单独满足推荐目标（Primary Macro F1 0.874 ≥ 0.85，Safety 达标），
按 §29 跳过，避免为"看起来高级"强行加复杂度。

### 4. LLM Fallback 是否真的需要？

**保留但非必需。** Benchmark 上 ambiguity gate 触发率仅 1.4%（5/362），远低于 20% 上限。
保留作为生产 ambiguous case（rec 粒度、info+resource 冲突）的兜底，不参与达标判定。

### 5. Safety Recall / Safety Target Accuracy 是多少？

- Safety Recall = **1.0**（oracle，67/67）｜目标 ≥0.98 ✅
- Safety Target Acc = **1.0**（oracle）｜目标 ≥0.95 ✅
- Safety 类错误（P05/P06/P07）= **0**

### 6. Primary Action Accuracy 是多少？

**0.9475**（oracle）｜目标 ≥0.90 ✅
（Predicted-state 由感知瓶颈限制，见 Q11）

### 7. Tool Action F1 是多少？

**Micro F1 = 0.8679**，Macro F1 = 0.8694，Exact = 0.9088｜目标 ≥0.80 ✅

### 8. Recommendation Mode Accuracy 是多少？

**0.8867**（oracle）｜目标 ≥0.80 ✅
（剩余误差集中在 soft 粒度，已定位，见 error_analysis §2.1）

### 9. Policy Exact Match 是多少？

**0.7818**（oracle）｜目标 ≥0.75 ✅

### 10. LLM Fallback Rate / Latency / Cost 是多少？

- LLM rate：**1.4%**（oracle ambiguity）｜目标 ≤20% ✅
- Latency：Deterministic 决策 0.05ms；生产 LLM 兜底仅在有真实 ambiguous case 时触发
- Cost：Deterministic 0 LLM 调用；LLM 兜底可忽略（1.4%）

### 11. Oracle-state vs Predicted-state 差距是多少？

| 指标 | Oracle | Predicted(rule) | Predicted(bert) | 差距来源 |
|---|---:|---:|---:|---|
| Primary Acc | 0.9475 | 0.6547 | 0.5249 | 感知（P16） |
| Safety Recall | 1.0 | 0.3134 | 0.6119 | Risk 感知（P16） |

**Oracle-state 下 Policy 很高 → 瓶颈主要在 perception（Raw BERT FNR/FPR + 粗规则通道），
与 Phase 0.5 定位一致（Phase 5 修复）。** Policy 本身在给定信号下已做最优路由。

### 12. 是否可以替换 Legacy Policy？

**可以（Shadow 阶段先行）。** 原因：
- Deterministic 全部指标碾压 Legacy（oracle：Primary 0.95 vs 0.80，Tool 0.87 vs 0.20，Exact 0.78 vs 0.11）
- 不改变感知模块（Intent/Emotion/Risk 未动）
- Shadow 记录差异，无行为回归（State Benchmark 1.0、Intent 0.844 未变）

### 13. Rollback 是否可用？

**可用。** `PolicyConfig.mode: legacy | shadow | hybrid` 三态切换：
- 生产保持 `shadow`（New Policy 旁路记录）
- 任何异常 `hybrid → legacy` 一键回退
- Shadow 全程 try/except，绝不干扰生产

### 14. AgentPolicy v1 是否 Freeze？

**YES。** 契约（schema）已冻结，Deterministic 达标，Shadow 集成完成。

---

## 2. 指标汇总（对 §19 目标）

| 指标 | 目标 | Oracle-state 实测 | 判定 |
|---|---:|---:|---|
| Primary Action Accuracy | ≥0.90 | 0.9475 | ✅ |
| Primary Macro F1 | ≥0.85 | 0.8742 | ✅ |
| Safety Action Recall | ≥0.98 | 1.0 | ✅ |
| Safety Target Accuracy | ≥0.95 | 1.0 | ✅ |
| Tool Action Micro F1 | ≥0.80 | 0.8679 | ✅ |
| Rec Mode Accuracy | ≥0.80 | 0.8867 | ✅ |
| Policy Exact Match | ≥0.75 | 0.7818 | ✅ |
| LLM Fallback Rate | ≤0.20 | 0.014 | ✅ |

**最低 PASS 全部满足。** Safety Invariants = 100%（Validator 全绿，单元测试 34/34）。

---

## 3. 交付物

```text
backend/policy/
├── schema.py                  # 契约（Preflight 冻结）
├── engine.py                  # Policy Core + 配置
├── safety.py                  # P0 Safety（S01/S02/S03 + INV-11）
├── deterministic.py           # P1 Primary/Tool/Rec 规则
├── ambiguity.py               # P3 Ambiguity Gate
├── classifier.py              # P2 Learned（占位，禁用）
├── llm_fallback.py            # P4 LLM Fallback
├── validator.py               # INV-01~11 校验
├── hybrid.py                  # 分层编排
├── shadow.py                  # Policy Shadow
└── legacy_policy_adapter.py   # Legacy 审计适配（Preflight）

evaluation/policy/
├── run_policy_eval.py         # single-cell eval（predicted/oracle）
├── run_ablation.py            # 全 ablation（3 方法 × 3 状态）
├── policy_edge_cases_v1.jsonl # 11 edge cases
└── reports/policy_ablation_*.json

evaluation/tests/test_policy.py # 34 个单元测试

docs/
├── phase3_policy_design.md
├── phase3_policy_ablation.md
├── phase3_policy_error_analysis.md
└── phase3_final_review.md     # 本文
```

---

## 4. 最终裁决

```text
AgentPolicy v1 = FREEZE（PASS）

推荐后续：
- 生产保持 shadow 模式观察（logs/policy_shadow_trace.jsonl）
- 端到端指标提升依赖 Phase 5（Risk 感知修复），Policy 无需改动
- Phase 6 重构 Recommendation 时，rec 粒度（soft）由 LLM 兜底吸收
```

---

## 5. 系统状态（对齐 §31）

```text
Perception（Intent 0.844 / Emotion / Risk BERT）
    ↓
AgentState v1
    ↓
AgentPolicy v1  ← 本阶段冻结
    ↓
ActionPlan v1
├── Primary Action      0.9475
├── Tool Actions        0.8679
├── Safety Target       1.0
└── Recommendation Mode 0.8867
    ↓
Execution Layer（Legacy，shadow 阶段）
```
