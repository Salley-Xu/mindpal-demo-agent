# Phase 0 Final Review

> 对应 Phase 0.5 Task 0.5.6（§19）
> 日期：2026-08-18
> 范围：Phase 0 + Phase 0.5（Schema 修正、指标口径修正、Risk Ablation、Benchmark 扩充、Baseline 重跑）

---

## 1. Benchmark 是否可以冻结？—— **PASS** ✅

### 验收条件逐项核对（§23 + §19.1）

| 条件 | 要求 | 实际 | 结果 |
|---|---|---:|---:|---|
| Schema v1.1 语义稳定 | 语义冻结 | PrimaryAction/ToolAction/SafetyTarget 拆分，迁移 helper 验证一致 | ✅ |
| Case 数量 | ≥ 250 | **362** | ✅ |
| L2+L3 | ≥ 50 | **67** | ✅ |
| 每个 Intent 有覆盖 | 10 类 | 10/10 | ✅ |
| 每个 Primary Action 有覆盖 | 4 类 | 4/4 | ✅ |
| Schema 校验 | 全通过 | 0 errors | ✅ |
| 重复污染 | 无严重重复 | Exact=0, MinHash=0 | ✅ |

### 软目标分布（建议，已记录偏差）

| 分布 | 建议 | 实际 | 状态 |
|---|---:|---:|---|
| L0 | 130–150 | 214 | ⚠️ 偏高（intent 长尾覆盖优先） |
| L1 | 70–80 | 81 | ✅ |
| L2 / L3 | 30–35 / 30–35 | 31 / 36 | ✅ |
| ask_clarification | ≥ 25 | 21 | ⚠️ 略低（软目标） |
| retrieve_memory / retrieve_knowledge / recommend_resource | 40 / 30 / 50 | 40 / 30 / 50 | ✅ |

**结论**：硬性验收全部通过，软目标 2 处轻微偏差已记录。**Benchmark v1.1 冻结**，Phase 1–7 作为日常 regression 基准，Phase 8 再升级到 v2（500–800 条）。

---

## 2. Risk 问题真正在哪一层？—— **Raw BERT 模型本身（情况 B）**

### 分层 Ablation（362 条 v1.1）

| Variant | Macro F1 | High-risk Recall | FPR | FNR |
|---|---:|---:|---:|---:|
| A: Raw BERT | 0.364 | 0.627 | **0.431** | **0.373** |
| B: +Rule Override | 0.364 | 0.627 | 0.431 | 0.373 |
| C: +Context Rules | 0.373 | 0.612 | 0.420 | 0.388 |
| D: Full Pipeline | 0.371 | 0.612 | 0.437 | 0.388 |
| E: Rule-only | 0.376 | 0.313 | **0.041** | 0.687 |

### 逐层判定

| 层 | 贡献 | 判定 |
|---|---|---|
| **Raw BERT** | 本身 FPR=0.43 + FNR=0.37，双高；`"你在吗？"` → L3 @ 0.9985 置信度 | ❌ **主要问题层** |
| Rule Override | 对 benchmark 几乎无影响（B==A，20 条高危 pattern 覆盖不了隐式表达） | 次要 |
| Context Rules | 微改善（Macro F1 0.364→0.373），discussion/第三方降级有效但覆盖面小 | 次要 |
| Session Aggregator | 几乎无影响（C≈D） | 次要 |
| Engineering Mapping | **无 Bug**：label 映射、tokenizer、架构均与训练一致；但文档化的 P1 融合规则 `binary_prob>0.5 AND 4分类<2 → level_2` **未实现** | ⚠️ 文档-实现不一致 |

### 结论（决策规则 §11 → **情况 B**）

**Raw BERT 本身 FPR 高 + FN 高，模型需要重新校准 / 重训 / 切 checkpoint。**

- 安全底线要求 High-risk Recall ≥ 0.98；当前 BERT 0.63 / 规则 0.31，**均远不达标**。
- 规则基线精度优秀（FPR=0.04）但召回灾难（FNR=0.69）——隐式表达是其主要盲区。
- Phase 5（Risk 2.0）方向：
  1. 评估切换 `v4_3_coral` 或重新校准 v4.2（最便宜路径）；
  2. 扩充规则兜底覆盖隐式表达（如"不想醒来/撑不下去/一了百了"）；
  3. 实现文档化的 binary 融合（修复文档-实现不一致）。

---

## 3. Phase 1 IntentResult 接口（冻结）

```python
IntentResult = {
    "labels": List[str],          # Intent Taxonomy v1 的 multi-label 子集
    "confidence": float,          # 0-1（多标签时为最低正类置信度）
    "is_open_set": bool,          # True = 落在已见标签外（应走 LLM fallback）
    "source": str,                # "classifier" | "llm" | "rule"
}
```

- **消费方**：Phase 2 AgentState 的 `intent` / `intent_confidence` 字段
- **基准**：Legacy Intent Coverage（当前 user_intent 规则 Macro F1=0.16，6/10 标签零召回）作为对照基线
- **目标**：Macro F1 ≥ 0.85（开发计划 §6.6）

## 4. Phase 3 ActionPlan 接口（冻结）

```python
ActionPlan = {
    "primary_action": str,        # PrimaryAction: continue_chat | ask_clarification | information_response | safety_intervention
    "tool_actions": List[str],    # ToolAction: [retrieve_memory, retrieve_knowledge, recommend_resource] 任意子集
    "safety_target": str,         # SafetyTarget: none | self | third_party
    "recommendation_mode": str,   # RecommendationAction: hard | soft | none | safety_only
}
```

- **设计依据**：`expected.primary_action` / `expected.tool_actions` / `expected.safety_target` / `expected.recommendation_action`（Benchmark v1.1 同构）
- **当前状态**：`ModulePredictor._derive_primary_action / _derive_tool_actions / _derive_safety_target` 已作为确定性信号推导的参考实现
- **验收**：Phase 3 Policy 的 Action Accuracy ≥ 90%，Safety Action Recall ≥ 98%（开发计划 §8.8）——当前 BERT baseline 为 0.53 / 0.63，差距显著

---

## 5. v1.1 Baseline 主表（冻结值）

| 模块 | 指标 | Current (BERT) | Rule |
|---|---:|---:|---:|
| Risk | Macro F1 | 0.371 | 0.399 |
| Risk | High-risk Recall / Precision | 0.612 / 0.241 | 0.313 / 0.636 |
| Risk | FPR / FNR | 0.437 / 0.388 | 0.041 / 0.687 |
| Emotion | Coarse Acc / Macro F1 | 0.597 / 0.475 | 0.483 / 0.332 |
| Legacy Intent | Coverage Macro F1 | 0.161 | 0.143 |
| Recommendation | Trigger P/R / Mode Acc | 0.172 / 0.329 / 0.282 | 0.333 / 0.408 / 0.611 |
| Primary Action | Accuracy / Safety Recall | 0.528 / 0.627 | 0.655 / 0.313 |
| Tool Action | Micro F1 / Exact Match | 0.161 / 0.354 | 0.178 / 0.193 |
| Memory Behavior | Precision / Recall / Over-rate | 0.147 / 0.609 / 0.853 | 0.128 / 0.913 / 0.872 |
| Risk Trend | Acc（n=23） | 0.304 | 0.000 |

---

## 6. 对 Phase 1 的启动指令

Benchmark 已冻结（PASS），Risk 根因已定位（Raw BERT）。按开发计划 §24：

```text
Phase 1：Intent Recognition
  顺序：Intent Dataset → LLM-only Baseline → Small Model Baseline
        → Confidence Calibration → LLM Fallback → Hybrid Intent
        → Agent Benchmark v1.1 Regression
```

- Phase 1 不负责修复 Risk（Risk 进入 Phase 5）。
- 每次 Phase 改动必须跑 `run_agent_eval.py --cases agent_benchmark_v1_1.jsonl` 回归。
