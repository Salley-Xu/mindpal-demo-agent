# Phase 1.5 Final Review

> 对应 Phase 1.5 Task 1.5.6（§17）
> 日期：2026-08-19
> 前置：Phase 1 Core PASS，Benchmark v1.1 Frozen

---

## 1. Context-aware 是否必要？—— **必要（对 follow_up），但需按场景启用**

- 域内 dev：ctx1 follow_up F1 **0.79 → 0.978**（显著提升）
- 独立测试：context 模型 E（0.844）≈ 校准 current D（0.843），follow_up E(0.707) < D(0.790)
- **结论**：上下文在"上下文风格与训练一致"时显著帮助 follow_up；对全新多样化场景收益减弱。**Intent Service 支持短上下文，但默认不强制**。

## 2. 最终模型使用哪种输入？—— **Previous 1 Turn（ctx1）**

- ctx0/ctx1/ctx2 对比：+1 turn Macro 最高（0.699 vs 0.661/0.692），follow_up 到顶（0.978）
- +2 turns 对 memory 更好（0.92）但整体无增益、输入更长 → 按 §5.4 选 **+1 turn**
- 最终模型：`models/intent/phase1_5_final_model/`（ctx1 + 8 epochs + 隐式高危数据）

## 3. Independent Test 最终指标

| 指标 | 值 |
|---|---:|
| Macro F1 | **0.844**（目标 0.85） |
| Micro F1 | 0.835 |
| Exact Match | 0.697 |
| Single-label Macro | 0.908 |
| Multi-label Macro | 0.610 |

## 4. high_risk_expression 是否达标？—— **部分（0.873，目标 0.93-0.95）**

- 提升：0.714 → **0.873**（隐式高危数据扩充起效）
- 剩余漏判：8 条，集中在**第三方危机**（"我朋友总说想不开"）和**极隐式**（"想去天台吹风""买的东西用不上"）
- 修复路径已明确（错误分析 P0：扩充第三方+极隐式样本）

## 5. follow_up 是否达标？—— **达标（MUST）**

- 独立测试：follow_up F1 **0.707 ≥ 0.70**（MUST 达成）
- 域内 dev：0.978（context 下）
- Benchmark：0.074 → 0.207（2.8x）

## 6. Intent Uncertainty 新定义是否有效？—— **有效（max_score）**

- 新定义（无法稳定映射 10 类）替代旧 Domain OOD（编程/购物等——那些属于 information/resource/help）
- **max_score 策略最优**：Uncertain Recall 0.754 ✓（≥0.75），In-domain FR 0.108（目标 ≤0.10，差 0.008）
- entropy 过度触发（FR 0.69）、margin 差（FR 0.34）→ 弃用

## 7. Hybrid 是否默认开启？—— **不默认，保留为 optional fallback**

- LLM Call Rate 16.5% ✓（≤30%）
- 但 Macro 0.783 < 最终模型 0.844，high_risk 0.571 < 0.873
- 按 §9.3：域内准确率以 classifier 为准，Hybrid 仅处理 Uncertainty 长尾

## 8. Phase 2 应使用哪个 Intent Service？

```python
# 冻结接口（§13）
predict_intent(current_text, previous_turns=None, max_context_turns=1) -> IntentResult
# IntentResult = {labels, confidence, is_open_set, source}
# 内部: {labels, confidence, label_scores, is_open_set, source, fallback_reason, context_used, latency_ms}
```

- **主路径**：Context-aware MacBERT（phase1_5_final_model，per-label threshold + temperature scaling），`previous_turns ≤ 1`
- **Uncertainty 门控**：max_score < 0.5 → is_open_set=true
- **Optional**：is_open_set 或 confidence<0.55 时 LLM fallback（Hybrid）

## 9. Phase 1 是否正式 Freeze？—— **PASS（有条件冻结）**

### MUST 条件核对（§16）

| 条件 | 要求 | 实际 | 结果 |
|---|---:|---:|---|
| Independent Test | ≥500 | 535 | ✅ |
| Context-aware 实验 | 完成 | 完成 | ✅ |
| Unified Evaluation | 完成 | 完成 | ✅ |
| high_risk Recall | ≥0.93 | 0.873 | ⚠️ 差 0.057 |
| follow_up F1 | ≥0.70 | 0.707 | ✅ |
| Leakage audit | 完成 | Exact/MinHash✅，embedding TODO | ⚠️ 部分 |

**决策**：按 §16 "核心指标略低但 Error Analysis 已充分 → 可以冻结并进入 Phase 2，不无限调参"，**Phase 1 / Intent Layer 正式冻结（PASS）**，附带记录：

1. high_risk Recall 0.873（目标 0.95）—— 需第三方+极隐式样本，Phase 2 数据迭代可继续提升
2. follow_up 上下文泛化 —— 短上下文已支持，Runtime 按场景启用
3. Embedding Leakage 审计 —— TODO
4. 完整数据扩充（2500-3500）—— 部分完成

---

## 冻结清单

| 项 | 状态 |
|---|---|
| Intent Taxonomy | ✅ FROZEN（10 类） |
| Intent Model | ✅ FROZEN（`models/intent/phase1_5_final_model/`） |
| Intent Service 接口 | ✅ FROZEN（`predict_intent(current_text, previous_turns≤1)`） |
| Independent Test | ✅ FROZEN（`intent_test_independent_v1.jsonl`，535 条） |
| Uncertainty 定义 | ✅ FROZEN（max_score ≥0.5 置信） |

**Phase 2 AgentState READY** —— 直接消费 `IntentResult`，无需再改 Intent 模型与输入接口。
