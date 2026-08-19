# Intent Dataset Report — v1 正式版（轻量扩充）

> 对应 Phase 1 Task 1.7 / §21-24 / §43
> 日期：2026-08-19
> 说明：完整版 Task 1.7（3000-5000 条 LLM 扩充）为 TODO；当前为轻量版（确定性场景替换扩充）。

---

## 1. 数据集组成

**文件**：`data/intent/intent_expanded_v1.jsonl`（1052 条）

| 组成 | 数量 | 来源 | 说明 |
|---|---:|---|---|
| Seed（手写模板） | 806 | template | 人工逐条撰写 |
| Variant（场景替换） | 246 | template_variant | 同域场景词替换（考试↔面试等），标签继承 |

**Source 分布**：`template=806, template_variant=246`
**Difficulty 分布**：`easy=233, medium=511, hard=308`

## 2. 标签分布（multi-label，共 1418 个标签）

| 标签 | 数量 | 占比 |
|---|---:|---:|
| information_request | 286 | 20.2% |
| emotional_expression | 234 | 16.5% |
| memory_reference | 145 | 10.2% |
| explicit_help_request | 133 | 9.4% |
| follow_up | 132 | 9.3% |
| feedback | 121 | 8.5% |
| resource_request | 117 | 8.3% |
| high_risk_expression | 89 | 6.3% |
| casual_chat | 89 | 6.3% |
| meta_question | 73 | 5.1% |

- **multi-label 占比**：33.3%
- **平均标签数/样本**：1.35
- **文本重复**：0 组

## 3. 数据划分（§19，Group Split 防泄漏）

`data/intent/intent_train_v1_exp.jsonl` / `intent_dev_v1_exp.jsonl` / `intent_test_v1_exp.jsonl`

| Split | 数量 | 分组方式 |
|---|---:|---|
| train | 508 | 模板桶级 Group Split |
| dev | 258 | 同上 |
| test | 286 | 同上 |

> ⚠️ **已知问题**：桶级 Group Split 导致 train 标签严重不均衡（emotional=16、high_risk=11、memory=5），训练出的 exp_model 指标差（test Macro 0.38）。**结论：seed 手写模板用分层切分（balanced，best_model 用），LLM 扩充数据才用 Group Split**（且需更细粒度分组）。

## 4. Leakage Audit（§24）

| 检查 | 结果 |
|---|---|
| Exact Match | 0 重复 ✅ |
| MinHash（sim≥0.8） | 0 对 ✅ |
| 变体与源模板同组 | ✅（split 时按 variant_of 归组） |
| Embedding Similarity | TODO（需 bge embedding 审计） |

## 5. 正式 Test Set（§23）

- `data/intent/intent_test_v1.jsonl`（122 条，分层切分，标签平衡）—— 模型开发用
- 正式独立 Test Set（500-800 条，不参与训练/阈值/LLM seed）—— **TODO**（Task 1.7 完整版）
- `data/intent/intent_ood_v1.jsonl`（105 条，9 域）—— OOD 评测用

## 6. 与目标对比（§22）

| 组成 | 建议 | 实际 | 状态 |
|---|---:|---:|---|
| Human/Template Seed | 15-20% | 77% | ⚠️ 偏高（轻量版未做 LLM 扩充） |
| LLM Expansion | 50-60% | 0% | ❌ TODO |
| Hard Negative | 15-20% | 9.6%（101/1052） | ⚠️ 偏低 |
| Adversarial/Ambiguous | 10-15% | ~5% | ⚠️ 偏低 |

**结论**：轻量版满足"跑通全流程"，完整版（LLM 扩充到 3000-5000、正式 Test Set、Leakage Embedding 审计）留作 Task 1.7 后续。
