# Risk Pipeline Audit

> 日期：2026-08-18 23:36:55
> 对象：`bert_data\models\v4_2_domain_only_v2\best_model`（MultiTaskBERT(v4.2)）

## 1. 工程配置审计

| 项目 | 值 | 判定 |
|---|---|---|
| checkpoint path | `bert_data\models\v4_2_domain_only_v2\best_model` | 存在 ✅ |
| tokenizer path | `bert_data\models\v4_2_domain_only_v2\best_model` | 存在 ✅ |
| 模型架构 | MultiTaskBERT(v4.2) | 自动检测 ✅ |
| 4 分类头输出 | 4 | 对齐 4 级 ✅ |
| 推理 label 映射 | {0: level_0, 1: level_1, 2: level_2, 3: level_3} | 与训练 `cssrs_lite_level 0-3` 一致 ✅ |
| 训练 label 映射 | cssrs_lite_level 0-3（SOS-1K 0-10 映射，见技术报告 §2.3） | 技术报告 §2.3 一致 ✅ |
| max_length / truncation | 128 / True | 标准 ✅ |
| softmax dim | -1 | 标准 ✅ |
| binary_threshold 融合（P1） | 存储但未在 predict() 使用 | **文档-实现不一致 ⚠️** |

**结论：未发现 label 映射 / tokenizer / 架构 层面的工程 Bug。** 文档化的 `binary_prob>0.5 AND 4分类<2 → level_2` 三级融合规则在 `bert_risk_predictor.predict()` 中**未实现**（参数存储未使用）。

## 2. 分层 Ablation 指标

| Variant | Macro F1 | High-risk Recall | FPR | FNR |
|---|---:|---:|---:|---:|
| A: Raw BERT | 0.3639 | 0.6269 | 0.4305 | 0.3731 |
| B: +Rule Override | 0.3639 | 0.6269 | 0.4305 | 0.3731 |
| C: +Context Rules | 0.3731 | 0.6119 | 0.4203 | 0.3881 |
| D: Full Pipeline | 0.371 | 0.6119 | 0.4373 | 0.3881 |
| E: Rule-only | 0.3756 | 0.3134 | 0.0407 | 0.6866 |

## 3. 明显错误样本审计

**输入**：`你在吗？我就随便聊聊。`

```json
{
  "raw_label": 3,
  "raw_confidence": 0.9985,
  "probs_4": [
    0.0005,
    0.0003,
    0.0007,
    0.9985
  ],
  "binary_prob": 0.9979
}
```

Raw BERT 输出：**Level 3 @ 高置信度**。

## 4. 根因结论（决策规则 §11）

### 情况 B：Raw BERT 本身 FPR 高 + FN 高
- 模型本身需要重新校准 / 重训 / 切 checkpoint，Phase 5 进入模型重构。

（详细量化依据见下方数据）

```json
{
  "raw_bert": {
    "n": 362,
    "accuracy": 0.384,
    "macro_precision": 0.4524,
    "macro_recall": 0.4305,
    "macro_f1": 0.3639,
    "micro_f1": 0.384,
    "per_class": {
      "0": {
        "tp": 53,
        "fp": 2,
        "fn": 161,
        "precision": 0.9636,
        "recall": 0.2477,
        "f1": 0.3941
      },
      "1": {
        "tp": 61,
        "fp": 77,
        "fn": 20,
        "precision": 0.442,
        "recall": 0.7531,
        "f1": 0.5571
      },
      "2": {
        "tp": 6,
        "fp": 109,
        "fn": 25,
        "precision": 0.0522,
        "recall": 0.1935,
        "f1": 0.0822
      },
      "3": {
        "tp": 19,
        "fp": 35,
        "fn": 17,
        "precision": 0.3519,
        "recall": 0.5278,
        "f1": 0.4222
      }
    },
    "high_risk_recall": 0.6269,
    "high_risk_precision": 0.2485,
    "false_negative_rate": 0.3731,
    "false_positive_rate": 0.4305,
    "hr_tp": 42,
    "hr_fn": 25,
    "hr_fp": 127,
    "hr_tn": 168
  },
  "full_pipeline": {
    "n": 362,
    "accuracy": 0.3895,
    "macro_precision": 0.4378,
    "macro_recall": 0.429,
    "macro_f1": 0.371,
    "micro_f1": 0.3895,
    "per_class": {
      "0": {
        "tp": 57,
        "fp": 8,
        "fn": 157,
        "precision": 0.8769,
        "recall": 0.2664,
        "f1": 0.4086
      },
      "1": {
        "tp": 59,
        "fp": 68,
        "fn": 22,
        "precision": 0.4646,
        "recall": 0.7284,
        "f1": 0.5673
      },
      "2": {
        "tp": 6,
        "fp": 111,
        "fn": 25,
        "precision": 0.0513,
        "recall": 0.1935,
        "f1": 0.0811
      },
      "3": {
        "tp": 19,
        "fp": 34,
        "fn": 17,
        "precision": 0.3585,
        "recall": 0.5278,
        "f1": 0.427
      }
    },
    "high_risk_recall": 0.6119,
    "high_risk_precision": 0.2412,
    "false_negative_rate": 0.3881,
    "false_positive_rate": 0.4373,
    "hr_tp": 41,
    "hr_fn": 26,
    "hr_fp": 129,
    "hr_tn": 166
  }
}
```
