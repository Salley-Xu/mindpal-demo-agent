# Intent Seed Dataset 质量审计报告

> 文件：data/intent/intent_seed_v1.jsonl

## 1. 总体统计

| 指标 | 值 |
|---|---|
| 总样本数 | 806 |
| multi-label 占比 | 242 (0.3) |
| 平均标签数/样本 | 1.31 |
| 文本长度分布 | {'短(<10)': 141, '中(10-30)': 665} |
| Source 分布 | {'template': 806} |
| Difficulty 分布 | {'easy': 191, 'medium': 367, 'hard': 248} |

## 2. 每标签 Positive 数

| 标签 | 数量 | 达标(>=60) |
|---|---|---|
| casual_chat | 83 | ✅ |
| emotional_expression | 146 | ✅ |
| resource_request | 79 | ✅ |
| explicit_help_request | 101 | ✅ |
| information_request | 200 | ✅ |
| high_risk_expression | 87 | ✅ |
| feedback | 95 | ✅ |
| follow_up | 100 | ✅ |
| memory_reference | 101 | ✅ |
| meta_question | 67 | ✅ |

## 3. 标签共现矩阵（Top 12）

| 标签对 | 共现数 |
|---|---|
| follow_up + information_request | 56 |
| explicit_help_request + high_risk_expression | 21 |
| feedback + memory_reference | 20 |
| emotional_expression + memory_reference | 17 |
| information_request + meta_question | 17 |
| emotional_expression + information_request | 16 |
| emotional_expression + explicit_help_request | 15 |
| emotional_expression + resource_request | 14 |
| emotional_expression + feedback | 11 |
| feedback + resource_request | 10 |
| emotional_expression + follow_up | 10 |
| explicit_help_request + memory_reference | 9 |

### 共现率（共现数 / 较小标签总数）—— 检查是否几乎重叠

| 标签对 | 共现率 |
|---|---|
| follow_up+information_request | 0.56  |
| explicit_help_request+high_risk_expression | 0.241  |
| feedback+memory_reference | 0.211  |
| emotional_expression+memory_reference | 0.168  |
| information_request+meta_question | 0.254  |
| emotional_expression+information_request | 0.11  |
| emotional_expression+explicit_help_request | 0.149  |
| emotional_expression+resource_request | 0.177  |
| emotional_expression+feedback | 0.116  |
| feedback+resource_request | 0.127  |

## 4. 查重审计

- **Exact Duplicate**：0 组
- **MinHash 近似重复（sim>=0.8）**：0 对

## 5. 结论

- 硬负样本 ≥100：见 generator 统计（101 条）
- 无标签对共现率 > 0.8 → Taxonomy 无明显冗余（需结合 §6 分析）
- 下一步：Taxonomy / Annotation 通过后进入 Legacy Rule + LLM-only Baseline
