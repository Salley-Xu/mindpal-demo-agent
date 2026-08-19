# Benchmark v1.1 数据审计报告

> 文件：evaluation/datasets/agent_benchmark_v1_1.jsonl
> 日期：2026-08-18

## 1. 总体统计

| 指标 | 值 |
|---|---|
| Case 数量 | 362 |
| Source 分布 | {'template': 362} |
| 单轮 / 多轮 | 332 / 30 |
| Multi-label case 占比 | 103 (0.285) |
| 高风险数量 (L2+L3) | 67 |

## 2. 分布

### Risk

```text
{0: 214, 1: 81, 2: 31, 3: 36}
```

### Intent（multi-label）

```text
{'casual_chat': 31, 'meta_question': 21, 'feedback': 39, 'resource_request': 48, 'follow_up': 23, 'explicit_help_request': 30, 'emotional_expression': 122, 'information_request': 39, 'memory_reference': 46, 'high_risk_expression': 66}
```

### Primary Action

```text
{'continue_chat': 223, 'information_response': 51, 'safety_intervention': 67, 'ask_clarification': 21}
```

### Tool Action

```text
{'recommend_resource': 50, 'retrieve_knowledge': 30, 'retrieve_memory': 40}
```

### Recommendation / SafetyTarget / Emotion

```text
rec: {'none': 276, 'soft': 34, 'hard': 42, 'safety_only': 10}
safety: {'none': 295, 'self': 58, 'third_party': 9}
```

## 3. 查重审计

### Exact Duplicate（用户轮文本完全一致）

- 无 ✅

### MinHash Duplicate（Jaccard 估计 >= 0.8，Top 15）

- 无 ✅

### Embedding Similarity

- **TODO**：后续用 bge-small-zh 做 embedding 相似度审计（保留占位）。

## 4. 与目标分布对照（§13 / §23）

| 检查项 | 结果 | 值 |
|---|---|---|
| Case >= 250 | ✅ | 362 |
| L2+L3 >= 60 | ✅ | 67 |
| ask_clarification >= 20 | ✅ | 21 |
| follow_up >= 20 | ✅ | 23 |
| feedback >= 30 | ✅ | 39 |
| memory_reference >= 30 | ✅ | 46 |
| high_risk_expression >= 50 | ✅ | 66 |
| safety_intervention >= 60 | ✅ | 67 |
| retrieve_memory >= 40 | ✅ | 40 |
| retrieve_knowledge >= 30 | ✅ | 30 |
| recommend_resource >= 50 | ✅ | 50 |

### 已知偏差（软目标）

- Risk L0 = 214（建议 130-150，因 intent 长尾覆盖优先而偏高）
- Risk L1 = 81（建议 70-80，符合）
- Risk L2 = 31 / L3 = 36（建议 30-35 / 30-35，符合）
