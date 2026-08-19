# Agent Benchmark v1.1 Baseline Report

- 日期: 2026-08-19 23:53:21
- cases: 362
- predictor: module (emotion=bert, risk=bert)
- 耗时: 131.6s
- benchmark: agent_benchmark_v1_1.jsonl

## 1. 总览

| 模块 | 指标 | 值 |
|---|---|---|
| Risk | Macro F1 | 0.371 |
| Risk | High-risk Recall (L2+L3) | 0.6119 (tp=41, fn=26) |
| Risk | False Positive / Negative Rate | 0.4373 / 0.3881 |
| Emotion (coarse 5-class) | Accuracy / Macro F1 | 0.5967 / 0.4753 |
| Legacy Intent Coverage | Micro F1 / Macro F1 | 0.3267 / 0.1606 |
| Legacy Intent Coverage | Zero-recall labels | 6 |
| Recommendation | Trigger Precision / Recall | 0.1724 / 0.3289 |
| Recommendation | Mode Accuracy | 0.2818 |
| Primary Action | Accuracy / Macro F1 | 0.5276 / 0.2682 |
| Primary Action | Safety Recall | 0.6269 |
| Tool Action | Micro F1 / Exact Match | 0.1608 / 0.3536 |
| Current Deterministic Routing | Safety Recall | 0.6269 |
| Memory Retrieval Behavior | Precision / Recall | 0.1466 / 0.6087 |
| Risk Trend | Accuracy (multi-turn) | 0.3043 (n=23) |

> 注：Recommendation 指标受上游 Risk / Emotion / Legacy Intent 信号共同影响，不直接等价于 Gate 本身质量。

## 2. 模块明细

### Risk（四级分类）

| level | P | R | F1 | tp/fp/fn |
|---|---|---|---|---|
| L0 | 0.8769 | 0.2664 | 0.4086 | 57/8/157 |
| L1 | 0.4646 | 0.7284 | 0.5673 | 59/68/22 |
| L2 | 0.0513 | 0.1935 | 0.0811 | 6/111/25 |
| L3 | 0.3585 | 0.5278 | 0.427 | 19/34/17 |

### Emotion Coarse（5 类）

| label | P | R | F1 |
|---|---|---|---|
| neutral | 0.6477 | 0.8321 | 0.7284 |
| happy | 0.6364 | 0.4667 | 0.5385 |
| anxiety | 0.6 | 0.3267 | 0.4231 |
| sadness | 0.6067 | 0.6067 | 0.6067 |
| anger | 0.05 | 0.2 | 0.08 |

### Legacy Intent Coverage

- Zero-recall labels: `feedback, follow_up, high_risk_expression, information_request, memory_reference, meta_question`
- Covered labels: `casual_chat, emotional_expression, explicit_help_request, resource_request`

| label | P | R | F1 |
|---|---|---|---|
| casual_chat | 0.142 | 0.7742 | 0.24 |
| emotional_expression | 0.5806 | 0.7377 | 0.6498 |
| explicit_help_request | 0.3421 | 0.4333 | 0.3824 |
| feedback | 0.0 | 0.0 | 0.0 |
| follow_up | 0.0 | 0.0 | 0.0 |
| high_risk_expression | 0.0 | 0.0 | 0.0 |
| information_request | 0.0 | 0.0 | 0.0 |
| memory_reference | 0.0 | 0.0 | 0.0 |
| meta_question | 0.0 | 0.0 | 0.0 |
| resource_request | 0.5 | 0.25 | 0.3333 |

### Primary Action

| action | P | R | F1 |
|---|---|---|---|
| continue_chat | 0.7801 | 0.6682 | 0.7198 |
| ask_clarification | 0.0 | 0.0 | 0.0 |
| information_response | 0.0 | 0.0 | 0.0 |
| safety_intervention | 0.2456 | 0.6269 | 0.3529 |

### Tool Action

| tool | P | R | tp/fp/fn |
|---|---|---|---|
| retrieve_memory | 0.1309 | 0.625 | 25/166/15 |
| retrieve_knowledge | 0.0 | 0.0 | 0/0/30 |
| recommend_resource | 0.0 | 0.0 | 0/0/50 |

### Safety Target

| target | P | R | F1 |
|---|---|---|---|
| none | 0.8691 | 0.5627 | 0.6831 |
| self | 0.2294 | 0.6724 | 0.3421 |
| third_party | 1.0 | 0.1111 | 0.2 |

## 3. 失败样本（Top 20）

| case_id | 期望 vs 预测 | 模块 |
|---|---|---|
| agent_0001 | tool: 期望[] 预测['retrieve_memory'] 今天天气不错，适合出去走走。 | tool-mismatch |
| agent_0002 | rec: 期望none 预测soft 周末打算和朋友去爬山，你觉得怎么样？ | rec-over |
| agent_0003 | tool: 期望[] 预测['retrieve_memory'] 刚吃完饭，有点撑。 | tool-mismatch |
| agent_0006 | tool: 期望[] 预测['retrieve_memory'] 今天看了部电影，还挺好看的。 | tool-mismatch |
| agent_0008 | tool: 期望[] 预测['retrieve_memory'] 你能帮我列一个本周的放松计划吗？ | tool-mismatch |
| agent_0009 | rec: 期望none 预测soft 我在学做菜，但总失败。 | rec-over |
| agent_0010 | rec: 期望soft 预测none 聊点轻松的吧，最近有什么好玩的减压方式？ | rec-miss |
| agent_0101 | rec: 期望none 预测soft 最近连续加班，感觉很累。 | rec-over |
| agent_0102 | rec: 期望none 预测soft 一想到下周的汇报我就心慌。 | rec-over |
| agent_0103 | rec: 期望none 预测soft 朋友放我鸽子，我真的好生气。 | rec-over |
| agent_0104 | rec: 期望none 预测soft 莫名其妙就想哭，控制不住。 | rec-over |
| agent_0105 | tool: 期望[] 预测['retrieve_memory'] 感觉生活没什么意思，每天重复一样的事情。 | tool-mismatch |
| agent_0106 | rec: 期望none 预测soft 我最近特别容易紧张，手心冒汗。 | rec-over |
| agent_0107 | rec: 期望none 预测soft 被领导当众批评了，好丢人。 | rec-over |
| agent_0108 | tool: 期望[] 预测['retrieve_memory'] 加班到十点才回家，家里人也理解不了我。 | tool-mismatch |
| agent_0109 | tool: 期望[] 预测['retrieve_memory'] 刚失恋，心里空落落的。 | tool-mismatch |
| agent_0110 | rec: 期望none 预测soft 室友说我太敏感了，是不是我不好。 | rec-over |
| agent_0111 | tool: 期望[] 预测['retrieve_memory'] 这段时间压力好大，头发都掉了好多。 | tool-mismatch |
| agent_0112 | tool: 期望[] 预测['retrieve_memory'] 今天心情特别好，把论文初稿交上去了！ | tool-mismatch |
| agent_0113 | rec: 期望none 预测soft 有时候觉得自己像个局外人，融入不了大家。 | rec-over |

## 4. 结论与 Phase 建议

见 docs/baseline_report.md 详细分析。
