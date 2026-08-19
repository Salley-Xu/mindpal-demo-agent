# Intent Baseline Report — legacy_rule

- 日期: 2026-08-19 00:32:54
- dataset: intent_seed_v1.jsonl (n=806, context=False)

## 指标

| 指标 | 值 |
|---|---|
| Macro F1 | 0.1425 |
| Micro F1 | 0.2066 |
| Exact Match | 0.1303 |
| Hamming Loss | 0.1867 |
| Avg Pred Labels | 1.04 |
| Avg Latency | 0.0 ms |
| LLM Call Rate | 0.0 |

## Per-label

| label | P | R | F1 | tp/fp/fn |
|---|---|---|---|---|
| casual_chat | 0.1256 | 0.9398 | 0.2216 | 78/543/5 |
| emotional_expression | 0.5909 | 0.4452 | 0.5078 | 65/45/81 |
| explicit_help_request | 0.52 | 0.3861 | 0.4432 | 39/36/62 |
| information_request | 0.0 | 0.0 | 0.0 | 0/0/200 |
| resource_request | 0.4375 | 0.1772 | 0.2523 | 14/18/65 |
| feedback | 0.0 | 0.0 | 0.0 | 0/0/95 |
| follow_up | 0.0 | 0.0 | 0.0 | 0/0/100 |
| memory_reference | 0.0 | 0.0 | 0.0 | 0/0/101 |
| high_risk_expression | 0.0 | 0.0 | 0.0 | 0/0/87 |
| meta_question | 0.0 | 0.0 | 0.0 | 0/0/67 |
