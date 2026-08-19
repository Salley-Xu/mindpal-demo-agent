# Intent Baseline Report — legacy_rule

- 日期: 2026-08-19 00:33:02
- dataset: intent_seed_v1.jsonl (n=362, context=False)

## 指标

| 指标 | 值 |
|---|---|
| Macro F1 | 0.1381 |
| Micro F1 | 0.228 |
| Exact Match | 0.1215 |
| Hamming Loss | 0.1815 |
| Avg Pred Labels | 1.066 |
| Avg Latency | 0.0 ms |
| LLM Call Rate | 0.0 |

## Per-label

| label | P | R | F1 | tp/fp/fn |
|---|---|---|---|---|
| casual_chat | 0.1077 | 0.9032 | 0.1924 | 28/232/3 |
| emotional_expression | 0.6875 | 0.3607 | 0.4731 | 44/20/78 |
| explicit_help_request | 0.3421 | 0.4333 | 0.3824 | 13/25/17 |
| information_request | 0.0 | 0.0 | 0.0 | 0/0/39 |
| resource_request | 0.5 | 0.25 | 0.3333 | 12/12/36 |
| feedback | 0.0 | 0.0 | 0.0 | 0/0/39 |
| follow_up | 0.0 | 0.0 | 0.0 | 0/0/23 |
| memory_reference | 0.0 | 0.0 | 0.0 | 0/0/46 |
| high_risk_expression | 0.0 | 0.0 | 0.0 | 0/0/66 |
| meta_question | 0.0 | 0.0 | 0.0 | 0/0/21 |
