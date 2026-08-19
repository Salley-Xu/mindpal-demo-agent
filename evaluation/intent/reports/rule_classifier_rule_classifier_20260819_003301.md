# Intent Baseline Report — rule_classifier

- 日期: 2026-08-19 00:33:01
- dataset: intent_seed_v1.jsonl (n=806, context=False)

## 指标

| 指标 | 值 |
|---|---|
| Macro F1 | 0.5603 |
| Micro F1 | 0.5831 |
| Exact Match | 0.2792 |
| Hamming Loss | 0.0999 |
| Avg Pred Labels | 1.082 |
| Avg Latency | 0.01 ms |
| LLM Call Rate | 0.0 |

## Per-label

| label | P | R | F1 | tp/fp/fn |
|---|---|---|---|---|
| casual_chat | 1.0 | 0.1205 | 0.2151 | 10/0/73 |
| emotional_expression | 0.6017 | 0.4863 | 0.5379 | 71/47/75 |
| explicit_help_request | 0.6145 | 0.505 | 0.5543 | 51/32/50 |
| information_request | 0.8947 | 0.34 | 0.4928 | 68/8/132 |
| resource_request | 0.5274 | 0.9747 | 0.6844 | 77/69/2 |
| feedback | 0.5126 | 0.6421 | 0.5701 | 61/58/34 |
| follow_up | 0.88 | 0.44 | 0.5867 | 44/6/56 |
| memory_reference | 0.8772 | 0.9901 | 0.9302 | 100/14/1 |
| high_risk_expression | 0.4426 | 0.3103 | 0.3649 | 27/34/60 |
| meta_question | 0.5684 | 0.806 | 0.6667 | 54/41/13 |
