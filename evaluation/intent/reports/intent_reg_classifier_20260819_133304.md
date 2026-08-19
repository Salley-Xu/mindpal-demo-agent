# Intent Baseline Report — classifier

- 日期: 2026-08-19 13:33:04
- dataset: intent_seed_v1.jsonl (n=362, context=False)

## 指标

| 指标 | 值 |
|---|---|
| Macro F1 | 0.716 |
| Micro F1 | 0.7683 |
| Exact Match | 0.6188 |
| Hamming Loss | 0.0517 |
| Avg Pred Labels | 0.945 |
| Avg Latency | 116.55 ms |
| LLM Call Rate | 0.0 |

## Per-label

| label | P | R | F1 | tp/fp/fn |
|---|---|---|---|---|
| casual_chat | 0.9333 | 0.4516 | 0.6087 | 14/1/17 |
| emotional_expression | 0.8491 | 0.7377 | 0.7895 | 90/16/32 |
| explicit_help_request | 0.875 | 0.7 | 0.7778 | 21/3/9 |
| information_request | 1.0 | 0.641 | 0.7813 | 25/0/14 |
| resource_request | 0.9535 | 0.8542 | 0.9011 | 41/2/7 |
| feedback | 1.0 | 0.6923 | 0.8182 | 27/0/12 |
| follow_up | 0.3333 | 0.0435 | 0.0769 | 1/2/22 |
| memory_reference | 0.9556 | 0.9348 | 0.9451 | 43/2/3 |
| high_risk_expression | 0.8462 | 0.5 | 0.6286 | 33/6/33 |
| meta_question | 1.0 | 0.7143 | 0.8333 | 15/0/6 |
