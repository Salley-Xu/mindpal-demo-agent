# Intent Baseline Report — llm

- 日期: 2026-08-19 00:52:00
- dataset: intent_seed_stratified_150.jsonl (n=150, context=False)

## 指标

| 指标 | 值 |
|---|---|
| Macro F1 | 0.6965 |
| Micro F1 | 0.6834 |
| Exact Match | 0.3333 |
| Hamming Loss | 0.1007 |
| Avg Pred Labels | 1.813 |
| Avg Latency | 3726.11 ms |
| LLM Call Rate | 1.0 |

## Per-label

| label | P | R | F1 | tp/fp/fn |
|---|---|---|---|---|
| casual_chat | 0.4118 | 0.9333 | 0.5714 | 14/20/1 |
| emotional_expression | 0.5897 | 0.8519 | 0.697 | 23/16/4 |
| explicit_help_request | 0.6538 | 0.8947 | 0.7556 | 17/9/2 |
| information_request | 0.4894 | 0.8214 | 0.6133 | 23/24/5 |
| resource_request | 1.0 | 0.5333 | 0.6957 | 8/0/7 |
| feedback | 0.8696 | 0.8 | 0.8333 | 20/3/5 |
| follow_up | 0.48 | 0.5455 | 0.5106 | 12/13/10 |
| memory_reference | 0.4737 | 1.0 | 0.6429 | 18/20/0 |
| high_risk_expression | 0.9375 | 0.7895 | 0.8571 | 15/1/4 |
| meta_question | 0.8125 | 0.7647 | 0.7879 | 13/3/4 |
