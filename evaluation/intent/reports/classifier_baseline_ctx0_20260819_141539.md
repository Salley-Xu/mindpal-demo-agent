# Intent Small Model Baseline

- 日期: 2026-08-19 14:15:39
- 模型: hfl/chinese-macbert-base epochs=5 lr=2e-05

## Test 指标

| 指标 | 值 |
|---|---|
| Macro F1 | 0.6607 |
| Micro F1 | 0.7153 |
| Exact Match | 0.4375 |
| Hamming Loss | 0.0641 |

## Per-label F1

| label | P | R | F1 |
|---|---|---|---|
| casual_chat | 0.0 | 0.0 | 0.0 |
| emotional_expression | 0.75 | 0.375 | 0.5 |
| explicit_help_request | 1.0 | 0.6667 | 0.8 |
| information_request | 0.8929 | 0.7812 | 0.8333 |
| resource_request | 1.0 | 0.5833 | 0.7368 |
| feedback | 0.8333 | 0.625 | 0.7143 |
| follow_up | 1.0 | 0.6522 | 0.7895 |
| memory_reference | 0.9375 | 0.75 | 0.8333 |
| high_risk_expression | 1.0 | 0.3333 | 0.5 |
| meta_question | 1.0 | 0.8182 | 0.9 |
