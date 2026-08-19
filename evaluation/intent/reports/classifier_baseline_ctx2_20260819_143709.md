# Intent Small Model Baseline

- 日期: 2026-08-19 14:37:09
- 模型: hfl/chinese-macbert-base epochs=5 lr=2e-05

## Test 指标

| 指标 | 值 |
|---|---|
| Macro F1 | 0.6922 |
| Micro F1 | 0.7559 |
| Exact Match | 0.5234 |
| Hamming Loss | 0.057 |

## Per-label F1

| label | P | R | F1 |
|---|---|---|---|
| casual_chat | 0.0 | 0.0 | 0.0 |
| emotional_expression | 0.7333 | 0.4583 | 0.5641 |
| explicit_help_request | 1.0 | 0.6667 | 0.8 |
| information_request | 0.8 | 0.75 | 0.7742 |
| resource_request | 1.0 | 0.5 | 0.6667 |
| feedback | 1.0 | 0.5625 | 0.72 |
| follow_up | 1.0 | 0.9565 | 0.9778 |
| memory_reference | 1.0 | 0.85 | 0.9189 |
| high_risk_expression | 1.0 | 0.3333 | 0.5 |
| meta_question | 1.0 | 1.0 | 1.0 |
