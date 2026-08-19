# Intent Small Model Baseline

- 日期: 2026-08-19 01:18:29
- 模型: hfl/chinese-macbert-base epochs=8 lr=2e-05

## Test 指标

| 指标 | 值 |
|---|---|
| Macro F1 | 0.8092 |
| Micro F1 | 0.8095 |
| Exact Match | 0.6311 |
| Hamming Loss | 0.0459 |

## Per-label F1

| label | P | R | F1 |
|---|---|---|---|
| casual_chat | 1.0 | 0.4545 | 0.625 |
| emotional_expression | 0.9286 | 0.5 | 0.65 |
| explicit_help_request | 1.0 | 0.8462 | 0.9167 |
| information_request | 0.92 | 0.697 | 0.7931 |
| resource_request | 1.0 | 0.7 | 0.8235 |
| feedback | 0.8889 | 0.6667 | 0.7619 |
| follow_up | 0.8235 | 0.875 | 0.8485 |
| memory_reference | 1.0 | 0.8824 | 0.9375 |
| high_risk_expression | 1.0 | 0.7647 | 0.8667 |
| meta_question | 1.0 | 0.7692 | 0.8696 |
