# Intent Small Model Baseline

- 日期: 2026-08-19 15:17:13
- 模型: hfl/chinese-macbert-base epochs=8 lr=2e-05

## Test 指标

| 指标 | 值 |
|---|---|
| Macro F1 | 0.9003 |
| Micro F1 | 0.9006 |
| Exact Match | 0.8056 |
| Hamming Loss | 0.025 |

## Per-label F1

| label | P | R | F1 |
|---|---|---|---|
| casual_chat | 0.8 | 0.7273 | 0.7619 |
| emotional_expression | 0.913 | 0.84 | 0.875 |
| explicit_help_request | 1.0 | 0.8182 | 0.9 |
| information_request | 0.8462 | 0.7586 | 0.8 |
| resource_request | 1.0 | 0.9167 | 0.9565 |
| feedback | 1.0 | 0.7647 | 0.8667 |
| follow_up | 1.0 | 1.0 | 1.0 |
| memory_reference | 1.0 | 0.8 | 0.8889 |
| high_risk_expression | 0.9394 | 0.9688 | 0.9538 |
| meta_question | 1.0 | 1.0 | 1.0 |
