# Intent Small Model Baseline

- 日期: 2026-08-19 14:53:54
- 模型: hfl/chinese-macbert-base epochs=8 lr=2e-05

## Test 指标

| 指标 | 值 |
|---|---|
| Macro F1 | 0.8485 |
| Micro F1 | 0.8625 |
| Exact Match | 0.6953 |
| Hamming Loss | 0.0344 |

## Per-label F1

| label | P | R | F1 |
|---|---|---|---|
| casual_chat | 1.0 | 0.3636 | 0.5333 |
| emotional_expression | 0.9091 | 0.8333 | 0.8696 |
| explicit_help_request | 1.0 | 0.8333 | 0.9091 |
| information_request | 0.8571 | 0.75 | 0.8 |
| resource_request | 1.0 | 0.9167 | 0.9565 |
| feedback | 1.0 | 0.5625 | 0.72 |
| follow_up | 1.0 | 0.9565 | 0.9778 |
| memory_reference | 1.0 | 0.85 | 0.9189 |
| high_risk_expression | 1.0 | 0.6667 | 0.8 |
| meta_question | 1.0 | 1.0 | 1.0 |
