# Intent Small Model Baseline

- 日期: 2026-08-19 14:27:06
- 模型: hfl/chinese-macbert-base epochs=5 lr=2e-05

## Test 指标

| 指标 | 值 |
|---|---|
| Macro F1 | 0.6985 |
| Micro F1 | 0.7413 |
| Exact Match | 0.5156 |
| Hamming Loss | 0.0578 |

## Per-label F1

| label | P | R | F1 |
|---|---|---|---|
| casual_chat | 1.0 | 0.3636 | 0.5333 |
| emotional_expression | 1.0 | 0.25 | 0.4 |
| explicit_help_request | 1.0 | 0.5 | 0.6667 |
| information_request | 0.8621 | 0.7812 | 0.8197 |
| resource_request | 1.0 | 0.6667 | 0.8 |
| feedback | 1.0 | 0.5 | 0.6667 |
| follow_up | 1.0 | 0.9565 | 0.9778 |
| memory_reference | 1.0 | 0.75 | 0.8571 |
| high_risk_expression | 1.0 | 0.2667 | 0.4211 |
| meta_question | 1.0 | 0.7273 | 0.8421 |
