# Agent Benchmark v1 Baseline Report

- 日期: 2026-08-18 22:59:36
- cases: 100
- predictor: module (emotion=bert, risk=bert)
- 耗时: 109.2s
- benchmark: evaluation/datasets/agent_benchmark_v1.jsonl

## 1. 总览

| 模块 | 指标 | 值 |
|---|---|---|
| Risk | Macro F1 | 0.2936 |
| Risk | High-risk Recall (L2+L3) | 0.5 (tp=5, fn=5) |
| Risk | False Negative Rate | 0.5 |
| Risk | False Positive Rate | 0.4222 |
| Emotion (fine) | Accuracy / Macro F1 | 0.38 / 0.1189 |
| Emotion (coarse 5-class) | Accuracy / Macro F1 | 0.54 / 0.4657 |
| Intent | Macro F1 / Micro F1 | 0.1908 / 0.4211 |
| Recommendation | Trigger Precision / Recall | 0.3913 / 0.4615 |
| Recommendation | Mode Accuracy | 0.32 |
| Routing (signal-derived) | Action Accuracy / Macro F1 | 0.43 / 0.151 |
| Routing | Safety Action Recall | 0.5 |
| Routing (current deterministic) | Safety Action Recall | 0.5 |
| Memory Gate | Precision / Recall | 0.1228 / 0.7 |
| Risk Trend | Accuracy (multi-turn only) | 0.0 (n=9) |

## 2. 模块明细

### Risk（四级分类）

| level | P | R | F1 | tp/fp/fn |
|---|---|---|---|---|
| L0 | 0.9412 | 0.2581 | 0.4051 | 16/1/46 |
| L1 | 0.475 | 0.6786 | 0.5588 | 19/21/9 |
| L2 | 0.0 | 0.0 | 0.0 | 0/30/4 |
| L3 | 0.1538 | 0.3333 | 0.2105 | 2/11/4 |

### Emotion

| label | P | R | F1 |
|---|---|---|---|
| anger | 0.25 | 0.25 | 0.25 |
| anxiety | 0.3158 | 0.4 | 0.3529 |
| calm | 0.0 | 0.0 | 0.0 |
| fatigue | 0.0 | 0.0 | 0.0 |
| grief | 0.0 | 0.0 | 0.0 |
| guilt | 0.0 | 0.0 | 0.0 |
| happy | 0.1429 | 0.25 | 0.1818 |
| hope | 0.0 | 0.0 | 0.0 |
| hopelessness | 0.0 | 0.0 | 0.0 |
| loneliness | 0.0 | 0.0 | 0.0 |
| neutral | 0.4694 | 0.9583 | 0.6301 |
| panic | 0.0 | 0.0 | 0.0 |
| sadness | 0.3333 | 0.4118 | 0.3684 |
| shame | 0.0 | 0.0 | 0.0 |
| stress | 0.0 | 0.0 | 0.0 |

### Intent（当前系统=user_intent 关键词规则映射）

| label | P | R | F1 |
|---|---|---|---|
| casual_chat | 0.2143 | 0.75 | 0.3333 |
| emotional_expression | 0.7143 | 0.6383 | 0.6742 |
| explicit_help_request | 0.5 | 0.5 | 0.5 |
| feedback | 0.0 | 0.0 | 0.0 |
| follow_up | 0.0 | 0.0 | 0.0 |
| high_risk_expression | 0.0 | 0.0 | 0.0 |
| information_request | 0.0 | 0.0 | 0.0 |
| memory_reference | 0.0 | 0.0 | 0.0 |
| meta_question | 0.0 | 0.0 | 0.0 |
| resource_request | 0.5 | 0.3333 | 0.4 |

### Recommendation（Gate 模式）

| mode | P | R | F1 |
|---|---|---|---|
| none | 0.7083 | 0.2881 | 0.4096 |
| soft | 0.3409 | 0.5769 | 0.4286 |
| hard | 0.0 | 0.0 | 0.0 |
| safety_only | 0.0 | 0.0 | 0.0 |
| third_party_support | 0.0 | 0.0 | 0.0 |

### Agent Routing（signal-derived vs current deterministic）

| action | P | R | F1 |
|---|---|---|---|
| continue_chat | 0.6667 | 0.7755 | 0.717 |
| ask_clarification | 0.0 | 0.0 | 0.0 |
| retrieve_memory | 0.0 | 0.0 | 0.0 |
| recommend_resource | 0.0 | 0.0 | 0.0 |
| information_response | 0.0 | 0.0 | 0.0 |
| safety_intervention | 0.1163 | 0.5 | 0.1887 |

## 3. 失败样本（Top 20）

| case_id | 期望 vs 预测 | 模块 |
|---|---|---|
| agent_0002 | rec: 期望none 预测soft 周末打算和朋友去爬山，你觉得怎么样？ | rec-over |
| agent_0009 | rec: 期望none 预测soft 我在学做菜，但总失败。 | rec-over |
| agent_0010 | rec: 期望soft 预测none 聊点轻松的吧，最近有什么好玩的减压方式？ | rec-miss |
| agent_0101 | rec: 期望none 预测soft 最近连续加班，感觉很累。 | rec-over |
| agent_0102 | rec: 期望none 预测soft 一想到下周的汇报我就心慌。 | rec-over |
| agent_0103 | rec: 期望none 预测soft 朋友放我鸽子，我真的好生气。 | rec-over |
| agent_0104 | rec: 期望none 预测soft 莫名其妙就想哭，控制不住。 | rec-over |
| agent_0106 | rec: 期望none 预测soft 我最近特别容易紧张，手心冒汗。 | rec-over |
| agent_0107 | rec: 期望none 预测soft 被领导当众批评了，好丢人。 | rec-over |
| agent_0110 | rec: 期望none 预测soft 室友说我太敏感了，是不是我不好。 | rec-over |
| agent_0113 | rec: 期望none 预测soft 有时候觉得自己像个局外人，融入不了大家。 | rec-over |
| agent_0114 | rec: 期望none 预测soft 考研失败了，感觉对不起爸妈。 | rec-over |
| agent_0116 | rec: 期望none 预测soft 和男朋友吵架了，冷静下来还是很难过。 | rec-over |
| agent_0203 | rec: 期望hard 预测none 我睡不着，帮帮我。 | rec-miss |
| agent_0208 | rec: 期望none 预测soft 我不知道怎么跟父母沟通我的选择，你能帮我分析吗？ | rec-over |
| agent_0402 | rec: 期望soft 预测none 我上次说想学的那个正念练习，现在可以开始了吗？ | rec-miss |
| agent_0403 | rec: 期望none 预测soft 和你说过我和妈妈关系有点紧张，今天又吵了。 | rec-over |
| agent_0406 | rec: 期望none 预测soft 我之前跟你说我膝盖受伤了，现在好点了，能恢复跑步吗？ | rec-over |
| agent_0407 | rec: 期望none 预测soft 你还记得我爸妈总拿我和别人比吗？今天又提了。 | rec-over |
| agent_0408 | rec: 期望none 预测soft 我上周跟你说搬了新家，最近失眠更严重了。 | rec-over |

## 4. 结论与 Phase 建议

见 baseline_report.md 详细分析。
