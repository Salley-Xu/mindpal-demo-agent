# Phase 6 Ranking Ablation

> 对应 Phase 6 Task 6.4 / 6.7
> 日期：2026-08-20

---

## 1. 方法

| Method | 说明 |
|---|---|
| B Retrieval only | 只召回，无序（baseline） |
| C Hybrid Retrieval | BM25 + semantic 混合召回 |
| D + Feature Ranking | 7 特征加权（semantic/BM25/intent/emotion/risk/profile/memory） |
| E + Feedback | reject→suppress / effective→boost |
| F + Cooldown | 同 item 硬阻断 + 同类抑制 |
| G Full Recommendation 2.0 | D+E+F |

## 2. 特征权重敏感性

Feature Ranker 权重（ranker.py `_FEATURE_WEIGHTS`）：

```text
semantic_similarity   0.20    核心语义匹配
risk_compatibility    0.20    安全放行（消费 Risk v2）
intent_match          0.15    资源请求对齐
emotion_fit           0.15    情绪安抚匹配
bm25                  0.10    lexical
profile_preference    0.10    长期画像
memory_preference     0.10    记忆偏好
```

## 3. 结果（场景化 benchmark）

| 能力 | B | D | E | F | G |
|---|---:|---:|---:|---:|---:|
| Trigger recall | 部分 | ✅ | ✅ | ✅ | ✅ |
| Safety violation | 高 | 低 | 低 | 低 | **0** |
| Repeat rate | 高 | 高 | 高 | **0** | **0** |
| Neg-feedback violation | 高 | 高 | **0** | **0** | **0** |
| Effective boost | 无 | 无 | ✅ | ✅ | ✅ |

## 4. LLM Rerank 决策

- 对比 Feature Ranker vs Feature + LLM Rerank
- **Feature Ranker 已达标**（safety violation 0 / repeat 0）→ LLM 无稳定增益
- 按 Phase 6 §7：**关闭 LLM Rerank**（不默认保留）

## 5. 结论

Recommendation 2.0 的核心价值 = Feature Ranking + Feedback 闭环 + Cooldown，
全部确定性规则，无 LLM 依赖，可解释可评测。
