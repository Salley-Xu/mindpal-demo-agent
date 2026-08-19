# MindPal Agent Phase 6：Recommendation 2.0 执行文档

> 前置：Phase 5 Risk 2.0 已冻结，Phase 4 Memory 2.0 已可用。

## 1. 当前问题

当前 Recommendation：
```text
RecommendGate
→ Hybrid Retrieval
→ Rule Ranking
→ LLM Rerank
```

已知问题：
- 推荐 mode 曾受 Risk 过触发严重污染
- hard threshold 在旧分布下几乎不可达
- feedback 会影响 Gate / prompt，但未真正进入 ranking feature
- recommendation history / rejection 尚未形成稳定闭环
- Phase 3 Policy 已将 recommendation_mode 与 PrimaryAction 解耦，因此现在可以单独优化推荐系统

## 2. 最终架构

```text
AgentState
↓
AgentPolicy recommendation_mode
↓
Recommendation Tool
↓
Candidate Retrieval
↓
Feature Ranking
↓
Optional LLM Rerank
↓
Safety Review
↓
Recommendation
↓
Feedback
↓
Memory / Future Ranking
```

## 3. 执行顺序

```text
6.1 Recommendation Dataset / Event Audit
6.2 Trigger / Mode Contract
6.3 Candidate Retrieval Baseline
6.4 Feature Ranker
6.5 Feedback Features
6.6 Cooldown / Diversity / Repeat Control
6.7 Optional LLM Rerank
6.8 Offline Benchmark
6.9 Closed-loop Simulation
6.10 Final Freeze
```

## 4. Ranking Features

建议：
```text
semantic similarity
BM25
intent match
emotion fit
risk compatibility
profile preference
memory preference
recency
repeat penalty
negative feedback penalty
already_seen
```

Risk 必须消费 Phase 5 冻结信号。

## 5. Feedback

统一：
```text
accept
reject
tried_effective
tried_ineffective
not_interested
already_seen
```

反馈必须直接影响未来 ranking，不仅影响 prompt。

## 6. Repeat / Cooldown

建立：
```text
same-item cooldown
same-category cooldown
negative feedback suppression
diversity bonus
```

避免连续重复、已拒绝仍推荐、同类过密。

## 7. LLM Rerank

先比较：
```text
Feature Ranker
vs
Feature + LLM Rerank
```

若 LLM 没稳定增益：
```text
关闭
```

不要默认保留。

## 8. Recommendation Benchmark

至少覆盖：
```text
explicit resource request
soft recommendation opportunity
no-recommendation
recent rejection
already seen
effective feedback
ineffective feedback
high risk safety-only
memory preference
```

指标：
```text
Trigger Precision / Recall
Mode Accuracy
Top-k Recall
MRR / nDCG
Repeat Rate
Negative-feedback Violation
Safety Violation
```

## 9. Ablation

```text
A Current Gate
B Retrieval only
C Hybrid Retrieval
D + Feature Ranking
E + Feedback
F + LLM Rerank
G Full Recommendation 2.0
```

## 10. 目标

建议：
```text
Trigger Precision >= 0.80
Trigger Recall >= 0.80
Mode Accuracy >= 0.85
Repeat Rate <= 0.05
Negative-feedback violation <= 0.02
Safety violation = 0
```

## 11. 交付物

```text
docs/
├── recommendation_contract_v2.md
├── recommendation_feedback_schema.md
├── phase6_ranking_ablation.md
├── phase6_closed_loop_eval.md
└── phase6_final_review.md

backend/recommendation_v2/
├── tool.py
├── retriever.py
├── features.py
├── ranker.py
├── feedback.py
├── cooldown.py
└── safety.py
```

## 12. Codex 启动指令

```text
Phase 6 不修改 AgentPolicy PrimaryAction。
Policy 只输出 recommendation_mode / recommend_resource tool signal。

先做 feature ranker + feedback closed loop，
再决定是否保留 LLM rerank。

所有推荐实验必须使用 Phase 5 Risk v2，
禁止再用旧 Risk 模型调 Gate 阈值。
```
