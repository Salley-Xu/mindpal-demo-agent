# AgentState Update Rules

> 对应 Phase 2 Task 2.4（§20-23）
> 日期：2026-08-19

---

## 1. 更新规则类型

| 规则 | 含义 |
|---|---|
| **REPLACE** | 每轮用当前输入覆盖 |
| **APPEND** | 追加，有上限 |
| **ROLLING_WINDOW** | 滚动窗口，超出上限丢弃最旧 |
| **ACCUMULATE** | 累积计数（可条件累积） |
| **DERIVE** | 由其他字段确定性推导 |
| **RESET** | 条件满足时归零 |

## 2. 字段级更新规则（含生命周期）

| 字段 | 规则 | 生命周期 | Owner | 来源 |
|---|---|---|---|---|
| `intent.*` | REPLACE | Turn | Intent Service | IntentAdapter |
| `conversation.recent_intents` | ROLLING_WINDOW(5) | Session | conversation_manager | ConversationAdapter |
| `emotion.current_emotion` | REPLACE | Turn | emotion_analyzer | EmotionAdapter |
| `emotion.intensity` | REPLACE | Turn | emotion_analyzer | EmotionAdapter |
| `emotion.trend` | DERIVE | Session | conversation_manager | ConversationAdapter |
| `risk.level` | REPLACE | Turn | risk_evaluator | RiskAdapter |
| `risk.score` | REPLACE | Turn | risk_evaluator | RiskAdapter |
| `risk.recent_risk_levels` | ROLLING_WINDOW(5) | Session | StateUpdater | 派生自当前+历史 |
| `risk.persistence` | ACCUMULATE conditional | Session | StateUpdater | L2+ → +1，否则 RESET 0 |
| `risk.evidence` | REPLACE | Turn | risk_evaluator | RiskAdapter |
| `risk.baseline` | REPLACE | Cross-session | RiskMemory | RiskAdapter |
| `risk.historical_high_risk_count` | REPLACE | Session | orchestrator | 派生自 mood_events |
| `conversation.stage` | REPLACE | Session | conversation_manager | ConversationAdapter |
| `conversation.turn_count` | REPLACE | Session | conversation_manager | ConversationAdapter |
| `conversation.key_concerns` | REPLACE | Session | conversation_manager | ConversationAdapter |
| `user_context.*` | REPLACE | Cross-session | UserProfileTool | ProfileAdapter |
| `recommendation.recent_ids` | ROLLING_WINDOW(10) | Session | conversation_manager | RecommendationAdapter |
| `recommendation.turns_since_last` | ACCUMULATE | Session | StateUpdater | 有推荐→0，否则+1 |
| `recommendation.recent_feedback` | APPEND(10) | Session | record_feedback | RecommendationAdapter |
| `recommendation.rejection_detected` | REPLACE | Turn | rejection_detector | RecommendationAdapter |
| `memory.explicit_memory_reference` | DERIVE | Turn | StateBuilder | 由 intent.memory_reference 派生 |
| `memory.last_retrieved_ids` | REPLACE | Turn | memory_context_builder | memory_signal 输入 |
| `derived.*` | DERIVE | Turn | StateBuilder | 由 intent/emotion 派生 |
| `meta.state_version` | ACCUMULATE | Session | StateUpdater | +1 |
| `meta.trace_id` | REPLACE | Turn | orchestrator | 生成 |

## 3. 重点规则说明

### 3.1 Risk Persistence（条件累积）

```python
if current.risk.level >= 2:
    persistence = previous.persistence + 1
else:
    persistence = 0   # RESET
```

只表示连续满足 L2+ 的 turn 数，**不据此修改 Risk Level**（§22）。

### 3.2 Recommendation turns_since_last（累积）

```python
if 本轮有推荐(有 recent_recommendation_ids 新增):
    turns_since_last = 0
else:
    turns_since_last = previous.turns_since_last + 1
```

### 3.3 滚动窗口配置（§23）

| 字段 | 窗口 |
|---|---|
| `recent_intents` | 5 turns |
| `recent_risk_levels` | 5 turns |
| `recent_feedback` | 10 items |
| `recent_recommendation_ids` | 10 items |

## 4. 生命周期映射

| 生命周期 | 持久化 |
|---|---|
| Turn-only | 不持久化（每轮重建） |
| Session | 存 session storage（latest state） |
| Cross-session Reference | 引用 DB 真值（不复制第二套库） |
