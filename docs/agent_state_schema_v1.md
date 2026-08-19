# AgentState Schema v1

> 对应 Phase 2 Task 2.2（§6-17）
> 日期：2026-08-19
> 代码：`backend/state/schema.py`
> 状态：**设计 v1**（Task 2.10 冻结）

---

## 1. 设计原则

1. **State ≠ Action**：不包含 should_recommend / should_retrieve_memory / primary_action / tool_actions / recommendation_mode（属于 Phase 3 Policy）。
2. **State ≠ Raw History**：只保存结构化摘要 + 引用 ID。
3. **生命周期分层**：Turn / Session / Cross-session Reference / Derived。
4. **单一所有权**：每字段有唯一 source / writer。

## 2. 顶层结构

```python
class AgentState(BaseModel):
    schema_version: str = "1.0"
    identity: IdentityState
    turn: TurnState
    intent: IntentState
    emotion: EmotionState
    risk: RiskState
    conversation: ConversationState
    user_context: UserContextState
    recommendation: RecommendationState
    memory: MemorySignalState
    derived: DerivedState
    meta: StateMeta
```

## 3. 各子状态

### IdentityState
`user_id` / `session_id`（不存敏感原始资料）

### TurnState
`turn_index` / `current_text` / `previous_user_text` / `previous_assistant_text`（short context ≤1 轮）

### IntentState（来源：Phase 1.5 Intent Service）
`labels` / `confidence` / `label_scores` / `is_open_set` / `source` / `fallback_reason` / `context_used`

### EmotionState（来源：emotion_analyzer）
`current_emotion`（中文）/ `confidence` / `intensity` / `context_emotion` / `trend` / `stress_source` / `legacy_user_intent`（4 类，兼容引用，不与 Intent 10 类混用）

### RiskState（来源：risk_evaluator，Phase 2 不修复 FPR/FNR）
`level` / `confidence` / `score` / `trend` / `persistence`（派生：连续 L2+ turn 数）/ `subject` / `is_discussion` / `is_third_party` / `safe_denial` / `evidence` / `escalation_reasons` / `baseline` / `historical_high_risk_count`

### ConversationState（消费 conversation_manager 结构化结果）
`stage` / `turn_count` / `current_topic` / `key_concerns` / `recent_intents`（滚动窗口 5）/ `emotion_trend` / `summary_version`

### UserContextState（画像摘要，非全量）
`preferred_support_style` / `avoid_styles` / `main_stress_sources` / `profile_risk_level`

### RecommendationState（历史状态，非决策）
`recent_recommendation_ids`（窗口 10）/ `last_recommendation_turn` / `turns_since_last_recommendation` / `recent_feedback`（窗口 10）/ `rejection_detected`

### MemorySignalState（信号，不执行 Memory 2.0）
`explicit_memory_reference`（由 intent.memory_reference 派生）/ `personalization_relevant` / `available_memory_count` / `last_retrieved_memory_ids`

### DerivedState（确定性派生，不涉及 Policy）
`help_seeking` / `resource_seeking` / `information_seeking` / `high_risk_intent_signal` / `memory_reference_signal` / `intent_uncertain` / `negative_emotion`

### StateMeta
`created_at` / `updated_at` / `trace_id` / `state_version` / `source_versions`（如 intent=phase1_5_final / risk=v4_2_domain_only_v2）

## 4. 生命周期归属

| 生命周期 | 字段 |
|---|---|
| **Turn-only** | current_text、intent、current emotion、risk evidence、derived signals |
| **Session** | turn_count、stage、recent_intents、risk persistence、recommendation history、short context |
| **Cross-session Reference** | user_context、risk.baseline、memory signals（AgentState 只存本轮快照，不复制真值库） |

## 5. 明确禁止的 State 字段

```text
should_recommend / should_retrieve_memory / should_safety_intervene
primary_action / tool_actions / recommendation_mode / current_action
```

这些属于 Phase 3 Agent Policy 输出（ActionPlan），见 docs/benchmark_schema_v1_1.md 的 ActionPlan 冻结接口。
