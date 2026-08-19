# Agent State Source Audit

> 对应 Phase 2 Task 2.1（§5）
> 日期：2026-08-19
> 目的：逐项审计现有状态来源，找出字段分散、语义重复、生命周期不清晰等问题，为 AgentState v1 提供设计依据。

---

## 1. 状态来源总览

当前状态信号分散在 8 处：

| 来源 | 位置 | 生命周期 |
|---|---|---|
| orchestrator 局部变量 | `agent_orchestrator.py` 的 run_agent 内十几个变量 | Turn |
| session dict | `conversation_manager.sessions` | Session（内存）+ SQLite |
| emotion 输出 | `emotion_analyzer` | Turn |
| risk 输出 | `risk_evaluator` + `session_risk_aggregator` | Turn |
| user profile | `database.user_profile` | Cross-session |
| risk memory | `risk_memory_store`（baseline/events） | Cross-session |
| recommendation history | `conversation_manager`（events/accepted/rejected） | Session |
| memory subsystem | `memory_store` / `memory_context_builder` | Cross-session |

## 2. 逐字段审计

| 字段 | 当前来源 | 当前存储位置 | 是否持久化 | 更新方 | 是否重复 |
|---|---|---|---|---|---|
| `intent.labels`（10 类） | **Intent Service（Phase 1.5，未接入 orchestrator）** | 无（模块独立） | 否 | Intent Service | 与 legacy `user_intent` 语义不同 |
| `user_intent`（4 类） | `emotion_analyzer._detect_user_intent` | emotion_state / mood_events / memory_items | 是 | emotion_analyzer | ⚠️ 与 intent 语义重复（表达意图 vs 决策意图） |
| `current_emotion` | `emotion_analyzer` | summary / history | 是 | emotion_analyzer | — |
| `context_emotion` | `emotion_analyzer` | history | 是 | emotion_analyzer | — |
| `emotion_type` | `emotion_analyzer` | emotion_state | 否 | emotion_analyzer | ⚠️ 与 current_emotion 近似 |
| `emotion_intensity` | `emotion_analyzer` | emotion_state | 是 | emotion_analyzer | — |
| `confidence` | `emotion_analyzer` / BERT | emotion_state | 否 | emotion_analyzer | — |
| `stress_source` | `emotion_analyzer._detect_stress_source` | emotion_state | 是 | emotion_analyzer | — |
| `negative_trend` | `emotion_analyzer` | emotion_state | 否 | emotion_analyzer | — |
| `emotion_trend` | `conversation_manager` 维护 | summary | 是 | conversation_manager | — |
| `risk.level` | `risk_evaluator`（BERT+状态机） | urgent_issue dict | 每轮重算 | risk_evaluator | — |
| `risk.score` | `risk_evaluator` | urgent_issue | 否 | risk_evaluator | — |
| `risk.trend` | `session_risk_aggregator` | urgent_issue | 否 | risk_evaluator | — |
| `risk.persistence` | 无独立字段（可由 recent_risk_levels 派生） | summary.recent_risk_levels | 是 | conversation_manager | ⚠️ 无统一字段 |
| `risk.subject` / discussion / third_party | `risk_evaluator._analyze_context` | urgent_issue.risk_context | 否 | risk_evaluator | — |
| `risk.safe_denial` | `risk_evaluator._analyze_context` | risk_context | 否 | risk_evaluator | — |
| `risk.escalation_reasons` | `risk_evaluator` | urgent_issue | 否 | risk_evaluator | — |
| `risk.baseline` | `RiskMemoryReader.get_baseline` | risk_baselines 表 | 是 | RiskMemoryWriter | — |
| `risk.historical_high_risk_count` | `MoodTrackingTool.get_recent_trend` | 派生 | 否 | orchestrator | — |
| `conversation.stage` | `conversation_manager` | sessions 表 | 是 | conversation_manager | — |
| `conversation.turn_count` | `conversation_manager` | sessions | 是 | conversation_manager | — |
| `conversation.current_topic` | `conversation_manager` | summary | 是 | conversation_manager | — |
| `conversation.key_concerns` | `conversation_manager` | sessions | 是 | conversation_manager | — |
| `conversation.recent_intents` | `conversation_manager` | summary | 是 | conversation_manager | — |
| `profile.preferred_support_style` | `UserProfileTool` | user_profile 表 | 是 | agent 工具 | — |
| `profile.avoid_style` | `UserProfileTool` | user_profile | 是 | agent 工具 | — |
| `profile.main_stress_sources` | `UserProfileTool` | user_profile | 是 | conversation_manager 累积 | — |
| `profile.risk_level` | `UserProfileTool` | user_profile | 是 | RiskMemoryWriter | ⚠️ 与 risk.baseline 概念重复 |
| `recommendation.recent_ids` | `conversation_manager` | recommendation_events | 是 | conversation_manager | — |
| `recommendation.feedback` | `conversation_manager` / `/content/feedback` | recommendation_feedback 表 | 是 | record_recommendation_feedback | — |
| `recommendation.rejection` | `rejection_detector` | summary.has_rejected_recommendation | 是 | orchestrator | — |
| `memory.explicit_reference` | 无（可由 intent.memory_reference 派生） | — | — | — | ⚠️ 缺失 |
| `memory.last_retrieved_ids` | `memory_context_builder` | 无（只在 prompt 文本） | 否 | memory_context_builder | ⚠️ 无结构化记录 |
| `memory.available_count` | `memory_store` | — | — | memory_store | ⚠️ 未暴露 |

## 3. 发现的问题

### 3.1 语义重复 / 字段不一致

| 问题 | 说明 |
|---|---|
| **user_intent vs intent.labels** | 旧 4 类（表达意图）vs 新 10 类（决策意图），语义不同但共享"intent"名称 |
| **profile.risk_level vs risk.baseline** | 画像里的长期风险 vs RiskMemory 的 baseline，两个来源可能漂移 |
| **emotion_type vs current_emotion** | 一个英文规范标签、一个中文标签，含义近似但命名不同 |

### 3.2 生命周期不清晰

| 问题 | 说明 |
|---|---|
| **risk.persistence 缺失** | 有 recent_risk_levels 可派生，但无统一字段/计算规则 |
| **memory.last_retrieved_ids 无结构化记录** | 记忆检索命中只在 prompt 文本，不可 Trace |
| **profile.main_stress_sources 由多模块累积** | conversation_manager / UserProfileTool 都在写，无单一 owner |

### 3.3 局部变量应提升为 Session State

| 变量 | 现状 |
|---|---|
| `historical_high_risk_count` | orchestrator 每轮重算（从 mood_events），应为 session 派生 |
| `has_rejected_recommendation` | 存在 summary，但写入点分散 |
| `risk_baseline` | orchestrator 每轮读 RiskMemory，应为 session 快照 |

### 3.4 应该每轮重算却被持久化 / 应该累积却每轮覆盖

| 问题 | 说明 |
|---|---|
| `emotion_trend` | conversation_manager 维护，但 emotion_analyzer 也在判断趋势，双写 |
| `recent_intents` | 累积合理（滚动窗口），但当前无窗口上限控制 |

## 4. 对 AgentState v1 的设计输入

1. **Intent 用新 Intent Service 结果**，legacy user_intent 标记为 legacy 引用，不混入正式 intent 字段。
2. **Risk 只包装当前输出**（level/score/trend/context），persistence 作为派生字段（连续 L2+ turn 数）。
3. **Recommendation 只存历史状态**（recent_ids/feedback/rejection），不存 should_recommend。
4. **Memory 信号显式化**：explicit_memory_reference（由 intent 派生）+ last_retrieved_ids。
5. **每字段定义 owner**：意图→IntentService，情绪→emotion_analyzer，风险→risk_evaluator，会话→conversation_manager，画像→UserProfileTool。
6. **Turn-only vs Session vs Cross-session 区分**（见 schema v1 文档）。

## 5. 语义冲突记录（§48 state_semantic_conflicts.md）

| 冲突 | 兼容方案 |
|---|---|
| `user_intent`（4 类）vs `intent.labels`（10 类） | AgentState 使用新 Intent 10 类；legacy user_intent 保留在 EmotionState.legacy_user_intent 供兼容 |
| `profile.risk_level` vs `risk.baseline` | AgentState 两者都保留但明确来源：profile.risk_level=画像冗余快照，risk.baseline=RiskMemory 权威 |
| `emotion_type` vs `current_emotion` | AgentState.emotion.current_emotion 用中文标签（系统现行）；emotion_type 保留英文规范标签引用 |
