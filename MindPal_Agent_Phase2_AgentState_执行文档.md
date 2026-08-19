# MindPal Agent Phase 2：AgentState 执行文档

> 版本：v1.0  
> 日期：2026-08-19  
> 阶段：Phase 2 — Unified Agent State  
> 执行对象：Codex  
> 前置状态：
> - Phase 0 / 0.5：PASS
> - Phase 1 / 1.5：PASS WITH KNOWN LIMITATIONS
> - Agent Benchmark v1.1：Frozen（362 cases）
> - Intent Taxonomy：Frozen
> - Intent Model / Runtime API：Frozen
> - Independent Intent Test：Frozen（535 cases）
> - ActionPlan Schema：Frozen
> - Phase 3 Agent Policy：尚未实现
>
> 本阶段原则：**只统一状态，不在 Phase 2 实现新的决策策略。**

---

# 1. Phase 2 目标

当前 Agent 已有大量状态信号，但分散在：

```text
agent_orchestrator 局部变量
conversation_manager session dict
emotion_analyzer 输出
risk_evaluator 输出
RiskMemory
UserProfile
recommendation_history
memory subsystem
```

当前主要问题不是“没有状态”，而是：

```text
状态分散
字段语义重复
生命周期不清晰
更新责任不明确
无法统一 Trace
Policy 无统一输入
```

Phase 2 的目标是建立：

```text
AgentState
```

作为后续 Agent Policy 的唯一结构化状态输入。

目标架构：

```text
User Input
    ↓
Perception
├── IntentResult
├── EmotionResult
└── RiskResult
    ↓
StateBuilder
    ↓
AgentState_t
    ↓
Current Agent Flow（Phase 2 仍保持原逻辑）
    ↓
StateUpdater
    ↓
AgentState_t+1
```

Phase 2 结束后：

```text
Phase 3 Policy
```

只需要消费：

```text
AgentState
```

而不是重新读取多个模块的零散变量。

---

# 2. 本阶段不做什么

Phase 2 禁止：

```text
重新训练 Intent
重新训练 Emotion
重新训练 Risk
Risk 2.0
Memory 2.0
Recommendation Gate 调参
新 Agent Policy
LLM Router
Planner / Multi-Agent
```

尤其禁止：

```text
if state.xxx:
    return SAFETY_INTERVENTION
```

这类真正的 State → Action 决策。

它属于：

```text
Phase 3 Agent Policy
```

Phase 2 可以提供 derived signals，但不能决定最终 ActionPlan。

---

# 3. 核心设计原则

## 3.1 State ≠ Action

例如：

```text
memory_reference = true
```

属于 State。

而：

```text
retrieve_memory
```

属于 ToolAction。

因此 AgentState 中禁止写：

```text
current_action
should_retrieve_memory
should_recommend
```

作为正式状态字段。

这些属于 Phase 3 Policy 输出。

## 3.2 State ≠ Raw History

AgentState 不是整个 conversation history。

不要把：

```text
完整聊天记录
完整长期记忆
完整推荐候选
```

塞入 State。

State 只保存：

```text
决策所需的结构化摘要信号
+
必要引用 ID
```

## 3.3 State 必须区分生命周期

至少分为：

```text
Turn State
Session State
Cross-session Reference
Derived State
```

避免所有字段永久累积。

## 3.4 State 必须有单一所有权

每个字段必须明确：

```text
source
writer
update rule
reset rule
```

不能多个模块随意修改同一字段。

---

# 4. 总体执行顺序

严格按：

```text
Task 2.1  State Source Audit
        ↓
Task 2.2  AgentState Schema v1
        ↓
Task 2.3  StateBuilder
        ↓
Task 2.4  StateUpdater
        ↓
Task 2.5  Session Lifecycle / Persistence
        ↓
Task 2.6  Shadow-mode Integration
        ↓
Task 2.7  State Trace / Debug
        ↓
Task 2.8  Unit + Integration + Regression
        ↓
Task 2.9  State Consistency Evaluation
        ↓
Task 2.10 Phase 2 Final Review
```

---

# 5. Task 2.1：State Source Audit

新增：

```text
docs/agent_state_source_audit.md
```

逐项审计现有状态来源。

至少覆盖：

```text
Intent
Emotion
Risk
Conversation Stage
User Profile
Risk Baseline
Recent Mood Trend
Recommendation History
Recommendation Feedback
Memory Reference
Session Summary
Turn Count
```

每个字段记录：

| 字段 | 当前来源 | 当前存储位置 | 是否持久化 | 更新方 | 是否重复 |
|---|---|---|---|---|---|

重点找：

```text
同一概念多个字段
字段命名不同但含义相同
局部变量但实际上应为 session state
应该每轮重算却被持久化
应该累积却每轮覆盖
```

---

# 6. Task 2.2：AgentState Schema v1

建议新增：

```text
agent/state/
├── schema.py
├── builder.py
├── updater.py
├── persistence.py
└── adapters.py
```

如果当前工程结构不同，适配已有目录即可。

推荐顶层结构：

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

---

# 7. IdentityState

```python
class IdentityState(BaseModel):
    user_id: str
    session_id: str
```

不要在 State 中保存用户真实姓名、敏感原始资料等非必要信息。

---

# 8. TurnState

```python
class TurnState(BaseModel):
    turn_index: int
    current_text: str

    previous_user_text: str | None = None
    previous_assistant_text: str | None = None
```

只保留 short context，不保存完整 history。

---

# 9. IntentState

必须直接消费 Phase 1.5 冻结的 IntentResult。

```python
class IntentState(BaseModel):
    labels: list[str]
    confidence: float
    label_scores: dict[str, float]

    is_open_set: bool
    source: str

    fallback_reason: str | None = None
    context_used: int = 0
```

来源唯一：

```text
Intent Service
```

其他模块禁止自行改写 Intent。

---

# 10. EmotionState

建议：

```python
class EmotionState(BaseModel):
    current_emotion: str
    confidence: float

    intensity: float | None = None
    context_emotion: str | None = None
    trend: str | None = None
    stress_source: str | None = None
```

如果当前系统存在 emotion_type / current_emotion / context_emotion，必须在 Source Audit 中明确语义，不要为了 Schema 强行合并。

---

# 11. RiskState

Phase 2 不改 Risk 模型，仅包装当前输出。

```python
class RiskState(BaseModel):
    level: int
    confidence: float | None = None
    score: float | None = None

    trend: str | None = None
    persistence: int = 0

    subject: str | None = None
    is_discussion: bool = False
    is_third_party: bool = False
    safe_denial: bool = False

    evidence: list[str] = []
    escalation_reasons: list[str] = []

    baseline: str | None = None
    historical_high_risk_count: int = 0
```

当前 Risk 已知质量问题保持原样，Phase 2 不修复 FPR/FNR。

---

# 12. ConversationState

```python
class ConversationState(BaseModel):
    stage: str
    turn_count: int

    current_topic: str | None = None
    key_concerns: list[str] = []

    recent_intents: list[str] = []
    emotion_trend: str | None = None

    summary_version: str | None = None
```

conversation_manager 仍是 Session 历史事实来源；AgentState 只是消费其结构化结果。

---

# 13. UserContextState

存本轮决策真正需要的长期画像摘要，而不是整个 profile。

```python
class UserContextState(BaseModel):
    preferred_support_style: str | None = None
    avoid_styles: list[str] = []
    main_stress_sources: list[str] = []
    profile_risk_level: str | None = None
```

Phase 2 不新增 Profile 推断。

---

# 14. RecommendationState

表示推荐历史状态，不表示推荐决策。

```python
class RecommendationState(BaseModel):
    recent_recommendation_ids: list[str] = []
    last_recommendation_turn: int | None = None
    turns_since_last_recommendation: int | None = None
    recent_feedback: list[str] = []
    rejection_detected: bool = False
```

禁止在 State 放：

```text
should_recommend
recommend_mode
```

---

# 15. MemorySignalState

Phase 2 不执行 Memory 2.0，只记录可供 Policy 使用的 Memory 信号。

```python
class MemorySignalState(BaseModel):
    explicit_memory_reference: bool = False
    personalization_relevant: bool = False
    available_memory_count: int = 0
    last_retrieved_memory_ids: list[str] = []
```

`explicit_memory_reference` 可以由 Intent 的 `memory_reference` 派生，但 `retrieve_memory` 不能放进 State。

---

# 16. DerivedState

只放无需额外模型、由已有信号确定性推导的字段。

```python
class DerivedState(BaseModel):
    help_seeking: bool = False
    resource_seeking: bool = False
    information_seeking: bool = False

    high_risk_intent_signal: bool = False
    memory_reference_signal: bool = False
    intent_uncertain: bool = False

    negative_emotion: bool = False
```

例如：

```python
help_seeking = "explicit_help_request" in intent.labels
```

合法。

禁止派生：

```text
should_recommend
should_retrieve_memory
should_safety_intervene
primary_action
tool_actions
recommendation_mode
```

这些属于 Policy。

---

# 17. StateMeta

```python
class StateMeta(BaseModel):
    created_at: str
    updated_at: str
    trace_id: str
    state_version: int
    source_versions: dict[str, str] = {}
```

例如：

```json
{
  "intent": "phase1_5_final",
  "risk": "v4_2_domain_only_v2",
  "emotion": "bert_emotion_vX"
}
```

---

# 18. Task 2.3：StateBuilder

新增：

```text
agent/state/builder.py
```

接口：

```python
build_agent_state(
    request,
    session,
    intent_result,
    emotion_result,
    risk_result,
    user_profile,
    recommendation_history,
    risk_baseline,
) -> AgentState
```

Builder 负责：

```text
字段归一化
结构转换
Derived State
默认值
版本信息
```

不负责：

```text
业务决策
Tool 调用
LLM 调用
Recommendation
Safety Routing
```

---

# 19. Builder 输入适配

不要强制所有旧模块立刻改输出 Schema。

使用 Adapter：

```python
IntentAdapter
EmotionAdapter
RiskAdapter
ConversationAdapter
ProfileAdapter
```

流程：

```text
Legacy Output
↓
Adapter
↓
AgentState Schema
```

---

# 20. Task 2.4：StateUpdater

接口：

```python
update_state(
    previous_state: AgentState,
    current_inputs: StateInputs
) -> AgentState
```

同时输出：

```text
docs/agent_state_update_rules.md
```

每个字段标记：

```text
REPLACE
APPEND
ROLLING_WINDOW
ACCUMULATE
DERIVE
RESET
```

---

# 21. 示例更新规则

Intent：

```text
REPLACE every turn
```

`conversation.recent_intents` 保留滚动窗口。

Emotion：

```text
current_emotion → REPLACE
emotion_trend → source/derive
```

Risk：

```text
level → REPLACE current
risk history → ROLLING_WINDOW
persistence → ACCUMULATE conditionally
```

Recommendation：

```text
recent IDs → ROLLING_WINDOW
turns_since_last → ACCUMULATE
feedback → APPEND limited
```

Memory Signal：

```text
explicit_memory_reference → REPLACE
last_retrieved_memory_ids → REPLACE
```

---

# 22. Risk Persistence

Phase 2 可以维护 `risk.persistence`，但只表示连续满足条件的 turn 数。

例如：

```python
if current.level >= 2:
    persistence = previous.persistence + 1
else:
    persistence = 0
```

不要在 Phase 2 基于 persistence 修改 Risk Level。

---

# 23. Rolling Window

建议配置化：

```text
recent_intents: 5 turns
risk_history: 5 turns
recent_feedback: 10 items
recent_recommendation_ids: 10 items
```

---

# 24. Task 2.5：State Lifecycle / Persistence

必须明确：

```text
哪些字段只活一轮
哪些字段活一个 Session
哪些字段来自 Cross-session DB
```

Turn-only：

```text
current_text
intent
current emotion
current risk evidence
derived signals
```

Session：

```text
turn_count
stage
recent_intents
risk persistence
recommendation history
short context
```

Cross-session Reference：

```text
user profile
risk baseline
long-term memory
```

Cross-session 数据不复制为第二套真值数据库；AgentState 只保留本轮 snapshot。

---

# 25. Persistence Strategy

推荐保存 AgentState Snapshot 到 session storage，不新增第二套长期数据库。

示例：

```json
{
  "session_id": "...",
  "turn_index": 12,
  "state_version": 12,
  "state": {}
}
```

推荐：

```text
每轮保存 latest state
```

Debug 模式可保存 full state history；Production 默认不无限保留。

---

# 26. Task 2.6：Shadow-mode Integration

Phase 2 第一阶段必须使用：

```text
Shadow Mode
```

即：

```text
Legacy variables
        ↓
Current Agent Logic

同时

Legacy outputs
        ↓
StateBuilder
        ↓
AgentState
```

AgentState 暂时不驱动生产决策。

---

# 27. Shadow Mode 目标

验证：

```text
AgentState 是否完整
AgentState 是否与旧变量一致
State update 是否正确
是否存在字段漂移
```

---

# 28. Consistency Check

每轮检查：

```text
legacy risk level == state.risk.level
legacy emotion == state.emotion.current_emotion
legacy conversation stage == state.conversation.stage
```

Intent 是新系统结果时，只检查 Intent Service → State 的传递一致性，不与 legacy 4 类 user_intent 强行一致。

---

# 29. Shadow Trace

建议：

```text
logs/agent_state_shadow.jsonl
```

每轮：

```json
{
  "trace_id": "...",
  "session_id": "...",
  "turn_index": 5,
  "state": {},
  "consistency": {
    "risk_match": true,
    "emotion_match": true,
    "stage_match": true
  }
}
```

---

# 30. Task 2.7：State Trace / Debug

Phase 7 才做完整 Observability。

Phase 2 只做最小 State Trace：

```python
state.to_debug_dict()
```

必须支持：

```text
trace_id
state_version
source_versions
```

Debug Trace 默认不记录：

```text
长期记忆原文
用户 Profile 全量
完整 conversation history
```

优先记录 ID、类型、结构化摘要。

---

# 31. Task 2.8：测试

至少新增：

```text
test_agent_state_schema.py
test_state_builder.py
test_state_updater.py
test_state_lifecycle.py
test_state_adapters.py
test_state_shadow_consistency.py
```

必测：

```text
普通闲聊
情绪表达
信息请求
高风险
emotion change
intent change
risk rising
risk falling
follow_up
memory_reference
session isolation
```

---

# 32. State Transition Tests

至少覆盖：

```text
Intent:
help → information

Emotion:
anxiety → calm

Risk:
0 → 1 → 2 → 1

Recommendation:
none → recommended → rejected

Memory:
no reference → explicit reference
```

---

# 33. Task 2.9：State Consistency Evaluation

新增：

```text
evaluation/state/
```

可从 Agent Benchmark v1.1 抽取或独立新增：

```text
100–150 multi-turn state cases
```

不得修改被冻结的 Agent Benchmark v1.1。

建议文件：

```text
state_transition_benchmark_v1.jsonl
```

示例：

```json
{
  "case_id": "state_001",
  "turns": [
    {
      "user": "最近压力有点大",
      "expected_state": {
        "intent.labels": ["emotional_expression"],
        "risk.level": 0
      }
    },
    {
      "user": "你刚才说的第二个方法怎么做？",
      "expected_state": {
        "intent.labels": ["follow_up", "information_request"],
        "conversation.turn_count": 2
      }
    }
  ]
}
```

---

# 34. State Metrics

至少：

```text
Field Accuracy
Transition Accuracy
Persistence Accuracy
Reset Accuracy
Session Isolation Pass Rate
Legacy Consistency Rate
```

重点看：

```text
State Transition Accuracy
```

Intent 本身的预测错误不计为 StateBuilder 错误；State Evaluation 只判断 Intent Service 输出是否被正确写入 State。

---

# 35. Phase 2 目标指标

建议：

```text
Schema Validation Pass = 100%
State Transition Accuracy >= 95%
Session Isolation = 100%
Legacy Consistency >= 99%
```

---

# 36. Agent Benchmark Regression

完成 Shadow Integration 后必须重新跑：

```text
agent_benchmark_v1_1.jsonl
```

要求：

```text
生产逻辑指标不发生有意义变化
```

因为 Phase 2 还没有改变 Policy。

允许变化：

```text
日志
Trace
State Snapshot
内部结构
```

不允许：

```text
Risk 指标变化
Recommendation 指标变化
Routing 指标变化
回复行为变化
```

---

# 37. 如果出现行为变化

立即停止并检查：

```text
Adapter side effect
旧变量被新 State 覆盖
默认值改变
数据类型变化
Session persistence 改变
```

Phase 2 必须是：

```text
behavior-preserving refactor
```

---

# 38. Final Integration

Shadow 验证通过后，可逐步让 orchestrator 内部参数改为消费 AgentState，但仍保留旧业务逻辑。

例如可以增加：

```python
recommend_gate.decide_from_state(state)
```

内部仍调用旧 `recommend_gate.decide(...)`。

不要在 Phase 2 修改 Gate 逻辑。

---

# 39. Compatibility Layer

建议：

```text
agent/state/legacy_adapter.py
```

提供：

```python
state_to_legacy_emotion(state)
state_to_legacy_risk(state)
state_to_legacy_summary(state)
state_to_legacy_profile(state)
```

目标是减少 orchestrator 局部变量，而不是一次性大改。

---

# 40. 推荐代码目录

```text
agent/
├── state/
│   ├── __init__.py
│   ├── schema.py
│   ├── builder.py
│   ├── updater.py
│   ├── adapters.py
│   ├── persistence.py
│   └── debug.py
│
├── orchestrator.py
└── ...

evaluation/
└── state/
    ├── state_transition_benchmark_v1.jsonl
    ├── metrics.py
    ├── run_state_eval.py
    └── reports/

docs/
├── agent_state_source_audit.md
├── agent_state_schema_v1.md
├── agent_state_update_rules.md
├── phase2_state_evaluation.md
└── phase2_final_review.md
```

---

# 41. Config

建议：

```yaml
agent_state:
  schema_version: "1.0"

  recent_intents_window: 5
  risk_history_window: 5
  recommendation_history_window: 10
  feedback_window: 10

  persist_latest_state: true
  persist_state_history_debug: false

  enable_shadow_mode: true
  enable_consistency_check: true
```

不得散落硬编码。

---

# 42. Phase 2 交付物

必须有：

```text
docs/
├── agent_state_source_audit.md
├── agent_state_schema_v1.md
├── agent_state_update_rules.md
├── phase2_state_evaluation.md
└── phase2_final_review.md

agent/state/
├── schema.py
├── builder.py
├── updater.py
├── adapters.py
├── persistence.py
└── debug.py

evaluation/state/
├── state_transition_benchmark_v1.jsonl
├── metrics.py
├── run_state_eval.py
└── reports/

tests/
├── test_agent_state_schema.py
├── test_state_builder.py
├── test_state_updater.py
├── test_state_lifecycle.py
├── test_state_adapters.py
└── test_state_shadow_consistency.py
```

---

# 43. Phase 2 Final Review 必答问题

`docs/phase2_final_review.md` 必须回答：

1. AgentState Schema 是否稳定：PASS / FAIL
2. 每个字段是否有唯一 source / owner / update rule：YES / NO
3. State Transition Accuracy 是多少
4. Session Isolation 是否 100%
5. Shadow Mode 与旧流程一致率是多少
6. Phase 2 是否改变了生产 Agent 行为
7. Phase 3 Agent Policy 是否可以只消费 AgentState
8. 最终冻结哪个 Schema：AgentState v1

---

# 44. Phase 2 停止条件

PASS 最低要求：

```text
Schema validation = 100%
Session isolation = 100%
Transition accuracy >= 95%
Legacy consistency >= 99%
Agent Benchmark 无行为回归
```

如果 Transition Accuracy 不足，优先检查：

```text
update rule
field ownership
lifecycle
```

而不是调模型。

---

# 45. 本阶段不追求的指标

不要在 Phase 2 追求：

```text
Agent Action Accuracy
Recommendation Precision
Safety Recall
Memory Retrieval Precision
```

因为 Phase 2 只统一 State，不改变 Policy。

这些属于后续 Phase 3 / 4 / 5 / 6。

---

# 46. Git Branch 建议

```text
feat/agent-state-schema
feat/agent-state-builder
feat/agent-state-updater
feat/agent-state-persistence
feat/agent-state-shadow
eval/agent-state
```

Commit：

```text
feat(state): add AgentState v1 schema
feat(state): add legacy output adapters
feat(state): add state builder and derived signals
feat(state): add deterministic state updater
feat(state): add session state persistence
feat(state): add shadow-mode consistency logging
eval(state): add state transition benchmark
docs(state): freeze AgentState v1
```

---

# 47. 建议执行周期

## Day 1

```text
State Source Audit
AgentState Schema
```

## Day 2

```text
Adapters
StateBuilder
```

## Day 3

```text
StateUpdater
Lifecycle
```

## Day 4

```text
Persistence
Shadow Mode
```

## Day 5

```text
State Benchmark
Unit / Integration Tests
```

## Day 6

```text
Agent Benchmark Regression
Final Review
Schema Freeze
```

---

# 48. Codex 启动指令

```text
请执行 Phase 2：AgentState。

前置条件：
- Phase 0 / 0.5 PASS
- Phase 1 / 1.5 已冻结
- Intent Service 已冻结：
  predict_intent(current_text, previous_turns=None, max_context_turns=1)
- Agent Benchmark v1.1 Frozen
- ActionPlan Schema Frozen

本阶段目标：
将当前分散在 orchestrator / conversation_manager / perception / profile /
recommendation history 中的结构化信号统一收敛到 AgentState v1。

严格按：
Task 2.1 → Task 2.10
执行。

重要限制：

1. 不实现新的 Agent Policy。
2. 不修改 Risk / Emotion / Intent 模型。
3. 不调 Recommendation Gate。
4. 不重构 Memory 2.0。
5. 不允许 AgentState 直接生成 ActionPlan。
6. Phase 2 必须是 behavior-preserving refactor。

第一阶段必须使用 Shadow Mode：
旧流程继续驱动生产逻辑，
AgentState 只旁路构建、更新、记录和一致性检查。

必须先输出：
- agent_state_source_audit.md
- agent_state_schema_v1.md
- agent_state_update_rules.md

然后才能接入 orchestrator。

AgentState 必须区分：
- Turn State
- Session State
- Cross-session Reference
- Derived State

每个字段必须定义：
- source
- owner
- update rule
- reset rule
- persistence scope

禁止把以下字段定义为 State：
- should_recommend
- should_retrieve_memory
- primary_action
- tool_actions
- recommendation_mode

这些属于 Phase 3 Policy。

完成后必须运行：
1. State Transition Benchmark
2. Session Isolation Test
3. Shadow Consistency Test
4. Frozen Agent Benchmark v1.1 Regression

最终输出：
- phase2_state_evaluation.md
- phase2_final_review.md

phase2_final_review.md 必须明确：
- AgentState Schema 是否稳定
- State Transition Accuracy
- Session Isolation
- Legacy Consistency
- 是否发生生产行为回归
- Phase 3 是否可以只消费 AgentState
- AgentState v1 是否 Freeze

如果发现旧系统字段语义冲突：
不要私自选择其中一个覆盖。
记录在 state_semantic_conflicts.md，并给出兼容方案。
```

---

# 49. Phase 2 完成后的预期状态

完成前：

```text
Intent
Emotion
Risk
Profile
Session
Recommendation History
Memory Signals
        ↓
分散在多个局部变量 / dict
        ↓
Legacy if-else / ReAct
```

完成后：

```text
Intent
Emotion
Risk
Profile
Session
Recommendation History
Memory Signals
        ↓
Adapters
        ↓
AgentState v1
        ↓
Legacy Logic（Phase 2）
        ↓
Phase 3 Hybrid Agent Policy
```

最终技术边界：

```text
Perception
    ↓
AgentState
    ↓
Policy
    ↓
ActionPlan
```

Phase 2 的核心价值不是增加业务能力，而是：

> **把 Agent 从“多个模块共享零散变量”升级为“所有决策统一消费显式状态”的架构，为后续 Policy、Memory、Risk 与 Recommendation 的可解释决策和系统级评测建立稳定接口。**
