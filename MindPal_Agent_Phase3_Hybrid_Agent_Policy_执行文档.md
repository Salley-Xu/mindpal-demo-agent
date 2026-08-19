# MindPal Agent Phase 3：Hybrid Agent Policy 执行文档

> 版本：v1.0
> 日期：2026-08-19
> 阶段：Phase 3 — Hybrid Agent Policy
> 执行对象：Codex
> 前置：Phase 3 Preflight PASS、AgentState v1 Frozen、ActionPlan v1 Frozen。

## 1. 核心目标

将当前散落在 orchestrator、risk routing、RecommendGate、ReAct 前置逻辑里的决策统一收敛为：

```text
AgentState
    ↓
AgentPolicy
    ↓
ActionPlan
```

本阶段不是 Multi-Agent、Planner 或 LLM Router Demo，而是统一 Agent 决策层。

---

## 2. 最终架构

```text
AgentState
    ↓
P0 Safety Invariants
    ↓
P1 Deterministic Rules
    ↓
P2 Learned Policy（仅必要时）
    ↓
P3 Confidence / Ambiguity Gate
    ↓
P4 LLM Fallback
    ↓
ActionPlan
```

原则：

```text
高优先级决策不可被低优先级覆盖
```

---

## 3. 执行顺序

```text
Task 3.1  Policy Core
Task 3.2  Safety Policy
Task 3.3  Deterministic Primary Action
Task 3.4  Deterministic Tool Actions
Task 3.5  Recommendation Mode
Task 3.6  Policy Ambiguity
Task 3.7  Learned Policy Baseline（可选）
Task 3.8  LLM Fallback
Task 3.9  Hybrid Policy Orchestrator
Task 3.10 Shadow Integration
Task 3.11 Policy Benchmark / Ablation
Task 3.12 Error Analysis
Task 3.13 Final Review / Cutover / Freeze
```

---

## 4. 代码结构

```text
backend/policy/
├── schema.py
├── engine.py
├── safety.py
├── deterministic.py
├── ambiguity.py
├── classifier.py
├── llm_fallback.py
├── validator.py
├── hybrid.py
└── legacy_policy_adapter.py
```

---

## 5. PolicyResult

```python
class PolicyResult(BaseModel):
    action_plan: ActionPlan
    confidence: float
    source: str
    matched_rules: list[str] = []
    fallback_reason: str | None = None
    policy_version: str = "1.0"
```

必须可 Trace。

---

## 6. Task 3.2 Safety Policy

最高优先级输入：
- `state.risk`
- high-risk intent signal
- risk subject
- third-party flag
- discussion flag
- safe denial

典型 self：

```python
ActionPlan(
    primary_action="safety_intervention",
    tool_actions=[],
    safety_target="self",
    recommendation_mode="safety_only",
)
```

第三方：

```python
ActionPlan(
    primary_action="safety_intervention",
    tool_actions=[],
    safety_target="third_party",
    recommendation_mode="safety_only",
)
```

必须保证：
- Safety 不被 LLM 覆盖
- Safety 不进入普通推荐
- third-party target 正确
- discussion / safe denial 不被强行升级

---

## 7. Task 3.3 Primary Action Policy

输出互斥 PrimaryAction：

```text
continue_chat
ask_clarification
information_response
safety_intervention
```

可使用：
- intent labels / scores / uncertainty
- emotion / intensity
- risk
- conversation stage
- help_seeking
- information_seeking
- short context

初版只写高确定性规则，例如：

```text
risk requires intervention
→ safety_intervention

clear information_request
→ information_response

intent uncertain / reference unresolved
→ ask_clarification

otherwise
→ continue_chat
```

实际规则必须以 legacy audit + benchmark gold 为依据，不凭感觉堆规则。

---

## 8. Task 3.4 Tool Action Policy

ToolAction 可多选：

```text
retrieve_memory
retrieve_knowledge
recommend_resource
```

### retrieve_memory
信号：
- memory_reference
- personalization relevance
- available memory
- unresolved historical reference

注意：

```text
memory_reference != 必然 retrieve_memory
```

### retrieve_knowledge
主要针对明确 knowledge-dependent information request。

### recommend_resource
信号：
- resource_request
- explicit help
- recommendation history
- feedback/rejection
- cooldown

Phase 3 只决定是否调用 capability，不重构 Memory/RAG/Recommendation 内部排序。

---

## 9. Task 3.5 Recommendation Mode

输出：

```text
none
soft
hard
safety_only
```

建议：
- explicit resource request → hard/direct
- appropriate help context → soft
- not appropriate → none
- safety → safety_only

继续尊重 cooldown / rejection / recent recommendation history。

---

## 10. Policy 与 RecommendGate 的边界

Phase 3 不调 Recommendation Ranking。

第一版可：

```text
Policy 决定是否考虑 recommendation
↓
Legacy RecommendGate / adapter
↓
recommendation_mode
```

或者复刻其 routing contract。

Phase 6 再正式重构 Recommendation。

---

## 11. Task 3.6 Ambiguity Detection

只把真正难决定的 State 送给后续层。

触发：
- no rule matched
- conflicting rules
- intent uncertain
- low-confidence perception
- primary action tie
- tool action uncertainty

```python
class PolicyAmbiguity(BaseModel):
    is_ambiguous: bool
    reasons: list[str]
    confidence: float
```

---

## 12. Task 3.7 Learned Policy Baseline

这是可选任务。

先跑 Deterministic Policy。

如果：

```text
Primary Action Macro F1 >= 0.90
Safety Recall 达标
```

可以跳过 learned classifier。

如果需要，优先：
- Logistic Regression
- LightGBM
- Small MLP

输入必须是 AgentState structured features，不直接输入 raw conversation。

建议将：
- Primary Action classifier
- Tool Action multi-label classifier

分开建模。

---

## 13. Learned Policy 特征

例如：

```text
intent one-hot / scores
intent confidence
emotion
emotion intensity
risk level
risk trend
conversation stage
help seeking
memory reference
recent recommendation
rejection
turn count
```

全部来自 AgentState。

---

## 14. Task 3.8 LLM Fallback

LLM 仅用于 ambiguous cases。

输入：

```text
AgentState compact serialization
+
Allowed Action Schema
+
Policy Invariants
```

输出严格 JSON：

```json
{
  "primary_action": "continue_chat",
  "tool_actions": ["retrieve_memory"],
  "safety_target": "none",
  "recommendation_mode": "soft"
}
```

必须经过：
- Schema validation
- Policy invariant validation
- Safety override validation

LLM 永远不能覆盖 Safety。

---

## 15. Task 3.9 Hybrid Policy

接口：

```python
decide(state: AgentState) -> PolicyResult
```

推荐逻辑：

```python
def decide(state):
    safety = safety_policy(state)
    if safety.matched:
        return safety

    deterministic = deterministic_policy(state)
    if deterministic.confident:
        return deterministic

    if learned_policy_enabled:
        learned = learned_policy(state)
        if learned.confident:
            return learned

    return llm_fallback(state)
```

每次必须记录：
- source
- confidence
- matched_rules
- fallback_reason
- policy_version

---

## 16. Task 3.10 Shadow Integration

第一阶段：

```text
Legacy Policy → 实际生产行为
New Policy    → Shadow ActionPlan
```

记录：

```text
legacy_action_plan
new_action_plan
gold_action_plan（benchmark 时）
```

差异分类：
- same
- new_better
- legacy_better
- both_wrong
- semantic_equivalent

未通过 Benchmark 前禁止直接 cutover。

---

## 17. Task 3.11 Policy Benchmark

主要使用：

```text
Agent Benchmark v1.1
```

可额外新增：

```text
policy_edge_cases_v1.jsonl
```

覆盖：
- ambiguous
- safety
- memory + recommendation
- info + follow-up
- feedback
- third-party
- uncertainty

不得修改 Benchmark v1.1 Gold。

---

## 18. Policy Metrics

必须报告：

```text
Primary Action Accuracy
Primary Action Macro F1
Safety Primary Action Recall
Safety Target Accuracy

Tool Action Micro F1
Tool Action Macro F1
Tool Action Exact Match
Per-tool Precision/Recall

Recommendation Mode Accuracy
Policy Exact Match
LLM Fallback Rate
Average Decision Latency
```

Policy Exact Match 定义为：

```text
primary_action
+ tool_actions
+ safety_target
+ recommendation_mode
全部正确
```

---

## 19. 目标指标

推荐：

```text
Primary Action Accuracy >= 0.90
Primary Action Macro F1 >= 0.85

Safety Action Recall >= 0.98
Safety Target Accuracy >= 0.95

Tool Action Micro F1 >= 0.80
Recommendation Mode Accuracy >= 0.80

Policy Exact Match >= 0.75
LLM Fallback Rate <= 20%
```

最低 PASS：

```text
Safety invariants = 100%
Safety Recall >= 0.98
Primary Accuracy >= 0.88
Tool Micro F1 >= 0.75
Policy Exact >= 0.70
LLM Rate <= 30%
```

---

## 20. 必做 Ablation

至少：

```text
A Current Legacy Policy
B Pure LLM Policy
C Deterministic Policy
D Deterministic + Learned（如实现）
E Deterministic + LLM Fallback
F Final Hybrid
```

主表：

| Method | Primary Acc | Safety Recall | Tool Micro F1 | Rec Mode Acc | Exact | LLM Rate | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy | | | | | | | |
| Pure LLM | | | | | | 100% | |
| Deterministic | | | | | | 0% | |
| + Learned | | | | | | 0% | |
| + LLM | | | | | | | |
| Final Hybrid | | | | | | | |

Pure LLM 只是 baseline，不是默认方案。

---

## 21. Oracle-state Evaluation

必须增加：

```text
Predicted AgentState → Policy
vs
Gold / Oracle State → Policy
```

若 Oracle State 下 Policy 很高：
> 瓶颈主要在 perception。

若 Oracle State 下仍差：
> Policy 本身仍需优化。

这个实验用于区分 upstream error 和 policy error。

---

## 22. Task 3.12 Error Analysis

输出：

```text
docs/phase3_policy_error_analysis.md
```

错误类型：

```text
P01 Wrong Primary Action
P02 Missing Tool Action
P03 Extra Tool Action
P04 Wrong Recommendation Mode
P05 Safety Miss
P06 Safety Over-trigger
P07 Wrong Safety Target
P08 Clarification Miss
P09 Memory Retrieval Miss
P10 Knowledge Retrieval Miss
P11 Recommendation Over-trigger
P12 Recommendation Under-trigger
P13 Rule Conflict
P14 Learned Policy Error
P15 LLM Fallback Error
P16 Upstream Perception Error
```

必须区分：

```text
Policy-intrinsic error
vs
Upstream-caused error
```

---

## 23. Production Cutover

Final Review PASS 后才允许：

```text
New Policy → Actual ActionPlan
```

保留 feature flag：

```yaml
agent_policy:
  mode: legacy | shadow | hybrid
```

必须支持：

```text
hybrid → legacy
```

快速 rollback。

---

## 24. Config

```yaml
agent_policy:
  version: "1.0"
  mode: "shadow"

  enable_safety_rules: true
  enable_deterministic: true
  enable_learned_policy: false
  enable_llm_fallback: true

  deterministic_confidence_threshold: 0.85
  learned_confidence_threshold: 0.80
  max_llm_fallback_rate: 0.20
```

---

## 25. 测试

至少：

```text
test_policy_schema.py
test_safety_policy.py
test_primary_action_policy.py
test_tool_action_policy.py
test_recommendation_mode_policy.py
test_policy_ambiguity.py
test_policy_validator.py
test_hybrid_policy.py
test_policy_shadow.py
```

安全场景必须覆盖：
- self high-risk
- third-party high-risk
- discussion
- safe denial
- implicit high-risk
- risk + resource request
- risk + memory reference

Safety invariant 必须全部 PASS。

---

## 26. Regression

必须重新跑：
- Agent Benchmark v1.1
- Intent Independent Test
- State Transition Benchmark

Phase 3 只允许 Policy/routing 指标发生变化。

---

## 27. Phase 3 交付物

```text
docs/
├── agent_policy_contract_v1.md
├── legacy_policy_audit.md
├── policy_invariants_v1.md
├── phase3_policy_design.md
├── phase3_policy_ablation.md
├── phase3_policy_error_analysis.md
└── phase3_final_review.md

backend/policy/
├── schema.py
├── engine.py
├── safety.py
├── deterministic.py
├── ambiguity.py
├── classifier.py
├── llm_fallback.py
├── validator.py
├── hybrid.py
└── legacy_policy_adapter.py

evaluation/policy/
├── metrics.py
├── run_policy_eval.py
├── policy_edge_cases_v1.jsonl
└── reports/
```

---

## 28. Final Review 必答

```text
1. Final Policy 架构是什么？
2. Deterministic Policy 单独能达到多少？
3. Learned Policy 是否真的需要？
4. LLM Fallback 是否真的需要？
5. Safety Recall / Safety Target Accuracy 是多少？
6. Primary Action Accuracy 是多少？
7. Tool Action F1 是多少？
8. Recommendation Mode Accuracy 是多少？
9. Policy Exact Match 是多少？
10. LLM Fallback Rate / Latency / Cost 是多少？
11. Oracle-state vs Predicted-state 差距是多少？
12. 是否可以替换 Legacy Policy？
13. Rollback 是否可用？
14. AgentPolicy v1 是否 Freeze？
```

---

## 29. 停止原则

如果 Deterministic Policy 已满足推荐目标：

```text
不要为了“看起来高级”强行加 Learned Policy。
```

如果 Deterministic + LLM 明显优于 learned classifier：

```text
直接采用。
```

Phase 3 的目标是：

> 最小复杂度下获得最可靠、可解释、可回退的 Agent 决策。

---

## 30. Codex 启动指令

```text
请执行 Phase 3：Hybrid Agent Policy。

前置要求：
Phase 3 Preflight 已 PASS。

正式输入：
AgentState v1

正式输出：
ActionPlan v1

原则：
Safety > Deterministic > Learned > LLM
低优先级不得覆盖高优先级 Safety。

严格执行：
Task 3.1 → Task 3.13

限制：
1. 不修改 Intent / Emotion / Risk 模型。
2. 不重构 Memory 2.0。
3. 不调 Recommendation Ranking。
4. 不修改 AgentState Schema。
5. 不修改 ActionPlan Schema。
6. LLM 不得覆盖 Safety。
7. Learned Policy 非必做；Deterministic 达标则跳过。
8. 第一阶段必须 Shadow Mode。
9. 所有 Policy 决策记录 source/confidence/matched_rules/fallback_reason。
10. 必须区分 Policy intrinsic error 与 upstream perception error。

必须完成：
- Legacy Policy baseline
- Deterministic baseline
- Pure LLM baseline
- Final Hybrid
- Policy ablation
- Oracle-state evaluation
- Frozen Agent Benchmark regression
- Rollback validation

最终输出：
phase3_final_review.md

并明确：
AgentPolicy v1 = PASS / FAIL / FREEZE。
```

---

## 31. Phase 3 完成后的系统状态

```text
Perception
↓
AgentState v1
↓
AgentPolicy v1
↓
ActionPlan v1
├── Primary Action
├── Tool Actions
├── Safety Target
└── Recommendation Mode
↓
Execution Layer
```

Phase 3 是 MindPal 从“多个模块拼接的对话系统”升级为：

> **显式 State → Policy → Action 的 Stateful Adaptive Agent**

的核心阶段。
