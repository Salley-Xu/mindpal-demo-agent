# MindPal Phase 2 Closeout / Phase 3 Preflight 执行文档

> 版本：v1.0
> 日期：2026-08-19
> 执行对象：Codex
> 目标：不新增 Phase 2.5，完成 Phase 2 收尾并为 Phase 3 冻结输入输出契约和 baseline。

## 1. 当前状态

```text
Phase 0 / 0.5      PASS
Phase 1 / 1.5      PASS
Phase 2            PASS
AgentState v1      FROZEN
IntentResult       FROZEN
ActionPlan Schema  FROZEN
```

Phase 2 已达到：
- State Transition Accuracy = 100%
- Session Isolation = 100%
- Shadow Consistency = 100%

正式进入 Phase 3 前只补 4 个动作：

```text
P3-0.1 Frozen Agent Benchmark Regression
P3-0.2 独立修复 content_recommender missing import
P3-0.3 冻结 Phase 3 Policy Contract
P3-0.4 Legacy Policy Audit + Current Policy Baseline
```

---

## 2. P3-0.1 Frozen Agent Benchmark Regression

### 目标
实际重跑冻结的 Agent Benchmark v1.1（362 cases），证明 Phase 2 是 behavior-preserving refactor，而不是仅通过代码审查推断。

### 必须对比
```text
Risk Macro F1
High-risk Recall
Risk FPR/FNR
Emotion
Intent
Recommendation
Primary Action
Tool Action
Memory Behavior
Risk Trend
```

输出：

```text
docs/phase2_regression_closeout.md
```

### 验收
原则上：

```text
after - before = 0
```

允许浮点显示误差、日志和 Trace 差异；不允许 Routing / Risk / Recommendation / Memory 行为发生实际变化。

若出现真实业务指标变化，停止 Phase 3 并定位 side effect。

---

## 3. P3-0.2 独立修复 content_recommender Bug

Phase 2 审计发现 orchestrator 的 `content_recommender` 未 import，特定 LLM failure/fallback path 会触发 NameError。

单独 commit：

```text
fix(orchestrator): import content recommender fallback dependency
```

禁止与 Policy 重构混合。

至少测试：
- fallback path smoke test
- LLM failure simulation
- normal path regression

输出：

```text
docs/known_bug_001_content_recommender.md
```

记录：触发条件、修复、before/after、测试结果。

---

## 4. P3-0.3 冻结 Policy Contract

### 输入
唯一正式输入：

```python
AgentState
```

Phase 3 Policy 禁止绕过 AgentState 直接读取 perception/profile/database 私有变量。

### 输出
冻结：

```python
ActionPlan = {
    "primary_action": str,
    "tool_actions": list[str],
    "safety_target": str,
    "recommendation_mode": str,
}
```

PrimaryAction：

```text
continue_chat
ask_clarification
information_response
safety_intervention
```

ToolAction：

```text
retrieve_memory
retrieve_knowledge
recommend_resource
```

SafetyTarget：

```text
none
self
third_party
```

RecommendationMode：

```text
none
soft
hard
safety_only
```

推荐增加运行时包装：

```python
PolicyResult = {
    "action_plan": ActionPlan,
    "confidence": float,
    "source": "rule|classifier|llm|hybrid",
    "matched_rules": list[str],
    "fallback_reason": str | None,
    "policy_version": str,
}
```

输出：

```text
docs/agent_policy_contract_v1.md
```

---

## 5. P3-0.4 Legacy Policy Audit

新增：

```text
docs/legacy_policy_audit.md
```

至少审计：
- L3 crisis route
- L2 high-risk route
- third-party crisis route
- RecommendGate
- knowledge retrieval decision
- memory retrieval behavior
- ReAct entry
- clarification behavior
- information response behavior
- recommendation trigger
- safety_only behavior

每条规则记录：

| Rule ID | Source File | Input Signal | Condition | Output | Priority |
|---|---|---|---|---|---|

---

## 6. Legacy → ActionPlan 映射

新增：

```text
backend/policy/legacy_policy_adapter.py
```

实现：

```text
AgentState
↓
LegacyPolicyAdapter
↓
ActionPlan
```

这里只解释当前行为，不修改当前行为。

---

## 7. Current Policy Baseline

在冻结 Agent Benchmark v1.1 上跑 LegacyPolicyAdapter。

输出：

```text
evaluation/policy/reports/current_policy_baseline.md
```

指标：
- Primary Action Accuracy
- Primary Action Macro F1
- Safety Action Recall
- Tool Action Micro F1
- Tool Action Exact Match
- Recommendation Mode Accuracy
- Safety Target Accuracy
- Policy Exact Match

该结果作为 Phase 3 正式 baseline。

---

## 8. Policy Invariants / Precedence

输出：

```text
docs/policy_invariants_v1.md
```

至少冻结：

```text
Safety > Normal Conversation
Safety > Recommendation
Safety-only 不允许普通推荐
Primary Action 互斥
Tool Actions 可多选
Recommendation != Primary Action
Memory Retrieval != Primary Action
Knowledge Retrieval != Information Primary Action
```

优先级：

```text
P0 Hard Safety Invariants
P1 Deterministic Policy
P2 Learned Policy（若需要）
P3 Confidence/Ambiguity
P4 LLM Fallback
```

低优先级层不得覆盖高优先级 Safety。

---

## 9. Preflight 交付物

```text
docs/
├── phase2_regression_closeout.md
├── known_bug_001_content_recommender.md
├── agent_policy_contract_v1.md
├── legacy_policy_audit.md
└── policy_invariants_v1.md

backend/policy/
└── legacy_policy_adapter.py

evaluation/policy/reports/
└── current_policy_baseline.md
```

---

## 10. PASS 条件

```text
Frozen Benchmark 无行为回归
content_recommender Bug 已独立修复
AgentState → PolicyResult → ActionPlan Contract 已冻结
Legacy Policy 已完整映射
Current Policy Baseline 已生成
Safety precedence 已冻结
```

全部通过后：

```text
PHASE 3 IMPLEMENTATION = GO
```

---

## 11. Codex 启动指令

```text
请先执行 Phase 2 Closeout / Phase 3 Preflight。

严格按：
P3-0.1 → P3-0.4

执行。

不要提前实现新 Policy。

要求：
1. 实际重跑 Frozen Agent Benchmark v1.1 并生成 regression diff。
2. 单独修复 content_recommender missing import。
3. 冻结 AgentState v1 → PolicyResult → ActionPlan v1 契约。
4. 审计全部现有 State → Action 行为。
5. 将 legacy 行为映射到 ActionPlan。
6. 在冻结 Benchmark 上生成 Current Policy Baseline。
7. 冻结 Policy Invariants 和 Safety Precedence。

全部 PASS 后再执行 Phase 3 正式文档。
```
