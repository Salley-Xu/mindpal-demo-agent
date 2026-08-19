# Agent Policy Contract v1

> 对应 Phase 3 Preflight P3-0.3
> 日期：2026-08-19
> 代码：`backend/policy/schema.py`
> 状态：**冻结**

---

## 1. 输入：AgentState（唯一）

```text
AgentState v1（backend/state/schema.py）
```

Phase 3 Policy **禁止绕过 AgentState** 直接读取 perception/profile/database 私有变量。

## 2. 输出：ActionPlan（冻结）

```python
ActionPlan = {
    "primary_action": str,        # continue_chat | ask_clarification | information_response | safety_intervention
    "tool_actions": list[str],    # [retrieve_memory, retrieve_knowledge, recommend_resource] 任意子集
    "safety_target": str,         # none | self | third_party
    "recommendation_mode": str,   # none | soft | hard | safety_only
}
```

## 3. 运行时包装：PolicyResult（可 Trace）

```python
PolicyResult = {
    "action_plan": ActionPlan,
    "confidence": float,
    "source": "rule|classifier|llm|hybrid",
    "matched_rules": list[str],
    "fallback_reason": str | None,
    "policy_version": str,   # "1.0"
}
```

## 4. 枚举合法值

| 字段 | 合法值 |
|---|---|
| primary_action | continue_chat / ask_clarification / information_response / safety_intervention |
| tool_actions | retrieve_memory / retrieve_knowledge / recommend_resource（multi-select） |
| safety_target | none / self / third_party |
| recommendation_mode | none / soft / hard / safety_only |

## 5. 对齐

- 与 Benchmark v1.1 Schema 的 `ExpectedOutcome.primary_action / tool_actions / safety_target / recommendation_action` 同构
- 与 `docs/phase0_final_review.md` 冻结的 ActionPlan 接口一致

## 6. 约束

1. Safety 决策（safety_intervention）不得被低优先级层覆盖
2. LLM 永远不能覆盖 Safety
3. `primary_action` 互斥；`tool_actions` 可多选
4. `recommendation_mode` 是 Policy 输出，不是 State 字段
5. 所有决策必须记录 `source / confidence / matched_rules / fallback_reason`
