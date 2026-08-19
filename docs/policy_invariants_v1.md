# Policy Invariants v1

> 对应 Phase 3 Preflight P3-0.4（§8）
> 日期：2026-08-19
> 状态：**冻结**

---

## 1. 硬性不变量（任何 Policy 层都必须满足）

```text
INV-01  Safety > Normal Conversation    安全决策优先于普通对话
INV-02  Safety > Recommendation         安全时不允许普通推荐
INV-03  Safety-only 不允许普通推荐       recommendation_mode=safety_only 时禁 hard/soft
INV-04  Primary Action 互斥             一个 State 只能有一个 primary_action
INV-05  Tool Actions 可多选             tool_actions 是集合，不互斥
INV-06  Recommendation ≠ Primary        recommendation_mode 独立于 primary_action
INV-07  Memory Retrieval ≠ Primary      retrieve_memory 是 tool，不是 primary
INV-08  Knowledge Retrieval ≠ Info Primary  retrieve_knowledge 可与 information_response 共存
INV-09  Safety 不被 LLM 覆盖            LLM 输出必须经过 Safety override 验证
INV-10  third-party target 正确          第三方危机必须 safety_target=third_party
INV-11  discussion / safe denial 不强行升级  讨论/安全否认不被 Safety 规则误伤
```

## 2. 优先级（低层不得覆盖高层 Safety）

```text
P0  Hard Safety Invariants      （INV-01~03, 09~11）
P1  Deterministic Policy
P2  Learned Policy（若需要）
P3  Confidence / Ambiguity
P4  LLM Fallback
```

## 3. Safety 行为规范

```text
self 高风险（L2/L3 或 high-risk intent）:
    primary=safety_intervention, target=self, rec=safety_only/none

third-party 危机:
    primary=safety_intervention, target=third_party, rec=safety_only

讨论语境 / 安全否认:
    不得强行升级为 safety_intervention
```

## 4. Validator 职责

每个 PolicyResult 必须通过：

```text
1. Schema validation（枚举合法）
2. Policy invariant validation（INV-01~11）
3. Safety override validation（LLM 结果不得覆盖 Safety）
```
