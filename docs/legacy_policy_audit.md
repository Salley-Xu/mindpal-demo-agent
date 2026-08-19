# Legacy Policy Audit

> 对应 Phase 3 Preflight P3-0.4
> 日期：2026-08-19
> 目的：审计当前全部 State→Action 行为，映射到 ActionPlan，作为 Phase 3 规则依据。

---

## 1. 规则清单

| Rule ID | Source File | 输入信号 | Condition | Output | Priority |
|---|---|---|---|---|---|
| R01 | agent_orchestrator.py | risk.level | level_3（紧急） | Safety intervention, target=self | P0 |
| R02 | agent_orchestrator.py | risk_context.subject + is_help_request | 第三方危机 + 求助 | Safety intervention, target=third_party | P0 |
| R03 | agent_orchestrator.py | risk.level | level_2（高风险） | Safety intervention（高危支持模式）, target=self | P0 |
| R04 | recommend_gate.py | emotion/risk/intent/trend/preference 加权 | score ≥ hard_threshold 等 | recommend_mode=hard/soft/none | P1 |
| R05 | recommend_gate.py | risk.level | level_2 | recommend_mode=safety_only（禁普通推荐） | P0 |
| R06 | recommend_gate.py | risk.level | level_3 | recommend_mode=none（禁推荐） | P0 |
| R07 | agent_orchestrator.py | 正常路由 | 非 L2/L3/第三方 | 进入 ReAct（continue_chat + tool 由 LLM 决定） | P2 |
| R08 | memory_context_builder | 正常路由 | 每轮无条件 | retrieve_memory（无检索门控） | P2 |
| R09 | agent_orchestrator._inject_knowledge_context | 正常路由 | 每轮尝试 | retrieve_knowledge（知识注入） | P2 |
| R10 | (无显式) | - | 无信息请求路由 | information_response 无确定性规则（LLM 决定） | - |
| R11 | (无显式) | - | 无澄清路由 | ask_clarification 无确定性规则（LLM 决定） | - |
| R12 | rejection_detector | 推荐反馈 | 检测到拒绝 | 影响后续 Gate 偏好分（非直接 action） | P1 |

## 2. 现有行为的 ActionPlan 映射

### 2.1 安全路由（确定性）

```text
L3           → ActionPlan(safety_intervention, [], self, none)
第三方危机    → ActionPlan(safety_intervention, [], third_party, safety_only)
L2           → ActionPlan(safety_intervention, [], self, safety_only)
```

### 2.2 正常路由（ReAct，LLM 决定内部动作）

```text
非安全     → ActionPlan(continue_chat, tool_actions=LLM 决定, none, recommend_mode=Gate 决定)
```

- memory/knowledge 检索每轮无条件触发（无门控）
- information_response / ask_clarification **无确定性规则**（当前依赖 LLM）

### 2.3 RecommendGate 输出 → recommendation_mode

```text
Gate hard   → recommendation_mode=hard
Gate soft   → recommendation_mode=soft
Gate none   → recommendation_mode=none
L2 锁      → recommendation_mode=safety_only
L3 锁      → recommendation_mode=none
```

## 3. 差距（Phase 3 需补的确定性规则）

| 现状 | 差距 |
|---|---|
| 无显式 information_response 规则 | Phase 3 需补：明确 information_request → information_response |
| 无显式 ask_clarification 规则 | Phase 3 需补：intent uncertain / 模糊 → ask_clarification |
| memory 每轮无条件检索 | Phase 3 需补：memory_reference 信号 → retrieve_memory 门控 |
| knowledge 每轮注入 | Phase 3 需补：knowledge-dependent info request → retrieve_knowledge |
| tool 动作由 LLM 自由决定 | Phase 3 需补：确定性 Tool Action 规则 |
