# Phase 3 Policy Design — Hybrid Agent Policy v1

> 对应 Phase 3 执行文档 §2-§9
> 日期：2026-08-20
> 状态：**已实现，冻结中**

---

## 1. 设计目标

把散落在 orchestrator、risk routing、RecommendGate、ReAct 前置逻辑里的决策，
统一收敛为显式分层决策引擎：

```text
AgentState v1
    ↓
AgentPolicy v1
    ↓
ActionPlan v1
```

**不是** Multi-Agent / Planner / LLM Router Demo，而是统一 Agent 决策层。

---

## 2. 最终架构

```text
AgentState
    ↓
P0 Safety Invariants   （backend/policy/safety.py）
    ↓
P1 Deterministic Rules （backend/policy/deterministic.py）
    ↓
P2 Learned Policy      （backend/policy/classifier.py，Deterministic 达标则关闭）
    ↓
P3 Confidence/Ambiguity（backend/policy/ambiguity.py）
    ↓
P4 LLM Fallback        （backend/policy/llm_fallback.py）
    ↓
ActionPlan             （backend/policy/schema.py + validator.py + hybrid.py）
```

原则：**高优先级决策不可被低优先级覆盖**。

---

## 3. 分层规则

### 3.1 P0 Safety（backend/policy/safety.py）

| Rule | 条件 | Primary | Safety Target | Rec Mode |
|---|---|---|---|---|
| S01 | risk.level >= 3 | safety_intervention | self | none |
| S02 | 第三方危机（is_third_party + 高风险信号） | safety_intervention | third_party | safety_only |
| S03 | risk.level == 2 | safety_intervention | self | none |

INV-11 保护：`safe_denial` / `discussion` 语境不强行升级。

设计依据（Frozen Benchmark v1.1 gold 分析）：
- `risk >= 2 ⟺ primary=safety_intervention`（362 中 0 反例）
- third_party 全部带 `high_risk_expression` 且文本含第三方主体词
- L2 self 的 gold rec 以 none 为主（24/31）；safety_only 仅用于第三方危机（9/10）

### 3.2 P1 Deterministic Primary Action

| Rule | 条件 | 输出 | gold 依据 |
|---|---|---|---|
| D01 | intent 含 information_request | information_response | 36/39 |
| D02 | intent 含 meta_question | information_response | 18/21 |
| D03 | 含糊表达（省略号/犹豫/极短句） | ask_clarification | gold 21 case 主信号 |
| D04 | 否则 | continue_chat | 223/362 主导 |

### 3.3 P1 Deterministic Tool Actions

| Rule | 条件 | Tool |
|---|---|---|
| D05 | resource_request | recommend_resource（45/48） |
| D06 | memory_reference | retrieve_memory（40/46） |
| D07 | information_request | retrieve_knowledge（30/39） |

### 3.4 P1 Recommendation Mode

gold：rec 以 none 为主（276/362），soft 是少数派。

| Rule | 条件 | Rec Mode |
|---|---|---|
| D08 | resource_request | hard（37/48） |
| D09 | rejection_detected | none（抑制） |
| D10 | 否则 | none |

### 3.5 P3 Ambiguity Gate

触发（backend/policy/ambiguity.py）：
- 规则冲突（info_primary_with_hard_rec）
- 主动作置信 < 0.72
- intent uncertain / open-set
- primary action tie

### 3.6 P4 LLM Fallback

仅 ambiguous case 调用。输入 AgentState 压缩序列化 + 允许动作 Schema + 不变量，
输出严格 JSON，必须通过 Validator（Schema + INV + Safety override）。

LLM **永远不能覆盖 Safety**：risk>=2 时 P0 先行返回，P4 不被调用。

---

## 4. Validator（backend/policy/validator.py）

每个 PolicyResult 校验 INV-01 ~ INV-11：
- INV-03 safety_only 不允许普通推荐
- INV-01/02 safety_intervention 必须带 safety_target
- INV-10 third_party 必须 target=third_party
- INV-11 safe_denial/discussion 不强行升级

---

## 5. Config（backend/policy/engine.py PolicyConfig）

```yaml
agent_policy:
  version: "1.0"
  mode: "shadow"              # legacy | shadow | hybrid
  enable_safety_rules: true
  enable_deterministic: true
  enable_learned_policy: false   # Deterministic 达标，不启用
  enable_llm_fallback: true
  max_llm_fallback_rate: 0.20
```

---

## 6. Shadow 集成（backend/policy/shadow.py）

- Legacy Policy → 实际生产行为（不动）
- New Policy → Shadow ActionPlan（旁路）
- 记录 legacy/new ActionPlan + diff 分类（same / different / semantic_equivalent）
- 写入 `logs/policy_shadow_trace.jsonl`
- orchestrator 中 Shadow 部分嵌套 try/except，**绝不干扰生产**

---

## 7. 关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| Learned Policy | 跳过 | Deterministic 单独达标（§29），不强行加复杂度 |
| LLM Fallback | 保留但默认低触发 | 生产 robustness，benchmark LLM rate 仅 ~1.4% |
| L2 rec | none（非 safety_only） | gold 24/31，且满足 INV-02（无普通推荐） |
| 含糊表达 | 确定性 D03 + 文本标记 | 覆盖 gold ask_clarification 主信号，不依赖 LLM |
| Eval 双轨 | oracle + predicted | 区分 Policy-intrinsic 与 upstream perception error（§21） |
