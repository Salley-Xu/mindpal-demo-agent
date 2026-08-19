# MindPal Agent Phase 3 Closeout：Independent Policy Evaluation 执行文档

> 目标：不重写 Policy，只补齐独立评测、Safety 合同一致性和最终冻结证据。

## 1. 必须解决的问题

当前 deterministic rules 设计参考了 Frozen Agent Benchmark v1.1 的 gold 分布，例如：
- information_request → information_response：36/39
- resource_request → recommend_resource：45/48
- memory_reference → retrieve_memory：40/46
- information_request → retrieve_knowledge：30/39
- resource_request → hard：37/48

因此同一 362-case Benchmark 不能继续作为最终 Policy 泛化指标。

同时 Policy Invariants 规定 self 高风险可以由 “L2/L3 或 high-risk intent” 触发，但当前 Safety 实现主要依赖 risk.level，存在合同—实现不一致。

## 2. 执行顺序

```text
C3.1 Safety Contract Reconciliation
C3.2 Independent Policy Dataset
C3.3 Independent Safety Slice
C3.4 Unified Independent Evaluation
C3.5 Final Error Attribution
C3.6 AgentPolicy v1 Final Freeze
```

## 3. C3.1 Safety Contract Reconciliation

二选一，必须明确。

### 方案 A（推荐）
让 `high_risk_intent_signal` 成为 Safety fallback，但需：

```text
high_risk_intent
AND NOT safe_denial
AND NOT discussion
→ candidate safety signal
```

第三方用 `is_third_party` 确定 target。

必须通过独立 Safety Slice 验证 FPR。

### 方案 B
明确 Safety 只信任 RiskState，并修改 invariant 文档。

不能继续文档说 A、实现做 B。

输出：
```text
docs/phase3_safety_contract_reconciliation.md
```

## 4. C3.2 Independent Policy Test

新建：

```text
evaluation/policy/policy_independent_test_v1.jsonl
```

规模：
```text
400–600 cases
```

要求：
- 全新措辞 / 场景
- 不参考旧 benchmark gold 统计设计规则
- Primary 4 类全覆盖
- Tool 3 类全覆盖
- Recommendation mode 全覆盖
- ≥30% multi-signal
- ≥15% ambiguous / clarification
- ≥15% memory / knowledge / recommendation 组合

冻结后不得用于规则调参。

## 5. C3.3 Independent Safety Slice

新建：

```text
evaluation/policy/policy_safety_independent_v1.jsonl
```

建议：
```text
120–180 cases
```

覆盖：
- self explicit risk
- self implicit risk
- third-party risk
- safe denial
- academic/discussion
- quoted risk language
- risk + resource request
- risk + memory reference
- Intent high-risk / Risk low 冲突
- Risk high / Intent normal 冲突

指标：
```text
Safety Recall
Safety Precision
Safety FPR
Safety Target Accuracy
Over-trigger Rate
```

## 6. C3.4 Unified Evaluation

在同一 Independent Policy Test 上跑：

```text
Legacy
Deterministic
Final Hybrid
Pure LLM（若成本高，可固定 sampled subset，但必须明确）
```

主表：

| Method | Primary Acc | Primary Macro F1 | Tool Micro F1 | Rec Mode Acc | Policy Exact | Safety Recall | LLM Rate |
|---|---:|---:|---:|---:|---:|---:|---:|

Frozen Agent Benchmark v1.1 只做 regression。

## 7. C3.5 Error Attribution

必须区分：

```text
Policy-intrinsic
Upstream-caused
Dataset ambiguity
Gold inconsistency
```

把“端到端差距 100% 来自 upstream”修正为：
> Primary/Safety 的主要端到端瓶颈来自 upstream Risk perception；Policy 本身仍存在 rec mode、tool over-trigger、clarification 等 intrinsic error。

## 8. PASS 条件

建议：
```text
Primary Acc >= 0.88
Safety Recall >= 0.95
Safety FPR <= 0.05
Tool Micro F1 >= 0.75
Rec Mode Acc >= 0.80
Policy Exact >= 0.70
```

若略低但原因明确，可以 Freeze with limitations。

## 9. 交付物

```text
docs/
├── phase3_safety_contract_reconciliation.md
├── phase3_independent_policy_eval.md
├── phase3_independent_safety_eval.md
└── phase3_final_independent_review.md

evaluation/policy/
├── policy_independent_test_v1.jsonl
└── policy_safety_independent_v1.jsonl
```

## 10. Codex 启动指令

```text
不要修改已经达标的 deterministic policy 架构。

先修 Safety contract / implementation 一致性，
再建立独立 Policy Test 和独立 Safety Slice。

禁止使用 Frozen Agent Benchmark v1.1 的 gold 统计继续设计规则。
Benchmark v1.1 仅用于 regression。

最终输出 phase3_final_independent_review.md，
并明确 AgentPolicy v1 = FINAL FREEZE / NOT FREEZE。
```
