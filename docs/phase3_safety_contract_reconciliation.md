# Phase 3 Safety Contract Reconciliation

> 对应 Phase 3 Closeout C3.1
> 日期：2026-08-20
> 状态：**已对齐（方案 A）**

---

## 1. 合同 vs 实现的差距

`docs/policy_invariants_v1.md` §3 规定：

```text
self 高风险（L2/L3 或 high-risk intent）:
    primary=safety_intervention, target=self, rec=safety_only/none
```

而 Phase 3 实现（`backend/policy/safety.py`）在 Closeout 前只有：

```text
S01  risk.level >= 3
S02  third-party 危机
S03  risk.level == 2
```

**缺口**：合同允许 "high-risk intent" 单独触发 self safety，但实现只信任 `risk.level`。

---

## 2. 方案选择

### 方案 A（采用）— high_risk_intent 作为 Safety fallback

```text
high_risk_intent
AND NOT safe_denial
AND NOT discussion
→ candidate safety signal
```

- 第三方用 `is_third_party` 确定 target
- 已实现为 `S04_high_risk_intent`（self）/ `S04_high_risk_intent_third_party`（third_party）

### 方案 B — 只信任 RiskState 并改合同文档

不采用：保留 "high-risk intent" 触发符合心理安全领域常识
（意图层捕捉到的明确高风险表达不应被风险分级器漏检时丢弃）。

---

## 3. 实现（backend/policy/safety.py）

```python
# S04 high_risk_intent 兜底（对齐 Invariants §3）
if (state.derived.high_risk_intent_signal
        and not r.safe_denial
        and not r.is_discussion):
    if r.is_third_party:
        → safety_intervention, third_party, safety_only
    else:
        → safety_intervention, self, none
```

## 4. 一致性验证

| 检查 | 结果 |
|---|---|
| gold 中 high_risk_intent ⟹ risk>=2 反例数 | **0**（Frozen Benchmark v1.1） |
| S04 守卫（safe_denial/discussion）单元测试 | PASS（test_policy.py 新增） |
| Oracle-state 回归 | primary 0.9475 / safety recall 1.0 / exact 0.7818 不变 |
| FPR 独立验证 | **C3.3 Independent Safety Slice**（下文） |

## 5. 结论

**合同与实现已对齐。** 文档描述（high-risk intent 可触发 Safety）与实现（S04 fallback）一致，
不再有"文档说 A、实现做 B"的问题。FPR 风险交由 C3.3 独立 Safety Slice 量化。
