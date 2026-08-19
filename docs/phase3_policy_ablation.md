# Phase 3 Policy Ablation

> 对应 Phase 3 执行文档 §20
> 日期：2026-08-20
> 数据：Frozen Agent Benchmark v1.1（362 cases）
> 来源：`evaluation/policy/run_ablation.py` → `evaluation/policy/reports/policy_ablation_*.json`

---

## 1. 主表（Predicted-state 与 Oracle-state 分开报告）

### 1.1 Oracle-state（Policy 上限，gold 感知信号）

| Method | Primary Acc | Primary Macro F1 | Safety Recall | Safety Target Acc | Tool Micro F1 | Rec Mode Acc | Exact | LLM Rate | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A Legacy | 0.8011 | 0.4652 | 1.0 | 0.9917 | 0.1972 | 0.5414 | 0.1105 | 0% | 0.05ms |
| C Deterministic | **0.9475** | **0.8742** | **1.0** | **1.0** | **0.8679** | **0.8867** | **0.7818** | 0% | 0.05ms |
| F Final Hybrid | 0.9475 | 0.8742 | 1.0 | 1.0 | 0.8679 | 0.8867 | 0.7818 | **1.4%** | 0.05ms |

> Learned Policy（D）未实现 —— Deterministic 单独达标，按 §29 跳过。
> Pure LLM（B）为 100% LLM 的对照；由于 Deterministic 已达标且 LLM rate 目标 ≤20%，B 仅作参考不推荐。

### 1.2 Predicted-state（端到端，含感知误差）

| Method | State Channel | Primary Acc | Safety Recall | Tool Micro F1 | Exact |
|---|---:|---:|---:|---:|---:|
| A Legacy | rule | 0.6547 | 0.3134 | 0.1528 | 0.0193 |
| C Deterministic | rule | 0.6547 | 0.3134 | 0.1528 | 0.3591 |
| A Legacy | bert | 0.5249 | 0.6119 | 0.0625 | 0.0580 |
| C Deterministic | bert | 0.5249 | 0.6119 | 0.0625 | 0.3453 |

---

## 2. 关键观察

### 2.1 Policy 自身质量（oracle-state）

C Deterministic 与 F Hybrid **全部达到推荐目标**：

| 指标 | 目标 | Deterministic | 状态 |
|---|---:|---:|---|
| Primary Action Accuracy | ≥0.90 | 0.9475 | ✅ |
| Primary Macro F1 | ≥0.85 | 0.8742 | ✅ |
| Safety Recall | ≥0.98 | 1.0 | ✅ |
| Safety Target Acc | ≥0.95 | 1.0 | ✅ |
| Tool Micro F1 | ≥0.80 | 0.8679 | ✅ |
| Rec Mode Acc | ≥0.80 | 0.8867 | ✅ |
| Policy Exact Match | ≥0.75 | 0.7818 | ✅ |
| LLM Fallback Rate | ≤0.20 | 0.014 | ✅ |

### 2.2 端到端差距 = 感知瓶颈（不是 Policy）

Predicted-state 下，**Legacy 与 Deterministic 的 primary/safety/tool 指标完全相同**（同输入同输出）：
- 规则通道：safety recall 0.3134（两者一致）—— 粗规则无法识别隐式高风险
- BERT 通道：safety recall 0.6119（两者一致）—— 生产风险模型 FNR 39%（漏检）+ FPR（过度触发）

结论：**Phase 3 Policy 在给定感知信号下已做到最优路由；端到端短板在 Risk 感知（Raw BERT FNR/FPR，Phase 0.5 已定位，Phase 5 修复范围）。**

### 2.3 Exact Match 提升（Predicted-state）

- rule：0.0193 → 0.3591（+18×）
- bert：0.0580 → 0.3453（+6×）

Policy 收敛 tool/rec 决策后，即使感知信号受限，整决策匹配度也大幅提升。

### 2.4 LLM Fallback Rate

Oracle-state 下 ambiguity gate 仅触发 1.4%（5 cases）：
- 全部为 info+resource 冲突（`info_primary_with_hard_rec`）或低置信
- 远低于 20% 上限；生产 robustness 有充足余量

---

## 3. 结论

| 决策 | 结论 |
|---|---|
| Deterministic Policy | **达标，采用** |
| Learned Policy | **跳过**（§29：不强行加复杂度） |
| LLM Fallback | 保留，benchmark 触发率仅 1.4%，作为生产 ambiguous 兜底 |
| Final Hybrid | **= Deterministic（safety+deterministic）+ 1.4% LLM 兜底** |
