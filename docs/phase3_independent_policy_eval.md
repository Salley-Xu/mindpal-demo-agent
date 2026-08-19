# Phase 3 Independent Policy Evaluation

> 对应 Phase 3 Closeout C3.2 / C3.4 / C3.5
> 日期：2026-08-20
> 数据：`evaluation/policy/policy_independent_test_v1.jsonl`（430 cases，全新措辞）

---

## 1. 为什么需要独立测试

Phase 3 deterministic rules 设计参考了 Frozen Agent Benchmark v1.1 的 gold 分布
（如 information_request → information_response：36/39），因此 362-case Benchmark
属于 **design-benchmark performance**，不能作为最终泛化指标。

`policy_independent_test_v1.jsonl` 为全新措辞/场景，语义真实标注，**未参与规则设计**。
Frozen Benchmark v1.1 仅用于 regression。

## 2. 数据集构成（430 cases）

| 维度 | 覆盖 | 要求 |
|---|---|---|
| 总量 | 430 | 400-600 ✓ |
| Primary 4 类 | continue 223 / info 80 / safety 59 / ask 68 | 全覆盖 ✓ |
| Tool 3 类 | memory / knowledge / resource | 全覆盖 ✓ |
| Rec Mode 4 类 | none 232 / hard 113 / safety_only 42 / soft 43 | 全覆盖 ✓ |
| multi-signal | 0.41 | ≥0.30 ✓ |
| ambiguous | 0.16 | ≥0.15 ✓ |
| tool 组合 | 0.22 | ≥0.15 ✓ |

## 3. 主表（C3.4）

### 3.1 Oracle-state（Policy 泛化上限）

| Method | Primary Acc | Safety Recall | Tool Micro F1 | Rec Mode Acc | Exact | LLM Rate |
|---|---:|---:|---:|---:|---:|---:|
| A Legacy | 0.6558 | 1.0 | 0.3468 | 0.6721 | 0.0721 | 0% |
| C Deterministic | **0.8837** | **1.0** | **0.995** | **0.9000** | **0.7767** | 0% |
| F Final Hybrid | 0.8837 | 1.0 | 0.995 | 0.9000 | 0.7767 | **6.1%** |

### 3.2 Predicted-state（rule 通道，端到端）

| Method | Primary Acc | Safety Recall | Tool Micro F1 | Rec Mode Acc | Exact |
|---|---:|---:|---:|---:|---:|
| A Legacy | 0.5488 | 0.2542 | 0.3209 | 0.4186 | 0.0047 |
| C Deterministic | 0.5488 | 0.2542 | 0.1016 | 0.5698 | 0.1302 |
| F Final Hybrid | 0.5488 | 0.2542 | 0.1016 | 0.5698 | 0.1302 |

## 4. 结论

1. **Deterministic 泛化到全新措辞依然显著优于 Legacy**：
   oracle Exact 0.7767 vs 0.0721，Rec 0.90 vs 0.67，Tool 0.995 vs 0.35。
   证明 Policy 规则不是 memorize benchmark 模板。

2. **Oracle-state 全部达到 PASS 条件**：
   Primary ≥0.88（0.8837）/ Safety Recall ≥0.95（1.0）/ Tool ≥0.75（0.995）/
   Rec ≥0.80（0.90）/ Exact ≥0.70（0.7767）→ **PASS**。

3. **Predicted-state 端到端差距 = 感知瓶颈**（与 Frozen Benchmark 结论一致）：
   legacy 与 deterministic 在 predicted 下 primary/safety 完全相同（0.5488 / 0.2542），
   规则通道无法识别隐式风险（FNR）+ 关键词对否定盲区（FPR）。

## 5. 误差归因（C3.5，oracle-state）

| 错误类型 | 数量 | 归因 |
|---|---:|---|
| PA ask_clarification→continue | 50 | **Policy-intrinsic**：含糊标记（省略号/犹豫）不泛化到新措辞；LLM fallback 设计场景 |
| RM soft→none | 43 | **Policy-intrinsic**：soft 是少数类（10%），无单一强信号；LLM 兜底场景 |
| Tool extra | 3 | Policy-intrinsic（保守多检索） |
| PA info_extra | 0 | gold 标签修正后归零（见 §6） |
| Safety miss / over-trigger | 0 | **零安全误差** |

### 5.1 修正后的归因表述

> ~~端到端差距 100% 来自 upstream~~
>
> **Primary/Safety 的主要端到端瓶颈来自 upstream Risk perception；
> Policy 本身仍存在 ask_clarification 泛化不足与 rec soft 粒度等 intrinsic error
> （两者合计 ~21% 决策误差，均为 LLM fallback 的目标场景）。**

## 6. 数据修正记录

生成初期 `G10_mem_info`（"还记得X吗？想深入了解下Y"）gold 误标为 continue_chat，
但其 intent 含 information_request 且文本明确请求信息 → 修正为 information_response。
属**生成数据 gold 标签不一致**，非 Policy 行为问题。修正后独立 oracle primary 0.81 → 0.8837。
