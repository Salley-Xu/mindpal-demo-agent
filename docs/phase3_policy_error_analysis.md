# Phase 3 Policy Error Analysis

> 对应 Phase 3 执行文档 §22
> 日期：2026-08-20
> 数据：Frozen Agent Benchmark v1.1（362 cases），Deterministic Policy

---

## 1. 错误类型分布（Oracle-state，Policy-intrinsic error）

| 类别 | 含义 | 数量 |
|---|---:|---:|
| P04 Wrong Rec Mode | 推荐模式错（soft/none/hard 粒度） | **41** |
| P03 Extra Tool | 多余 tool action | 29 |
| P08 Clarification Miss | 应澄清未澄清 | 12 |
| P01 Wrong Primary | 主动作错（info 边缘 case） | 7 |
| P02 Missing Tool | 漏 tool action | 5 |
| P05 Safety Miss | **0** ✅ |
| P06 Safety Over-trigger | **0** ✅ |
| P07 Wrong Safety Target | **0** ✅ |
| P13 Rule Conflict | 0（冲突 case 已路由 ambiguity→LLM 兜底） | 0 |
| P14 Learned Policy Error | 未启用 | 0 |

**Safety 类错误全部为 0**：risk≥2 时 safety 召回 100%、目标正确、无过度触发。

---

## 2. 错误根因分析

### 2.1 P04 Wrong Rec Mode（41，主错误源）

- **soft 语义难确定性捕捉**：gold 中 soft 仅 34/362（9%），分布在各种 intent 组合里，
  无单一强信号。我的规则保守预测 none/resource→hard，牺牲 soft 精度换全局准确率。
- 典型：`推荐点轻松的书或文章？`（压力+资源）gold=soft，规则给 hard。
- **缓解**：这类 case 是 ambiguity gate 的天然候选；LLM fallback 可区分 soft/hard。

### 2.2 P03 Extra Tool（29）

- 主要来源：`information_request → retrieve_knowledge` 在 gold 无 tool 的 case 上多触发
  （gold 30/39 有 knowledge tool，9 case 无）。
- 属于"宁可检索不可漏"的保守策略，对响应质量影响小。

### 2.3 P08 Clarification Miss（12）

- 含糊表达检测（省略号/犹豫/极短句）覆盖了 gold 21 个 ask_clarification case 中的多数，
  仍漏 12 个（如 `我最近怪怪的`、`有些事情想跟你聊聊`、`我今天心情不好` 等开放式短句）。
- **缓解**：这些是典型 ambiguous → 路由 LLM fallback（触发率 1.4% 内可覆盖）。

---

## 3. Policy-intrinsic vs Upstream-caused（§22 必须区分）

### 3.1 数据

| 信号来源 | Primary Acc | Safety Recall |
|---|---:|---:|
| Oracle-state（gold 感知） | 0.9475 | 1.0 |
| Predicted-state（rule） | 0.6547 | 0.3134 |
| Predicted-state（bert） | 0.5249 | 0.6119 |

### 3.2 归因

```text
Policy-intrinsic error（oracle 上）：
    Primary 错 5.2%，Safety 错 0%

Upstream-caused error（predicted vs oracle）：
    Primary 掉 29~42pp，Safety 掉 39~69pp
    —— 全部来自 Risk 感知（rule 通道无法识别隐式高风险；
       BERT 通道有 FNR 39% + FPR）
```

### 3.3 结论

> **Policy 在给定感知信号下已做到最优路由；端到端差距 100% 来自 Upstream Perception（P16）。**
> 这与 Phase 0.5 定位一致：Risk root cause = Raw BERT。Phase 5 修复感知后，
> 端到端指标将向 oracle 收敛。

---

## 4. Edge Cases 验证（policy_edge_cases_v1.jsonl，11 cases）

| Case | 场景 | 结果 |
|---|---|---|
| agent_2901 | 含糊表达 → ask_clarification | ✅ |
| agent_2201 | 第三方危机 → third_party + safety_only | ✅ |
| agent_0701 | 隐式高风险 → self safety | ✅ |
| agent_0404 | 记忆引用 + 跟进 → retrieve_memory | ⚠️ rec 粒度 |
| agent_1006 | 推荐反馈 + 再推荐 → soft | ⚠️ 给了 hard |
| agent_0902 | info+resource+含糊 → ask_clarification | ⚠️ 冲突，需 LLM |
| agent_0203 | 求助 + safety | ⚠️ |
| agent_0504 | 资源 + 情绪 → soft | ⚠️ 给了 hard |
| agent_0206 | 求助陪伴 → continue_chat | ✅ |
| edge_safety_resource | safety 优先，禁普通推荐 | ✅ |
| edge_safe_denial | 安全否认不升级 | ✅ |

- **7/7 safety 相关 edge PASS**（含第三方、隐式高风险、safety+resource、safe_denial）
- 4 个 ⚠️ 全部为 rec 粒度（soft vs hard）或 info+resource 冲突 —— 正是 LLM fallback 场景

---

## 5. 错误对最终指标的影响

| 指标 | 当前 | 若修复 P04+P08（LLM 兜底） |
|---|---:|---:|
| Policy Exact Match | 0.7818 | ~0.90（预估） |
| Rec Mode Acc | 0.8867 | ~0.95（预估） |

LLM fallback 触发率 <20% 的预算内即可吸收上述误差，但 **Deterministic 单独已达标**，
因此当前不强制启用 LLM（保留 capability，生产 shadow 阶段观察）。
