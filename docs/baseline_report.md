# Agent Benchmark v1 Baseline Report

> 对应开发计划 Phase 0（Task 0.2 – 0.6）
> 日期：2026-08-18
> Benchmark：`evaluation/datasets/agent_benchmark_v1.jsonl`（100 条，人工撰写）
> 评测框架：`evaluation/`（schema / predictors / metrics / runners）

---

## ⚠️ 重要更新（Phase 0.5，2026-08-18）

本报告原始内容基于 v1（100 条）。Phase 0.5 已完成 **Schema v1.1 修正** 与 **Benchmark 扩充至 362 条**，正式 baseline 以 v1.1 为准：

- Schema v1.1：Action 拆分为 `primary_action` + `tool_actions` + `safety_target`（见 `docs/benchmark_schema_v1_1.md`）
- Benchmark v1.1：`evaluation/datasets/agent_benchmark_v1_1.jsonl`（362 条，L2+L3=67）
- 完整 v1.1 指标：`evaluation/reports/baseline_v1_1_bert_bert_*.md` / `baseline_v1_1_rule_rule_rule_*.md` / `risk_ablation_*.md`
- 根因结论：见 `docs/risk_pipeline_audit.md`（Raw BERT 本身 FPR=0.43/FNR=0.37 → 情况 B）
- 最终决策：见 `docs/phase0_final_review.md`（Benchmark Freeze = PASS）

### v1.1 主表（362 条）

| 模块 | 指标 | Current (BERT) | Rule Baseline |
|---|---:|---:|
| Risk | Macro F1 | **0.371** | 0.399 |
| Risk | High-risk Recall | **0.612** | 0.313 |
| Risk | FPR | **0.437** | 0.041 |
| Emotion | Coarse Accuracy / Macro F1 | **0.597 / 0.475** | 0.483 / 0.332 |
| Legacy Intent | Coverage Macro F1 | **0.161** | 0.143 |
| Recommendation | Trigger Precision / Recall | **0.172 / 0.329** | 0.333 / 0.408 |
| Primary Action | Accuracy | **0.528** | 0.655 |
| Safety | Primary Action Recall | **0.627** | 0.313 |
| Tool Action | Micro F1 / Exact Match | **0.161 / 0.354** | 0.178 / 0.193 |
| Memory Behavior | Precision / Recall | **0.147 / 0.609** | 0.128 / 0.913 |
| Risk Trend | Accuracy（多轮 n=23） | **0.304** | 0.000 |

**v1.1 关键新发现**：
1. 规则基线的高风险召回从 v1 的 0.60 **暴跌到 0.31** —— 证明新增的隐式高风险表达（"如果明天不用醒来就好了"等）确实提升了难度。
2. BERT 与规则呈 P/R 权衡：BERT 召回高（0.61）但 FPR 高（0.44）；规则精度高（FPR 0.04）但召回低（0.31）。**两者都远未达到安全召回 ≥0.98 的目标。**
3. 风险趋势判定（多轮）在 23 条上 Acc=0.30（BERT）/0.0（规则）—— SessionRiskAggregator 的趋势信号在短窗口下不可靠。

---


---

## 1. 执行摘要

本轮是改造前的**系统级评测基线**，回答四个核心问题：

1. **Agent 是否做出正确 Action？** → 当前无显式 Action Space；确定性路由仅覆盖安全分级。
2. **Agent 是否使用了正确 Memory？** → 当前"每轮都检索"（无检索门控），门控行为 Precision=0.12。
3. **Agent 是否在正确时机推荐/干预？** → 推荐 Trigger Recall=0.46，Over-Recommendation 26/100；安全干预 Recall=0.50。
4. **模块能力如何？** → 见下表。

**最核心的发现：当前 BERT 风险模型（v4_2_domain_only_v2，生产配置）在该 Benchmark 上表现劣于简单关键词规则。** 它把大量无害文本判为高风险（`"你在吗？我就随便聊聊。"` → Level 3，置信度 0.999），同时漏判一半真实高风险。这与此前 `eval_output/RISK_REPORT.md`（2026-07-06）"模型尚未达到投产标准"的结论一致，且问题仍然存在。

| 模块 | 指标 | **BERT 通道（当前系统）** | Rule 通道（对照） |
|---|---:|---:|---:|
| Risk | Macro F1 | **0.294** | 0.519 |
| Risk | High-risk Recall (L2+L3) | **0.500** | 0.600 |
| Risk | False Positive Rate | **0.422** | 0.044 |
| Risk | False Negative Rate | **0.500** | 0.400 |
| Emotion | Coarse 5-class Accuracy | **0.54** | 0.51 |
| Emotion | Fine Macro F1 | **0.119** | 0.219 |
| Intent | Macro F1 / Micro F1 | **0.191 / 0.421** | 0.188 / 0.397 |
| Recommendation | Trigger Precision / Recall | **0.391 / 0.462** | 0.513 / 0.513 |
| Recommendation | Mode Accuracy | **0.32** | 0.51 |
| Routing | Action Accuracy | **0.43** | 0.51 |
| Routing | Safety Action Recall | **0.50** | 0.60 |
| Memory Gate | Precision / Recall | **0.123 / 0.700** | 0.111 / 1.000 |
| Risk Trend | Accuracy（多轮 n=9） | **0.444** | - |

> 注：BERT 通道 = `--emotion bert --risk bert`（真实模型）；Rule 通道 = 复刻当前关键词规则。两者都是**离线确定性**评测，不涉及 LLM。

---

## 2. 评测设置

### 2.1 Benchmark 数据（100 条）

- 分布：L0=62 / L1=28 / L2=4 / L3=6；单轮 90 / 多轮 10
- Intent Taxonomy v1 的 10 个标签全部覆盖（multi-label）
- Agent Action Space v1 的 6 个动作全部覆盖
- 场景标签：情绪表达 / 显式求助 / 信息请求 / Memory 引用 / 推荐 / 反馈 / 高风险 / 模糊多标签 / 长多轮
- 全部为人工撰写模板（source=template），可审查

### 2.2 评测对象与限制

- **评测对象**：当前系统确定性决策模块（emotion→risk→gate→route），离线调用真实模块：
  - 情绪：`bert_emotion_predictor`（5 类）
  - 风险：`risk_evaluator`（BERT v4.2 + SessionRiskAggregator v5.0 状态机）
  - 意图：复刻 `emotion_analyzer._detect_user_intent` 关键词规则（当前系统唯一意图信号）
  - 门控：`recommend_gate`（真实规则加权）
- **未覆盖**（需 LLM / 需后续 Phase）：
  - ReAct Agent Loop 的 LLM 行为（LLM 是否调用工具、回复质量）
  - LLM 情绪深层通道（仅测 BERT 表层）
  - 记忆检索质量（memory_items 未预置，只测"门控行为"）
- **多轮处理**：逐轮累积 `recent_risk_levels` 到 summary 模拟会话级聚合；最终轮为预测对象。

### 2.3 复现命令

```bash
# 完整 BERT baseline
python evaluation/runners/run_agent_eval.py --predictor module --emotion bert --risk bert --report-name v1

# Rule 对照
python evaluation/runners/run_agent_eval.py --predictor module --emotion rule --risk rule --report-name rule

# 从已存 JSON 重算报告（不重跑模型）
python evaluation/runners/run_agent_eval.py --from-json evaluation/reports/baseline_v1_bert_bert_<ts>.json
```

---

## 3. 分模块分析

### 3.1 Risk（最差模块 ❌）

| level | P | R | F1 | tp/fp/fn |
|---|---:|---:|---:|---:|
| L0 | 0.941 | 0.258 | 0.405 | 16/1/46 |
| L1 | 0.475 | 0.679 | 0.559 | 19/21/9 |
| L2 | **0.000** | **0.000** | **0.000** | **0/30/4** |
| L3 | 0.154 | 0.333 | 0.211 | 2/11/4 |

**问题**：
- **Level 2 完全不可用**：30 个预测全部落空，0 命中 → 系统实际在 L1 与 L3 之间跳跃。
- **严重过触发**：62 个期望 L0 中 29 个被判 L2/L3（FPR=0.42）。直接导致：
  - 大量正常对话被路由到高风险支持模式（用户体验差）
  - `recommend_gate` 被 `safety_only` 锁死，正常推荐被抑制
- **漏判真实高风险**：10 个期望 L2/L3 中 5 个被压到 L0/L1（如"最近总想伤害自己"→L1、"快崩溃了"→L1）。

**结论**：风险模块信号不可靠，是**全系统最薄弱环节**，且劣于简单关键词基线。Phase 5（Risk 2.0）必须优先。

### 3.2 Emotion

- 细粒度 Macro F1 = 0.119（当前 BERT 只能输出 5 类，而 Benchmark 有 15+ 细粒度标签）
- 粗粒度 5 类 Accuracy = 0.54，主要错误：
  - `stress` → neutral / anxiety / sadness（无归属）
  - `hopelessness` → sadness（合理近似但丢失严重程度）
  - `fatigue` → neutral（完全漏掉）
- 对比：Rule 通道细粒度 Macro F1 = 0.219 > BERT 0.119

**结论**：当前 BERT 情绪模型对"焦虑/抑郁"大类的粗分类可用，但无法表达压力/疲劳/绝望/孤独等细粒度情绪，且 `stress` 类大量归入 neutral。Phase 1 情绪侧需扩展标签空间或依赖 LLM 通道。

### 3.3 Intent（架构缺口 ⚠️）

- 当前系统**没有独立 intent 分类器**，唯一信号是嵌入在情绪分析中的 4 类关键词规则。
- Benchmark 10 标签中 **6 个零召回**：`meta_question / feedback / follow_up / information_request / memory_reference / high_risk_expression`
- Macro F1 = 0.191

**结论**：这是明确的**架构缺口**而非模型调优问题。Phase 1（Intent Recognition）需从零建立分类器 + LLM fallback。

### 3.4 Recommendation

- Trigger Precision / Recall = 0.39 / 0.46；Mode Accuracy = 0.32
- **Over-Recommendation 26 条**：期望 none，Gate 却触发 soft —— 根因是 BERT 风险过触发 + 情绪强度偏高，导致 `safety_only`/`soft` 过度触发
- **Miss-Recommendation 7 条**：显式求助但未触发
- **hard 从未命中**（0/13）：Gate 的 hard 阈值（0.58）在当前输入特征分布下几乎不可达

**结论**：推荐门控依赖的情绪/风险信号不可靠，导致门控决策整体漂移。Phase 6（Recommendation 2.0）需先在 Phase 5 修复风险信号。

### 3.5 Agent Routing / Safety

- Action Accuracy = 0.43；Safety Action Recall = 0.50（10 个期望安全干预中漏 5 个）
- 当前确定性路由 = 仅安全分级（L2/L3→安全干预，否则 ReAct 循环），**无显式 Action Space**
- 漏判清单：`伤害自己`、`朋友想走极端`、`情绪低落+拒食`、`快崩溃`、`活着真累` 均被判 continue_chat

**结论**：安全底线 Recall=0.50 **不达标**（目标 ≥0.98）。风险信号 + 路由两层都需要重建。

### 3.6 Memory Gate

- Precision = 0.12，Recall = 0.70
- 当前系统**每轮都检索记忆**（MemoryContextBuilder 在正常路由无条件调用），造成严重过度检索

**结论**：印证审计发现的"无检索门控"。Phase 4（Memory 2.0）的 Retrieval Gate 是明确改进点。

### 3.7 Risk Trend（多轮）

- Accuracy = 0.444（n=9）。SessionRiskAggregator 的趋势判定在短窗口下不稳定。

---

## 4. 主要错误类型（对齐 Error Taxonomy §12.4）

| 错误码 | 类型 | 数量/现象 | 根因 |
|---|---|---|---|
| E03 | Risk False Negative | 5/10 高风险漏判 | BERT 风险模型 + 会话聚合不足 |
| E04 | Risk False Positive | 29/62 低风险误报 L2/L3 | BERT 风险模型过触发 |
| E01 | Intent Error | 6 类零召回 | 无独立 intent 分类器 |
| E09 | Over Recommendation | 26 条 | 风险过触发 + 门控信号漂移 |
| E10 | Missing Recommendation | 7 条 | hard 阈值不可达 + 门控抑制 |
| E13 | Wrong Routing | Action Acc=0.43 | 无显式 Action Space |
| E15 | Under Safety | Safety Recall=0.50 | 风险漏判传导到路由 |
| E08 | Memory Hallucination / 过度检索 | 门控 P=0.12 | 无检索门控 |

---

## 5. 最差 Top 3 模块

1. **Risk（BERT 风险模型）**：Macro F1=0.29，既过触发又漏判，劣于规则基线 —— **最高优先级**
2. **Intent**：架构性缺失，6/10 标签零召回 —— **Phase 1 必做**
3. **Recommendation Gate**：依赖不可靠信号，Mode Acc=0.32，over-rec 26 条 —— **依赖 Risk 修复**

---

## 6. 对后续 Phase 的启示与优先级

| 优先级 | 行动 | 依据 |
|---|---|---|
| P0 | **Phase 5 Risk 2.0**：多轮状态 + 校准/融合（先对比 v4_3_coral 或增强规则兜底） | Risk 是最差模块，且直接拖垮 Gate 与 Routing |
| P1 | **Phase 1 Intent**：独立分类器 + 置信度 + LLM fallback | 6 类零召回，架构缺口 |
| P1 | **Phase 2 AgentState + Phase 3 Policy**：统一状态 + Hybrid 路由 | Routing Acc=0.43，Safety Recall 不达标 |
| P1 | **Phase 4 Memory 2.0**：检索门控 + 冲突消解 | 门控 P=0.12，无治理 |
| P2 | **Phase 6 Recommendation 2.0**：反馈闭环 + 特征排序 | 依赖 Risk 修复后才有意义 |
| P2 | **Phase 7 Observability**：统一 Trace | 本轮已发现 Replay 需求 |

**关键结论**：改造的第一优先级不是加模块，而是**修复/重建风险信号链路**（Risk 2.0）。在风险信号修复之前，推荐门控、安全路由、Agent Policy 的验证都会被信号噪声污染。

---

## 7. 评测框架交付物

| 文件 | 说明 |
|---|---|
| `evaluation/benchmark_schema.py` | Schema（Pydantic + 枚举），供生成与校验 |
| `evaluation/datasets/agent_benchmark_v1.jsonl` | 100 条人工 Benchmark |
| `evaluation/datasets/generate_benchmark_v1.py` | 数据集生成器（可复现） |
| `evaluation/predictors/module.py` | ModulePredictor（BERT/Rule 双通道） |
| `evaluation/metrics/agent_metrics.py` | 指标库 |
| `evaluation/runners/run_agent_eval.py` | 评测 runner（含 `--from-json` 重算） |
| `evaluation/reports/baseline_*.json / .md` | 评测结果（可回放） |
| `docs/current_agent_architecture.md` | Task A 架构审计 |
| `docs/current_agent_dataflow.md` | Task A 数据流审计 |
