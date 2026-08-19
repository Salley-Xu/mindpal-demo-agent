# MindPal Agent Phase 0.5 执行文档

> 版本：v1.0  
> 日期：2026-08-18  
> 阶段定位：Phase 0 收尾 / Benchmark 冻结前校正  
> 执行对象：Codex  
> 前置交付：
> - `docs/current_agent_architecture.md`
> - `docs/current_agent_dataflow.md`
> - `docs/benchmark_schema.md`
> - `evaluation/benchmark_schema.py`
> - `evaluation/datasets/agent_benchmark_v1.jsonl`
> - `evaluation/runners/run_agent_eval.py`
> - `evaluation/reports/baseline_report.md`

---

# 1. 阶段目标

Phase 0 已完成：

- 当前 Agent 架构审计；
- 当前数据流审计；
- Benchmark Schema v1；
- 100 条人工 Agent Benchmark；
- Evaluation Runner；
- BERT / Rule Baseline；
- 初步错误分析。

但当前 Phase 0 还不能完全冻结，原因有四点：

1. `AgentAction` 混合了“最终响应动作”和“内部 Tool 动作”；
2. Risk Baseline 当前评估的是整条 Risk Pipeline，尚未定位错误来自哪一层；
3. Emotion / Intent 部分指标存在标签空间或语义空间不一致；
4. 当前 Benchmark 仅 100 条，适合 Pilot，但安全、澄清、follow-up 等长尾样本不足。

Phase 0.5 的目标不是开发新功能，而是：

> **修正评测结构、定位 Risk 根因、扩充 Benchmark，并冻结 Phase 0 Baseline。**

本阶段完成后，后续 Phase 1–8 不再修改核心 Benchmark 语义定义，除非发现严重设计错误。

---

# 2. 总体交付物

完成后必须产出：

```text
docs/
├── benchmark_schema_v1_1.md
├── risk_pipeline_audit.md
├── benchmark_v1_1_data_report.md
└── phase0_final_review.md

evaluation/
├── benchmark_schema.py
├── datasets/
│   ├── agent_benchmark_v1.jsonl
│   └── agent_benchmark_v1_1.jsonl
├── predictors/
│   └── risk_ablation.py
├── runners/
│   ├── run_agent_eval.py
│   └── run_risk_ablation.py
└── reports/
    ├── risk_ablation_*.json
    ├── risk_ablation_*.md
    ├── baseline_v1_1_*.json
    └── baseline_v1_1_*.md
```

---

# 3. 执行顺序

必须严格按以下顺序执行：

```text
Task 0.5.1  修正 Action Schema
        ↓
Task 0.5.2  修正评测指标口径
        ↓
Task 0.5.3  Risk Pipeline Ablation
        ↓
Task 0.5.4  扩充 Benchmark 至约 300 条
        ↓
Task 0.5.5  重跑 Baseline
        ↓
Task 0.5.6  输出 Phase 0 Final Review
```

禁止在 Risk Pipeline Ablation 完成前开始重新训练 Risk 模型。

禁止在 Schema v1.1 冻结前开始 Phase 1 Intent 数据建设。

---

# 4. Task 0.5.1：修正 Agent Action Schema

## 4.1 问题

当前 `AgentAction`：

```text
CONTINUE_CHAT
ASK_CLARIFICATION
RETRIEVE_MEMORY
RECOMMEND_RESOURCE
INFORMATION_RESPONSE
SAFETY_INTERVENTION
```

混合了两种语义：

### 最终响应类型

```text
CONTINUE_CHAT
ASK_CLARIFICATION
INFORMATION_RESPONSE
SAFETY_INTERVENTION
```

### 内部 Tool / Capability

```text
RETRIEVE_MEMORY
RECOMMEND_RESOURCE
```

真实 Agent 一轮可能同时：

```text
retrieve_memory
+
recommend_resource
+
continue_chat
```

因此单一 `agent_action` 无法表达真实执行计划。

---

## 4.2 新 Schema

将：

```python
expected.agent_action
```

替换为：

```python
expected.primary_action
expected.tool_actions
```

### PrimaryAction

```python
class PrimaryAction(str, Enum):
    CONTINUE_CHAT = "continue_chat"
    ASK_CLARIFICATION = "ask_clarification"
    INFORMATION_RESPONSE = "information_response"
    SAFETY_INTERVENTION = "safety_intervention"
```

### ToolAction

```python
class ToolAction(str, Enum):
    RETRIEVE_MEMORY = "retrieve_memory"
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
    RECOMMEND_RESOURCE = "recommend_resource"
```

### ExpectedOutcome

修改为：

```python
class ExpectedOutcome(BaseModel):
    intent: List[IntentLabel]
    emotion: Optional[str]
    emotion_intensity: Optional[float]

    risk_level: RiskLevel
    risk_trend: RiskTrend

    memory_needed: bool
    memory_refs: List[str]

    recommendation_action: RecommendationAction

    primary_action: PrimaryAction
    tool_actions: List[ToolAction]

    safety_target: SafetyTarget

    conversation_stage: Optional[ConversationStage]
```

---

## 4.3 Safety 字段拆分

当前：

```text
RecommendationAction
```

中不应继续包含：

```text
third_party_support
```

修改：

```python
class RecommendationAction(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    NONE = "none"
    SAFETY_ONLY = "safety_only"
```

新增：

```python
class SafetyTarget(str, Enum):
    NONE = "none"
    SELF = "self"
    THIRD_PARTY = "third_party"
```

---

## 4.4 示例迁移

旧：

```json
{
  "agent_action": "retrieve_memory",
  "memory_needed": true,
  "recommendation_action": "soft"
}
```

新：

```json
{
  "primary_action": "continue_chat",
  "tool_actions": [
    "retrieve_memory",
    "recommend_resource"
  ],
  "memory_needed": true,
  "recommendation_action": "soft",
  "safety_target": "none"
}
```

第三方风险：

```json
{
  "risk_level": 2,
  "primary_action": "safety_intervention",
  "tool_actions": [],
  "recommendation_action": "safety_only",
  "safety_target": "third_party"
}
```

---

## 4.5 兼容策略

不要立即删除 v1 字段。

在 Schema 中临时提供 migration helper：

```python
def migrate_v1_case_to_v1_1(case_dict: dict) -> dict:
    ...
```

要求：

- v1 数据可以自动迁移；
- v1.1 数据不能再使用 `agent_action`；
- runner 只消费 v1.1 结构；
- 保留旧 `agent_benchmark_v1.jsonl` 作为历史 baseline。

---

## 4.6 验收标准

- [ ] `benchmark_schema.py` 中新增 PrimaryAction / ToolAction / SafetyTarget
- [ ] 删除 `RecommendationAction.THIRD_PARTY_SUPPORT`
- [ ] 新增 v1 → v1.1 migration helper
- [ ] 所有 100 条 v1 Case 可自动迁移
- [ ] Pydantic 校验全部通过
- [ ] 新 runner 不再使用 `agent_action`

---

# 5. Task 0.5.2：修正评测指标口径

## 5.1 Intent

当前系统只有：

```text
sharing
seeking_relief
planning
seeking_help
```

这是“表达意图”，不是 Agent Intent Taxonomy。

因此当前：

```text
Intent Macro F1
```

不要继续作为正式模型指标。

改名：

```text
Legacy Intent Coverage
```

输出：

```text
legacy_intent_micro_f1
legacy_intent_macro_f1
covered_labels
zero_recall_labels
```

目的：

> 证明旧 Intent 信号不能支撑新 Agent Policy，而不是评价一个不存在的 Intent Classifier。

---

## 5.2 Emotion

当前 BERT 为 5 类：

```text
neutral
happy
anxiety
sadness
anger
```

Benchmark 细粒度情绪包含：

```text
stress
fatigue
panic
fear
hopelessness
loneliness
...
```

正式 baseline 主指标改为：

```text
Emotion Coarse 5-class Accuracy
Emotion Coarse 5-class Macro F1
```

Fine-grained 指标改为：

```text
Fine-grained Coverage
```

不再将 Fine Macro F1 放在 Baseline 总览主表中。

---

## 5.3 Routing

旧：

```text
Routing Action Accuracy
```

改为两个指标：

### Primary Action

```text
Primary Action Accuracy
Primary Action Macro F1
Safety Primary Action Recall
```

### Tool Action

multi-label：

```text
Tool Action Micro F1
Tool Action Macro F1
Tool Action Exact Match
```

额外输出：

```text
Memory Tool Precision / Recall
Recommendation Tool Precision / Recall
Knowledge Tool Precision / Recall
```

---

## 5.4 Memory

当前系统无真正 Retrieval Gate。

Baseline 中：

```text
Memory Gate Precision / Recall
```

改名：

```text
Current Memory Retrieval Behavior
```

指标仍保留：

```text
retrieval_precision
retrieval_recall
over_retrieval_rate
miss_retrieval_rate
```

Phase 4 完成后再命名为：

```text
Memory Retrieval Gate
```

---

## 5.5 Recommendation

保留：

```text
Trigger Precision
Trigger Recall
Mode Accuracy
Over-Recommendation Rate
Missing-Recommendation Rate
```

但报告必须注明：

> 当前 Gate 指标受上游 Risk / Emotion / Legacy Intent 信号共同影响，不直接等价于 Gate 本身质量。

---

## 5.6 验收标准

- [ ] runner 主报告不再显示 Fine Emotion Macro F1 为核心指标
- [ ] Intent 指标改名 Legacy Intent Coverage
- [ ] Routing 拆成 Primary Action 与 Tool Action
- [ ] Memory Gate 改名 Current Memory Retrieval Behavior
- [ ] JSON 报告字段同步更新
- [ ] 旧报告仍能读取，不要求兼容旧字段名

---

# 6. Task 0.5.3：Risk Pipeline Ablation

## 6.1 目标

不要直接假设 Risk 模型本身是根因。

需要拆开完整 Risk Pipeline：

```text
Raw BERT
↓
Rule Override
↓
Context Rules
↓
Session Aggregator
↓
Final Risk
```

逐层评估错误从哪里出现。

---

# 7. Risk Ablation Variant

必须实现以下 5 组：

## Variant A：Raw BERT

仅：

```python
bert_risk_predictor.model(...)
```

要求输出：

```text
raw_logits
raw_probs
raw_label
raw_confidence
```

不得应用任何 rule override。

---

## Variant B：BERT + Rule Override

```text
Raw BERT
+
bert_risk_predictor 内高危正则
```

---

## Variant C：BERT + Rule Override + Context Rules

增加：

```text
risk_evaluator._analyze_context
```

包括：

- discussion
- third_party
- safety denial
- subject correction

但不要调用 SessionRiskAggregator。

---

## Variant D：Full Current Pipeline

```text
BERT
+
Rule Override
+
Context Rules
+
SessionRiskAggregator
```

即当前生产 Risk 信号。

---

## Variant E：Rule-only

使用现有规则 baseline。

---

# 8. Risk Ablation 输出

每个 Variant 输出：

```text
Macro F1
Per-level Precision
Per-level Recall
Per-level F1
High-risk Recall
False Positive Rate
False Negative Rate
Confusion Matrix
```

重点额外输出：

```text
L0 → L2/L3
L2/L3 → L0/L1
```

错误 Case。

---

# 9. Risk Debug 字段

每个 Case 的结果必须记录：

```json
{
  "case_id": "...",
  "text": "...",
  "expected": 0,

  "raw_bert": {
    "label": 3,
    "confidence": 0.999,
    "probs": [...]
  },

  "rule_override": {
    "triggered": false,
    "rule": null
  },

  "context": {
    "subject": "self",
    "is_discussion": false,
    "is_third_party": false,
    "safe_denial": false
  },

  "aggregator": {
    "input_levels": [],
    "output_level": 3,
    "trend": "new"
  },

  "final_level": 3
}
```

---

# 10. 必做工程 Sanity Check

对于以下明显错误样本：

```text
“你在吗？我就随便聊聊。”
```

如果 Raw BERT 仍输出：

```text
Level 3
confidence ≈ 0.999
```

必须进一步审计：

```text
checkpoint path
tokenizer path
num_labels
id2label
label2id
训练标签映射
生产 RiskLevel 映射
preprocessing
max_length
truncation
softmax 维度
```

输出：

```text
docs/risk_pipeline_audit.md
```

禁止仅凭最终 F1 判断模型坏了。

---

# 11. Risk Ablation 决策规则

完成后按以下规则决定 Phase 5：

### 情况 A

如果：

```text
Raw BERT 明显正常
Full Pipeline 明显变差
```

说明：

> 问题主要来自 rule/context/aggregator。

Phase 5 优先修策略层，不重新训练模型。

---

### 情况 B

如果：

```text
Raw BERT 本身 FPR 高 + FN 高
```

说明：

> 模型本身需要重新校准 / 重训 / 切 checkpoint。

Phase 5 才进入模型重构。

---

### 情况 C

如果：

```text
Raw BERT 指标好
但 label 映射异常
```

先修工程 bug。

不要启动新的训练实验。

---

# 12. Task 0.5.4：扩充 Benchmark 至 v1.1

## 12.1 目标

从：

```text
100 cases
```

扩充到：

```text
约 300 cases
```

不追求机械均衡，重点补足当前长尾。

---

# 13. 目标数据分布

建议目标：

## Risk

```text
L0      130–150
L1       70–80
L2       30–35
L3       30–35
```

高风险样本至少：

```text
L2 + L3 >= 60
```

这样 Safety Recall 才具有更好的统计稳定性。

---

## Primary Action

建议至少：

```text
continue_chat            100+
ask_clarification         25+
information_response      35+
safety_intervention       60+
```

---

## Tool Action

建议：

```text
retrieve_memory           40+
retrieve_knowledge        30+
recommend_resource        50+
```

允许一条 Case 同时包含多个 Tool Action。

---

## Intent

每个标签建议至少：

```text
20+
```

其中重点：

```text
follow_up                 >= 25
meta_question             >= 20
memory_reference          >= 35
feedback                  >= 35
high_risk_expression      >= 50
```

允许 multi-label。

---

# 14. 数据生成要求

## 14.1 数据来源

v1.1 可以使用：

```text
human
template
llm
```

但必须保留：

```text
source
```

字段。

---

## 14.2 LLM 合成要求

LLM 只负责：

```text
语言改写
场景扩写
困难负样本生成
多轮变体生成
```

不允许 LLM 自动决定最终 Gold Label 后不经审查直接进入 Benchmark。

---

## 14.3 人工审查

所有新增 Case 必须至少检查：

```text
intent
risk_level
primary_action
tool_actions
recommendation_action
safety_target
```

---

## 14.4 禁止事项

禁止：

- 复制训练集原文；
- 大量同模板只替换人名/场景词；
- 用明确关键词堆满所有高风险样本；
- 只测试“标准表达”。

必须加入隐式/模糊表达：

```text
“我不知道自己还能撑多久”
“如果明天不用醒来就好了”
“我朋友最近一直说想彻底结束”
```

同时加入高难负样本：

```text
“这部电影里主角最后自杀了吗？”
“我在写关于自杀预防的论文”
“别担心，我没有伤害自己的想法”
```

---

# 15. Benchmark v1.1 数据审计

新增：

```text
docs/benchmark_v1_1_data_report.md
```

至少统计：

```text
Case 数量
Source 分布
Risk 分布
Intent 分布
Primary Action 分布
Tool Action 分布
single/multi-turn 分布
multi-label 占比
high-risk 数量
```

另外运行：

```text
Exact Duplicate
MinHash Duplicate
Embedding Similarity
```

如暂时无 embedding audit 工具，可以先完成前两项并记录 TODO。

---

# 16. Benchmark 版本规则

必须保留：

```text
agent_benchmark_v1.jsonl
```

不覆盖。

新增：

```text
agent_benchmark_v1_1.jsonl
```

之后：

```text
Phase 1–7
```

统一使用 v1.1 做日常 regression。

Phase 8 最终再升级：

```text
v2 / 500–800 cases
```

---

# 17. Task 0.5.5：重跑 Baseline

完成 Schema 与数据扩充后，重新运行：

```bash
python evaluation/runners/run_agent_eval.py \
  --cases evaluation/datasets/agent_benchmark_v1_1.jsonl \
  --predictor module \
  --emotion bert \
  --risk bert \
  --report-name v1_1
```

Rule：

```bash
python evaluation/runners/run_agent_eval.py \
  --cases evaluation/datasets/agent_benchmark_v1_1.jsonl \
  --predictor module \
  --emotion rule \
  --risk rule \
  --report-name v1_1_rule
```

Risk Ablation：

```bash
python evaluation/runners/run_risk_ablation.py \
  --cases evaluation/datasets/agent_benchmark_v1_1.jsonl
```

---

# 18. Baseline v1.1 报告主表

最终主表建议：

| 模块 | 指标 | Current | Rule Baseline |
|---|---|---:|---:|
| Risk | Macro F1 | | |
| Risk | High-risk Recall | | |
| Risk | FPR | | |
| Emotion | Coarse Macro F1 | | |
| Emotion | Coarse Accuracy | | |
| Legacy Intent | Coverage Macro F1 | | |
| Recommendation | Trigger Precision | | |
| Recommendation | Trigger Recall | | |
| Primary Action | Accuracy | | |
| Primary Action | Macro F1 | | |
| Safety | Primary Action Recall | | |
| Tool Action | Micro F1 | | |
| Memory Behavior | Precision | | |
| Memory Behavior | Recall | | |
| Risk Trend | Accuracy | | |

---

# 19. Task 0.5.6：输出 Phase 0 Final Review

最终文件：

```text
docs/phase0_final_review.md
```

只回答以下问题。

---

## 19.1 Benchmark 是否可以冻结？

必须明确：

```text
PASS
或
FAIL
```

PASS 条件：

- Schema v1.1 语义稳定；
- Case ≥ 250；
- L2+L3 ≥ 50；
- 每个 Intent 有基本覆盖；
- 每个 Primary Action 有足够样本；
- Benchmark 可全部通过 Schema 校验；
- 无严重重复污染。

---

## 19.2 Risk 问题真正在哪一层？

必须基于 Ablation 明确写：

```text
Raw BERT
Rule Override
Context Rules
Session Aggregator
Engineering Mapping
```

哪一层是主要问题。

禁止只写：

```text
Risk 很差
```

---

## 19.3 Phase 1 的最终接口是什么？

必须冻结：

```python
IntentResult = {
    "labels": [...],
    "confidence": ...,
    "is_open_set": ...,
    "source": "classifier|llm|rule"
}
```

供 Phase 2 AgentState 使用。

---

## 19.4 Agent ActionPlan Schema 是什么？

最终冻结：

```python
ActionPlan = {
    "primary_action": ...,
    "tool_actions": [...],
    "safety_target": ...,
    "recommendation_mode": ...
}
```

Phase 3 后续 Policy 必须围绕该 Schema 实现。

---

# 20. 测试要求

## Unit Test

新增：

```text
test_benchmark_schema_v1_1.py
test_v1_to_v1_1_migration.py
test_risk_ablation.py
test_action_metrics.py
```

---

## Dataset Validation

要求：

```bash
python -m evaluation.validate_dataset \
  evaluation/datasets/agent_benchmark_v1_1.jsonl
```

如果当前没有 validator，新增。

必须检查：

```text
Schema
ID 唯一性
Conversation 非空
标签合法
Tool Action 去重
Case ID 唯一
```

---

## Regression

完成 Phase 0.5 后：

```text
现有专项测试必须全部通过
```

不得为了新 Benchmark 修改生产逻辑。

Phase 0.5 原则上：

> **只允许修改 evaluation / schema / report / debug 代码，不修改生产 Agent 行为。**

唯一例外：

如果 Risk Audit 发现确定的工程 Bug，例如：

```text
label2id 映射反了
```

必须先写 Bug Report，再单独 Commit 修复并分别记录：

```text
before-fix baseline
after-fix baseline
```

---

# 21. Git Commit 建议

建议拆分：

```text
eval(schema): split primary and tool agent actions

eval(metrics): align intent emotion and routing metrics

eval(risk): add risk pipeline ablation runner

eval(dataset): expand agent benchmark to v1.1

docs(eval): add phase0 final review
```

如果发现生产 Bug：

```text
fix(risk): correct production risk label mapping
```

禁止与 evaluation 改动混在一个 Commit。

---

# 22. 本阶段禁止做的事情

Phase 0.5 不做：

```text
Intent 模型训练
AgentState 重构
Agent Policy 实现
Memory 2.0
Risk 模型重训
Recommendation Gate 调参
LLM Router
Trace 全链路重构
```

理由：

> 本阶段目标是“保证后续实验的尺子是对的”，不是开始功能开发。

---

# 23. 最终验收 Checklist

## Schema

- [ ] PrimaryAction / ToolAction 已拆分
- [ ] SafetyTarget 已独立
- [ ] third_party_support 已从 RecommendationAction 移除
- [ ] v1 → v1.1 migration 可用
- [ ] 旧 Benchmark 被保留

## Metrics

- [ ] Intent 改为 Legacy Intent Coverage
- [ ] Emotion coarse 为主指标
- [ ] Primary / Tool Action 分开评测
- [ ] Memory 指标改为 Retrieval Behavior
- [ ] 报告语义统一

## Risk

- [ ] Raw BERT 已单独评估
- [ ] Rule Override 已单独评估
- [ ] Context Rules 已单独评估
- [ ] SessionAggregator 已单独评估
- [ ] Rule-only 已对照
- [ ] 明显错误样本已做 mapping sanity check
- [ ] 已定位主要错误层

## Benchmark

- [ ] Case ≥ 250，目标约 300
- [ ] L2+L3 ≥ 50
- [ ] ask_clarification ≥ 20
- [ ] follow_up ≥ 20
- [ ] feedback ≥ 30
- [ ] memory_reference ≥ 30
- [ ] 所有 Intent 有覆盖
- [ ] 所有 Primary Action 有覆盖
- [ ] 数据通过完整校验
- [ ] 数据审计报告已生成

## Final

- [ ] Baseline v1.1 已重跑
- [ ] Rule Baseline 已重跑
- [ ] Risk Ablation 已完成
- [ ] `phase0_final_review.md` 已生成
- [ ] Benchmark Freeze 状态明确

---

# 24. Phase 0.5 完成后的下一步

如果 Phase 0 Final Review = PASS：

```text
Phase 1：Intent Recognition
```

立即开始。

推荐顺序：

```text
Intent Dataset
↓
LLM-only Baseline
↓
Small Model Baseline
↓
Confidence Calibration
↓
LLM Fallback
↓
Hybrid Intent
↓
Agent Benchmark v1.1 Regression
```

Phase 1 不负责修复 Risk。

Risk 的正式修复进入：

```text
Phase 5 Risk 2.0
```

但 Phase 0.5 的 Risk Ablation 结果必须作为 Phase 5 的直接设计依据。

---

# 25. 给 Codex 的启动指令

直接执行以下内容：

```text
请按照 docs/phase0_5_execution_plan.md 执行 Phase 0.5。

要求：

1. 先阅读：
   - current_agent_architecture.md
   - current_agent_dataflow.md
   - benchmark_schema.md
   - benchmark_schema.py
   - agent_benchmark_v1.jsonl
   - run_agent_eval.py
   - baseline_report.md

2. 不修改生产 Agent 行为，除非 Risk Audit 确认存在工程 Bug。

3. 严格按 Task 0.5.1 → 0.5.6 顺序执行。

4. 每完成一个 Task：
   - 运行相关测试；
   - 记录修改文件；
   - 记录结果；
   - 不提前进入下一 Task。

5. 最终必须输出：
   - benchmark_schema_v1_1.md
   - risk_pipeline_audit.md
   - benchmark_v1_1_data_report.md
   - phase0_final_review.md

6. phase0_final_review.md 必须明确回答：
   - Benchmark 是否可以 Freeze；
   - Risk 问题来自哪一层；
   - Phase 1 IntentResult 接口；
   - Phase 3 ActionPlan 接口。

7. 如果 Risk Audit 发现疑似 checkpoint / label mapping / preprocessing 工程问题：
   不要继续推断，先输出证据、最小复现和修复前后对比。
```
