# MindPal Agent 系统开发任务计划

> 版本：v1.0  
> 日期：2026-08-18  
> 目标：将当前压力管理对话 Agent 从“多模块功能集合”升级为“可决策、可评测、可解释、可迭代的 Stateful Adaptive Agent”，并形成可直接用于大模型算法 / Agent 算法 / 搜推算法求职展示的完整项目闭环。

---

## 1. 项目背景

当前 MindPal Agent 已具备以下核心能力：

- 情绪识别
- 风险判断
- Session 短期工作记忆
- 跨会话长期记忆
- FAISS + BM25 混合召回
- Recommendation Gate
- LLM Rerank
- 高风险安全资源路由
- 基于用户画像与历史信息的个性化回应

当前系统的主要问题不是“功能不够多”，而是：

1. 缺少统一的 Agent State；
2. Intent、Emotion、Risk、Memory、Recommendation 之间尚未形成统一决策链；
3. Memory 仍偏向“存储 + 检索”，缺少写入治理、冲突解决和生命周期管理；
4. 风险识别仍需从单轮分类升级为多轮动态风险状态；
5. Recommendation 缺少显式反馈闭环；
6. 缺少系统级 Benchmark、Trace、Error Taxonomy 与 Ablation；
7. 简历目前更容易体现“做了哪些模块”，还不足以体现“为什么这样设计、效果如何、如何验证”。

因此，本轮开发的核心不是增加更多 Tool，而是将现有能力收敛到统一的 Agent 架构和评测体系中。

---

# 2. 总体目标

最终系统目标：

```text
User Input
    ↓
Perception Layer
├── Intent
├── Emotion
├── Risk
└── Conversation Signals
    ↓
Conversation State
    ↓
Agent Policy / Router
├── Continue Chat
├── Ask Clarification
├── Retrieve Memory
├── Recommend Resource
├── Information Response
└── Safety Intervention
    ↓
Context Builder
├── Session Memory
├── Long-term Memory
└── Retrieved Resources
    ↓
LLM Response
    ↓
Post-processing
├── Memory Write
├── State Update
├── Recommendation Feedback
├── Risk Update
└── Trace / Evaluation
```

最终需要回答四个核心问题：

1. Agent 是否做出了正确 Action？
2. Agent 是否使用了正确 Memory？
3. Agent 是否在正确时机进行了推荐 / 安全干预？
4. Agent 的决策改造是否通过 Benchmark 和 Ablation 得到验证？

---

# 3. 开发原则

## 3.1 不优先增加新 Tool

本阶段原则上不投入：

- Multi-Agent
- Planner / Executor / Critic 多 Agent
- 天气、搜索、日历等通用 Tool
- MCP 工具堆叠
- AutoGen / CrewAI Demo

除非后续发现这些能力能够直接支撑现有 Benchmark。

## 3.2 优先提升可评测性

任何重要模块升级前必须回答：

```text
当前 baseline 是什么？
目标指标是什么？
如何判断新方案更好？
```

没有 Benchmark 的模块优先补 Benchmark，而不是直接重构。

## 3.3 优先采用 Hybrid Architecture

总体原则：

```text
Rules
    ↓
保证确定性与安全边界

Small Model / Classifier
    ↓
处理稳定、高频、低成本决策

LLM
    ↓
处理复杂语义、长尾和开放场景
```

避免所有模块都依赖 LLM。

## 3.4 所有决策必须可 Trace

任何一轮对话都应该可以回放：

```text
User Input
→ Perception
→ State
→ Policy
→ Tool / Memory
→ Context
→ Response
→ State Update
```

---

# 4. 总体阶段规划

| Phase | 名称 | 优先级 | 预计周期 | 核心产出 |
|---|---|---:|---:|---|
| 0 | Baseline & Evaluation | P0 | 4–5 天 | Agent Benchmark、Metrics、Trace Schema |
| 1 | Intent Recognition | P1 | 3–4 天 | Intent Taxonomy、Classifier、Fallback |
| 2 | Agent State | P1 | 3 天 | State Schema、State Updater |
| 3 | Agent Policy / Router | P1 | 4–5 天 | Action Space、Hybrid Policy |
| 4 | Memory 2.0 | P1 | 5–7 天 | Write Gate、Memory Type、Conflict Resolution |
| 5 | Risk 2.0 | P1 | 4–5 天 | Multi-turn Risk State、Trend Detection |
| 6 | Recommendation 2.0 | P2 | 4–5 天 | Feedback Loop、Ranking Features |
| 7 | Observability | P2 | 3–4 天 | Trace、Replay、Error Taxonomy |
| 8 | Benchmark & Ablation | P1 | 5–7 天 | 实验报告、对比表、最终 README |

总周期约 6 周。

---

# 5. Phase 0：Baseline & Agent Evaluation Framework

## 5.1 目标

在修改现有 Agent 架构前，先建立系统级评测基线。

这一阶段必须优先完成，因为后续所有改造都需要统一 Benchmark 验证。

---

## 5.2 Task 0.1：梳理当前系统决策链

### 工作内容

审计当前代码，输出完整调用链：

```text
input
→ emotion
→ risk
→ memory
→ recommend gate
→ retrieval
→ rerank
→ response
```

需要明确：

- 每个模块输入
- 每个模块输出
- 是否调用 LLM
- 是否存在规则
- 是否存在持久状态
- 模块间依赖
- 异常处理
- fallback

### 输出文件

```text
docs/
  current_agent_architecture.md
  current_agent_dataflow.md
```

### 验收标准

必须能够从一轮输入完整追踪到最终回复。

---

## 5.3 Task 0.2：定义 Benchmark Schema

建议统一为 JSONL。

示例：

```json
{
  "case_id": "case_0001",
  "conversation": [
    {"role": "user", "content": "最近连续加班，感觉很累"}
  ],
  "expected": {
    "intent": ["emotional_expression"],
    "emotion": "fatigue",
    "risk_level": 0,
    "memory_needed": false,
    "recommendation_action": "none",
    "agent_action": "continue_chat"
  },
  "tags": [
    "single_turn",
    "low_risk",
    "emotion"
  ]
}
```

多轮：

```json
{
  "case_id": "case_0101",
  "conversation": [
    {"role": "user", "content": "最近一直睡不好"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "每天都觉得什么都没意义"}
  ],
  "expected": {
    "risk_level": 2,
    "risk_trend": "rising",
    "agent_action": "safety_intervention"
  }
}
```

---

## 5.4 Task 0.3：建设 Agent Benchmark v1

### 数量目标

第一版：

```text
500–800 cases
```

后续扩展：

```text
1000+ cases
```

### 数据分布建议

| 场景 | 数量 |
|---|---:|
| 普通闲聊 | 80 |
| 情绪表达 | 100 |
| 显式求助 | 80 |
| 信息请求 | 60 |
| Memory 引用 | 80 |
| 推荐场景 | 100 |
| 推荐反馈 | 60 |
| 低风险 | 80 |
| 中风险 | 80 |
| 高风险 | 80 |
| Intent 模糊 / 多标签 | 60 |
| 长多轮状态变化 | 80 |

允许一个 Case 属于多个标签。

### 数据来源

优先：

1. 自己定义场景模板；
2. LLM 合成；
3. 人工抽检与修正；
4. 禁止直接复用训练数据；
5. 对评测集做 MinHash / embedding 相似度泄漏审计。

---

## 5.5 Task 0.4：实现 Evaluation Harness

目录建议：

```text
evaluation/
├── datasets/
│   ├── agent_benchmark_v1.jsonl
│   └── retrieval_benchmark_v1.jsonl
├── metrics/
│   ├── intent_metrics.py
│   ├── risk_metrics.py
│   ├── memory_metrics.py
│   ├── recommendation_metrics.py
│   └── routing_metrics.py
├── runners/
│   ├── run_agent_eval.py
│   └── run_retrieval_eval.py
└── reports/
```

---

## 5.6 指标定义

### Intent

- Macro F1
- Micro F1
- Per-class F1
- Open-set Recall

### Risk

- High-risk Recall
- Macro F1
- False Negative Rate
- False Positive Rate

### Memory

- Retrieval Precision
- Retrieval Recall
- Memory Usefulness
- Memory Hallucination Rate
- Memory Conflict Rate

### Recommendation

- Trigger Precision
- Trigger Recall
- Over-Recommendation Rate
- Missing-Recommendation Rate

### Retrieval / Ranking

- Recall@5
- Recall@10
- MRR
- NDCG@5
- NDCG@10

### Agent Routing

- Action Accuracy
- Macro F1
- Safety Action Recall

### End-to-End

使用 Rubric / LLM-as-a-Judge：

- Helpfulness
- Safety
- Personalization
- Naturalness
- Memory Consistency
- Over-intervention

---

## 5.7 Phase 0 验收标准

必须产出：

```text
docs/baseline_report.md
```

内容至少包括：

- 当前各模块指标
- 当前 E2E 指标
- 主要错误类型
- 最差 Top 3 模块
- 后续 Phase 优先级

---

# 6. Phase 1：Intent Recognition

## 6.1 目标

补齐 Agent 决策中的 Intent 信号，使系统能够区分：

```text
“用户现在是什么情绪”
```

与：

```text
“用户希望 Agent 做什么”
```

---

## 6.2 Intent Taxonomy v1

推荐初始标签：

```text
casual_chat
emotional_expression
explicit_help_request
information_request
resource_request
feedback
follow_up
memory_reference
high_risk_expression
meta_question
```

支持 multi-label。

例如：

```text
“我最近特别焦虑，有没有什么办法？”
```

标签：

```text
emotional_expression
explicit_help_request
```

---

## 6.3 Task 1.1：构建 Intent Dataset

目标：

```text
3000–5000 条
```

方法：

- 模板生成基础样本；
- LLM 生成语义改写；
- 加入多标签样本；
- 加入模糊 Intent；
- 加入困难负样本；
- 加入对抗表达。

划分：

```text
train / dev / test
```

Test 必须与 Agent Benchmark 独立。

---

## 6.4 Task 1.2：建立三种 Baseline

### Baseline A

LLM-only

### Baseline B

BERT / RoBERTa Classifier

### Baseline C

Small Model + LLM Fallback

逻辑：

```text
if confidence >= threshold:
    use classifier
else:
    use llm
```

---

## 6.5 Task 1.3：置信度与 Open-set

需要增加：

```text
intent_confidence
```

低置信度：

```text
UNKNOWN / LLM fallback
```

避免分类器对未见场景强行输出错误标签。

---

## 6.6 验收指标

建议目标：

```text
Macro F1 >= 0.85
```

并证明：

```text
Hybrid
>
Small Model Only
```

在长尾场景下更稳定。

另外统计：

- 平均延迟
- LLM 调用比例
- 推理成本

---

# 7. Phase 2：Agent State

## 7.1 目标

将各模块信号统一进入 Conversation State。

---

## 7.2 State Schema

建议：

```python
class AgentState:
    intent: list[str]
    intent_confidence: float

    emotion: str
    emotion_intensity: float
    emotion_trend: str

    risk_level: int
    risk_trend: str
    risk_persistence: int

    help_seeking: bool

    memory_needed: bool
    memory_refs: list[str]

    recommendation_history: list[str]
    recommendation_feedback: list

    conversation_stage: str

    last_action: str
    current_action: str
```

---

## 7.3 Task 2.1：State 初始化

第一轮：

```text
User Input
↓
Perception
↓
Initial State
```

---

## 7.4 Task 2.2：State Update

下一轮：

```text
State_t
+
User_t+1
+
Perception_t+1
↓
State_t+1
```

必须区分：

```text
瞬时信号
```

和：

```text
持久信号
```

例如：

```text
emotion
```

可以快速变化；

```text
risk_history
```

需要累积。

---

## 7.5 Task 2.3：State Persistence

至少提供：

```text
session state
```

并能够在 debug 模式输出。

---

## 7.6 验收标准

通过 50+ 多轮测试 Case 验证：

- 状态不会错误丢失；
- 新输入能够覆盖过时状态；
- 累积字段正确；
- session 隔离正确。

---

# 8. Phase 3：Agent Policy / Router

## 8.1 目标

建立统一的：

```text
State → Action
```

决策层。

---

## 8.2 Action Space

第一版建议：

```text
CONTINUE_CHAT
ASK_CLARIFICATION
RETRIEVE_MEMORY
RECOMMEND_RESOURCE
INFORMATION_RESPONSE
SAFETY_INTERVENTION
```

后续可扩展。

---

## 8.3 Task 3.1：Rule Policy

必须先写确定性高优规则。

例如：

```python
if state.risk_level >= HIGH:
    return SAFETY_INTERVENTION

if "memory_reference" in state.intent:
    return RETRIEVE_MEMORY

if "resource_request" in state.intent:
    return RECOMMEND_RESOURCE
```

规则层重点解决：

- 安全底线
- 明确用户指令
- 明确 Memory 引用

---

## 8.4 Task 3.2：Classifier Policy

对于稳定高频场景：

```text
State Features
↓
Action Classifier
```

输入特征：

```text
intent
emotion
emotion_intensity
risk
help_seeking
memory_needed
recommendation_frequency
conversation_stage
```

---

## 8.5 Task 3.3：LLM Router

仅处理：

- 冲突信号；
- 复杂多 Intent；
- classifier 低置信度；
- 长尾场景。

---

## 8.6 Hybrid Policy

完整流程：

```text
Safety Rules
↓
Deterministic Rules
↓
Classifier
↓
Confidence Check
↓
LLM Fallback
```

---

## 8.7 Routing Experiment

对比：

```text
LLM-only
Small Model-only
Rules + Small Model
Rules + Small Model + LLM fallback
```

测：

- Action Accuracy
- Safety Recall
- Avg Latency
- LLM Call Rate
- Cost

---

## 8.8 验收目标

```text
Action Accuracy >= 90%
Safety Action Recall >= 98%
```

同时 Hybrid LLM 调用率显著低于 LLM-only。

---

# 9. Phase 4：Memory 2.0

## 9.1 目标

从：

```text
Storage + Retrieval
```

升级为：

```text
Write
→ Organize
→ Retrieve
→ Use
→ Update
→ Forget
```

---

## 9.2 Memory Type

建议：

```text
PROFILE
PREFERENCE
EVENT
GOAL
COPING_STRATEGY
COPING_FEEDBACK
RELATIONSHIP
INTERACTION
```

---

## 9.3 Task 4.1：Memory Candidate Extraction

每轮对话结束后：

```text
Conversation
↓
Memory Candidate Extractor
```

输出：

```json
{
  "type": "COPING_FEEDBACK",
  "content": "呼吸训练效果不好",
  "importance": 0.8,
  "confidence": 0.92
}
```

---

## 9.4 Task 4.2：Memory Write Gate

判断维度：

```text
importance
stability
future_usefulness
confidence
privacy
duplication
```

输出：

```text
WRITE
UPDATE
IGNORE
```

---

## 9.5 Task 4.3：Memory Schema

建议字段：

```text
memory_id
user_id
memory_type
content
structured_value
created_at
updated_at
confidence
importance
source_turn
status
valid_from
valid_until
supersedes
```

---

## 9.6 Task 4.4：Conflict Resolution

支持：

```text
ADD
UPDATE
MERGE
SUPERSEDE
EXPIRE
```

案例：

```text
旧：
用户喜欢跑步

新：
用户膝盖受伤，近期无法跑步
```

不能简单删除旧偏好。

应：

```text
running_preference = positive
running_availability = temporarily_unavailable
```

---

## 9.7 Task 4.5：Memory Retrieval Gate

不是每轮都检索 Memory。

根据：

```text
intent
conversation context
memory reference
personalization need
```

判断：

```text
retrieve / skip
```

---

## 9.8 Task 4.6：Memory Rerank

Ranking Feature：

```text
semantic_similarity
memory_type_match
recency
importance
confidence
intent_match
```

---

## 9.9 Memory Benchmark

至少覆盖：

```text
正确回忆
不应回忆
过期记忆
冲突记忆
偏好更新
事件时间
错误归属
```

---

## 9.10 Ablation

比较：

```text
No Memory
Naive Vector Memory
Memory + Retrieval Gate
Memory 2.0
```

指标：

- Retrieval Precision
- Retrieval Recall
- Memory Usefulness
- Personalization
- Conflict Error
- Hallucination Rate

---

# 10. Phase 5：Risk 2.0

## 10.1 目标

将风险识别从单轮：

```text
Message → Risk
```

升级为：

```text
Current Turn
+
Recent Context
+
Risk History
+
Emotion Trend
↓
Dynamic Risk State
```

---

## 10.2 Risk State

字段：

```text
risk_level
risk_confidence
risk_history
risk_trend
risk_persistence
risk_escalation
```

---

## 10.3 Task 5.1：Risk Trend

至少：

```text
stable
rising
falling
fluctuating
```

---

## 10.4 Task 5.2：Context-aware Risk

输入：

```text
current message
recent N turns
historical risk state
```

比较：

```text
single-turn
vs
context-aware
```

---

## 10.5 Task 5.3：Early Warning

设计多轮 case：

```text
0 → 0 → 1 → 2 → 3
```

评估系统是否能够：

- 提前识别风险升级；
- 降低高风险漏判；
- 不显著增加低风险误报。

---

## 10.6 指标

```text
High-risk Recall
Early Detection Rate
False Alarm Rate
Escalation Accuracy
```

---

## 10.7 Safety Rule

无论模型结果如何：

```text
确定性高风险规则
```

继续保留作为 Safety Guardrail。

---

# 11. Phase 6：Recommendation 2.0

## 11.1 目标

将现有推荐流程：

```text
Gate
→ Retrieval
→ Rerank
→ Recommend
```

升级为：

```text
Policy
→ Recommendation Tool
→ Retrieval
→ Rerank
→ Feedback
→ Memory Update
→ Future Ranking
```

---

## 11.2 Task 6.1：Recommendation Tool 化

统一接口：

```python
recommend(
    agent_state,
    user_profile,
    memory,
    history
)
```

Agent Policy 决定是否调用。

---

## 11.3 Task 6.2：Recommendation Feedback

识别：

```text
accept
reject
tried_effective
tried_ineffective
not_interested
already_seen
```

并写入长期 Memory。

---

## 11.4 Task 6.3：Ranking Features

候选特征：

```text
semantic_score
bm25_score
intent_match
emotion_match
risk_match
profile_match
memory_match
repeat_penalty
negative_feedback_penalty
```

---

## 11.5 Task 6.4：Ranking Baseline

比较：

```text
FAISS
FAISS + BM25
Hybrid + LLM Rerank
Hybrid + Feature Rerank
Hybrid + Feature + LLM Rerank
```

---

## 11.6 指标

```text
Recall@5
MRR
NDCG@5
Trigger Precision
Trigger Recall
Over-Recommendation Rate
Repeat Recommendation Rate
```

---

# 12. Phase 7：Observability & Error Analysis

## 12.1 Trace Schema

每轮必须保存：

```json
{
  "trace_id": "...",
  "input": "...",
  "perception": {
    "intent": [],
    "emotion": "...",
    "risk": 0
  },
  "state_before": {},
  "policy": {
    "action": "...",
    "source": "rule|classifier|llm"
  },
  "memory": {
    "retrieved": [],
    "written": []
  },
  "recommendation": {
    "triggered": false,
    "candidates": []
  },
  "response": "...",
  "state_after": {}
}
```

---

## 12.2 Task 7.1：Trace Logger

要求：

- JSONL；
- 每轮唯一 trace_id；
- 支持关闭敏感文本；
- 支持 debug 模式。

---

## 12.3 Task 7.2：Replay

实现：

```text
trace_id
↓
replay
↓
查看每个中间决策
```

---

## 12.4 Error Taxonomy

建议：

```text
E01 Intent Error
E02 Emotion Error
E03 Risk False Negative
E04 Risk False Positive

E05 Missing Memory
E06 Wrong Memory
E07 Memory Conflict
E08 Memory Hallucination

E09 Over Recommendation
E10 Missing Recommendation
E11 Wrong Recommendation
E12 Repeated Recommendation

E13 Wrong Routing
E14 Over Safety
E15 Under Safety

E16 Personalization Error
E17 Conversation Inconsistency
```

---

## 12.5 Task 7.3：自动错误归因

Evaluation Runner 输出：

```text
case_id
failed_metric
predicted
expected
error_type
trace_id
```

便于批量分析。

---

# 13. Phase 8：Final Benchmark & Ablation

## 13.1 必做实验

### Experiment A：Intent Router

```text
LLM-only
vs
Small Model
vs
Hybrid
```

---

### Experiment B：Memory

```text
No Memory
vs
Naive Memory
vs
Memory 2.0
```

---

### Experiment C：Risk

```text
Single-turn
vs
Multi-turn
```

---

### Experiment D：Retrieval / Rerank

```text
FAISS
vs
FAISS + BM25
vs
Hybrid + Rerank
```

---

### Experiment E：Recommendation Gate

```text
No Gate
vs
Recommendation Gate
```

---

### Experiment F：Agent Policy

```text
Pure LLM Router
vs
Hybrid Policy
```

---

# 14. 最终报告模板

输出：

```text
docs/
├── final_system_design.md
├── final_evaluation_report.md
├── ablation_report.md
├── error_analysis.md
└── project_summary.md
```

其中：

## final_evaluation_report.md

至少包括：

```text
1. Benchmark
2. Metrics
3. Baselines
4. Final Results
5. Per-module Analysis
6. Failure Cases
```

## ablation_report.md

必须说明：

```text
每一个新增组件是否真的有用
```

而不是只报告最终模型。

---

# 15. 推荐代码目录

```text
mindpal/
├── agent/
│   ├── state.py
│   ├── policy.py
│   ├── actions.py
│   └── orchestrator.py
│
├── perception/
│   ├── intent/
│   ├── emotion/
│   └── risk/
│
├── memory/
│   ├── schema.py
│   ├── extractor.py
│   ├── write_gate.py
│   ├── conflict.py
│   ├── retriever.py
│   └── reranker.py
│
├── recommendation/
│   ├── gate.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── feedback.py
│   └── tool.py
│
├── safety/
│   ├── rules.py
│   ├── risk_state.py
│   └── intervention.py
│
├── tracing/
│   ├── logger.py
│   ├── schema.py
│   └── replay.py
│
├── evaluation/
│   ├── datasets/
│   ├── metrics/
│   ├── runners/
│   └── reports/
│
└── tests/
```

---

# 16. Codex 执行规范

每一个 Phase 都要求 Codex：

## 开始前

先读取：

```text
README
当前模块代码
相关技术文档
上一阶段报告
```

然后输出：

```text
implementation_plan.md
```

内容：

- 当前代码结构
- 修改文件
- 新增文件
- 接口变化
- 风险
- 测试方法

未经计划不要直接大规模重构。

---

## 开发中

原则：

1. 尽量保持接口兼容；
2. 一个 Commit 只解决一个逻辑问题；
3. 先补测试，再重构核心模块；
4. 所有新策略都支持 config 开关；
5. 所有模型都支持 baseline fallback；
6. 所有决策都写 Trace；
7. 不允许删除旧 baseline，必须方便 Ablation。

---

## 完成后

每个 Phase 输出：

```text
phaseX_review.md
```

模板：

```text
# Phase X Review

## 1. 完成内容

## 2. 修改文件

## 3. 新增功能

## 4. 测试结果

## 5. Benchmark

## 6. 当前问题

## 7. 下一阶段建议
```

---

# 17. 测试要求

至少包含：

## Unit Test

每个关键模块：

```text
intent
state
policy
memory
risk
recommendation
```

都需要单测。

---

## Integration Test

覆盖：

```text
User
→ Perception
→ State
→ Policy
→ Memory / Tool
→ Response
```

---

## Regression Test

每完成一个 Phase：

```text
必须跑完整 Agent Benchmark
```

避免某个模块提升后破坏其他模块。

---

# 18. Git 开发建议

Branch：

```text
feat/evaluation-framework
feat/intent-recognition
feat/agent-state
feat/agent-policy
feat/memory-v2
feat/risk-v2
feat/recommendation-v2
feat/observability
exp/final-ablation
```

Commit 示例：

```text
feat(memory): add memory write gate
feat(policy): add rule-based safety routing
eval(agent): add routing benchmark
fix(memory): resolve superseded preference conflict
```

---

# 19. 六周执行排期

## Week 1

完成：

```text
Phase 0
Phase 1
```

交付：

- Agent Benchmark v1
- Evaluation Harness
- Intent Classifier
- Intent Baseline Report

---

## Week 2

完成：

```text
Phase 2
Phase 3
```

交付：

- Agent State
- Hybrid Policy
- Routing Benchmark

---

## Week 3

完成：

```text
Phase 4 前半
```

交付：

- Memory Schema
- Memory Write Gate
- Memory Type
- Candidate Extraction

---

## Week 4

完成：

```text
Phase 4 后半
Phase 5
```

交付：

- Memory Conflict
- Memory Retrieval Gate
- Multi-turn Risk
- Risk Trend

---

## Week 5

完成：

```text
Phase 6
Phase 7
```

交付：

- Recommendation Feedback Loop
- Ranking Experiments
- Trace
- Replay
- Error Taxonomy

---

## Week 6

完成：

```text
Phase 8
```

交付：

- Full Benchmark
- Ablation
- Error Analysis
- README
- Demo
- Resume Metrics

---

# 20. 最低可交付版本

如果开发时间不足，只完成以下四项：

## Must Have 1：Agent Evaluation

```text
500+ Agent Cases
+
Routing / Memory / Safety / Recommendation Metrics
```

## Must Have 2：Agent State + Policy

```text
Perception
→ State
→ Hybrid Policy
→ Action
```

## Must Have 3：Memory 2.0

```text
Write Gate
+
Memory Type
+
Conflict Resolution
+
Feedback Update
```

## Must Have 4：Multi-turn Risk

```text
Single-turn Risk
+
Context
+
Trend
+
History
```

这四项完成后，项目已经足以形成一次明显的简历升级。

---

# 21. 最终项目验收标准

最终版本至少满足：

### Architecture

- [ ] 所有模块统一通过 Agent State 协作
- [ ] Agent Action Space 明确定义
- [ ] 存在 Hybrid Policy
- [ ] Memory 支持完整生命周期
- [ ] Risk 支持多轮状态
- [ ] Recommendation 支持反馈闭环
- [ ] 全链路可 Trace

### Evaluation

- [ ] Agent Benchmark ≥ 500 cases
- [ ] Intent 有 Macro F1
- [ ] Risk 有 High-risk Recall
- [ ] Memory 有 Precision / Recall
- [ ] Retrieval 有 Recall@K / MRR / NDCG
- [ ] Recommendation 有 Trigger Precision / Recall
- [ ] Routing 有 Action Accuracy
- [ ] E2E 有 Rubric Evaluation

### Experiment

- [ ] Intent Ablation
- [ ] Memory Ablation
- [ ] Risk Ablation
- [ ] Retrieval Ablation
- [ ] Recommendation Gate Ablation
- [ ] Agent Policy Ablation

### Engineering

- [ ] Unit Test
- [ ] Integration Test
- [ ] Regression Test
- [ ] Trace Replay
- [ ] Error Taxonomy
- [ ] Config Switch
- [ ] Baseline 保留

---

# 22. 最终简历可提炼的成果方向

最终不要只写：

```text
实现 Intent、Memory、Risk、Recommendation 等模块
```

而要尽量形成量化结果。

目标表述方向：

### Agent Policy

```text
设计状态驱动的 Hybrid Agent Policy，
融合 Intent、Emotion、Risk、Memory 与 Recommendation Signals，
统一完成对话、记忆检索、推荐与安全干预路由；
在自建 XXX 条 Agent Benchmark 上，
Action Routing Accuracy 达 XX%，
高风险 Action Recall 达 XX%。
```

### Memory

```text
重构长期记忆系统，
实现 Memory Write Gate、类型化存储、
冲突消解与反馈驱动更新；
Memory Retrieval Precision / Recall 达 XX% / XX%，
多轮个性化评分提升 XX%。
```

### Risk

```text
设计多轮动态风险状态，
融合单轮风险结果、上下文与历史风险趋势，
高风险 Recall 达 XX%，
风险提前识别率提升 XX%。
```

### Recommendation

```text
构建 Recommendation Feedback Loop，
将用户显式/隐式反馈写入长期记忆并参与后续排序；
在推荐 Benchmark 上 NDCG@5 提升 XX%，
重复推荐率下降 XX%。
```

---

# 23. 最终技术故事线

整个项目最终应形成一条清晰技术主线：

```text
模型理解用户
↓
State 表示用户当前状态
↓
Policy 决定 Agent 应采取的动作
↓
Memory 提供长期个性化上下文
↓
Recommendation / Safety 执行动作
↓
Feedback 更新状态与记忆
↓
Evaluation 判断系统是否真的变好
```

这比继续增加 Agent Tool 更能体现算法设计、系统设计、实验验证和工程闭环能力。

---

# 24. 推荐立即开始的第一批任务

Codex 第一轮建议只做以下任务：

```text
Task A
审计当前代码并生成 current_agent_architecture.md

Task B
定义 Agent Benchmark Schema

Task C
生成第一版 100 条人工可审查 Benchmark

Task D
实现 evaluation runner

Task E
跑当前系统 Baseline

Task F
输出 baseline_report.md
```

在 Baseline Report 完成之前，不建议直接开始 Agent State 或 Memory 大规模重构。

原因是：

```text
没有 baseline
=
后续无法证明架构升级有效
```

因此整个开发计划的真正起点不是“写新模块”，而是先建立：

```text
可复现的 Agent Evaluation Pipeline
```
