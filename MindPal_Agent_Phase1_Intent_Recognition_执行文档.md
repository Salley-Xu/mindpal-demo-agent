# MindPal Agent Phase 1：Intent Recognition 执行文档

> 版本：v1.0  
> 日期：2026-08-19  
> 阶段：Phase 1 — Intent Recognition  
> 执行对象：Codex  
> 前置状态：
> - Phase 0：PASS
> - Phase 0.5：PASS
> - Agent Benchmark：v1.1 Frozen（362 cases）
> - ActionPlan Schema：Frozen
> - IntentResult Runtime Interface：Frozen
> - Risk Root Cause：已定位为 Raw BERT，本阶段不处理 Risk

---

# 1. Phase 1 目标

当前系统只有嵌入在 `emotion_analyzer` 中的浅层关键词意图：

```text
sharing
seeking_relief
planning
seeking_help
```

该信号属于“压力情境下的表达意图”，不能覆盖 Agent Policy 所需要的决策意图。

Benchmark v1.1 中当前 Legacy Intent Coverage：

```text
Macro F1 ≈ 0.161
```

且多个 Agent Intent 标签当前系统无法识别。

Phase 1 的目标是构建独立 Intent Recognition Layer：

```text
User Input
    ↓
Intent Classifier
    ↓
Confidence / Open-set Detection
    ↓
if confident:
    classifier result
else:
    LLM fallback
    ↓
IntentResult
```

最终为 Phase 2 AgentState 提供稳定、结构化、可评测的 Intent 信号。

---

# 2. 最终冻结接口

Phase 1 输出必须严格遵循：

```python
IntentResult = {
    "labels": List[str],
    "confidence": float,
    "is_open_set": bool,
    "source": str,
}
```

其中：

```text
source ∈ {
    "classifier",
    "llm",
    "rule"
}
```

推荐 Runtime 内部额外保留：

```python
IntentResultInternal = {
    "labels": List[str],
    "confidence": float,
    "label_scores": Dict[str, float],
    "is_open_set": bool,
    "source": str,
    "fallback_reason": Optional[str],
    "latency_ms": float,
}
```

注意：

> `label_scores` 仅作为运行时调试 / calibration / trace 字段，不修改 Benchmark v1.1 Schema。

---

# 3. Intent Taxonomy v1

沿用 Benchmark v1.1 已冻结的 10 类：

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

支持：

```text
multi-label
```

不新增标签，除非 Annotation Audit 发现严重不可分问题。

Phase 1 原则：

> 先证明现有 Taxonomy 可以稳定标注，再训练模型。

---

# 4. 本阶段核心问题

Phase 1 必须解决四个问题：

## Q1：标签边界是否足够清晰？

例如：

```text
explicit_help_request
vs
resource_request
```

以及：

```text
follow_up
vs
information_request
```

必须通过 Annotation Guideline 明确。

## Q2：Small Model 能否稳定完成高频 Intent？

目标：

```text
Macro F1 >= 0.85
```

## Q3：如何处理低置信度 / 未知 Intent？

需要：

```text
confidence calibration
+
open-set detection
```

## Q4：是否需要所有请求都调用 LLM？

目标：

```text
Small Model
+
LLM Fallback
```

在保证效果的同时降低：

```text
LLM Call Rate
Latency
Cost
```

---

# 5. 总体执行顺序

必须按以下顺序执行：

```text
Task 1.1  Taxonomy Audit + Annotation Guideline
        ↓
Task 1.2  Seed Dataset（500–800）
        ↓
Task 1.3  Dataset Quality Audit
        ↓
Task 1.4  Legacy Rule Baseline
        ↓
Task 1.5  LLM-only Baseline
        ↓
Task 1.6  Small Model Baseline
        ↓
Task 1.7  Dataset 扩充到 3000–5000
        ↓
Task 1.8  Threshold + Calibration
        ↓
Task 1.9  Open-set Detection
        ↓
Task 1.10 LLM Fallback
        ↓
Task 1.11 Hybrid Intent
        ↓
Task 1.12 Agent Benchmark v1.1 Regression
        ↓
Task 1.13 Ablation + Final Review
```

---

# 6. Task 1.1：Taxonomy Audit

## 6.1 目标

在训练前确认 10 类标签：

```text
定义清晰
边界稳定
可 multi-label
可被模型学习
```

必须新增：

```text
docs/intent_annotation_guideline.md
```

每个标签必须包含：

```text
定义
正例
反例
容易混淆的标签
multi-label 规则
边界规则
```

---

# 7. 标签定义建议

## 7.1 casual_chat

定义：

> 不以压力管理、信息获取、求助、资源推荐、安全表达或历史引用为主要目的的普通交流。

正例：

```text
“你在吗？”
“今天还挺开心的。”
“陪我聊会儿吧。”
```

推荐规则：

> 一旦存在更具体任务意图，原则上不额外标 casual_chat。

## 7.2 emotional_expression

定义：

> 用户主要在表达、描述、倾诉自己的情绪或心理体验，没有明确提出“希望 Agent 做什么”。

正例：

```text
“最近压力特别大。”
“我现在好焦虑。”
“今天真的很累。”
```

如果：

```text
“我很焦虑，有什么办法吗？”
```

标注：

```text
emotional_expression
+
explicit_help_request
```

## 7.3 explicit_help_request

定义：

> 用户明确请求帮助、建议、方法、行动方案或支持。

正例：

```text
“我现在应该怎么办？”
“怎么才能缓解这种焦虑？”
“你能帮我想想办法吗？”
```

## 7.4 resource_request

定义：

> 用户明确请求可消费或可执行的资源，如练习、音频、文章、课程、工具、内容推荐。

正例：

```text
“推荐几个缓解压力的练习。”
“有没有适合睡前听的冥想音频？”
```

边界：

```text
“我应该怎么办？”
→ explicit_help_request

“推荐一个呼吸练习给我。”
→ resource_request
```

## 7.5 information_request

定义：

> 用户主要请求事实、解释、概念、机制或知识性回答，而不是请求个性化支持方案。

正例：

```text
“什么是正念？”
“焦虑和压力有什么区别？”
```

## 7.6 feedback

定义：

> 用户对之前的建议、推荐、练习或 Agent 行为给出评价。

正例：

```text
“你上次推荐的呼吸法没什么用。”
“那个冥想我试了，感觉不错。”
```

## 7.7 follow_up

定义：

> 当前请求依赖上一轮对话语境，属于对上一轮内容的追问、继续或细化。

正例：

```text
“刚才那个方法具体怎么做？”
“你说的第二点是什么意思？”
```

## 7.8 memory_reference

定义：

> 用户显式或隐式引用过去跨轮 / 跨会话已表达的事实、事件、偏好或经历，当前回复需要历史记忆支撑。

正例：

```text
“我上次跟你说的那个项目黄了。”
“还是之前那个室友的问题。”
```

规则：

```text
“刚才……”
→ follow_up

“上次 / 之前我跟你说过……”
→ memory_reference
```

## 7.9 high_risk_expression

定义：

> 当前文本存在自伤、自杀、严重危机或明确危险倾向的高风险表达。

注意：

```text
high_risk_expression = Intent Signal
risk_level = Risk Module Output
```

两者不得合并。

## 7.10 meta_question

定义：

> 用户询问 Agent 本身、能力、身份、数据、系统行为。

正例：

```text
“你是机器人吗？”
“你会记住我说的话吗？”
```

---

# 8. Annotation Ambiguity Rules

必须明确以下混淆对：

```text
emotional_expression vs explicit_help_request
explicit_help_request vs resource_request
information_request vs resource_request
follow_up vs memory_reference
feedback vs follow_up
high_risk_expression vs emotional_expression
meta_question vs information_request
```

每组至少提供：

```text
5 个边界例子
```

---

# 9. Annotation Quality 验收

从 Seed Dataset 随机抽：

```text
100 cases
```

做：

```text
人工 Gold
vs
第二标注者或独立 LLM Review
```

如果有双人工标注，计算：

```text
Cohen's Kappa
```

目标：

```text
Kappa >= 0.80
```

如果使用人工 Gold vs LLM Review：

```text
冲突率 <= 10%
```

若：

```text
冲突率 > 15%
```

必须先修改 Guideline，不进入模型训练。

---

# 10. Task 1.2：Seed Intent Dataset

## 10.1 规模

先做：

```text
500–800 条
```

推荐：

```text
600–700
```

不要一开始直接生成 5000 条。

## 10.2 Seed Dataset 分布

每个标签至少：

```text
60 positive examples
```

建议：

| Intent | Positive 数量 |
|---|---:|
| casual_chat | 70 |
| emotional_expression | 100 |
| explicit_help_request | 90 |
| information_request | 70 |
| resource_request | 70 |
| feedback | 60 |
| follow_up | 60 |
| memory_reference | 60 |
| high_risk_expression | 80 |
| meta_question | 60 |

Multi-label：

```text
>= 30%
```

---

# 11. Seed Dataset Schema

新增：

```text
data/intent/
├── intent_seed_v1.jsonl
└── intent_schema.py
```

示例：

```json
{
  "id": "intent_seed_0001",
  "conversation": [
    {
      "role": "user",
      "content": "最近真的很焦虑，有什么办法能让我缓一缓吗？"
    }
  ],
  "text": "最近真的很焦虑，有什么办法能让我缓一缓吗？",
  "labels": [
    "emotional_expression",
    "explicit_help_request"
  ],
  "source": "human",
  "difficulty": "medium",
  "notes": null
}
```

---

# 12. 必须覆盖的数据类型

Seed 必须包含：

```text
直接表达
隐式表达
口语
短句
长句
否定句
反问
模糊表达
多 Intent
上一轮追问
历史引用
第三方表述
安全讨论型负样本
```

---

# 13. Hard Negative

必须专门加入：

```text
至少 100 条
```

Hard Negative。

例如：

```text
“我在写一篇关于自杀预防的论文。”
```

不要标：

```text
high_risk_expression
```

例如：

```text
“你刚才说焦虑会影响睡眠，那为什么？”
```

标：

```text
follow_up
+
information_request
```

而不是：

```text
explicit_help_request
```

---

# 14. Task 1.3：Dataset Quality Audit

输出：

```text
docs/intent_seed_data_report.md
```

至少统计：

```text
总样本数
每标签 positive 数
multi-label 占比
平均标签数
文本长度分布
source 分布
difficulty 分布
Exact Duplicate
MinHash Duplicate
标签共现矩阵
```

重点分析：

```text
哪些标签共现最多
哪些标签几乎完全重叠
哪些标签样本不足
```

如果两类：

```text
>80% 共现
```

需要重新检查 taxonomy 是否冗余。

---

# 15. Task 1.4：Legacy Rule Baseline

目的：

> 固化当前系统意图能力的正式 baseline。

当前 Benchmark v1.1：

```text
Legacy Intent Coverage Macro F1 ≈ 0.161
```

需要在 Intent Seed/Test 上重新跑一次。

输出：

```text
evaluation/intent/reports/legacy_rule_baseline.md
```

---

# 16. Task 1.5：LLM-only Baseline

LLM-only 不作为最终方案，而作为：

```text
性能上界参考
+
数据质量检查器
+
fallback baseline
```

LLM 只输出结构化 JSON：

```json
{
  "labels": [
    "emotional_expression",
    "explicit_help_request"
  ],
  "confidence": 0.91
}
```

禁止创建新标签。

记录：

```text
Macro F1
Micro F1
Exact Match
Per-label F1
Average Labels / Sample
Invalid JSON Rate
Latency
Cost / 1k Samples
```

---

# 17. Task 1.6：Small Model Baseline

## 17.1 模型路线

第一版只比较：

```text
BERT / RoBERTa family
```

推荐：

```text
Chinese RoBERTa / MacBERT / 当前项目已有中文 BERT backbone
```

优先：

```text
本地可稳定训练
复现简单
推理快
```

## 17.2 模型结构

```text
Encoder
↓
[CLS]
↓
Linear(10)
↓
Sigmoid
```

Loss：

```text
BCEWithLogitsLoss
```

不要使用 softmax。

---

# 18. 上下文输入实验

第一版 Baseline：

```text
current-turn only
```

然后单独实验：

```text
current-turn only
vs
context-aware input
```

Context-aware 重点看：

```text
follow_up
memory_reference
feedback
```

不要默认所有 Intent 都需要历史上下文。

---

# 19. 数据划分

Seed 阶段：

```text
train 70%
dev   15%
test  15%
```

要求尽可能：

```text
按场景模板 Group Split
```

禁止同模板改写同时进入 train/test。

---

# 20. Small Model 指标

必须记录：

```text
Macro F1
Micro F1
Exact Match
Per-label Precision
Per-label Recall
Per-label F1
Hamming Loss
Subset Accuracy
Latency
Throughput
Model Size
```

---

# 21. Task 1.7：扩充正式 Intent Dataset

Seed Guideline 通过后，再扩充到：

```text
3000–5000 条
```

推荐：

```text
约 4000 条
```

流程：

```text
Seed
↓
LLM Scenario Expansion
↓
Paraphrase
↓
Hard Negative Generation
↓
Human Review
```

禁止：

```text
LLM 生成 text + labels
↓
未经审核直接训练
```

---

# 22. 正式数据组成建议

```text
Human / Template Seed      15–20%
LLM Expansion              50–60%
Hard Negative              15–20%
Adversarial / Ambiguous    10–15%
```

允许重叠。

---

# 23. 正式 Test Set

单独建设：

```text
500–800 条
```

要求：

```text
不参与 prompt 示例
不参与训练
不参与 threshold 拟合
不参与 LLM 数据生成 seed
```

推荐文件：

```text
intent_test_v1.jsonl
```

---

# 24. Leakage Audit

必须检查：

```text
Exact Match
MinHash
Embedding Similarity
Template Group Leakage
```

Train/Test：

```text
cosine_similarity >= 0.92
```

进入人工审查队列。

---

# 25. Task 1.8：Threshold + Calibration

multi-label 不能只用固定：

```text
0.5
```

比较：

```text
A. global threshold = 0.5
B. optimized global threshold
C. per-label threshold
```

推荐优先：

```text
per-label threshold
```

如果 Macro F1 有稳定提升。

---

# 26. high_risk_expression 特殊阈值

对：

```text
high_risk_expression
```

优先 Recall。

允许单独使用较低 threshold。

但注意：

> Intent 的 high_risk_expression 不能替代 Risk Module，只是给后续 Policy 的辅助信号。

---

# 27. Calibration

至少比较：

```text
Raw Sigmoid
Temperature Scaling
```

可选：

```text
Platt Scaling / Logistic Calibration
```

输出：

```text
ECE
Brier Score
Reliability Plot
```

---

# 28. Task 1.9：Open-set Detection

目标：

> 识别不属于现有 10 类或模型不确定的输入。

例如：

```text
代码请求
购物咨询
旅行请求
复杂跨域问题
完全无意义输入
```

Open-set 不新增：

```text
other
```

标签。

而输出：

```text
is_open_set = true
```

---

# 29. OOD Dataset

额外构造：

```text
300–500 条
```

来源：

```text
编程
数学
旅行
购物
天气
娱乐
行政
跨域请求
乱码/无意义
```

---

# 30. Open-set Strategy

第一版优先：

```text
max positive score < threshold
```

可进一步比较：

```text
entropy / uncertainty
embedding prototype distance
```

指标：

```text
OOD Recall
OOD Precision
In-domain False Reject Rate
AUROC
```

目标：

```text
OOD Recall >= 0.80
In-domain False Reject <= 0.10
```

---

# 31. Task 1.10：LLM Fallback

触发逻辑建议：

```python
if is_open_set:
    fallback = True
elif confidence < CONF_THRESHOLD:
    fallback = True
else:
    fallback = False
```

第一版不要写复杂冲突规则。

LLM 只能输出已有 10 类。

---

# 32. Task 1.11：Hybrid Intent

最终流程：

```text
User Input
↓
Small Model
↓
label_scores
↓
Calibration
↓
Open-set / Confidence Gate
     ├─ high confidence → classifier
     └─ uncertain      → LLM fallback
↓
IntentResult
```

---

# 33. Hybrid 实验矩阵

必须对比：

```text
A Legacy Rule
B LLM-only
C Small Model
D Small Model + Calibration
E Small Model + LLM Fallback
```

可选：

```text
F Context-aware Small Model
```

最终主表：

| Method | Macro F1 | Micro F1 | Exact Match | OOD Recall | LLM Call Rate | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|
| Legacy Rule | | | | | 0% | |
| LLM-only | | | | | 100% | |
| Small Model | | | | | 0% | |
| + Calibration | | | | | 0% | |
| Hybrid | | | | | X% | |

---

# 34. 成本 / 延迟指标

至少记录：

```text
Classifier latency
LLM latency
Hybrid average latency
LLM fallback rate
Average cost / 1000 requests
```

Phase 1 要证明的是：

> Hybrid 在接近 LLM-only 效果时，以明显更低调用率、延迟和成本完成 Intent Recognition。

---

# 35. Phase 1 目标指标

建议：

```text
Macro F1 >= 0.85
Micro F1 >= 0.90
```

重点标签：

```text
high_risk_expression Recall >= 0.95
memory_reference F1 >= 0.85
feedback F1 >= 0.85
resource_request F1 >= 0.85
```

Open-set：

```text
OOD Recall >= 0.80
In-domain False Reject <= 0.10
```

Hybrid：

```text
LLM Call Rate <= 30%
```

如果达不到，不要无限调参，进入 Error Analysis。

---

# 36. Task 1.12：Agent Benchmark v1.1 Regression

Intent 改造完成后必须重新跑冻结的：

```text
agent_benchmark_v1_1.jsonl
```

但本阶段只允许 Intent 信号发生变化。

不得修改：

```text
Risk
Memory
Recommendation Gate
Agent Policy
Benchmark v1.1
```

重点观察：

```text
Intent Macro F1
Tool Action signal-derived metrics
Recommendation signal-derived metrics
Primary Action signal-derived metrics
```

注意：

> Phase 1 尚未实现正式 Agent Policy，因此 Routing 改善只能称为 signal-derived proxy。

---

# 37. 负向回归检查

必须确认：

```text
Risk 指标不变
Emotion 指标不变
Benchmark cases 不变
Memory 行为不变
```

如果变化，说明 Phase 1 污染了其他模块。

---

# 38. Task 1.13：Error Analysis

输出：

```text
docs/phase1_intent_error_analysis.md
```

错误类型：

```text
I01 Label Boundary Error
I02 Missing Secondary Intent
I03 Over-labeling
I04 Under-labeling
I05 Follow-up Context Missing
I06 Memory Reference Confusion
I07 Resource vs Help Confusion
I08 Feedback vs Follow-up Confusion
I09 High-risk Intent Miss
I10 Open-set False Accept
I11 Open-set False Reject
I12 LLM Fallback Error
```

每类至少分析：

```text
数量
占比
典型 Case
根因
是否可修
修复优先级
```

重点关注：

```text
高置信度错误
```

---

# 39. 必做 Ablation

## Experiment A

```text
Legacy Rule
vs
Small Model
```

## Experiment B

```text
0.5 Threshold
vs
Optimized Threshold
```

## Experiment C

```text
No Calibration
vs
Calibration
```

## Experiment D

```text
Small Model
vs
Small Model + LLM Fallback
```

## Experiment E

```text
Current-turn only
vs
Context-aware
```

主要看：

```text
follow_up
memory_reference
feedback
```

---

# 40. 推荐代码目录

```text
mindpal/
├── perception/
│   └── intent/
│       ├── __init__.py
│       ├── schema.py
│       ├── classifier.py
│       ├── predictor.py
│       ├── calibration.py
│       ├── open_set.py
│       ├── llm_fallback.py
│       └── service.py
│
├── training/
│   └── intent/
│       ├── dataset.py
│       ├── train.py
│       ├── evaluate.py
│       └── config.yaml
│
├── evaluation/
│   └── intent/
│       ├── datasets/
│       ├── metrics.py
│       ├── run_legacy.py
│       ├── run_llm.py
│       ├── run_classifier.py
│       ├── run_hybrid.py
│       └── reports/
│
└── docs/
```

如果现有目录结构不同，优先适配现有工程。

---

# 41. Config

关键参数必须配置化：

```yaml
intent:
  classifier_model: "..."
  global_threshold: 0.5
  per_label_thresholds: {}
  open_set_threshold: 0.35
  fallback_confidence_threshold: 0.55
  enable_llm_fallback: true
  max_context_turns: 3
```

不得散落硬编码。

---

# 42. 测试要求

Unit Test：

```text
test_intent_schema.py
test_intent_threshold.py
test_intent_calibration.py
test_open_set.py
test_llm_fallback.py
```

Dataset Test：

```text
Schema
Label 合法
ID 唯一
空文本
重复
Train/Test 泄漏
```

Integration Test：

```text
high confidence classifier
low confidence fallback
open-set fallback
multi-label
follow-up context
memory reference
high-risk expression
```

Regression：

```text
Agent Benchmark v1.1 / 362 cases
```

必须跑。

---

# 43. Phase 1 交付物

最终必须有：

```text
docs/
├── intent_annotation_guideline.md
├── intent_seed_data_report.md
├── intent_dataset_report.md
├── phase1_intent_error_analysis.md
├── phase1_ablation_report.md
└── phase1_final_review.md

data/intent/
├── intent_seed_v1.jsonl
├── intent_train_v1.jsonl
├── intent_dev_v1.jsonl
├── intent_test_v1.jsonl
└── intent_ood_v1.jsonl

evaluation/intent/reports/
├── legacy_rule_baseline.*
├── llm_only_baseline.*
├── classifier_baseline.*
├── calibrated_classifier.*
└── hybrid_intent.*

models/
└── intent/
    └── best_model/
```

---

# 44. Phase 1 Final Review 必答问题

`docs/phase1_final_review.md` 必须回答：

1. Taxonomy 是否可稳定标注：PASS / FAIL
2. 最佳 Small Model 是什么：模型、参数、Macro F1、Micro F1
3. Calibration 是否有效：before vs after
4. Open-set 是否有效：OOD Recall / Precision / False Reject
5. Hybrid 是否值得：效果 / 延迟 / LLM Call Rate / 成本
6. Phase 2 AgentState 应接入哪个 IntentResult 实现

---

# 45. 阶段停止条件

## 条件 A

如果：

```text
Small Model Macro F1 >= 0.85
Hybrid LLM Call Rate <= 30%
```

Phase 1 通过。

## 条件 B

如果：

```text
Small Model Macro F1 = 0.80–0.85
Hybrid Macro F1 >= 0.88
```

且成本可接受，也可以 PASS。

## 条件 C

如果：

```text
Taxonomy Agreement < 0.80
```

退回 Annotation Guideline，不继续训练。

## 条件 D

如果大量高置信度错误存在：

> 优先修数据和标签边界，不继续调 fallback threshold。

---

# 46. Git Branch 建议

```text
feat/intent-taxonomy
feat/intent-dataset
exp/intent-llm-baseline
exp/intent-small-model
feat/intent-calibration
feat/intent-open-set
feat/intent-llm-fallback
exp/intent-hybrid
```

Commit 示例：

```text
docs(intent): add annotation guideline
data(intent): add reviewed seed dataset
eval(intent): add legacy and llm baselines
feat(intent): add multilabel classifier
feat(intent): add confidence calibration
feat(intent): add open-set detection
feat(intent): add llm fallback
eval(intent): add hybrid ablation report
```

---

# 47. 本阶段禁止事项

Phase 1 不做：

```text
Risk 2.0
AgentState 重构
Agent Policy
Memory 2.0
Recommendation Gate 调参
Recommendation Ranking 重构
Trace 全链路重构
```

尤其：

> 不要因为新 Intent 信号出来后 Gate 指标仍然差，就在 Phase 1 顺手调 Recommendation Gate。

---

# 48. 建议时间安排

## Day 1

```text
Taxonomy Audit
Annotation Guideline
```

## Day 2–3

```text
Seed Dataset
Quality Audit
```

## Day 4

```text
Legacy Rule
LLM-only Baseline
```

## Day 5–6

```text
Small Model Baseline
```

## Day 7–8

```text
Dataset Expansion
正式训练
```

## Day 9

```text
Threshold
Calibration
```

## Day 10

```text
Open-set
```

## Day 11

```text
LLM Fallback
Hybrid
```

## Day 12

```text
Regression
Ablation
Error Analysis
Final Review
```

---

# 49. Codex 启动指令

```text
请执行 Phase 1：Intent Recognition。

前置条件：

Phase 0 / 0.5 已冻结。
不得修改：
- agent_benchmark_v1_1.jsonl
- Benchmark v1.1 Schema 语义
- Risk 模块
- Memory 模块
- Recommendation Gate
- Agent Policy

首先阅读：

1. docs/phase0_final_review.md
2. docs/benchmark_schema_v1_1.md
3. evaluation/benchmark_schema.py
4. evaluation/datasets/agent_benchmark_v1_1.jsonl
5. 当前 emotion_analyzer 中 legacy user_intent 实现
6. 当前 evaluation runner / metrics
7. 本执行文档

严格按照：
Task 1.1 → Task 1.13
执行。

第一步不要训练模型。

先完成：
- intent_annotation_guideline.md
- 500–800 条 Seed Dataset
- Seed Quality Audit

只有当 Taxonomy / Annotation 通过审核后，才能进入 Small Model Training。

训练阶段必须同时保留：
- Legacy Rule baseline
- LLM-only baseline
- Small Model baseline
- Hybrid baseline

不得删除任何 baseline。

最终必须输出：
- intent_annotation_guideline.md
- intent_dataset_report.md
- phase1_intent_error_analysis.md
- phase1_ablation_report.md
- phase1_final_review.md

phase1_final_review.md 必须回答：

1. Taxonomy 是否稳定；
2. Small Model 最佳性能；
3. Calibration 是否有效；
4. Open-set 是否有效；
5. Hybrid 是否值得；
6. Phase 2 应接入哪个 IntentResult 实现。

所有模型改动完成后必须跑冻结的：
agent_benchmark_v1_1.jsonl
进行 regression。

如果发现 Agent Benchmark 需要修改：
不要直接修改。
先记录为 benchmark_change_request.md，
留到后续统一处理。
```

---

# 50. Phase 1 完成后的预期状态

完成后系统应从：

```text
User
↓
emotion_analyzer._detect_user_intent()
↓
4 类关键词规则
```

升级为：

```text
User
↓
Intent Recognition Layer
├── Small Multi-label Classifier
├── Calibration
├── Open-set Detection
└── LLM Fallback
↓
IntentResult
↓
Phase 2 AgentState
```

Phase 1 的核心价值不是“多训练一个 BERT”，而是：

> **为整个 Agent 决策系统建立一个独立、低成本、可校准、可 fallback、可评测的 Intent 感知层。**
