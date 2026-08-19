# MindPal Agent Phase 1.5：Intent Recognition 收尾执行计划

> 版本：v1.0  
> 日期：2026-08-19  
> 阶段定位：Phase 1 Core 已通过后的冻结前收尾  
> 执行对象：Codex  
> 目标：解决 Phase 1 当前剩余的上下文、泛化、独立评测与 uncertainty 定义问题，并正式冻结 Intent Recognition Layer。

---

# 1. 前置状态

Phase 1 已完成并验证：

- 10 类 multi-label Intent Taxonomy
- Annotation Guideline
- Seed Dataset：806 条
- Expanded Dataset：1052 条
- MacBERT multi-label classifier
- BCEWithLogitsLoss
- per-label threshold
- temperature scaling
- Legacy Rule baseline
- LLM-only baseline
- Small Model baseline
- Hybrid fallback 原型
- Intent Error Analysis
- Phase 1 Ablation

当前最佳开发集结果：

```text
Model: hfl/chinese-macbert-base
Macro F1: 0.854
Micro F1: 0.845
Exact Match: 0.689
Latency: ~100 ms CPU
```

但 Phase 1 尚未完全冻结，主要剩余问题：

```text
1. follow_up 对上下文高度依赖，current-turn-only F1 极低
2. 正式独立 Test Set 尚未建立
3. Expanded Dataset 仍以模板派生为主，泛化能力未充分验证
4. Open-set 当前把 domain OOD 与 intent OOD 混在一起
5. Hybrid / LLM / Classifier 并未在完全相同 Test Set 上严格比较
6. high_risk_expression Recall 0.88，低于 0.95 目标
```

---

# 2. Phase 1.5 总目标

本阶段不新增 Intent 标签，不重构 AgentState，不做 Agent Policy。

只解决四个问题：

```text
A. Context-aware Intent
B. Targeted Data Expansion
C. Independent Test Set
D. Intent Uncertainty / Open-set Redefinition
```

最终目标：

```text
User + Short Context
        ↓
Context-aware Intent Classifier
        ↓
Calibration + Per-label Threshold
        ↓
Intent Uncertainty Gate
        ├─ confident → classifier
        └─ uncertain → LLM fallback
        ↓
IntentResult
```

完成后：

```text
Phase 1 = FINAL PASS
Intent Layer = FROZEN
Phase 2 = READY
```

---

# 3. 本阶段禁止事项

Phase 1.5 不做：

```text
Risk 2.0
AgentState
Agent Policy
Memory 2.0
Recommendation Gate 调参
Recommendation Rerank
全链路 Trace
新 Intent Taxonomy
Multi-Agent
```

原则：

> 只修正 Intent Layer 本身，不污染其他模块。

---

# 4. 执行顺序

必须严格按以下顺序：

```text
Task 1.5.1  Context-aware Intent
        ↓
Task 1.5.2  Targeted Data Expansion
        ↓
Task 1.5.3  Independent Test Set
        ↓
Task 1.5.4  Intent Uncertainty / Open-set Redefinition
        ↓
Task 1.5.5  Unified Evaluation
        ↓
Task 1.5.6  Error Analysis + Final Freeze
```

---

# 5. Task 1.5.1：Context-aware Intent

## 5.1 目标

解决以下标签在 current-turn-only 模式下信息不足的问题：

```text
follow_up
memory_reference
feedback
```

特别是 `follow_up`，当前在 Agent Benchmark 上表现极弱，主要原因不是分类器本身，而是输入缺少上一轮上下文。

## 5.2 Context 输入格式

必须比较三种输入形式。

### Variant A：Current Turn Only

```text
[current_user]
```

作为现有 baseline。

### Variant B：Previous 1 Turn

推荐：

```text
[PREV_ASSISTANT] ...
[CURRENT_USER] ...
```

或者 tokenizer pair：

```python
tokenizer(
    previous_assistant,
    current_user,
    ...
)
```

### Variant C：Previous 2 Turns

```text
[PREV_USER]
[PREV_ASSISTANT]
[CURRENT_USER]
```

不建议超过 2 个历史 turn。

## 5.3 Context 实验要求

保持：

```text
同一 backbone
同一 train/dev/test
同一 threshold search
同一 calibration 方法
```

只改变：

```text
输入上下文
```

比较：

| Input | Macro F1 | follow_up F1 | memory_reference F1 | feedback F1 | Latency |
|---|---:|---:|---:|---:|---:|
| Current only | | | | | |
| +1 turn | | | | | |
| +2 turns | | | | | |

## 5.4 Context-aware 验收

优先关注：

```text
follow_up F1 >= 0.75
```

同时：

```text
memory_reference F1 不下降 > 0.03
feedback F1 不下降 > 0.03
整体 Macro F1 不下降 > 0.02
```

如果 `+2 turns` 相比 `+1 turn` 没有明显收益：

> 选择 Previous 1 Turn，减少输入长度与推理成本。

---

# 6. Task 1.5.2：Targeted Data Expansion

## 6.1 目标

当前 1052 条数据已经能跑通流程，但模板占比高。

本阶段不机械追求 5000 条。

目标扩充至：

```text
2500–3500 条
```

推荐：

```text
约 3000 条
```

重点提升：

```text
多标签覆盖
上下文依赖
隐式高风险表达
hard negative
模板外语言多样性
```

## 6.2 P0：Implicit High-risk

重点扩：

```text
隐式告别
未来绝望
消失表达
不想醒来
撑不到下次
最后一次联系
负担感
彻底离开
```

例如：

```text
“以后可能不会再跟你聊了。”
“我觉得自己可能撑不到周末。”
“如果明天不用醒来就好了。”
“这是最后一次和别人说这些。”
```

目标：

```text
新增 150–250 条
```

要求：不能只靠固定关键词。

## 6.3 P0：follow_up Context Cases

必须构造真实上下文：

```json
{
  "conversation": [
    {"role": "assistant", "content": "你可以先试试呼吸训练和渐进式肌肉放松。"},
    {"role": "user", "content": "第二个具体怎么做？"}
  ],
  "labels": [
    "follow_up",
    "information_request"
  ]
}
```

目标：

```text
新增 250–350 条
```

## 6.4 P1：Multi-label Combination

重点补：

```text
feedback + memory_reference
follow_up + information_request
emotional_expression + explicit_help_request
emotional_expression + resource_request
high_risk_expression + explicit_help_request
memory_reference + emotional_expression
feedback + resource_request
```

目标：

```text
新增 400–600 条
```

## 6.5 P1：Hard Negative

重点：

```text
风险讨论
否定风险
影视/论文语境
第三方引用但非求助
引用“死/结束”等词但无风险
```

目标新增：

```text
150–200 条
```

## 6.6 P2：语言多样性

加入：

```text
口语
缩写
不完整句
反问
网络表达
长句
轻微错别字
省略主语
```

## 6.7 数据来源建议

```text
Human / Existing Seed        25–30%
LLM Scenario Expansion       40–50%
Hard Negative / Adversarial  15–20%
Context Cases                15–20%
```

可以重叠。

LLM 只能用于：

```text
生成候选
改写
扩写
hard negative 生成
```

最终标签必须经过 review。

## 6.8 Dataset Split

必须放弃粗粒度模板桶 Group Split 导致的严重标签失衡。

采用：

```text
Stratified Multi-label Group Split
```

目标：既防模板泄漏，又保证标签分布。

建议：

```text
train 70%
dev   15%
test  15%
```

这里的 test 仍是开发测试，不替代 Task 1.5.3 的 Independent Test。

## 6.9 Leakage Audit

必须执行：

```text
Exact Match
MinHash
Embedding Similarity
Template / variant group leakage
```

Embedding 推荐：

```text
bge-small-zh / 当前项目已有 embedding 模型
```

阈值：

```text
cosine >= 0.92
```

进入人工审查。

输出：

```text
docs/phase1_5_dataset_report.md
```

---

# 7. Task 1.5.3：Independent Test Set

## 7.1 目标

构建可以真正用于：

```text
模型选择后的最终评测
简历指标
Phase 1 Freeze
```

的独立 Test。

## 7.2 Independent Test 规模

目标：

```text
500–600 条
```

推荐：

```text
550 条
```

## 7.3 Independent Test 禁止来源

不能：

```text
从训练模板做词替换
从 Seed 直接 paraphrase
从 expanded dataset 改写
使用 training prompt 中的 few-shot 原句
```

必须：

```text
全新场景
全新措辞
全新组合
```

## 7.4 内容比例

建议：

```text
普通单意图             30%
multi-label            30%
context-aware          20%
hard negative          10%
implicit high-risk     10%
```

Intent 标签全部覆盖。

## 7.5 Gold 标注

必须至少完成人工审核。

推荐抽取：

```text
150 条
```

做第二标注。

如果可行：

```text
Cohen's Kappa >= 0.80
```

否则至少：

```text
Disagreement <= 10%
```

## 7.6 Freeze

一旦完成：

```text
intent_test_independent_v1.jsonl
```

立即 Freeze。

后续不得用于：

```text
threshold tuning
calibration
prompt 调整
训练
```

只允许最终 evaluation。

---

# 8. Task 1.5.4：重新定义 Intent Open-set / Uncertainty

## 8.1 当前问题

当前 OOD 使用：

```text
编程
数学
旅游
购物
天气
娱乐
```

作为 Intent OOD。

但这些请求仍可能属于现有 Intent：

```text
information_request
resource_request
explicit_help_request
```

因此：

```text
Domain OOD ≠ Intent OOD
```

必须拆开。

## 8.2 新定义

Phase 1.5 不再使用 Domain OOD 评价 Intent Model。

将 `is_open_set` 重新定义为：

> **Intent Uncertainty：当前输入无法被稳定映射到已有 10 类 Intent。**

## 8.3 Intent Uncertainty Dataset

构造：

```text
200–300 条
```

类型包括：

```text
乱码
信息不足
极端省略
纯符号
语义不完整
多个互相冲突的任务
无法判断行为目的
模型不可确定表达
```

例如：

```text
“那个怎么办”
“？？？”
“反正就那样吧你看着办”
“这个还有别的？”
```

注意：如果有足够 context 可以判断，则不是 open-set。

## 8.4 Domain Scope 留待后续

新增设计说明：

```text
Intent
= 用户想做什么

Domain Scope
= MindPal 是否应该处理

Risk
= 是否安全相关

Emotion
= 用户心理/情绪状态
```

本阶段只实现 Intent Uncertainty，不实现完整 Domain Classifier。

## 8.5 Uncertainty Strategy

优先比较：

```text
A. max calibrated score
B. prediction entropy
C. top1 - top2 margin
```

可选：

```text
embedding prototype distance
```

第一版不强制 embedding OOD。

## 8.6 指标

评估：

```text
Uncertain Recall
Uncertain Precision
In-domain False Reject
AUROC
LLM Fallback Rate
```

目标：

```text
Uncertain Recall >= 0.75
In-domain False Reject <= 0.10
```

不再使用旧 Domain OOD Recall >= 0.80 作为 Phase 1 验收条件。

---

# 9. Task 1.5.5：Unified Evaluation

这是本阶段最关键的实验。

必须使用完全相同的：

```text
intent_test_independent_v1.jsonl
```

比较：

```text
A Legacy Rule
B LLM-only
C Small Model
D Calibrated Small Model
E Context-aware Small Model
F Hybrid
```

## 9.1 Unified Evaluation 主表

| Method | Macro F1 | Micro F1 | Exact Match | High-risk Recall | follow_up F1 | Latency | LLM Call Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy Rule | | | | | | | 0% |
| LLM-only | | | | | | | 100% |
| Small Model | | | | | | | 0% |
| + Calibration | | | | | | | 0% |
| + Context | | | | | | | 0% |
| Hybrid | | | | | | | X% |

## 9.2 Hybrid 重新定义

Hybrid 不再为了提高域内 F1 而设计。

其目标是：

```text
处理 Intent Uncertainty / 长尾
```

触发条件：

```python
if intent_uncertain:
    use_llm = True
elif confidence < fallback_threshold:
    use_llm = True
else:
    use_llm = False
```

## 9.3 Hybrid 目标

推荐：

```text
LLM Call Rate <= 20–30%
```

同时：

```text
Macro F1 不低于最佳 Small Model > 0.02
```

如果 Hybrid F1 更差，但显著提升 uncertain case：

> 可以保留为 optional fallback，不强制设为默认主路径。

---

# 10. 关键标签验收

## 10.1 High-risk Intent

在 Independent Test 上：

```text
high_risk_expression Recall >= 0.95
```

如果仍低于 0.95，继续分析：

```text
显式风险
隐式风险
告别表达
第三方
否定语境
```

不要只调 threshold。

## 10.2 follow_up

在包含 context 的 Independent Test 上：

```text
follow_up F1 >= 0.75
```

如果 current-turn-only << context-aware，则正式冻结：

```text
Intent Service 必须接收 short context
```

---

# 11. Task 1.5.6：Error Analysis

新增：

```text
docs/phase1_5_error_analysis.md
```

重点错误：

```text
C01 Context Missing
C02 Context Noise
C03 Secondary Intent Missing
C04 Implicit High-risk Miss
C05 Hard-negative False Positive
C06 Intent Uncertainty False Accept
C07 Intent Uncertainty False Reject
C08 LLM Fallback Regression
C09 Template Generalization Error
C10 High-confidence Error
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

---

# 12. 必做 Slice Evaluation

Independent Test 除总体指标外，必须单独报告：

```text
single-label
multi-label
context-aware
implicit high-risk
hard-negative
short utterance
long utterance
```

主表：

| Slice | n | Macro F1 | Exact Match |
|---|---:|---:|---:|
| Single-label | | | |
| Multi-label | | | |
| Context-aware | | | |
| Implicit high-risk | | | |
| Hard-negative | | | |

---

# 13. Runtime 接口冻结

Phase 1.5 完成后，正式冻结：

```python
IntentResult = {
    "labels": List[str],
    "confidence": float,
    "is_open_set": bool,
    "source": str,
}
```

内部建议：

```python
IntentResultInternal = {
    "labels": List[str],
    "confidence": float,
    "label_scores": Dict[str, float],
    "is_open_set": bool,
    "source": str,
    "fallback_reason": Optional[str],
    "context_used": int,
    "latency_ms": float,
}
```

Service 接口：

```python
predict_intent(
    current_text: str,
    previous_turns: list | None = None,
) -> IntentResult
```

默认：

```text
previous_turns <= 2
```

不得把完整 conversation history 全塞给 Intent Classifier。

---

# 14. 推荐最终架构

```text
Current User
+
Last 1–2 Turns
      ↓
Context-aware MacBERT
      ↓
Temperature Scaling
      ↓
Per-label Threshold
      ↓
Uncertainty Detector
      ├─ confident
      │     ↓
      │ classifier IntentResult
      │
      └─ uncertain
            ↓
         LLM fallback
            ↓
         IntentResult
```

---

# 15. Phase 1.5 目标指标

最终 Independent Test 建议目标：

```text
Macro F1 >= 0.85
Micro F1 >= 0.88
Exact Match >= 0.70
```

关键标签：

```text
high_risk_expression Recall >= 0.95
follow_up F1 >= 0.75
memory_reference F1 >= 0.85
feedback F1 >= 0.85
resource_request F1 >= 0.85
```

Uncertainty：

```text
Recall >= 0.75
In-domain False Reject <= 0.10
```

Hybrid：

```text
LLM Call Rate <= 30%
```

---

# 16. Final Freeze 条件

Phase 1.5 可以 PASS 的最低条件：

## 必须满足

```text
Independent Test >= 500
Context-aware 实验完成
Unified Evaluation 完成
high_risk_expression Recall >= 0.93
follow_up F1 >= 0.70
Train/Test leakage audit 完成
```

## 推荐满足

```text
Macro F1 >= 0.85
high_risk Recall >= 0.95
follow_up >= 0.75
Hybrid call rate <= 30%
```

如果核心指标略低但 Error Analysis 已充分：

> 可以冻结并进入 Phase 2，不无限调参。

---

# 17. Final Review

必须输出：

```text
docs/phase1_5_final_review.md
```

必须回答：

```text
1. Context-aware 是否必要？
2. 最终模型使用 current / +1 / +2 turn 哪一种？
3. Independent Test 最终指标是多少？
4. high_risk_expression 是否达标？
5. follow_up 是否达标？
6. Intent Uncertainty 新定义是否有效？
7. Hybrid 是否默认开启？
8. Phase 2 应使用哪个 Intent Service？
9. Phase 1 是否正式 Freeze？
```

最后必须写：

```text
Phase 1 Freeze = PASS / FAIL
```

---

# 18. 交付物

必须产出：

```text
docs/
├── phase1_5_dataset_report.md
├── phase1_5_context_ablation.md
├── phase1_5_unified_evaluation.md
├── phase1_5_error_analysis.md
└── phase1_5_final_review.md

data/intent/
├── intent_train_v1_5.jsonl
├── intent_dev_v1_5.jsonl
├── intent_test_dev_v1_5.jsonl
├── intent_test_independent_v1.jsonl
└── intent_uncertainty_v1.jsonl

evaluation/intent/reports/
├── context_ablation.*
├── independent_legacy.*
├── independent_llm.*
├── independent_classifier.*
├── independent_context.*
└── independent_hybrid.*

models/intent/
└── phase1_5_best_model/
```

---

# 19. Git 建议

Branch：

```text
feat/intent-context
data/intent-targeted-expansion
eval/intent-independent-test
feat/intent-uncertainty
exp/intent-unified-eval
```

Commit：

```text
feat(intent): add short-context intent input

data(intent): add targeted multilabel and implicit-risk cases

eval(intent): add frozen independent test set

feat(intent): redefine intent uncertainty detection

eval(intent): add unified phase1.5 comparison

docs(intent): freeze phase1 intent layer
```

---

# 20. 测试要求

Unit：

```text
test_context_builder.py
test_context_intent_predictor.py
test_uncertainty_detector.py
test_hybrid_fallback.py
```

Dataset：

```text
schema
duplicate
MinHash
embedding similarity
split leakage
label distribution
context completeness
```

Integration：

```text
follow_up
memory_reference
feedback
implicit high-risk
hard negative
uncertain input
LLM fallback
```

Regression：

```text
Agent Benchmark v1.1 / 362 cases
```

必须再次运行。

---

# 21. Agent Benchmark Regression

Phase 1.5 允许 Intent 输入接口使用短上下文。

但禁止修改：

```text
Risk
Memory
Recommendation Gate
Agent Policy
Benchmark v1.1
```

报告：

```text
Legacy Intent Coverage
vs
Phase1 Intent
vs
Phase1.5 Context-aware Intent
```

并明确：

> Agent Routing 仍然只是 signal-derived proxy，真正 Policy 在 Phase 3。

---

# 22. 建议执行周期

## Day 1

```text
Context-aware data format
+1/+2 turn experiments
```

## Day 2

```text
Targeted expansion
implicit risk
multi-label
follow-up
```

## Day 3

```text
Independent Test construction
annotation audit
leakage audit
```

## Day 4

```text
Intent uncertainty redefinition
uncertainty detector
```

## Day 5

```text
Unified evaluation
Legacy / LLM / Small / Context / Hybrid
```

## Day 6

```text
Error analysis
Agent Benchmark regression
Final Review
Freeze
```

---

# 23. Codex 启动指令

```text
请执行 Phase 1.5：Intent Recognition 收尾。

前置状态：
- Phase 1 Core PASS
- Benchmark v1.1 Frozen
- 当前最佳 MacBERT + calibration 开发集 Macro F1 = 0.854

本阶段不得修改：
- Intent Taxonomy
- Agent Benchmark v1.1
- Risk 模块
- Memory 模块
- Recommendation Gate
- Agent Policy

严格按以下顺序：

1. Context-aware Intent
2. Targeted Data Expansion
3. Independent Test Set
4. Intent Uncertainty Redefinition
5. Unified Evaluation
6. Error Analysis + Final Freeze

重点要求：

A. Context-aware
必须比较：
- current only
- previous 1 turn
- previous 2 turns

B. 数据扩充
优先增加：
- follow_up
- multi-label combinations
- implicit high-risk
- hard negatives
- memory_reference / feedback

数据总量目标约 2500–3500，不要求机械达到 5000。

C. Independent Test
必须新建 500–600 条完全独立测试集。
不得从 train/seed 做简单词替换或 paraphrase。
一旦创建完成立即 freeze。

D. Open-set
不再把编程/旅游/购物等 domain OOD 当成 Intent OOD。
将 is_open_set 重新定义为：
“当前输入无法被稳定映射到现有 10 类 Intent”。

E. Unified Evaluation
Legacy / LLM-only / Small Model / Calibrated / Context-aware / Hybrid
必须全部在同一个 independent test 上评测。

F. Phase 1.5 Final Review
必须明确：
- context-aware 是否必要
- 最终输入窗口
- Independent Test 指标
- high_risk_expression Recall
- follow_up F1
- Hybrid 是否默认启用
- Intent Service 最终接口
- Phase 1 Freeze = PASS / FAIL

最后必须重新跑：
agent_benchmark_v1_1.jsonl

如果发现需要修改 Benchmark：
不要直接修改。
记录 benchmark_change_request.md。
```

---

# 24. Phase 1.5 完成后的预期状态

完成后：

```text
Phase 0      PASS
Phase 0.5    PASS
Phase 1      FINAL PASS
Phase 1.5    PASS

Intent Taxonomy     FROZEN
Intent Model        FROZEN
Intent Service      FROZEN
Independent Test    FROZEN

Phase 2 AgentState  READY
```

最终 Intent Layer 应具备：

```text
Multi-label
Context-aware
Calibrated
Per-label Threshold
High-risk Recall Priority
Intent Uncertainty Detection
Optional LLM Fallback
Independent Benchmark
```

这样 Phase 2 只需要消费：

```text
IntentResult
```

而不再重新修改 Intent 模型与输入接口。
