# MindPal Agent Phase 8：Final Evaluation / Ablation / Resume Evidence 执行文档

> 目标：建立最终独立系统评测，回答“整个 Agent 是否真的比初始版本更好”。

## 1. 原则

Phase 8 不再调模型。

一旦 Final Benchmark 创建：
```text
立即冻结
```

不得：
- 改规则
- 调阈值
- 改 prompt
- 根据结果重新训练

否则重新污染 final test。

## 2. Final Benchmark v2

建议：
```text
600–800 cases
```

组成：
```text
single-turn
multi-turn
intent ambiguity
memory
risk trajectories
recommendation
feedback
safety
hard negatives
third-party
cross-session personalization
```

同时建立：
```text
Balanced Safety Slice
Natural Distribution Slice
```

## 3. 评测层级

### Module
```text
Intent
Emotion
Risk
Memory
Recommendation
```

### State / Policy
```text
State Transition
Primary Action
Tool Action
Safety Target
Policy Exact
```

### End-to-end
```text
Response appropriateness
Personalization
Safety
Consistency
Over-intervention
Recommendation usefulness
```

## 4. 必做 Ablation

至少：

```text
Baseline Initial System
+ Intent
+ AgentState
+ AgentPolicy
+ Risk 2.0
+ Memory 2.0
+ Recommendation 2.0
Full System
```

专项：

```text
Memory: No / Always Retrieve / Gate / Full
Risk: old / new
Policy: legacy / deterministic / hybrid
Recommendation: old / new
```

## 5. 关键系统指标

建议最终报告：

```text
Safety Recall / FPR
Primary Action Accuracy
Policy Exact Match
Memory Retrieval P/R
Recommendation Trigger P/R
Recommendation Mode Accuracy
Multi-turn Consistency
Over-refusal / Over-safety
LLM Fallback Rate
Latency
Cost
```

## 6. Error Attribution

使用 Phase 7 Trace 对所有错误做：

```text
Perception
State
Policy
Memory
Recommendation
Execution
Response
```

归因。

输出 Top residual error categories。

## 7. Human / LLM Judge

若做 response-level 评测，使用冻结 rubric：
```text
helpfulness
appropriateness
personalization
safety
consistency
non-overreach
```

LLM Judge 必须：
- frozen prompt
- blind comparison
- order randomization
- subset human audit

## 8. 最终报告

必须生成：

```text
docs/final_agent_evaluation.md
docs/final_ablation_report.md
docs/final_error_analysis.md
docs/resume_evidence.md
```

`resume_evidence.md` 只放：
```text
有独立测试支撑
可复现
可解释
不夸大的指标
```

## 9. Resume Evidence 结构

每条经历：

```text
问题
→ 设计
→ 实验
→ 指标
→ 系统收益
```

不要只列模块名。

## 10. Phase 8 PASS

Phase 8 的 PASS 不是单指标，而是：
```text
Safety 改善
Policy 改善
Memory over-retrieval 显著下降
Recommendation 稳定提升
无严重 general regression
关键指标有独立 benchmark 支撑
```

## 11. Codex 启动指令

```text
Phase 8 开始后禁止继续调系统。

先冻结 Final Benchmark v2，再一次性跑所有版本与 ablation。
所有结果自动写 JSON + Markdown，并保留 trace_id。

最终输出可直接用于：
1. 项目 README
2. 技术总结
3. 简历
4. 面试讲解
的证据链。
```
