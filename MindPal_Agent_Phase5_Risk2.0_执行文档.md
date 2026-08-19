# MindPal Agent Phase 5：Risk 2.0 执行文档

> 阶段目标：修复当前 Agent 最大端到端瓶颈——Risk Perception，并建立单轮分类 + 多轮动态风险状态。

## 1. 已知问题

Phase 0.5 已定位：
```text
Raw BERT Macro F1 ≈ 0.364
High-risk Recall ≈ 0.627
FPR ≈ 0.431
FNR ≈ 0.373
```

典型 benign：
```text
“你在吗？我就随便聊聊。”
→ L3 @ 0.9985
```

Full Pipeline ≈ Raw BERT，说明问题主要来自模型本身，而非 SessionAggregator。

此外：
- binary_threshold 已存储但未进入 predict()
- v4_3_coral 未系统比较
- Risk Trend 多轮表现弱
- Phase 3 predicted-state Safety/Primary 被 Risk 明显拖累

## 2. 最终架构

```text
Current Turn
+ Recent Context
+ Risk History
+ Intent high-risk signal
        ↓
Single-turn Risk Model
        ↓
Calibration / Fusion
        ↓
Context Safety Rules
        ↓
Dynamic Risk State
        ↓
RiskResult v2
        ↓
AgentState.risk
```

## 3. 执行顺序

```text
5.1 Risk Dataset Audit
5.2 Checkpoint Shootout
5.3 Binary / 4-class Fusion
5.4 Calibration
5.5 Hard-negative / Implicit-risk Expansion
5.6 Multi-turn Dynamic Risk State
5.7 Early-warning / Trend Evaluation
5.8 Policy End-to-end Regression
5.9 Ablation + Final Freeze
```

## 4. Risk Dataset Audit

先审计：
- train/dev/test 来源
- label mapping
- class imbalance
- hard negatives
- implicit-risk coverage
- third-party / discussion / denial
- template leakage
- domain shift

禁止一上来重训。

输出：
```text
docs/risk_dataset_audit_v2.md
```

## 5. Checkpoint Shootout

至少比较：

```text
v4_2_domain_only_v2
v4_3_coral
rule-only
```

统一在独立 Risk Test 上评测：

```text
Macro F1
Per-level F1
High-risk Recall / Precision
FPR / FNR
ECE
Brier
```

## 6. Binary / 4-class Fusion

必须真实比较：

```text
4-class only
binary only
documented binary fusion
calibrated fusion
rule + model fusion
```

目标是找到 Recall/FPR tradeoff 最优的最小复杂度方案。

## 7. Calibration

至少：
```text
raw logits
temperature scaling
per-level / decision threshold
```

高风险优先 Recall，但必须同时限制 benign FPR。

## 8. Targeted Data Expansion

重点新增：
- implicit goodbye
- hopeless future
- burden
- disappearance
- third-party crisis
- academic/discussion
- explicit denial
- quoted risk language
- benign short utterances

Hard negative 必须足够，防止 benign→L3 高置信。

## 9. DynamicRiskState

建议：

```python
class DynamicRiskState:
    level: int
    confidence: float
    trend: str
    persistence: int
    escalation: bool
    subject: str
    discussion: bool
    safe_denial: bool
    recent_levels: list[int]
    reasons: list[str]
```

单轮模型负责 current risk；多轮状态机负责 persistence / trend / escalation / recovery。

每一步必须可 Trace。

## 10. Multi-turn Benchmark

新建：
```text
risk_trajectory_benchmark_v1.jsonl
```

覆盖：
```text
0→0→1→2→3
3→2→1→0
fluctuating
persistent medium
third-party escalation
false alarm recovery
```

指标：
```text
Trajectory Accuracy
Early Detection Rate
Escalation Accuracy
Recovery Accuracy
Trend Accuracy
Time-to-detect
```

## 11. Policy 联动评测

必须重新跑：

```text
Predicted Risk v2
→ AgentState
→ Frozen AgentPolicy v1
```

看：
```text
Safety Recall
Primary Action Accuracy
Safety FPR
Policy Exact
```

验证 Risk 修复能否真正转化为系统收益。

## 12. 目标

推荐：
```text
High-risk Recall >= 0.95
FPR <= 0.10
Macro F1 >= 0.70
L2 F1 >= 0.60
L3 F1 >= 0.80
Trend Acc >= 0.80
```

不要只追 Macro F1。

## 13. 交付物

```text
docs/
├── risk_dataset_audit_v2.md
├── phase5_checkpoint_ablation.md
├── phase5_fusion_calibration.md
├── phase5_multiturn_risk_eval.md
├── phase5_policy_end_to_end.md
└── phase5_final_review.md

backend/risk/
├── predictor_v2.py
├── fusion.py
├── calibration.py
└── dynamic_state.py
```

## 14. Codex 启动指令

```text
Phase 5 不允许直接“重训一个新 BERT”开始。

顺序必须是：
Dataset Audit → Checkpoint Shootout → Fusion → Calibration → Targeted Expansion → Multi-turn State。

最终必须同时报告：
1. module-level risk metrics
2. multi-turn trajectory metrics
3. Risk v2 接入 Frozen Policy 后的 end-to-end safety/routing metrics

通过后冻结 RiskResult v2 / DynamicRiskState 接口。
```
