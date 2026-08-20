# MindPal Stateful Adaptive Agent — 完整技术总结

> 日期：2026-08-20
> 范围：Phase 0-8 全部完成冻结
> 用途：技术文档 / 作品集 / 简历 / 面试讲解

---

## 1. 项目背景与目标

MindPal Pro 是一个基于 DeepSeek Chat + BERT 的中文 AI 心理支持对话系统。
本项目把它从"多模块拼接的对话系统"升级为 **State → Policy → Action 的 Stateful Adaptive Agent**：

```text
Before:  input → 各模块各自决策（routing / gate / ReAct 混杂）→ response
After:   perception → AgentState v1 → AgentPolicy v1 → ActionPlan v1 → execution
```

核心目标：让系统**可评测、可解释、可迭代、可回退**，并用独立 benchmark 证明每个改进真实有效。

---

## 2. 最终架构

```text
Perception（三层感知）
├── Intent Layer（Phase 1.5 冻结）：10 类意图，独立测试 Macro 0.844
├── Emotion：BERT + LLM 双通道
└── Risk v5.1（Phase 5 冻结）：单轮分类 + 多轮状态机，FPR 0.022
        ↓
AgentState v1（Phase 2 冻结，12 子状态，State≠Action）
        ↓
AgentPolicy v1（Phase 3 FINAL FREEZE）
├── P0 Safety Invariants（命中即返回，不可被覆盖）
├── P1 Deterministic Rules（primary/tools/rec，gold 为依据）
├── P3 Ambiguity Gate（conflict / low-confidence / open-set）
└── P4 LLM Fallback（仅 ambiguous，永不覆盖 Safety）
        ↓
ActionPlan v1（primary_action + tool_actions + safety_target + recommendation_mode）
        ↓
Execution
├── Memory 2.0（Phase 4）：Retrieval Gate + Conflict Resolver + Lifecycle
├── Recommendation 2.0（Phase 6）：Feature Ranker + Feedback 闭环 + Cooldown
└── Response
        ↓
Observability（Phase 7）：Unified Trace + Replay + Error Attribution
```

---

## 3. 各层技术细节

### 3.1 评测体系（贯穿所有阶段的设计哲学）

| 组件 | 规模 | 用途 |
|---|---|---|
| Agent Benchmark v1.1 | 362 条 | 冻结回归（**只做 regression，不再设计规则**） |
| Independent Policy Test | 430 条 | 全新措辞，验证 Policy 泛化 |
| Safety Slice | 124 条 | 10 类安全场景，验证 Safety 零误伤 |
| Risk 跨源独立测试 | 792 条 | benchmark + policy 对话型风险标注 |
| Memory Benchmark v2 | 23 条 | Retrieval Gate / Conflict |
| Risk Trajectory | 14 条 | 多轮风险状态机 |

> **关键决策**：规则设计一律用独立集，冻结 benchmark 只做回归。
> 避免"design-benchmark performance"——这是本项目的核心方法论之一。

### 3.2 Perception — Risk 2.0（Phase 5，最大改进）

**问题**：Raw BERT 在对话域严重过度触发（FPR 42%），良性对话被判高风险。

**诊断路径**（严格按文档顺序，不直接重训）：
1. **数据审计**：发现 test 与 train 同源（sos-1k），hard negatives 确认（"离开这个世界对我是解脱"→L0），denial/quoted 覆盖缺口
2. **Checkpoint Shootout**：v4_2/coral/ft 全部 FPR 0.42-0.81
3. **Fusion/Calibration 扫描**：证明既有模型无法达到 Recall≥0.95 & FPR≤0.10
4. **Targeted Data Expansion**：hard negatives（127 FP）+ 隐式风险 FN + 第三方风险正例 + 合成 benign → 两轮重训

**结果**：

| 指标 | v4_2 | v5.1 |
|---|---:|---:|
| FPR | 0.4204 | **0.0225** |
| High-risk Recall | 0.4762 | **0.9365** |
| Macro F1 | 0.3141 | **0.7060** |

多轮部分新增 **DynamicRiskState**（单轮模型 + v5.0 状态机）：轨迹 Accuracy 1.0 / 早检测 1.0 / 恢复 1.0。

### 3.3 State — AgentState v1（Phase 2）

- 12 子状态（identity/turn/intent/emotion/risk/conversation/profile/recommendation/memory/derived/meta）
- **State≠Action** 原则：禁止 should_recommend/primary_action 等 Policy 输出混入 State
- 每字段有唯一 source/owner/update rule
- 一致性：Transition Accuracy 1.0，Session Isolation 100%，Shadow Consistency 100%

### 3.4 Policy — Hybrid Agent Policy（Phase 3）

**架构**：分层优先级（P0 Safety > P1 Deterministic > P3 Ambiguity > P4 LLM），高优先级不可被低优先级覆盖。

**规则设计依据**：legacy 审计 + benchmark gold 分布（非拍脑袋）。

**Ablation**（Independent Policy Test，oracle-state）：

| Method | Primary Acc | Tool F1 | Rec Mode Acc | Exact |
|---|---:|---:|---:|---:|
| Legacy | 0.6558 | 0.3468 | 0.6721 | 0.0721 |
| **Deterministic** | **0.8837** | **0.995** | **0.9000** | **0.7767** |

**关键决策**：
- Learned Policy **跳过**（Deterministic 单独达标，§29 不强行加复杂度）
- LLM Fallback 保留但触发率仅 1.4%（远低于 20% 预算）
- Safety Contract 对齐：high_risk_intent 可独立触发 Safety（S04），独立 Safety Slice 验证 FPR=0

### 3.5 Memory 2.0（Phase 4）

**问题**：每轮都检索记忆（over-retrieval 100%），无 conflict/dedup 生命周期。

**方案**（治理层，不换向量库）：
- **Retrieval Gate**（最高优先级）：需要时才检索（memory_reference → 必然；info 知识查询 → 不检索）
- **Conflict Resolver**：MERGE（同义/强化）/ SUPERSEDE（偏好更新/矛盾）/ EXPIRE（TTL）
- **Lifecycle**：TTL 到期 / 长期未访问 decay

**结果**：检索量 **-61%**，over-retrieval **1.0→0.0**，Gate F1 0.965，Conflict Accuracy 1.0。

### 3.6 Recommendation 2.0（Phase 6）

**问题**：推荐被 Risk 污染、feedback 不进 ranking、无 cooldown。

**方案**：
- Feature Ranker（7 特征：semantic/BM25/intent/emotion/risk/profile/memory）
- Feedback 闭环（accept/reject/tried_effective/already_seen → 权重更新）
- Cooldown（同 item 硬阻断 + 同类抑制）
- Safety Review（消费 Risk v2，safety_only 禁普通推荐）

**结果**：Safety violation 0 / Repeat 0 / Neg-feedback violation 0 / Effective-boost 生效。
LLM Rerank **关闭**（feature ranker 达标，§7 不默认保留）。

### 3.7 Observability（Phase 7）

- **Unified Trace**：同一 turn 一个 trace_id，版本溯源（Intent/Risk/State/Policy/Memory/Rec）
- **Replay**：trace → 重建 AgentState → 重跑 Policy（same-version + counterfactual legacy）
- **Error Attribution**：失败 case 定位到最早错误层（perception/policy/safety 等）

---

## 4. 系统级改进（端到端）

冻结 Agent Benchmark v1.1（predicted-state）：

| 指标 | Baseline | **Full System** | 提升 |
|---|---:|---:|---:|
| Primary Action Acc | 0.5249 | **0.7541** | +0.23 |
| Safety Recall | 0.6119 | **0.7761** | +0.16 |
| Safety Target Acc | 0.5663 | **0.9420** | +0.38 |
| Policy Exact Match | 0.0580 | **0.4530** | +0.40 |

> Baseline = legacy policy + v4_2 risk；Full = hybrid policy + v5.1 risk（Policy 冻结未动）。

**端到端改进 100% 来自 Risk 2.0** —— 印证"先修感知瓶颈"的执行顺序正确。

---

## 5. 关键工程决策（作品集亮点）

| 决策 | 理由 |
|---|---|
| 冻结 benchmark 只做 regression | 避免 design-benchmark 过拟合 |
| Targeted Data Expansion 而非堆模型 | 数据审计定位 FPR 根因后精准修复 |
| 不强行加 learned model / LLM | Deterministic 达标即冻结（最小复杂度） |
| oracle-state vs predicted-state 双轨评测 | 区分 Policy-intrinsic 与 upstream error |
| 每阶段独立 benchmark + Final Review + Freeze | 证据链完整，改进可归因 |
| 先修 Risk 再动 Memory/Recommendation | 最大端到端瓶颈优先（路线图排序依据） |

---

## 6. 已知限制

| 限制 | 影响 | 处理 |
|---|---|---|
| Risk L3 F1 0.515 | L2/L3 混淆 | HR 合并 Recall 0.936 达标，Safety routing 影响小 |
| sos-1k 域 FPR 0.70 | 域标签病态（v4_2 亦 0.76） | 生产对话域 FPR 0.022 是目标场景 |
| Policy clarification 泛化有限 | 含糊表达 ~12% 依赖 LLM | LLM fallback 已就绪（预算充足） |
| HR Recall 0.936 ≈ 0.95 | 余量小 | 持续数据迭代 |

---

## 7. 复现方法

```bash
# 环境
D:\anaconda3\python.exe（PYTHONIOENCODING=utf-8, KMP_DUPLICATE_LIB_OK=TRUE, HF_HUB_OFFLINE=1）

# Risk 评测
python evaluation/risk/run_risk_shootout.py                 # checkpoint 对比
python evaluation/risk/run_fusion_calibration.py           # 阈值扫描
python evaluation/risk/run_trajectory_eval.py              # 多轮轨迹

# Policy 评测
python evaluation/policy/run_policy_eval.py --state oracle  # oracle 上限
python evaluation/policy/run_independent_eval.py            # 独立集
python evaluation/policy/run_safety_slice.py                # 安全切片

# Memory / Recommendation / Tracing
python evaluation/memory/run_memory_eval.py
python evaluation/recommendation/run_recommendation_eval.py
python evaluation/tracing/run_tracing_demo.py

# 全量测试
python -m pytest evaluation/tests/ -q
```

---

## 8. 关键资产索引

| 资产 | 位置 |
|---|---|
| 最终证据链 | `docs/resume_evidence.md` |
| 各阶段 Final Review | `docs/phase{2,3,3_closeout,4,5,6,7,8}_final_review.md` |
| 冻结接口 | `backend/state/schema.py` / `backend/policy/schema.py` / `backend/risk/` / `backend/memory_v2/` / `backend/recommendation_v2/` / `backend/tracing/` |
| 冻结数据集 | `evaluation/datasets/` / `evaluation/policy/policy_independent_test_v1.jsonl` / `policy_safety_independent_v1.jsonl` |
| Risk v5.1 权重 | `bert_data/models/v5_1_tuned/best_model/`（gitignore，仅 tokenizer 提交） |
| 进度日志 | `docs/execution_progress.md` |

---

## 9. 一句话总结

> 用"冻结基准回归 + 独立测试泛化 + 数据驱动的定向修复"方法论，
> 将心理支持对话系统重构为可评测可解释的 Stateful Adaptive Agent，
> 把最大端到端瓶颈（风险感知误报率）从 42% 降到 2%，其余各层均有独立数据支撑的改进。
