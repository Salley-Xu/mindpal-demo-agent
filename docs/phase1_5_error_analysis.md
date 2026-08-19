# Phase 1.5 Error Analysis

> 对应 Phase 1.5 Task 1.5.6（§11）
> 日期：2026-08-19
> 评测对象：Final Context-aware Model（phase1_5_final_model）on Independent Test（535 条）

---

## 1. 错误总体

| 指标 | 值 |
|---|---:|
| Macro F1 | 0.844 |
| Exact Match | 0.697 |
| 错误样本数 | 162/535（30%） |

## 2. 错误类型分布（§11 Taxonomy）

| 错误码 | 类型 | 数量 | 占比 | 典型 Case |
|---|---|---:|---:|---|
| **C03** | Secondary Intent Missing（次要意图漏标） | 130 | 80% | multi-label 场景漏掉一个次要标签 |
| **C05** | Hard-negative False Positive | 24 | 15% | 高危词安全语境被误判 |
| **C04** | Implicit High-risk Miss | 8 | 5% | 隐式告别/第三方危机漏判 |

## 3. 重点分析

### 3.1 C03 Secondary Intent Missing（130，最严重）

**根因**：multi-label 联合预测偏弱（slice：multi_label Macro 0.61）。

**子模式**：
1. **漏 emotional_expression**：`"我朋友最近状态很差，我很心疼，怎么帮他？"`（gold=emotional+help）
2. **漏 feedback**：`"你推荐的，我用了，效果不错。"`（gold=feedback）
3. **漏 high_risk 的附加标签**：`"压力大到崩溃，帮帮我。"`（gold=high_risk+help，可能漏 high_risk）

**修复方向**：
- 扩充 multi-label 组合训练样本（Task 1.5.2 P1，400-600 条，已部分）
- 用 label correlation 做后处理（已知 emotional+help 共现高时补标）

### 3.2 C05 Hard-negative False Positive（24）

**根因**：高危词出现在安全语境（讨论/学术/否定）时模型仍触发。

**典型**：独立测试 hard_negative slice（n=27, Macro 0.28）——说明 hard-negative 处理弱。

**修复方向**：
- 增加安全语境 hard-negative 训练样本
- 规则后处理：检测讨论/学术/否定标记词时抑制 high_risk

### 3.3 C04 Implicit High-risk Miss（8，安全关键）

**漏判案例**：
- **第三方危机**：`"我朋友最近老说想不开，我很担心"`、`"我弟弟总说想不开"`、`"我同事总说想走极端"` —— 模型在"第三方 + 危机表达"组合上漏标
- **极隐式**：`"最近总想去天台吹风"`（天台=危险信号）、`"我买的东西，可能用不上了"`（遗物）

**根因**：这些表达依赖背景知识（天台/买的东西=自杀计划的间接信号），训练数据中此类极隐式样本不足。

**现状**：high_risk Recall 0.873（已从 0.714 提升），目标 0.93/0.95 仍有差距。

**修复优先级（P0）**：
1. 扩充第三方危机 + 极隐式表达训练样本（如"天台""买的东西用不上""钥匙交出去"）
2. high_risk 专用低阈值（当前 0.35）
3. 结合 Risk 模块规则兜底（HIGH_RISK_OVERRIDE_PATTERNS 覆盖显式行为）

## 4. Slice 弱点总结

| Slice | Macro F1 | 弱点 |
|---|---:|---|
| Single-label | 0.908 | ✅ |
| Multi-label | 0.610 | C03 联合预测 |
| Context-aware | 0.377 | 全新上下文场景泛化 |
| Hard-negative | 0.280 | C05 安全语境误报 |

## 5. 修复优先级

| 优先级 | 动作 | 预期收益 |
|---|---|---|
| P0 | 扩充第三方危机 + 极隐式高危样本 | high_risk Recall 0.87→0.93 |
| P1 | 扩充 multi-label 组合样本 + label correlation 后处理 | C03 减少 |
| P1 | hard-negative 规则抑制（讨论/否定标记词） | C05 减少 |
| P2 | 上下文数据多样化（不同风格的 assistant 轮） | context slice 提升 |
