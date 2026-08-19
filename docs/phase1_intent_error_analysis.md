# Phase 1 Intent Error Analysis

> 对应 Phase 1 Task 1.13 / §38
> 日期：2026-08-19
> 评测对象：Calibrated Small Model（seed test v1，122 条）

---

## 1. 错误总体

| 指标 | 值 |
|---|---:|
| test Macro F1 | 0.854 |
| test Micro F1 | 0.845 |
| Exact Match | 0.689 |
| 错误样本数 | 38/122（31%） |

## 2. 错误类型分布（§38 Error Taxonomy）

| 错误码 | 类型 | 数量 | 占比 | 典型 Case |
|---|---|---:|---:|---|
| **I04** | Under-labeling（漏标次要意图） | 28 | 74% | "你说过让我别喝咖啡，我停了。"（gold=feedback+memory，pred=memory） |
| **I03** | Over-labeling（过标） | 8 | 21% | "我同学说要离开这个世界。"（gold=high_risk，pred 多了 casual_chat） |
| **I09** | High-risk Intent Miss | 2 | 5% | "这是最后一次跟你说这些了。"（gold=high_risk，pred=memory_reference） |

## 3. 重点分析

### 3.1 I04 Under-labeling（28 条，最严重）

**根因**：multi-label 场景下模型倾向"保守"输出单标签，漏掉次要标签。

**子模式**：
1. **漏 emotional_expression**：`"我朋友说想结束一切，我好担心，怎么办？"`（gold=emotional+help+high_risk，pred 漏 emotional）
2. **漏 feedback**：`"你说过让我别喝咖啡，我停了。"`（gold=feedback+memory，pred 漏 feedback）
3. **漏 resource/info 叠加**：`"我这种情况是不是很常见？"`（gold=emotional+info，pred 误判 follow_up+info）

**根因**：
- 训练数据 multi-label 占比 30%，模型对"次要标签"的联合预测偏弱
- 部分标签共现模式（如 feedback+memory）在 train 中偏少

**修复方向**：
- 增加 multi-label 样本比例（扩充数据时强化组合标签）
- per-label threshold 已经部分缓解（feedback threshold 0.4→ 召回 0.67），可进一步按错误类型调优

### 3.2 I03 Over-labeling（8 条）

**根因**：模型在高危/任务标签外叠加了 casual_chat 或 follow_up。

**典型**：`"我同学说要离开这个世界。"` → pred=[casual_chat, high_risk]。casual_chat 的关键词 "说" 触发，但标注指南明确"有具体任务意图不标 casual_chat"。

**修复方向**：
- 关键词分类器（RuleClassifier）已实现"有标签不标 casual"逻辑，可迁移到 Small Model 后处理
- 或在后处理中：若存在 high_risk/task 标签，强制去掉 casual_chat

### 3.3 I09 High-risk Intent Miss（2 条，安全关键）

**典型**：
- `"这是最后一次跟你说这些了。"`（隐式告别 → 被判 memory_reference）
- `"我可能坚持不到下次跟你说话了。"`（隐式告别 → 空预测）

**根因**：隐式告别/绝望表达在训练数据中的 high_risk 样本较少，模型未学会这类表达。

**影响**：high_risk_expression Recall = 0.882（目标 ≥0.95）。虽然这是 **Intent 信号**而非 Risk 模块输出，但 Policy 层依赖它作为辅助信号。

**修复方向（最高优先级）**：
1. 扩充隐式高危表达训练样本（"最后一次""坚持不到""不想醒"等告别/绝望句式）
2. high_risk_expression 使用更低阈值（Recall 优先，§26）——当前 threshold 0.35 已较低，可单独降到 0.25
3. 与 Risk 模块的 rule override 互补（Risk 侧的 `HIGH_RISK_OVERRIDE_PATTERNS` 覆盖显式行为）

## 4. 高置信度错误检查（§38）

抽取"高置信度但错误"的样本，确认不是校准问题：

| case | gold | pred | max-score |
|---|---|---|---|
| "有什么技巧能帮助入睡？" | explicit_help_request | [] | ~0.3（低置信空预测）|
| "这个回答是通用的吗？" | info+meta | follow_up+info | ~0.5 |

高置信度错误较少（calibration 后 ECE 0.059）。主要错误集中在**多标签联合**与**隐式表达**，非校准问题。

## 5. 修复优先级

| 优先级 | 动作 | 预期收益 |
|---|---|---|
| P0 | 扩充隐式高危表达样本 + high_risk 低阈值 | 高危 Recall 0.88→0.95 |
| P1 | 后处理：有 task/high_risk 标签时去 casual_chat | 消除 I03 |
| P1 | 扩充 multi-label 组合样本 | I04 减少 |
| P2 | Context-aware 重训（follow_up/memory） | benchmark follow_up 0.08→目标 0.5+ |
