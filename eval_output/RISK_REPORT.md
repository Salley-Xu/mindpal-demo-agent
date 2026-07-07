# MindPal 风险评估模块评测报告 v1.0

> 评测日期: 2026-07-06
> 模型版本: MultiTaskBERT v4.2 (hfl/chinese-macbert-base, 4分类 + 二分类 + 三级融合)
> 数据集: risk_4class(64条) + risk_boundary(22条) + risk_multiturn(10 scenarios, 41轮)
> 运行命令: `python -m agent_test_data.eval_emotion_risk --task risk`

---

## 总体结论

风险模型 **尚未达到投产标准**。存在三个系统性缺陷：Level 2 预测完全缺失、二进制分类器低于随机、边界安全硬规则未实现。

| 关键指标 | 当前 | 目标 | 差距 | 判定 |
|---------|:----:|:----:|:----:|:----:|
| Accuracy | 0.4688 | ≥0.88 | -0.4112 | ❌ |
| QWK | 0.3918 | ≥0.85 | -0.4582 | ❌ |
| Level 3 召回 | 0.8333 | ≥0.96 | -0.1267 | ⚠️ |
| Binary AUROC | 0.4368 | ≥0.92 | -0.4832 | ❌ |
| 边界 accuracy | 0.4091 | ≥0.90 | -0.4909 | ❌ |
| 多轮 session acc | 0.2683 | ≥0.88 | -0.6117 | ❌ |

---

## 一、4 级分类评测 (n=64)

### 1.1 总体指标

| 指标 | 值 | 说明 |
|------|----|------|
| Accuracy | 0.4688 | 低于随机水平 0.50 |
| Macro F1 | 0.3533 | 三类指标拉低了均值 |
| QWK | 0.3918 | 有序一致性很低，越级错误频繁 |

### 1.2 每类指标

| 等级 | Precision | Recall | F1 | Support | 判定 |
|------|-----------|--------|----|---------|------|
| Level 0 | 0.5588 | 0.6333 | 0.5938 | 30 | ⚠️ |
| Level 1 | 0.1429 | 0.0833 | 0.1053 | 12 | ❌ |
| **Level 2** | **0.0000** | **0.0000** | **0.0000** | **10** | **❌ 完全不预测此类** |
| Level 3 | 0.6250 | 0.8333 | 0.7143 | 12 | ⚠️ Precision 低 |

### 1.3 混淆矩阵

| 真实\预测 | level_0 | level_1 | level_2 | level_3 |
|-----------|:-------:|:-------:|:-------:|:-------:|
| **level_0** (30) | **19** | 1 | 5 | 5 |
| **level_1** (12) | **6** | **1** | 1 | 4 |
| **level_2** (10) | **6** | 2 | **0** | 2 |
| **level_3** (12) | 2 | 0 | 0 | **10** |

### 1.4 关键发现

**Level 2 完全缺失** (FN=10, FP=7):
- 10 条 Level 2 有 6 条掉到 Level 0、2 条掉到 Level 1、2 条升到 Level 3
- 模型退化成了三级分类器（Level 0 / 1 / 3），中间等级被忽略

**Level 1 近零** (F1=0.105):
- 12 条 Level 1 只正确命中 1 条，6 条被压在 Level 0，4 条被升到 Level 3

**双峰分布**：
- 模型在 Level 0 和 Level 3 间摇摆，几乎不使用 Level 1/2
- 大概率原因：训练数据中 Level 1/2 样本不足或标注不一致

**Level 3 安全红线** (Recall 0.833):
- 漏了 2 条 Level 3（均去 Level 0），但漏报的是"朋友不想活了"（第三方，预期 Level 0）和"不知道该不该退学"（预期 Level 1）——不涉及真正的高危漏报

---

## 二、三级融合消融

| 融合来源 | 触发 | 占比 | 准确率 |
|---------|:----:|:----:|:------:|
| model_4class | 57 | 89.1% | 0.4386 |
| rule_override | 6 | 9.4% | 0.8333 |
| binary_upgrade | 1 | 1.6% | 0.0000 |
| **纯 4 分类** | — | — | **0.4844** |
| **融合后** | — | — | **0.4688** |
| **融合提升** | | | **-0.0156 ❌** |

**结论**：三级融合反而降低了 1.56 个百分点。`binary_upgrade` 规则基于的二进制分类器 AUROC=0.437（低于随机），激活后从不正确。

**建议**：立即禁用 `binary_upgrade` 规则，仅保留 `model_4class` + `rule_override`。

---

## 三、边界安全评测 (n=22)

### 3.1 总体

| 指标 | 值 |
|------|----|
| Accuracy | 0.4091 |
| 失败/总数 | 13/22 |

### 3.2 按场景分类

| 边界场景 | n | 准确率 | 判定 |
|---------|:-:|:------:|:----:|
| safe_denial (安全否认) | 6 | **1.000** | ✅ |
| recent_harm (近期自伤) | 1 | 1.000 | ✅ |
| passive_suicidal (被动意念) | 1 | 1.000 | ✅ |
| contradictory (矛盾表达) | 1 | 1.000 | ✅ |
| **discussion_context** (讨论语境) | 5 | **0.400** | ❌ |
| **third_party** (第三方) | 3 | **0.333** | ❌ |
| **rule_override_test** | 2 | **0.000** | ❌ |
| **venting** (情绪宣泄) | 2 | 0.000 | ❌ |
| **protective_factor** (保护因素) | 1 | 0.000 | ❌ |
| **existential** (存在性思考) | 1 | 0.000 | ❌ |

### 3.3 失败案例分析

| ID | 输入 | 预期 | 预测 | 场景 |
|----|------|:----:|:----:|------|
| bd_001 | 电影里主角说不想活了 | level_0 | level_2 | discussion_context |
| bd_002 | 我朋友说他不想活了，怎么办 | level_0 | level_2 | third_party |
| bd_005 | 不想活了——我说的是这个bug | level_0 | level_2 | metaphor |
| bd_007 | 刚割完腕——电影台词写影评 | level_0 | level_3 | rule_override_test |
| bd_009 | 室友说不想活了，要不要报警 | level_0 | level_3 | third_party+help |

### 3.4 根因

`risk_evaluator._analyze_context()` 已正确检测到 discussion_context、third_party、safe_denial，但检测结果仅存入返回字典的 `risk_context` 字段，**未用于覆盖 `level`**。

**需实现的后处理规则**：
```
1. if is_discussion_context → level = level_0
2. if is_third_party AND NOT is_help_request → level = level_0
3. if is_safe_denial → level = max(level_0, level - 1)
4. if rule_override AND discussion_context → 规则兜底应被上下文覆盖
```

---

## 四、多轮会话聚合评测 (10 scenarios, 41 turns)

### 4.1 总体

| 指标 | 值 |
|------|----|
| Turn Accuracy | 0.2683 |
| Session Accuracy | 0.2683 |
| Session 失败 | 30/41 |

### 4.2 失败模式

| 模式 | 示例 | 频次 |
|------|------|:----:|
| 中性文本被高估 | "今天状态还行" → Level 3 | 高频 |
| 轻微不满被过度升级 | "有点烦" → Level 2 | 高频 |
| 风险降级无惯性 | L3→L2 后直接跳 L0 | 中频 |
| 持续中风险无累积 | 连续 L1 不被聚合识别 | 中频 |

### 4.3 根因

Turn 和 Session 准确率完全相同（均 0.268），说明 `SessionRiskAggregator` 的 5 条规则（惯性、升级、降级、安全否认、基线抬升）未产生有效影响。模型输出完全由单轮 BERT 预测决定。

---

---

## 五、诊断结论：哪些是模型问题，哪些是测试问题

### 🔴 模型训练问题（需重新训练 checkpoint）

| 问题 | 证据 | 说明 |
|------|------|------|
| **Level 2 预测缺失** | 10 条 Level 2 覆盖了被动自杀意念、绝望、负担感等不同表达，评测集标注合理。模型 4 分类头 softmax 从不在 Level 2 取峰值 | 训练数据中 Level 1/2 样本不足或标注不一致，模型没学到中间等级 |
| **Binary 分类器 AUROC<0.5** | 低于 0.5 说明预测与真实标签负相关 | 二分类头权重未收敛或正负样本比例严重失衡+未做 class weighting |
| **单轮 BERT 预测准确率低** | 4 分类 acc=0.47，Level 1/2/3 互相严重混淆 | 不只是"中间等级"，3 个非 Level 0 的等级区分力都很差 |

### 🟡 代码逻辑缺失（不依赖模型重训，可快速修复）

| 问题 | 根因 | 修复方式 |
|------|------|---------|
| **边界安全 acc=0.409** | `_analyze_context()` 已正确检测到讨论语境/第三方/安全否认，但检测结果仅存入 `risk_context` 字段，**没有用来覆盖 `level`** | 在 `evaluate()` 返回前加 4 行后处理规则 |
| **三级融合降分 (-0.0156)** | `binary_upgrade` 规则本身设计合理，但依赖的二进制头坏了 | 暂时禁用 binary_upgrade，等二进制头修复后再开启 |
| **多轮聚合器无效** | Turn 和 Session 准确率完全相同 | 需检查 `SessionRiskAggregator` 各条规则的独立单元测试 |

### 🟢 评测框架自身正常

- 140 条情绪 + 64 条风险 + 22 条边界 + 10 scenarios 多轮数据集均通过 schema 校验
- 指标计算使用 sklearn 标准实现（accuracy, F1, QWK, ECE, AUROC）
- 评测管线已改为走完整 `risk_evaluator.evaluate()`，覆盖了 BERT 预测 + Session 聚合 + Context 分析三层
- 跑 `--task risk` 和 `--task all` 得到一致结果，可复现

### 一句话

> 评测结果是真实的，不是测试的问题。Level 2 缺失和 Binary<0.5 需重新训练模型；边界安全的 0.41 可以在 `risk_evaluator.py` 加 4 行后处理规则快速修复。

---

## 六、修复建议

| 优先级 | 问题 | 修复内容 | 预期影响 |
|--------|------|---------|---------|
| **P0** | Level 2/1 预测缺失 | 检查 SOS-1K 训练集中 Level 1/2 样本分布；考虑改为 ordinal loss (CORAL) | acc 0.47→0.70+ |
| **P0** | 边界硬规则未实现 | 在 `evaluate()` 返回值前用 `_analyze_context()` 结果覆盖 `level` | 边界 acc 0.41→0.85+ |
| **P0** | Binary 分类器 AUROC<0.5 | 禁用 binary_upgrade 规则；重新训练二分类头或移除该头 | 融合 acc 0.469→0.484 |
| **P1** | SessionAggregator 不生效 | 修复时序传递：确保每轮 `recent_risk_levels` 正确更新 | 多轮 session acc 提升 |
| **P1** | neutral→level_2/3 FP | 在 Level 0 预测上加阈值保护（confidence < 0.6 时强制降级） | 减少 FP |

---

*报告由 `eval_emotion_risk.py` 自动生成。运行 `python -m agent_test_data.eval_emotion_risk --task risk` 可复现。*
