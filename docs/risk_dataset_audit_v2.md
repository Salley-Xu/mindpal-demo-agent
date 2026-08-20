# Risk Dataset Audit v2

> 对应 Phase 5 Task 5.1
> 日期：2026-08-20
> 原则：先审计数据，不重训。

---

## 1. 数据集来源与规模

| 集合 | 文件 | 规模 | 标签分布 (L0/L1/L2/L3) | 来源 |
|---|---|---:|---|---|
| train_domain_only | processed_v4/train_domain_only.jsonl | 800 | 200/200/200/200（完美均衡） | 域内合成+筛选 |
| train_v3 | processed_v4/train_v3.jsonl | 936 | 259/237/287/153 | sos-1k-fine |
| dev_v3 | processed_v4/dev_v3.jsonl | 191 | 55/47/59/30 | sos-1k-fine(186)+sos-hl-1k(5) |
| test_v3 | processed_v4/test_v3.jsonl | 125 | 35/32/38/20 | sos-1k-fine |
| silver_v3 | processed_v4/silver_v3.jsonl | 1247 | 346/316/382/203 | sos-1k-fine |
| gold_v3 | processed_v4/gold_v3.jsonl | **0** | - | **不存在** |

## 2. 关键发现

### 2.1 test 集与 train 同源 → 域漂移风险

`test_v3`（125）与 `train_v3`（936）全部来自 **sos-1k-fine**。
无完全重复文本（0 重复），但**同源模板风格**，模型可能学到源内偏置而非泛化能力。
需要一个**跨源独立 Risk Test**（本阶段使用 agent_benchmark_v1_1 + policy_independent_test_v1，
共 792 条对话型风险标注）。

### 2.2 hard negatives 确认存在

test_v3 中 L0 标签但含高风险语言的样本：

```text
"哎 离开这个世界对我是解脱啊"              → L0（含 L3 关键词"离开这个世界"）
"想要从10楼跳下去结束自己的一切，可是，胆小，不敢" → L0（含"跳下去"）
"感觉自己现在像在悬崖边上上吊一样"          → L0
"身边有人跳楼自杀了……"                    → L0
```

这些是 benign 与 L3 的高混淆样本，是 FPR 高企的根因之一。
**其中"离开这个世界"类正是 rule 通道与 BERT 都易误判的 case。**

### 2.3 场景覆盖缺口

| 场景 | test_v3 中数量 |
|---|---:|
| discussion（学术/讨论） | 1 |
| denial（安全否认） | 0 |
| quoted（引用风险语言） | 0 |
| third_party（第三方） | 6 |

**确认 Phase 5 §8 Targeted Expansion 的必要性**：denial/quoted/discussion 在训练数据中几乎缺席。

### 2.4 subject_context

train/dev/test 均以 `self_current` 为主（~93%），third_party 仅 ~4%，negated ~1%。

### 2.5 实现层问题（非数据）

`BertRiskPredictor.predict()` 中 `binary_threshold` 已存储但**未实际进入融合逻辑**
（docstring 声称"binary_prob > threshold → level_2"，但 predict() 只返回 4-class argmax）。
`v4_3_coral` 走 CORAL 序数回归（threshold 0.28），未被系统比较过。

---

## 3. 结论

| 问题 | 影响 | 处理 |
|---|---|---|
| test 同源 | 指标虚高 | 用 792 条跨源对话测试 + test_v3 双轨 |
| hard negatives | FPR 高 | Checkpoint Shootout + Fusion 重点看 FPR |
| denial/discussion/quoted 缺覆盖 | 无法评估 | Targeted Expansion（5.5）补齐 |
| binary_threshold 未生效 | 融合能力未用 | 5.3 真实比较 4-class vs binary vs 融合 |
| coral 未比较 | 潜在更好模型未知 | 5.2 纳入 shootout |
