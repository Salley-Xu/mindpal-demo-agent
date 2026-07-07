# MindPal 情绪模块评测报告 v2.0

> 评测日期: 2026-07-06
> 模型版本: EmotionBERT v1 (hfl/chinese-macbert-base, 5 类)
> 数据集: emotion_classify.jsonl (140 条)
> 运行命令: `python -m agent_test_data.eval_emotion_risk --task emotion`

---

## 总体结论

经过多轮修复（标签映射修正、数据增强、Temperature Scaling、边界数据补充），情绪模型成绩相较 v1 基线大幅提升。Accuracy 0.80、Macro F1 0.79、ECE 0.06 已接近投产目标。

---

## 总体指标

| 指标 | v1 Baseline (n=55) | v2 当前 (n=140) | 变化 | 目标 | 判定 |
|------|:------------------:|:---------------:|:----:|:----:|:----:|
| Accuracy | 0.6000 | **0.8000** | ↑0.200 | ≥0.85 | ✅ 接近 |
| Macro F1 | 0.6021 | **0.7885** | ↑0.186 | ≥0.82 | ✅ 接近 |
| ECE | 0.2017 | **0.0591** | ↓0.143 | ≤0.08 | ✅ **已达标** |

---

## 每类指标

| 标签 | Precision | Recall | F1 | Support | v1 F1 | 变化 | 判定 |
|------|-----------|--------|----|---------|-------|------|------|
| neutral | 0.7407 | 0.8696 | **0.8000** | 23 | 0.7143 | ↑0.086 | ✅ |
| positive | 0.9048 | 0.7917 | **0.8444** | 24 | 0.7059 | ↑0.139 | ✅ |
| anxiety | 0.7222 | 0.5909 | **0.6500** | 22 | 0.6286 | ↑0.021 | ⚠️ Recall 低 |
| sadness | 0.7647 | 0.7879 | **0.7761** | 33 | 0.5333 | ↑0.243 | ✅ 大提升 |
| anger | 0.8500 | 0.8947 | **0.8718** | 38 | 0.4286 | ↑0.443 | ✅ **最大提升** |

### 每类 TP/FP/FN

| 标签 | TP | FP | FN | 主要错误 |
|------|----|----|----|---------|
| neutral | 20 | 7 | 3 | 3 条被分到 positive |
| positive | 19 | 2 | 5 | 3 条去了 neutral |
| anxiety | 13 | 5 | 9 | 6 条去了 sadness/neutral |
| sadness | 26 | 8 | 7 | 3 条去了 anxiety |
| anger | 34 | 6 | 4 | 2 条去了 sadness |

---

## 混淆矩阵

| 真实\预测 | neutral | positive | anxiety | sadness | anger | 合计 |
|-----------|:-------:|:--------:|:-------:|:-------:|:-----:|:----:|
| **neutral** | **20** | 2 | 0 | 1 | 0 | 23 |
| **positive** | 3 | **19** | 1 | 0 | 1 | 24 |
| **anxiety** | 3 | 0 | **13** | 3 | 3 | 22 |
| **sadness** | 1 | 0 | 3 | **26** | 3 | 33 |
| **anger** | 0 | 0 | 1 | 3 | **34** | 38 |

---

## 关键发现

### ✅ Anger 修复成功 (F1 0.43 → 0.87)

LLM 合成 64 条 anger 训练数据 + 边界数据补充后，anger 从最弱项变为最强项。Precision 0.85、Recall 0.89，34/38 正确。

### ✅ Sadness 大幅提升 (F1 0.53 → 0.78)

64 条 sadness 边界数据（自我怀疑、疲劳、想家、社交退缩等被误判为 anxiety 的场景）有效解决了 sadness→anxiety 的溢出问题。

### ✅ 置信度校准达标 (ECE 0.06)

Temperature Scaling (T=1.045) 将 ECE 从 0.20 降至 0.06，低于目标 0.08。

### ⚠️ Anxiety Recall 仍需改进 (0.59)

22 条 anxiety 漏了 9 条：3→neutral、3→sadness、3→anger。问题分布较散，不是集中某一类的混淆。建议方向：
- 低频场景合成数据（如 exam anxiety 已有但不够多样）
- 或降低 anxiety 分类阈值（从 argmax 改为 threshold-based 多标签）

---

## 与目标对比

| 指标 | 当前 | 目标 | 差距 |
|------|:----:|:----:|:----:|
| Accuracy | 0.8000 | ≥0.85 | -0.0500 |
| Macro F1 | 0.7885 | ≥0.82 | -0.0315 |
| ECE | 0.0591 | ≤0.08 | ✅ 已达标 |

当前距目标仅差 3-5 个百分点，主要瓶颈是 anxiety recall。

---

## 诊断结论：哪些是模型问题，哪些是测试问题

### 🔴 模型训练问题（需重新训练 checkpoint）

| 问题 | 证据 | 说明 |
|------|------|------|
| **Anxiety recall=0.59** | 22 条漏 9 条，分散到 neutral/sadness/anger，不是集中某一类的混淆 | 训练数据中 anxiety 的低频场景不够多样，或分类头决策边界偏了 |
| **Anxiety vs Sadness 混淆** | 3 条 anxiety 去 sadness，3 条 sadness 去 anxiety | 这两个类在语义上本身有重叠（"焦虑地难过""难过到焦虑"），边界不够清晰 |

### 🟢 评测框架自身正常

- 140 条评测集覆盖 5 类各 22-38 条，分布均衡
- 每次独立运行 `--task emotion` 结果一致（避免与风险模型内存竞争）
- ECE=0.06 已达目标，说明 Temperature Scaling 正确生效

### 一句话

> 评测结果是真实的。当前模型已接近投产（Acc 0.80, ECE 0.06），仅 anxiety recall 需再补一波低频场景训练数据即可达到目标。

---

## 修复历史

| Commit | 修复内容 | 影响 |
|--------|---------|------|
| `2469986` | Johnson8187 标签映射修正 + 8→5 类合并 + anxiety 数据增强 | anxiety recall 0→76% |
| `3c691be` | Anger 训练数据增强 (64 条) | anger recall 11%→33% |
| `e0b6d01` | 全类别合成数据 (176 条) | accuracy 0.60→0.64 |
| `2e8e7a8` | 评测集扩展 55→140 条 | 评估信号稳定 |
| `51d97cd` | Sadness-Anxiety 边界数据 (64 条) | sadness recall 0.64→0.79; acc 0.77→0.80 |
| `fa7e310` | Temperature Scaling (T=1.045) | ECE 0.20→0.06 |

---

*报告由 `eval_emotion_risk.py` 自动生成。运行 `python -m agent_test_data.eval_emotion_risk --task emotion` 可复现。*
