# Phase 5 Fusion / Calibration

> 对应 Phase 5 Task 5.3 / 5.4
> 日期：2026-08-20
> 数据：792 条跨源独立 Risk Test（与 shootout 相同）

---

## 1. 扫描的决策规则

| 规则 | 含义 |
|---|---|
| F1 | 4-class argmax（当前行为） |
| F2 | binary_prob > t |
| F3 | P(L3) > t |
| F4 | P(L2)+P(L3) > t |
| F5 | argmax>=2 AND binary>t（联合门控） |
| F6 | rule 命中优先，否则 P(L2)+P(L3) > t |

## 2. 阈值扫描（v4_2_domain_only_v2）

| 规则 | HR Recall | FPR | Prec |
|---|---:|---:|---:|
| F1_4class_argmax | 0.4762 | 0.4204 | 0.1765 |
| F2_binary>0.3 | 0.5397 | 0.4129 | 0.1983 |
| F2_binary>0.5 | 0.5159 | 0.3964 | 0.1976 |
| F2_binary>0.7 | 0.4683 | 0.3694 | 0.1934 |
| F2_binary>0.9 | 0.4048 | 0.3153 | 0.1954 |
| F3_P3>0.1 | 0.2778 | **0.1216** | 0.3017 |
| F3_P3>0.2 | 0.2540 | 0.1051 | 0.3137 |
| F3_P3>0.5 | 0.2460 | **0.0946** | 0.3298 |
| F4_P23>0.1 | 0.5635 | 0.4685 | 0.1854 |
| F4_P23>0.5 | 0.4762 | 0.4189 | 0.1770 |
| F5_argmax2_and_binary>0.5 | 0.4762 | 0.3919 | 0.1869 |
| F6_rule_else_P23>0.3 | 0.5159 | 0.4354 | 0.1831 |

## 3. 操作点分析

```text
满足 Recall>=0.95 的点：无（所有规则最大 Recall 0.56）
满足 FPR<=0.10 的点：F3_P3>0.5（Recall 仅 0.246）
```

**结论：当前 v4_2 模型无法通过任何 fusion/calibration 达到目标
（Recall≥0.95 且 FPR≤0.10）。**

## 4. 其他发现

1. **binary_threshold 已存储但未进入 predict()**（docstring 声称融合，实现只有 4-class argmax）。
   本实验真实扫描了 binary fusion，证明其也无法救回。
2. **P(L3) 高阈值（F3）能压 FPR 到 ~0.10，但代价是 Recall 塌到 0.25** —— 模型对隐式风险
   （L2）与 benign 的高风险词（L3 词面）无法区分。
3. **规则优先级（F6）没有帮助** —— rule 命中的都是 L3 关键词，召回提升有限。

## 5. 对 5.5 的输入

- 需 Targeted Expansion：hard negatives（127 FP）+ 隐式风险 FN（25）+ denial/quoted 负例（55）。
- 训练后重新扫描阈值，选择 Recall≥0.95 下的最低 FPR 工作点。
