# Phase 5 Checkpoint Ablation

> 对应 Phase 5 Task 5.2
> 日期：2026-08-20
> 数据：跨源独立 Risk Test = Frozen Benchmark v1.1（362）+ Independent Policy Test（430）= 792 条对话型风险标注

---

## 1. 测试集

gold level 分布：L0 503 / L1 163 / L2 55 / L3 71（High-risk L2+L3 = 126，占 16%）。

## 2. 主表

| Method | Macro F1 | Acc | HR Recall | HR Prec | FPR | FNR | L0 F1 | L1 F1 | L2 F1 | L3 F1 | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rule_only | 0.3548 | 0.5922 | 0.2857 | 0.7200 | **0.0210** | 0.7143 | 0.7392 | 0.1200 | 0.1944 | 0.3654 | 0ms |
| v4_2_domain_only_v2（当前生产） | 0.3141 | 0.3485 | 0.4762 | 0.1765 | 0.4204 | 0.5238 | 0.3410 | 0.5354 | 0.0667 | 0.3133 | 77ms |
| v4_3_coral | 0.1256 | 0.1338 | 0.4762 | 0.1150 | 0.6937 | 0.5238 | 0.0305 | 0.3747 | 0.0694 | 0.0278 | 77ms |
| v4_2_ft_conversational | 0.1373 | 0.1275 | 0.5556 | 0.1149 | 0.8093 | 0.4444 | 0.1679 | 0.0000 | 0.0822 | 0.2992 | 77ms |

> FPR = benign(L0/L1) 被预测为 High-risk(L2/L3) 的比例；FNR = 漏检比例。
> （注：rule_only 的 Macro F1 高因 FPR 极低、且 L3 关键词召回部分；但 HR Recall 仅 0.286）

## 3. 关键结论

1. **全部 BERT checkpoint 严重过度触发**：当前生产 v4_2 FPR=0.42，coral 0.69，ft_conv 0.81。
   即 42%~81% 的良性对话被误判为高风险。
2. **HR Recall 全部不足**：0.48~0.56，远低于 0.95 目标。
3. **v4_3_coral 比 v4_2 更差**（FPR 0.69 vs 0.42），不采用。
4. **rule_only FPR 极低（0.02）但 Recall 也极低（0.29）**——关键词规则抓不到隐式风险。
5. **根因**：模型训练于 sos-1k（危机论坛戏剧化语言），在真实对话域（"离开这个世界"等
   短语出现在 benign 语境）严重 miscalibrated。Full Pipeline ≈ Raw BERT，
   SessionAggregator 无法修复单轮分类的根本问题。

## 4. 对 Phase 5 的启示

Fusion/Calibration（5.3/5.4）在既有模型上只能小幅 tradeoff，**无法达到
Recall≥0.95 & FPR≤0.10 目标** → 必须 Targeted Data Expansion + 重训（5.5）。
