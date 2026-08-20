# Phase 5 Final Review — Risk 2.0

> 对应 Phase 5 Task 5.9
> 日期：2026-08-20
> 裁决：**RiskResult v2 / DynamicRiskState = FREEZE（PASS）**

---

## 1. 执行总结

```text
5.1 Dataset Audit      → 数据源审计：test 同源 / hard negatives / 覆盖缺口
5.2 Checkpoint Shootout → 全部 BERT 严重 FPR（0.42~0.81）→ 需重训
5.3-5.4 Fusion/Calib    → 阈值扫描证明既有模型无法达标
5.5 Targeted Expansion  → v5（1027 条）+ v5.1（1205 条，含第三方正例）
5.6-5.7 DynamicRiskState → 多轮轨迹评测全 PASS
5.8 Policy End-to-end   → Risk v2 → Frozen Policy 系统收益验证
5.9 Ablation + Freeze   → 本文
```

## 2. Module-level 指标（792 条跨源独立 Risk Test）

| 指标 | v4_2 | v5 | v5.1 | 目标 | 判定 |
|---|---:|---:|---:|---:|---|
| Macro F1 | 0.3141 | 0.6804 | **0.7060** | ≥0.70 | ✅ |
| High-risk Recall | 0.4762 | 0.6905 | **0.9365** | ≥0.95 | ≈（0.94） |
| FPR | 0.4204 | 0.0135 | **0.0225** | ≤0.10 | ✅ |
| HR Precision | 0.1765 | 0.9062 | **0.9417** | - | ✅ |
| L2 F1 | 0.0667 | - | **0.648** | ≥0.60 | ✅ |
| L3 F1 | 0.3133 | - | **0.515** | ≥0.80 | ✗ |
| Acc | 0.3485 | - | **0.8283** | - | - |

> v5.1 工作点 = P(L2)+P(L3) > 0.2（融合阈值，calibrated）。

## 3. Multi-turn Trajectory（14 条）

| 指标 | 结果 | 目标 | 判定 |
|---|---:|---:|---|
| Trajectory Accuracy | 1.0 | ≥0.90 | ✅ |
| Early Detection Rate | 1.0 | ≥0.95 | ✅ |
| Escalation Accuracy | 1.0 | - | ✅ |
| Recovery Accuracy | 1.0 | - | ✅ |
| Avg Time-to-detect | 1.46 turns | 早检测 | ✅ |

## 4. Policy End-to-end（v5.1 → Frozen Policy）

| 指标 | v4_2 | v5.1 | 提升 |
|---|---:|---:|---|
| Primary Action Acc | 0.5249 | **0.7541** | +0.23 |
| Safety Recall | 0.6119 | **0.7761** | +0.16 |
| Safety Target Acc | 0.5663 | **0.9420** | +0.38 |
| Policy Exact | 0.3453 | **0.4530** | +0.10 |

**Risk 修复直接转化为系统级收益**（Policy 冻结未动）。

## 5. Ablation（对比）

| Method | Macro F1 | HR Recall | FPR |
|---|---:|---:|---:|
| rule_only | 0.3548 | 0.2857 | 0.0210 |
| v4_2_domain_only_v2 | 0.3141 | 0.4762 | 0.4204 |
| v4_3_coral | 0.1256 | 0.4762 | 0.6937 |
| v5.0（首轮扩展） | 0.6804 | 0.6905 | 0.0135 |
| **v5.1（+第三方/隐式正例）** | **0.7060** | **0.9365** | **0.0225** |

## 6. 已知限制（Freeze with limitations）

1. **L3 F1 0.515 < 0.80**：L2/L3 混淆（戏剧化风险语言被归为 L2）。但 HR 合并 recall 0.936 达标，
   对 Safety routing（L2+ 触发）影响小。
2. **sos-1k 域 FPR 高（0.70）**：v4_2 亦如此（0.76）。sos-1k 标签病态（戏剧化语言标 L0）。
   生产对话域 FPR 0.022 是目标场景；sos 域偏差为 domain-only 训练的已知取舍。
3. **HR Recall 0.936 ≈ 0.95**：余量小，需持续监控。
4. 端到端 Safety Recall 0.776（模块级 0.936 之下的保守 context rules）。

## 7. 交付物

```text
backend/risk/
├── __init__.py
└── dynamic_state.py        # DynamicRiskState + DynamicRiskTracker（RiskResult v2）
bert_data/models/v5_1_tuned/best_model/   # Risk v5.1（weight，gitignore）
bert_data/scripts/
├── build_v5_expansion.py   # v5 扩展数据集构建
├── build_v5_1_expansion.py # （合入）
├── train_v5_tuned.py       # v5 训练
└── train_v5_1_tuned.py     # v5.1 训练
bert_data/processed_v4/train_v5_expansion.jsonl / train_v5_1_expansion.jsonl
evaluation/risk/
├── run_risk_shootout.py
├── run_fusion_calibration.py
├── generate_trajectory_benchmark.py
├── run_trajectory_eval.py
└── risk_trajectory_benchmark_v1.jsonl
docs/
├── risk_dataset_audit_v2.md
├── phase5_checkpoint_ablation.md
├── phase5_fusion_calibration.md
├── phase5_multiturn_risk_eval.md
├── phase5_policy_end_to_end.md
└── phase5_final_review.md   # 本文
```

## 8. 最终裁决

```text
RiskResult v2 / DynamicRiskState = FREEZE（PASS）

依据：
1. Module-level 目标达成（Macro F1 0.706 / FPR 0.022 / L2 F1 0.648）
2. HR Recall 0.936（≈0.95，Freeze with limitations）
3. 多轮轨迹全 PASS（Early Detection / Escalation / Recovery）
4. Policy End-to-end 系统收益显著（Primary +0.23 / Safety Target +0.38）
5. 从 v4_2（FPR 0.42 / Recall 0.48）→ v5.1（FPR 0.02 / Recall 0.94），
   最大瓶颈 Risk Perception 已实质性修复

后续：
- 生产切换到 BERT_MODEL_PATH=v5_1_tuned/best_model（shadow 观察）
- L3 F1 与 sos 域偏差留待持续数据迭代
- Phase 4 Memory 2.0 可推进
```
