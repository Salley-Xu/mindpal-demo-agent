# Phase 1 Final Review — Intent Recognition

> 对应 Phase 1 Task 1.13 / §44-45 / §50
> 日期：2026-08-19
> 前置：Phase 0 / 0.5 PASS，Benchmark v1.1 Frozen

---

## 1. Taxonomy 是否可稳定标注？—— **PASS** ✅

| 证据 | 结果 |
|---|---|
| 标注指南 | `docs/intent_annotation_guideline.md`（10 类 + 7 组混淆对 + 边界规则） |
| Seed 数据 | 806 条，全标签≥60，multi-label 30%，硬负样本 101 |
| 查重审计 | Exact=0，MinHash=0，无标签对共现率>0.8（无冗余） |
| LLM↔Gold 一致性 | 多数标签 gold recall ≥0.76；follow_up 最低（0.55，未传 context） |
| 标签可分性 | 分类器按标签 F1 均 >0.63（除 follow_up），证明标签可学习 |

**结论**：10 类标签定义稳定、可区分、可标注。follow_up 是最难标签（依赖上下文），已在指南 §3.4 明确边界。

## 2. 最佳 Small Model

| 项 | 值 |
|---|---|
| 模型 | `hfl/chinese-macbert-base`（Multi-label 头，BCEWithLogitsLoss） |
| 训练数据 | Seed train（564 条，分层切分），8 epochs，lr 2e-5 |
| **Macro F1（test）** | **0.854**（per-label threshold 校准后） |
| Micro F1（test） | 0.845 |
| Exact Match（test） | 0.689 |
| Latency | ~100ms（CPU） |
| 模型文件 | `models/intent/best_model/` |

**关键标签**（test）：memory_reference F1=0.97、resource_request=0.95、high_risk_expression F1=0.91 / Recall=0.88、meta_question=0.87、follow_up 弱（需 context）。

## 3. Calibration 是否有效？—— **YES**

| 指标 | Before | After | 变化 |
|---|---:|---:|---:|
| Temperature T | 1.0 | **0.84** | - |
| ECE（dev） | 0.085 | **0.059** | ↓30% |
| Brier（dev） | 0.033 | **0.029** | ↓12% |
| Macro F1（per-label thr） | 0.809 | **0.854** | ↑0.045 |

**结论**：温度缩放 + per-label threshold 均有效。per-label threshold（§25 推荐路径）比全局阈值再提升 +0.011。

## 4. Open-set 是否有效？—— **部分，未达标**

| 指标 | 值 | 目标 | 状态 |
|---|---:|---:|---|
| AUROC | 0.657 | ≥0.80 | ❌ |
| OOD Recall（最优阈值 0.6） | 0.457 | ≥0.80 | ❌ |
| In-domain False Reject | 0.217 | ≤0.10 | ❌ |
| OOD 数据集 | 105 条（9 域） | 300-500 | ⚠️ 偏小 |

**根因**：
1. max-score 单一信号分离力弱（in-domain mean 0.72 vs OOD 0.62，重叠大）
2. OOD 中"什么是 X/怎么 Y"类问题与 information_request 语义重叠
3. OOD 数据集偏小且部分样本过于贴近域内

**结论**：max-score 策略（§30 第一版）**不足以满足 OOD Recall≥0.80**。需 §30 的 entropy / embedding-prototype-distance 替代策略，或构造更清晰的 OOD 数据。此为 Phase 1 主要未达标项，记录为已知限制。

## 5. Hybrid 是否值得？—— **值得（对 OOD/长尾），域内准确率以 classifier 为准**

| 项 | Small Model + Calibration | Hybrid（+LLM Fallback） |
|---|---:|---:|
| Macro F1（test 60 样本） | 0.802 | 0.791 |
| LLM Call Rate | 0% | **26.7%**（达标 ≤30%） |
| OOD 兜底 | 无（OOD 被判为域内） | 41% OOD 送 LLM |
| Avg Latency | ~100ms | ~1185ms |
| 成本/1k 请求 | ~0 | ~0.27×LLM-only |

**结论**：
- **域内准确率**：Calibrated Small Model 已超 LLM-only（0.854 vs 0.70），**无需 LLM**。
- **Hybrid 的价值在 OOD/长尾**：open-set 判定后送 LLM 兜底，弥补 classifier 对未知输入的过度自信。
- LLM call rate 26.7% 达标；若更保守，可将 fallback threshold 调低（更多 LLM 调用）换取鲁棒性。

## 6. Phase 2 AgentState 应接入哪个 IntentResult 实现？

**推荐：Calibrated Small Model + Hybrid 兜底（两段式）**

```python
# 运行时（HybridIntentPredictor，evaluation/intent/hybrid.py）
IntentResult = {
    "labels": [...],              # per-label threshold 校准后的标签
    "confidence": max_score,      # 经温度缩放
    "is_open_set": bool,          # MaxScoreOpenSet（后续升级 entropy/embedding）
    "source": "classifier" | "llm" | "rule",
}
```

- **默认路径**：`SmallModelPredictor`（per-label threshold [0.25,0.55,0.4,0.2,0.35,0.4,0.45,0.25,0.35,0.25]，温度 T=0.84）—— 0% LLM，延迟 ~100ms
- **兜底路径**：`is_open_set` 或 `confidence < 0.55` → LLM fallback
- **关键改进 TODO（接入前）**：
  1. follow_up/memory_reference 需 context-aware（传上轮）—— 否则 benchmark follow_up F1=0.08
  2. open-set 升级（entropy/embedding distance）
  3. high_risk_expression 单独低阈值（Recall 优先）

---

## 停止条件判定（§45）

| 条件 | 要求 | 实际 | 判定 |
|---|---|---:|---|
| **A** | Small Model Macro F1 ≥ 0.85 | **0.854** | ✅ |
| **A** | Hybrid LLM Call Rate ≤ 30% | **26.7%** | ✅ |
| C | Taxonomy Agreement < 0.80 退回 | 0.88+ | 未触发 |

**Phase 1 通过（条件 A 满足）**。

## 未达标的已知项（记录，不无限调参）

1. Micro F1 0.845（目标 ≥0.90）—— 接近，随数据扩充可提升
2. high_risk_expression Recall 0.88（目标 ≥0.95）—— 需隐式表达样本扩充（错误分析 P0）
3. Open-set OOD Recall 0.46（目标 ≥0.80）—— 需替代策略，Phase 1.5
4. follow_up 弱（无 context）—— 需 context-aware，Phase 1.5
5. 完整 Task 1.7（3000-5000 条 LLM 扩充）—— 轻量版已完成（1052 条），完整版 TODO

## Phase 2 接入建议

按 §50 目标状态，系统应从 `emotion_analyzer._detect_user_intent()`（4 类规则）升级为独立 Intent Recognition Layer。Phase 2 AgentState 的 `intent` 字段应消费上述 `IntentResult`（labels/confidence/is_open_set/source），并保留 `fallback_reason` 用于 trace。
