# Phase 1.5 Unified Evaluation

> 对应 Phase 1.5 Task 1.5.5（§9 / §12）
> 日期：2026-08-19
> 统一在冻结的独立测试集 `intent_test_independent_v1.jsonl`（535 条）上对比。

---

## 1. 统一主表（§9.1）

| Method | Macro F1 | Micro F1 | Exact | high_risk Recall | follow_up F1 | Latency | LLM Call Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| A Legacy Rule | 0.117 | 0.158 | 0.097 | 0.000 | 0.000 | ~0ms | 0% |
| C Small Model | 0.818 | 0.816 | 0.619 | 0.556 | 0.775 | 75ms | 0% |
| D + Calibration | 0.843 | 0.831 | 0.671 | 0.714 | 0.790 | 76ms | 0% |
| **E Context-aware** | **0.844** | **0.835** | **0.697** | **0.873** | 0.707 | 77ms | 0% |
| F Hybrid (+LLM) | 0.783 | 0.776 | 0.579 | 0.571 | 0.685 | 101ms | 16.5% |

> B LLM-only 未在独立测试全量运行（成本），以 Phase 1 分层 150 条结果（Macro 0.70）作参考；其值低于 E。

## 2. 关键结论

1. **E Context-aware（最终模型 phase1_5_final_model）是域内+独立测试综合最优**：Macro 0.844、Exact 0.697。
2. **隐式高危数据扩充显著提升 high_risk Recall**：0.714 → **0.873**（仍略低于目标 0.93）。
3. **follow_up 在全新独立场景下 context 收益减弱**（E 0.707 vs D 0.790）—— 上下文风格与训练不一致时泛化有限。
4. **Hybrid 是 optional fallback 而非主路径**（§9.3）：LLM call 16.5% 达标，但 Macro 0.783 < E 0.844。其价值在 Uncertainty 兜底。

## 3. Slice Evaluation（§12，E 方法）

| Slice | n | Macro F1 | Exact Match | 分析 |
|---|---:|---:|---:|---|
| Single-label | 304 | **0.908** | 0.885 | ✅ 强 |
| Multi-label | 231 | 0.610 | 0.450 | ⚠️ 联合预测弱（次要标签易漏） |
| Context-aware | 100 | 0.377 | 0.550 | ⚠️ 全新上下文场景难 |
| Implicit high-risk | ~50 | - | - | high_risk Recall 0.873 |
| Hard-negative | 27 | 0.280 | 0.815 | ⚠️ 高危词安全语境偶发误报 |

## 4. Agent Benchmark v1.1 回归（§21）

| 模型 | Macro F1 | follow_up F1 | high_risk Recall |
|---|---:|---:|---:|
| Legacy Rule（Phase 1） | 0.138 | - | - |
| Rule Classifier（current） | 0.745 | 0.074 | 0.712 |
| **Final Context-aware** | **0.757** | **0.207** | **0.758** |

- Benchmark Macro F1：0.138 → **0.757**（整体提升）。
- follow_up：0.074 → 0.207（上下文带来 2.8x 提升，仍偏弱——benchmark follow_up 案例的上下文信息有限）。
- 生产代码零改动（Risk/Memory/Gate/Policy/Benchmark 均未修改）。

## 5. 目标对照（§15）

| 指标 | 目标 | 实际 | 状态 |
|---|---:|---:|---|
| Macro F1 | ≥0.85 | **0.844** | ⚠️ 差 0.006 |
| Micro F1 | ≥0.88 | 0.835 | ⚠️ 差 0.045 |
| Exact Match | ≥0.70 | 0.697 | ⚠️ 差 0.003 |
| high_risk Recall | ≥0.95 | 0.873 | ❌ 差 0.077 |
| follow_up F1 | ≥0.75 | 0.707 | ⚠️ 差 0.043 |
| Hybrid LLM Call | ≤30% | 16.5% | ✅ |
| Uncertainty Recall | ≥0.75 | 0.754 | ✅ |
| In-domain FR | ≤0.10 | 0.108 | ⚠️ 差 0.008 |
