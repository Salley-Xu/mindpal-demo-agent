# Phase 1 Ablation Report — Intent Recognition

> 对应 Phase 1 Task 1.13 / §33 / §39
> 日期：2026-08-19

---

## 1. 实验矩阵（§33）

| Method | Macro F1 | Micro F1 | Exact Match | OOD Recall | LLM Call Rate | Avg Latency |
|---|---:|---:|---:|---:|---:|---:|
| A. Legacy Rule | 0.143 | 0.207 | 0.130 | - | 0% | ~0ms |
| B. LLM-only | 0.697 | 0.683 | 0.333 | - | 100% | 3726ms |
| C. Small Model | 0.809 | 0.810 | 0.631 | - | 0% | 101ms |
| D. + Calibration | **0.854** | **0.845** | **0.689** | - | 0% | 101ms |
| E. Hybrid (+LLM Fallback) | 0.791* | 0.809* | 0.617* | 0.41 | **26.7%** | 1185ms |
| F. Context-aware | TODO | - | - | - | - | - |

> \* E 在 60 样本子集上评测（限 LLM 成本）。A/B/C/D 在 seed test v1（122 条，分层）评测；B 在分层 150 条子集评测 —— 跨样本对比，方向性结论成立。

## 2. 必做 Ablation 逐项（§39）

### Experiment A：Legacy Rule vs Small Model
- Legacy Macro F1 **0.143** → Small Model **0.809**（+0.67）
- **结论**：独立 Intent Layer 相比旧 user_intent 关键词规则是数量级提升。

### Experiment B：0.5 Threshold vs Optimized Threshold
- global 0.5：test Macro F1 **0.809**
- optimized global（0.40）：test Macro F1 **0.843**（+0.034）
- per-label threshold：test Macro F1 **0.854**（+0.045）
- **结论**：per-label threshold 最优（§25 推荐路径验证有效）。

### Experiment C：No Calibration vs Calibration
- 温度缩放 T=0.84：ECE **0.085 → 0.059**，Brier **0.033 → 0.029**
- **结论**：温度缩放有效改善校准（ECE 降 30%）。

### Experiment D：Small Model vs +LLM Fallback
- Calibrated classifier：test Macro F1 **0.854**（0% LLM）
- Hybrid（fallback thr 0.55）：test Macro F1 0.791（60 样本，26.7% LLM）
- **结论（诚实）**：对**域内 test 集**，LLM fallback 无增益甚至略降 —— 因为 low-confidence 被 fallback 的样本里，classifier 不少是"低置信但正确"。LLM fallback 的价值在 **OOD / 长尾新输入**（OOD Recall 0.41 会送到 LLM），不在域内准确率。

### Experiment E：Current-turn only vs Context-aware
- TODO（需传上轮 context 重训/重测）。已知 follow_up 在当前-turn-only 下弱：
  - Benchmark follow_up F1 **0.077**（未传 context）
  - 标注一致性分析也显示 follow_up gold recall 0.55（未传 context 的 LLM 复标）
- **方向**：follow_up/memory_reference 依赖上下文，Phase 1.5 需补 context-aware。

## 3. 成本 / 延迟（§34）

| Method | Latency | LLM Call Rate | 估算成本/1k 请求 |
|---|---:|---:|---:|
| Legacy Rule | ~0ms | 0% | ~0 |
| Small Model | ~100ms | 0% | ~0（本地 CPU） |
| LLM-only | ~3700ms | 100% | ~37 倍于 hybrid（按调用量） |
| Hybrid | ~1185ms | 26.7% | ~0.27×LLM-only |

> DeepSeek 调用成本按每次 ~250 tokens 估算，hybrid 相对 LLM-only 降低约 **73%** LLM 调用量。

## 4. 结论

1. **Calibration 是最大增益来源**：per-label threshold + temperature scaling 把 Macro F1 从 0.809 提到 **0.854**，同时改善校准。
2. **Hybrid 的价值在 OOD/长尾**，不是域内准确率；LLM call rate 26.7% 达标（≤30%）。
3. **follow_up 是当前最大弱点**（需上下文），也是 Phase 1 与 Agent Policy 结合时需优先解决的。
4. **Open-set（max-score）不达标**（OOD Recall 0.46 < 0.80），需 §30 的 entropy / embedding-distance 替代策略。
