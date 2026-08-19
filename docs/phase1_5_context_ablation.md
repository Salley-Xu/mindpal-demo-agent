# Phase 1.5 Context-aware Ablation

> 对应 Phase 1.5 Task 1.5.1（§5）
> 日期：2026-08-19

## 1. 实验设置

- 数据：`intent_seed_v1_5.jsonl`（850→951 条，含上下文增强）
- 切分：stratified 70/15/15（train 595+ / dev 127 / test 128）
- 模型：macbert-base multi-label，5 epochs（同配置只变输入上下文）
- 输入格式：current only / +1 turn（prev_assistant+current）/ +2 turns（prev_user+prev_assistant+current）

## 2. 结果（dev/test v1_5）

| Input | Macro F1 | Micro F1 | Exact | follow_up F1 | memory_reference F1 | feedback F1 |
|---|---:|---:|---:|---:|---:|---:|
| Current only | 0.661 | 0.715 | 0.438 | 0.790 | 0.833 | 0.714 |
| **+1 turn** | **0.699** | 0.741 | **0.516** | **0.978** | 0.857 | 0.667 |
| +2 turns | 0.692 | **0.756** | 0.523 | 0.978 | **0.919** | 0.720 |

## 3. 结论

1. **上下文显著提升 follow_up**：F1 从 0.79 → **0.978**（目标 ≥0.75 达成）。
2. **+1 turn 是甜点**：Macro F1 最高（0.699），follow_up 已到顶（0.978）。
3. **+2 turns 对 memory_reference 更好**（0.92 vs 0.86），但整体 Macro 略降（0.692），且输入更长、推理成本更高。
4. 按 §5.4 规则（+2 相对 +1 无明显收益 → 选 +1），**最终选择 Previous 1 Turn**。

## 4. 泛化验证（独立测试集 535 条）

在全新独立测试集上，context-aware 模型（E）Macro 0.817 vs 校准 current-only（D）0.843：

| 视角 | 结果 |
|---|---|
| 域内 dev（同上下文风格） | ctx1 显著优于 current（follow_up 0.79→0.98） |
| 独立测试（全新场景措辞） | context 泛化收益减弱（E 0.817 < D 0.843） |

**解释**：上下文对 follow_up 的帮助在**上下文风格与训练一致**时最明显；全新场景的上下文格式多样时，context 模型可能把注意力分给不熟悉的历史轮，收益被稀释。

**决策**：正式 Intent Service 接口**支持短上下文**（`previous_turns ≤ 1`），但运行时默认在当前输入上预测，上下文作为可选增强（当对话历史存在且风格一致时启用）。
