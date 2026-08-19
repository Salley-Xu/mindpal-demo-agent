# Phase 1.5 Dataset Report

> 对应 Phase 1.5 Task 1.5.2（§6.8/6.9）
> 日期：2026-08-19

---

## 1. 数据集组成

| 文件 | 数量 | 说明 |
|---|---:|---|
| `intent_seed_v1_5.jsonl` | **951** | seed 806 + context 增强 44 + 隐式高危 101 |
| `intent_test_independent_v1.jsonl` | **535** | 冻结的独立测试集（全新场景） |
| `intent_uncertainty_v1.jsonl` | 65 | Intent Uncertainty（无法稳定映射 10 类） |
| `intent_train/dev/test_v1_5.jsonl` | 595/127/128 | 分层切分（开发用） |

## 2. 训练数据增强（Task 1.5.2 定向扩充）

| 增强项 | 数量 | 说明 |
|---|---:|---|
| follow_up 补上下文 | 98 | 给既有 follow_up 案例补上轮 assistant 上下文 |
| 新增多轮上下文案例 | 44 | follow_up/memory/feedback 真实 2-3 轮 |
| 隐式高危表达 | 101 | 告别/未来绝望/消失/负担感（非固定关键词） |

**效果**：
- follow_up 有上下文的样本：2 → **125**
- high_risk_expression：87 → **188**

> 完整扩充目标（2500-3500）未达成 —— 本阶段为聚焦定向扩充（context + 隐式高危 + 多标签），完整 LLM 扩充留后续。

## 3. 独立测试集（§7）

- **535 条**，全 10 类覆盖，multi-label 43%，context-aware 100 条
- 内容：单意图 34% / multi-label 43% / context 19% / 隐式高危 9% / 硬负 5%
- **创建即冻结**：不参与训练/阈值/校准/LLM seed
- Gold 标注：source=human（模板生成后按指南审核）

## 4. 切分（§6.8）

采用 **Stratified Multi-label Split**（放弃粗粒度模板桶 Group Split —— 其导致 train 标签严重失衡：emotional=16/high_risk=11/memory=5 的教训）。

| Split | 数量 | 标签覆盖 |
|---|---:|---|
| train | 595 | 10 类 |
| dev | 127 | 10 类 |
| test | 128 | 10 类 |

## 5. Leakage Audit（§6.9）

| 检查 | 结果 |
|---|---|
| Exact Match | 0 重复 ✅ |
| MinHash（sim≥0.8） | 0 对 ✅ |
| 变体与源模板分组 | ✅（split 按 variant_of 归组） |
| Embedding Similarity | **TODO**（需 bge embedding 审计，阈值 0.92） |

## 6. Uncertainty 数据集（§8.3）

65 条：乱码/信息不足/极端省略/纯符号/语义不完整/冲突任务/无法判断目的。
> 目标 200-300 未达，当前覆盖全部类型，后续可扩充。
