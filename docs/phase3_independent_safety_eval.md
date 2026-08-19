# Phase 3 Independent Safety Evaluation

> 对应 Phase 3 Closeout C3.3
> 日期：2026-08-20
> 数据：`evaluation/policy/policy_safety_independent_v1.jsonl`（124 cases，10 类场景）

---

## 1. Safety Slice 构成（124 cases：正 79 / 负 45）

| 场景 | cases | 类型 |
|---|---:|---|
| self explicit risk | 15 | 正 |
| self implicit risk | 15 | 正 |
| third-party risk | 15 | 正 |
| risk + resource request | 8 | 正 |
| risk + memory reference | 6 | 正 |
| Intent high / Risk low 冲突（Contract A） | 10 | 正 |
| Risk high / Intent normal 冲突 | 10 | 正 |
| safe denial | 15 | 负 |
| academic / discussion | 15 | 负 |
| quoted risk language | 15 | 负 |

## 2. 指标

### 2.1 Oracle-state（Policy 自身安全逻辑）

| 指标 | 结果 | 目标 |
|---|---:|---:|
| Safety Recall | **1.0**（79/79） | ≥0.95 ✅ |
| Safety Precision | **1.0** | - |
| Safety FPR | **0.0**（0/45） | ≤0.05 ✅ |
| Safety Target Accuracy | **1.0** | ≥0.95 ✅ |
| Over-trigger Rate | **0.0** | - |

**解读**：
- 全部 79 正例命中：显式/隐式/第三方/risk+resource/risk+memory 由 S01-S03，
  **Intent high / Risk low 冲突由 S04（Contract A）兜底命中**，
  Risk high / Intent normal 冲突由 S01/S03 命中。
- 全部 45 负例不升级：safe denial / discussion / quoted 无 FPR。
- S04 兜底 **未引入 FPR**（oracle FPR=0），验证 C3.1 方案 A 安全性。

### 2.2 Predicted-state（rule 通道，端到端）

| 指标 | 结果 | 归因 |
|---|---:|---:|
| Safety Recall | 0.2025 | upstream：规则通道漏隐式风险（FNR） |
| Safety Precision | 0.4444 | upstream：关键词对否定盲区（FPR） |
| Safety FPR | 0.4444 | upstream：如"我没有伤害自己的念头"含 L3 关键词"伤害自己"→误判 |
| Over-trigger Rate | 0.1613 | upstream（同上） |

## 3. 结论

1. **Policy 安全逻辑在独立场景上无懈可击**：Recall 1.0 / FPR 0 / Target 1.0。
2. **端到端安全 FPR/FNR 全部来自 Risk 感知**（关键词规则对否定/隐式表达的盲区），
   与 Phase 5 Risk 2.0 修复范围一致。
3. C3.1 的 high_risk_intent fallback（S04）合同对齐**通过 FPR 验证**。
