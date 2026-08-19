# Phase 3 Final Independent Review

> 对应 Phase 3 Closeout C3.6
> 日期：2026-08-20
> 裁决：**AgentPolicy v1 = FINAL FREEZE（PASS）**

---

## 1. 独立评测证据（非 design-benchmark）

在**全新措辞**独立测试集（430 cases）与独立 Safety Slice（124 cases）上的结果：

### 1.1 Independent Policy Test（oracle-state，Policy 泛化上限）

| 指标 | 目标 | 实测 | 判定 |
|---|---:|---:|---|
| Primary Action Acc | ≥0.88 | **0.8837** | ✅ |
| Safety Recall | ≥0.95 | **1.0** | ✅ |
| Safety FPR | ≤0.05 | **0.0**（Safety Slice） | ✅ |
| Tool Micro F1 | ≥0.75 | **0.995** | ✅ |
| Rec Mode Acc | ≥0.80 | **0.9000** | ✅ |
| Policy Exact | ≥0.70 | **0.7767** | ✅ |
| LLM Fallback Rate | ≤0.20 | 0.061 | ✅ |

### 1.2 Independent Safety Slice（124 cases：正 79 / 负 45）

| 指标 | 实测 |
|---|---:|
| Safety Recall | **1.0**（79/79） |
| Safety Precision | **1.0** |
| Safety FPR | **0.0**（0/45） |
| Safety Target Acc | **1.0** |
| Over-trigger Rate | **0.0** |

### 1.3 相对 Legacy 的泛化优势（oracle，独立集）

| 指标 | Legacy | Deterministic | 提升 |
|---|---:|---:|---:|
| Policy Exact | 0.0721 | 0.7767 | **+10.8×** |
| Rec Mode Acc | 0.6721 | 0.9000 | +0.23 |
| Tool Micro F1 | 0.3468 | 0.9950 | +0.65 |

---

## 2. C3.5 误差归因最终结论

| 归因 | 内容 | 占比 |
|---|---|---|
| **Policy-intrinsic** | ask_clarification 泛化不足（50）+ rec soft 粒度（43）+ tool extra（3） | ~22% |
| **Upstream-caused** | predicted vs oracle 差距（Risk 感知 FNR/FPR） | 主导端到端 |
| **Dataset ambiguity** | soft 是少数类，语义上本身两可 | 部分 soft |
| **Gold inconsistency** | G10_mem_info 32 case 已修正 | 0（修正后） |

**修正表述**：

> Primary/Safety 的主要端到端瓶颈来自 upstream Risk perception；
> Policy 本身存在 ask_clarification 泛化不足与 rec soft 粒度等 intrinsic error，
> 二者均为 LLM fallback 的设计目标场景（独立集 ambiguity 16% + soft 10% ≈ 26% 决策
> 落入 LLM 兜底候选，而当前 LLM rate 仅 6.1%，仍有 budget）。

---

## 3. Safety Contract 一致性

- 文档（Invariants §3：L2/L3 **或 high-risk intent** 触发）与实现（S04 fallback）**已对齐**
- S04 兜底在独立 Safety Slice 上 **FPR=0**，验证方案 A 无过度触发

---

## 4. 最终裁决

```text
AgentPolicy v1 = FINAL FREEZE（PASS）

依据：
1. 独立测试集（全新措辞）oracle 全部达到 PASS 条件
2. Safety Slice 零安全误差（Recall 1.0 / FPR 0 / Target 1.0）
3. Deterministic 泛化显著优于 Legacy（Exact +10.8×）
4. Frozen Benchmark v1.1 仅作 regression，未再用于规则设计
5. 合同-实现一致（S04 补齐），FPR 独立验证通过

已知限制（Freeze with limitations）：
- ask_clarification 确定性检测泛化有限（~50/68 漏检）→ 依赖 P4 LLM fallback
- rec soft 粒度确定性无法完全区分 → LLM 兜底候选
- 端到端 safety 依赖 Phase 5（Risk 感知修复）
```

---

## 5. 交付物

```text
evaluation/policy/
├── policy_independent_test_v1.jsonl      # 430 cases 新措辞
├── policy_safety_independent_v1.jsonl    # 124 cases 10 类场景
├── generate_independent_test.py
├── generate_safety_slice.py
├── run_independent_eval.py
├── run_safety_slice.py

backend/policy/safety.py                  # 新增 S04（Contract A fallback）

docs/
├── phase3_safety_contract_reconciliation.md
├── phase3_independent_policy_eval.md
├── phase3_independent_safety_eval.md
└── phase3_final_independent_review.md    # 本文
```

## 6. 下一步（对齐主路线图）

```text
Phase 5 Risk 2.0（端到端瓶颈根因：Raw BERT FNR/FPR）
  → Phase 4 Memory 2.0
  → Phase 6 Recommendation 2.0
  → Phase 7 Observability
  → Phase 8 Final Evaluation
```
