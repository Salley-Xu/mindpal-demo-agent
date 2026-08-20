# Final Error Analysis

> 对应 Phase 8 §6
> 日期：2026-08-20
> 基于 Phase 7 Trace + Oracle-state 归因

---

## 1. 错误归因框架

使用 Phase 7 ErrorAttributor + oracle-vs-predicted 对比，将端到端误差分解到最早错误层。

## 2. Policy 层残留误差（oracle-state，Independent Policy Test 430）

| 类别 | 数量 | 归因 |
|---|---:|---|
| ask_clarification 漏检 | 50 | Policy-intrinsic（含糊检测泛化）→ LLM fallback 目标 |
| rec soft 漏检 | 43 | Policy-intrinsic（rec 粒度）→ LLM 兜底候选 |
| tool extra | 3 | Policy-intrinsic（保守多检索） |

**Safety 类误差（P05/P06/P07）= 0**。

## 3. Upstream 误差（predicted vs oracle，Risk 感知）

| 层 | 指标 | 差距来源 |
|---|---|---|
| Risk | HR Recall 0.936（模块）→ 端到端 0.776 | context floor 规则保守性（讨论/否认降级） |
| Perception | predicted primary 0.52→0.75 | 单轮模型 + 意图粗通道（Rule 通道时更差） |

## 4. 端到端 Top Residual Error

1. **Risk 感知残余**（模块 HR Recall 0.936 ≈ 0.95，L3 F1 0.515）：隐式高风险 + L2/L3 混淆
2. **Policy clarification 泛化**：含糊表达（~12% 决策）依赖 LLM fallback
3. **Rec soft 粒度**：soft 是少数类，确定性难区分

## 5. 结论

- **Safety 已实质修复**：模块 FPR 0.42→0.02，HR Recall 0.48→0.94
- **剩余误差集中在**：L3 细分（0.515 F1）、clarification 泛化、rec 粒度 —— 均为
  可解释的 known-limitation，非系统性缺陷
- Phase 7 Trace 已能定位到具体层（perception/policy/safety）
