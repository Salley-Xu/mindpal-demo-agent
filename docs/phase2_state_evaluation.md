# Phase 2 State Consistency Evaluation

> 对应 Phase 2 Task 2.9（§33-35）
> 日期：2026-08-19

---

## 1. 评测设置

- **评测对象**：StateBuilder + StateUpdater（不评测感知模型本身）
- **输入**：已知的感知结果（intent/emotion/risk），检查是否正确写入 State
- **数据**：`evaluation/state/state_transition_benchmark_v1.jsonl`（36 case / 89 轮）
- **覆盖场景**：普通闲聊 / 情绪表达 / 信息请求 / 高风险升级(0→1→2→3) / 降级(2→1→0) / 波动 / follow_up / memory_reference / intent 变化 / 第三方危机 / 推荐拒绝 / session 隔离

## 2. 结果

| 指标 | 值 | 目标 | 状态 |
|---|---:|---:|---|
| Schema Validation Pass | 100% | 100% | ✅ |
| Field Accuracy | **1.0** | - | ✅ |
| **State Transition Accuracy** | **1.0** | ≥0.95 | ✅ |
| Session Isolation | **100%** | 100% | ✅ |
| 完全匹配 case | 36/36 | - | ✅ |

## 3. 验证的状态更新规则

| 规则 | 验证 |
|---|---|
| intent REPLACE | ✅（每轮新 intent 覆盖） |
| emotion REPLACE | ✅（情绪变化正确写入） |
| risk persistence ACCUMULATE/RESET | ✅（L2+ 连续累积，否则归零） |
| risk recent_levels ROLLING_WINDOW(5) | ✅（窗口正确，末位正确） |
| recommendation turns_since ACCUMULATE | ✅（有推荐→0，否则+1） |
| meta.state_version ACCUMULATE | ✅ |
| session isolation | ✅（各 case 独立，无串扰） |

## 4. Shadow-mode 一致性（§28）

每轮旁路检查 AgentState 与 legacy 变量：

```python
ConsistencyChecker.check(state, {"risk_level", "emotion", "stage", "turn_count"})
```

- risk_match / emotion_match / stage_match / turn_count_match
- 结果写入 `logs/agent_state_shadow.jsonl`（每轮一条，含 state 摘要 + consistency）

## 5. 生产行为回归确认（§36-37）

Shadow 集成是**行为保持重构**：
- 钩子包在 `try/except` 内，失败静默
- 不修改任何生产变量 / 不改变控制流
- Agent 端到端冒烟测试通过（正常完成，shadow 不干扰）

**结论**：Phase 2 未改变生产 Agent 行为（Risk/Recommendation/Routing 指标不会变化，因未触碰其逻辑）。
