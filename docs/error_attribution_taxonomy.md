# Error Attribution Taxonomy

> 对应 Phase 7 Task 7.7
> 日期：2026-08-20
> 实现：`backend/tracing/attribution.py`

---

## 1. 错误层

| 层 | 说明 | 检查信号 |
|---|---|---|
| Perception | 感知错误 | intent 高风险但 risk 极低（冲突） |
| State | 状态错误 | 状态字段不一致 |
| Policy | 策略错误 | risk≥2 但 primary≠safety（safety miss） |
| Memory | 记忆错误 | 该检索未检索 / 检索错误记忆 |
| Recommendation | 推荐错误 | 触发/模式错误 |
| Execution | 执行错误 | tool 调用失败 |
| LLM Response | LLM 响应错误 | 输出非法/超时 |
| Safety | 安全错误 | risk≥2 但 rec=hard/soft（violation） |

## 2. 归因原则

```text
每个失败 case 定位到最早错误层（first_error_layer）
```

- 按层顺序检查：perception → state → policy → memory → recommendation → execution → llm → safety
- 命中即返回该层（不继续更深层）

## 3. 当前实现检查器

| 检查 | 判定 |
|---|---|
| `_perception_error` | high_risk intent + risk=0 → perception_intent_risk_conflict |
| `_policy_error` | risk≥2 + primary≠safety → policy_safety_miss |
| `_safety_error` | risk≥2 + rec∈{hard,soft} → safety_rec_violation |

## 4. 与 Phase 8 结合

- Phase 8 Final Error Analysis 用 oracle-vs-predicted + Trace 归因
- Top residual：Risk 感知残余 / clarification 泛化 / rec 粒度
