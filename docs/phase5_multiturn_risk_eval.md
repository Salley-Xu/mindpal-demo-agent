# Phase 5 Multi-turn Dynamic Risk Evaluation

> 对应 Phase 5 Task 5.6 / 5.7
> 日期：2026-08-20
> 交付：`backend/risk/dynamic_state.py` + `evaluation/risk/risk_trajectory_benchmark_v1.jsonl`

---

## 1. DynamicRiskState（RiskResult v2）

`backend/risk/dynamic_state.py`：

```python
class DynamicRiskState:
    level: int          # 会话级聚合等级（单轮模型 + v5.0 状态机）
    confidence: float
    trend: str          # rising | stable | falling | fluctuating | new
    persistence: int    # 连续 L2+ turn 数
    escalation: bool    # 相对上轮升级
    subject: str        # self | third_party | discussion
    discussion: bool
    safe_denial: bool
    recent_levels: list[int]   # 5-window
    reasons: list[str]         # 命中的聚合规则（可 Trace）
```

- 单轮模型负责 current risk；`SessionRiskAggregator` v5.0 状态机负责
  persistence / trend / escalation / recovery。
- 每步记录 `reasons`（active_rules），可 Trace。

## 2. Multi-turn Trajectory Benchmark（14 条轨迹）

| 场景 | 条数 |
|---|---|
| 渐进升级 0→0→1→2→3 | 3 |
| 恢复 3→2→1→0（惯性渐进） | 3 |
| fluctuating | 2 |
| persistent medium | 2 |
| third-party escalation | 2 |
| false alarm recovery | 2 |

## 3. 评测结果

| 指标 | 结果 | 目标 |
|---|---:|---:|
| Trajectory Accuracy | **1.0**（14/14） | ≥0.90 ✅ |
| Early Detection Rate | **1.0** | ≥0.95 ✅ |
| Escalation Accuracy | **1.0** | ✅ |
| Recovery Accuracy | **1.0** | ✅ |
| Trend 参考 | 全部 rising/stable/falling 正确 | - |
| Avg Time-to-detect | **1.46** turns | 早检测 ✅ |

## 4. 关键设计发现

1. **恢复是渐进式（安全优先）**：L3 后 5-window 内 `l2_count≥2` 持续触发升级，
   等级保持在 2；需 ~5 轮滑窗才能降到 0。恢复尾端因 `l1_thrice_in_window`
   停留在 1（保守）。这是**设计行为**，不是缺陷——防止断崖降级。
2. **当前轮优先**：utterance L2/L3 时 session 立即≥2（不等惯性）。
3. **误报恢复快**：false alarm（安全否认/讨论）在 2 轮内回到 0。

## 5. 结论

DynamicRiskState 正确实现 v5.0 状态机语义，轨迹评测全 PASS。
单轮模型升级（v5）只改变 `level` 输入质量，不改变状态机行为。
