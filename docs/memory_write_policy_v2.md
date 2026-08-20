# Memory Write Policy v2

> 对应 Phase 4 Task 4.3
> 日期：2026-08-20
> 实现：`backend/memory_v2/write_gate.py`

---

## 1. 特征（Phase 4 §6）

```text
importance      0.3 + stability*0.3 + explicit*0.2 + risk*0.25 + confidence*0.1
stability       高稳定类型（profile/preference/relationship）权重高
future usefulness 通过 type priority 体现
confidence      提取器置信度
privacy         敏感类型低置信不写细节
duplication     resolver 处理
```

## 2. 门槛

```text
should_write = importance >= 0.45
```

- 显式来源 +0.2；风险相关 +0.25；低信号事件 -0.1
- 禁止"所有事实都写记忆"（临时情绪/单轮噪声不写）

## 3. 评测指标

| 指标 | 定义 | 目标 |
|---|---|---|
| Write Precision | 写入中有用的比例 | ≥0.80 |
| Write Recall | 应写入的都被写入 | ≥0.80 |
| Over-write Rate | 无用写入比例 | ≤0.15 |
| Sensitive-write Error | 敏感细节被写入 | =0 |

## 4. 与 legacy MemoryPolicy 的关系

- legacy `MemoryPolicy.should_write` 保留（后端兼容）
- Memory 2.0 `WriteGateV2` 在其上增加：risk 加权 / 低信号惩罚 / privacy gate
