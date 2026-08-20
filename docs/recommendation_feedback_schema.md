# Recommendation Feedback Schema

> 对应 Phase 6 Task 6.5
> 日期：2026-08-20
> 状态：**冻结**

---

## 1. Feedback 类型（统一）

| 类型 | 含义 | Ranking 影响 |
|---|---|---|
| `accept` | 接受/有用 | 该类 `category_weight +0.2` |
| `reject` | 拒绝/不想要 | 该类 `-0.25`，item 永久压制 |
| `tried_effective` | 试过且有效 | `category_weight +0.2` + `effective_boost +0.1` |
| `tried_ineffective` | 试过但无效 | 该类 `-0.25`，item 压制 |
| `not_interested` | 不感兴趣 | 该类 `-0.25`，item 压制 |
| `already_seen` | 已看过 | item `item_penalty +0.5` |

## 2. Event Schema

```json
{
  "user_id": "u_xxx",
  "item_id": "c_xxx",
  "item_category": "relax",
  "feedback": "tried_effective",
  "turn_id": "t_xxx",
  "timestamp": "..."
}
```

## 3. 权重闭环（FeedbackWeights）

| 字段 | 作用 |
|---|---|
| `category_weight` | 该类整体权重（0.2~2.0） |
| `item_penalty` | 单条压制 |
| `category_penalty` | 该类压制 |
| `effective_boost` | 有效反馈额外加分 |

## 4. 原则

- **反馈必须直接影响未来 ranking**（不仅影响 prompt）
- 负反馈 item 永久压制（不重复推荐）
- 正反馈提升同类（越用越懂）
- 与 Memory 2.0 `COPING_FEEDBACK` 类型对接
