# Memory Schema v2

> 对应 Phase 4 Task 4.2
> 日期：2026-08-20
> 状态：**冻结**
> 实现：`backend/memory_v2/schema.py`

---

## 1. Memory Types（8 类）

| Type | extract condition | stability | TTL | merge key | retrieval priority |
|---|---|---|---|---|---|
| `profile` | explicit | 0.9 | 长期 | topic | 1.0 |
| `preference` | explicit | 0.8 | 长期 | preference | 1.2 |
| `event` | inferred | 0.5 | 180 天 | event | 0.8 |
| `goal` | explicit | 0.7 | 长期 | goal | 1.1 |
| `coping_strategy` | inferred | 0.6 | 90 天 | strategy | 1.0 |
| `coping_feedback` | explicit | 0.7 | 60 天 | feedback | 0.9 |
| `relationship` | explicit | 0.85 | 长期 | person | 1.1 |
| `interaction` | inferred | 0.5 | 30 天 | pattern | 0.7 |

## 2. Memory Operations（6 种，冻结）

```text
ADD / UPDATE / MERGE / SUPERSEDE / EXPIRE / DELETE
```

每次写入生成 MemoryWriteEvent：

```json
{ "operation": "SUPERSEDE", "reason": "preference_supersede",
  "source_turn": "t_xxx", "confidence": 0.9,
  "candidate_type": "preference", "candidate_content": "..." }
```

## 3. 与 legacy MemoryItem 的关系

- 复用 `models.MemoryItem`（已含 memory_type/importance/confidence/expires_at/status）
- Memory 2.0 在其上增加治理层（Write Gate / Conflict / Lifecycle）
- 不换数据库 / embedding
