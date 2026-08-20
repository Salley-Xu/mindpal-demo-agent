# Memory Conflict Resolution

> 对应 Phase 4 Task 4.4
> 日期：2026-08-20
> 实现：`backend/memory_v2/resolver.py`

---

## 1. 处理场景

| 场景 | 示例 | 操作 |
|---|---|---|
| 精确重复 | 相同 content | ADD 忽略（不重复写） |
| 同义重复 | "觉得冥想不错" vs "觉得冥想挺好" | MERGE |
| 强化表达 | "继续坚持冥想"（reinforce） | MERGE |
| 偏好更新 | "现在不喝咖啡了" | SUPERSEDE（归档旧条目） |
| 矛盾事实 | "喜欢跑步" vs "完全不想动" | SUPERSEDE |
| 时间变化 | 旧偏好被新事实替代 | SUPERSEDE + 新条目 |

## 2. 判定逻辑

```text
1. 精确重复 → ADD skip
2. 字符重叠 > 0.70 → MERGE（synonym_merge）
3. 强化标记（继续/坚持/不错/保持）且同类型重叠>0.4 → MERGE（reinforce_merge）
4. 变化标记（现在不/戒/讨厌/放弃/不再/不想/改）→ SUPERSEDE（preference_supersede）
5. 同类型反向表述 → SUPERSEDE（contradiction_supersede）
6. 否则 → ADD
```

## 3. 关键设计

- **不能简单两条都永久保留**：SUPERSEDE 将旧条目 `status=archived`，新条目 `active`
- 强化 vs 更新区分：排除"不错/挺好的/坚持"等误判（修复"不错"含"不"的 bug）
- Conflict Accuracy = 1.0（Memory Benchmark v2）

## 4. 与 Lifecycle 配合

- SUPERSEDE 由 `updater.py` 执行归档
- TTL 到期由 `updater.apply_expiry` → EXPIRE
- 长期未访问 + 低重要性 → EXPIRE（decay）
