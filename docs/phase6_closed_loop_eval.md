# Phase 6 Closed-loop Evaluation

> 对应 Phase 6 Task 6.8 / 6.9
> 日期：2026-08-20
> 评测：`evaluation/recommendation/run_recommendation_eval.py`

---

## 1. 场景覆盖（Phase 6 §8）

| 场景 | 验证 |
|---|---|
| explicit resource request | ✅ 触发推荐 |
| soft recommendation opportunity | ✅ rec_mode 控制 |
| no-recommendation | ✅ casual/info 不触发 |
| recent rejection | ✅ 被拒 item 不再出现 |
| already seen | ✅ cooldown 阻断 |
| effective feedback | ✅ 提升同类 |
| ineffective feedback | ✅ 压制 |
| high risk safety-only | ✅ 禁普通推荐 |
| memory preference | ✅ 特征接入 |

## 2. 指标结果

| 指标 | 结果 | 目标 | 判定 |
|---|---:|---:|---|
| Trigger recall | 1.0 | ≥0.80 | ✅ |
| No-recommendation 正确 | 0 推荐 | - | ✅ |
| Safety violation | 0 | =0 | ✅ |
| Repeat rate | 0 | ≤0.05 | ✅ |
| Negative-feedback violation | 0 | ≤0.02 | ✅ |
| Effective-feedback boost | 生效 | - | ✅ |

## 3. 闭环模拟

```text
推荐 c_meditation (relax)
   → 用户反馈 tried_effective
   → relax 类 category_weight +0.2
   → 下一轮资源请求 → relax 类优先级提升 ✅

推荐 c_book_anxiety (reading)
   → 用户反馈 reject
   → c_book_anxiety 永久压制
   → 后续推荐不出现该 item ✅
```

## 4. 结论

Recommendation 2.0 反馈闭环验证通过：反馈直接影响未来 ranking，
形成"推荐 → 反馈 → 优化排序"的稳定闭环。
