# Recommendation Contract v2

> 对应 Phase 6 Task 6.1 / 6.2
> 日期：2026-08-20
> 状态：**冻结**

---

## 1. Trigger / Mode Contract

AgentPolicy 输出 `recommendation_mode`，Recommendation 2.0 只消费该信号，不修改 PrimaryAction。

| Policy mode | Recommendation Tool 行为 |
|---|---|
| `none` | 不推荐（返回空） |
| `soft` | 允许 soft 推荐（1-2 条，低侵入） |
| `hard` | 允许 hard 推荐（2-3 条，直接满足资源请求） |
| `safety_only` | 只放行 safety 资源（否则返回空） |

## 2. 触发条件（是否考虑推荐）

- explicit resource_request → hard/direct
- 合适的 help 上下文 → soft
- 非适当场景（casual 首轮 / info 知识查询）→ none
- 高风险（Risk v5.1 level≥2）→ safety_only / none（INV-02）

## 3. 边界（与 Policy 解耦）

- **Policy 决定**：`recommendation_mode` + `recommend_resource` tool signal
- **Recommendation 2.0 决定**：候选召回、排序、cooldown、反馈权重、safety 放行
- Memory 2.0 提供 preference 信号（`memory_preference` 特征）

## 4. 冻结内容

- 特征权重（ranker.py `_FEATURE_WEIGHTS`）：semantic 0.20 / risk 0.20 / intent 0.15 / emotion 0.15 / bm25 0.10 / profile 0.10 / memory 0.10
- Cooldown：同 item 8 轮硬阻断 / 同类 3 轮抑制
- 不依赖 LLM Rerank（feature ranker 达标，§7 原则）
