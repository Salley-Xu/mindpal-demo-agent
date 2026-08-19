# MindPal Agent Phase 4：Memory 2.0 执行文档

> 阶段目标：从“每轮都检索 + 局部 Write Gate”升级为完整 Memory Lifecycle。

## 1. 当前已知问题

现有系统：
- 正常路由每轮调用 MemoryContextBuilder
- Memory Retrieval Behavior Precision 很低，存在严重 over-retrieval
- 有 Write Gate / Retriever / Injector，但缺完整 conflict / dedup / expire 生命周期
- extractor 覆盖的 memory type 少于 schema 中定义的类型
- recommendation_feedback / coping_strategy 等没有完整写入链路
- memory embeddings/outbox 等存在未闭环结构

因此 Phase 4 重点不是“换向量库”，而是 Memory Governance。

## 2. 最终生命周期

```text
Observe
↓
Extract
↓
Write Gate
↓
Dedup / Conflict
↓
Store
↓
Retrieve Gate
↓
Candidate Retrieval
↓
Rerank
↓
Use
↓
Feedback
↓
Update / Merge / Supersede / Expire
```

## 3. 执行顺序

```text
4.1 Memory Schema Audit
4.2 Memory Type / Operation Freeze
4.3 Write Pipeline v2
4.4 Dedup + Conflict Resolution
4.5 Retrieval Gate
4.6 Retrieval / Rerank
4.7 Memory Use Trace
4.8 Feedback / Update Lifecycle
4.9 Memory Benchmark + Ablation
4.10 Final Freeze
```

## 4. Memory Types

建议至少：

```text
PROFILE
PREFERENCE
EVENT
GOAL
COPING_STRATEGY
COPING_FEEDBACK
RELATIONSHIP
INTERACTION
```

每类定义：
- extract condition
- stability
- TTL / expiry
- merge key
- privacy level
- retrieval priority

## 5. Memory Operations

冻结：

```text
ADD
UPDATE
MERGE
SUPERSEDE
EXPIRE
DELETE
```

每次写入必须生成：
```text
operation
reason
source_turn
confidence
```

## 6. Write Gate

特征：
```text
importance
stability
future usefulness
confidence
privacy
duplication
```

禁止“所有事实都写记忆”。

评测：
```text
Write Precision
Write Recall
Over-write Rate
Sensitive-write Error
```

## 7. Dedup / Conflict

必须处理：
```text
同义重复
偏好更新
旧事实被新事实替代
相互矛盾事实
时间变化
```

示例：
```text
旧：喜欢冥想
新：最近不想再做冥想
→ SUPERSEDE / context-aware update
```

不能简单两条都永久保留。

## 8. Retrieval Gate

这是 Phase 4 最高优先级。

输入：
```text
AgentState.intent
current query
memory_reference signal
personalization need
conversation context
```

输出：
```text
retrieve_memory: bool
retrieval_scope
memory_types
```

目标：
> 从“每轮都检索”变为“需要时才检索”。

## 9. Retrieval / Rerank

Baseline：
```text
Naive Vector
```

对比：
```text
Hybrid semantic + lexical
+ type match
+ recency
+ importance
+ confidence
+ intent match
```

第一版不需要上 LLM rerank。

## 10. Memory Benchmark

新建独立：
```text
memory_benchmark_v2.jsonl
```

覆盖：
```text
explicit reference
implicit personalization
conflict
updated preference
expired memory
irrelevant memory
no-memory-needed
multi-memory
```

指标：
```text
Retrieval Gate P/R/F1
Top-k Recall
MRR / nDCG
Wrong Memory Rate
Over-retrieval Rate
Conflict Resolution Accuracy
Memory Use Accuracy
```

## 11. Ablation

必须：
```text
No Memory
Naive Always Retrieve
Retrieval Gate
Gate + Rerank
Memory 2.0 Full
```

重点证明：
> Memory 2.0 不是“召回更多”，而是“只在必要时召回正确记忆”。

## 12. 与 Policy 的边界

Policy 只决定：
```text
retrieve_memory
```

Memory 2.0 决定：
```text
retrieval scope
which memories
how rank
how update
```

不要让 Memory 反向侵入 Policy。

## 13. 目标

建议：
```text
Retrieval Gate Precision >= 0.80
Recall >= 0.85
Over-retrieval <= 0.15
Top-5 Recall >= 0.90
Conflict Accuracy >= 0.90
```

## 14. 交付物

```text
docs/
├── memory_schema_v2.md
├── memory_write_policy_v2.md
├── memory_conflict_resolution.md
├── memory_retrieval_gate.md
├── phase4_memory_ablation.md
└── phase4_final_review.md

backend/memory_v2/
├── schema.py
├── extractor.py
├── write_gate.py
├── resolver.py
├── retrieval_gate.py
├── retriever.py
├── reranker.py
└── updater.py
```

## 15. Codex 启动指令

```text
不要先换数据库或 embedding 模型。

Phase 4 的第一目标是：
1. Write Lifecycle
2. Retrieval Gate
3. Conflict / Dedup
4. Retrieval quality

必须保留 No Memory / Always Retrieve baseline。
最终需要证明 Memory 2.0 相比当前“每轮都检索”显著降低 over-retrieval，
同时保持 memory_reference 场景高 Recall。
```
