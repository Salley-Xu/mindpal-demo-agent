# Agent Trace Schema v1

> 对应 Phase 7 Task 7.1
> 日期：2026-08-20
> 状态：**冻结**
> 实现：`backend/tracing/schema.py`

---

## 1. Schema

```json
{
  "trace_id": "...",          // 同一 turn 一个，模块共享
  "session_id": "...",
  "turn_id": 0,
  "input": {},                // 必要输入 + hash
  "perception": {},           // intent / emotion / risk
  "state": {},                // AgentState 摘要
  "policy": {},               // PolicyResult（source/matched_rules/confidence）
  "tools": {},                // tool 选择
  "memory": {},               // 记忆命中/丢弃
  "recommendation": {},       // 推荐决策
  "response": {},             // 响应摘要
  "feedback": {},             // 反馈
  "latency": {},              // 分阶段耗时
  "versions": {},             // 各模块版本
  "llm_calls": [],            // LLM 调用记录
  "error": {}                 // 错误 + 归因层
}
```

## 2. 版本溯源（versions）

```text
intent: phase1_5_final
risk:   v5_1
state:  agentstate_v1
policy: agentpolicy_v1
memory: memory_v2
recommendation: recommendation_v2
```

## 3. 记录原则

- 必要 input / hash（不存全量敏感 profile）
- 模型版本 / confidence / matched rules / fallback reason
- candidates / selected ids / action plan / execution result / latency
- 避免：全量敏感 profile、全量长期记忆原文、不必要完整 prompt

## 4. 存储

- JSONL（`logs/traces/{trace_id}.json`）
- 可离线读取 + Replay
