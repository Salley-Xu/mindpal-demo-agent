# MindPal Agent Phase 7：Observability / Trace / Replay 执行文档

> 目标：把当前零散 trace 升级为统一 Agent 决策链路可观测系统。

## 1. 当前缺口

现有系统能追踪部分：
```text
input → emotion → risk → gate → route → response
```

但缺：
- 统一 trace_id 贯穿 perception/state/policy/tool
- memory candidate hit/drop 结构化记录
- Policy 决策来源与 matched rules
- ActionPlan 与执行结果关联
- replay
- error attribution

## 2. Unified Trace Schema

建议：

```json
{
  "trace_id": "...",
  "session_id": "...",
  "turn_id": "...",
  "input": {},
  "perception": {},
  "state": {},
  "policy": {},
  "tools": {},
  "memory": {},
  "recommendation": {},
  "response": {},
  "feedback": {},
  "latency": {},
  "versions": {}
}
```

## 3. 执行顺序

```text
7.1 Trace Contract
7.2 Perception Trace
7.3 State / Policy Trace
7.4 Tool / Memory / Recommendation Trace
7.5 Response / Feedback Trace
7.6 Replay Runner
7.7 Error Attribution
7.8 Latency / Cost Dashboard
7.9 Regression / Sampling
7.10 Freeze
```

## 4. Trace 原则

必须：
```text
同一 turn 一个 trace_id
所有模块共享
```

记录：
- necessary input / hash
- model versions
- confidence
- matched rules
- fallback reason
- candidates / selected ids
- action plan
- execution result
- latency

避免写入：
- 全量敏感 profile
- 全量长期记忆原文
- 不必要完整 prompt

## 5. Replay

实现：

```text
trace → reconstruct AgentState → rerun Policy / Tool selection
```

支持：
```text
same-version replay
new-version counterfactual replay
```

用途：
- regression
- bug reproduction
- ablation
- offline policy comparison

## 6. Error Attribution

统一：
```text
Perception Error
State Error
Policy Error
Memory Error
Recommendation Error
Execution Error
LLM Response Error
Safety Error
```

每个失败 case 定位到最早错误层。

## 7. Latency / Cost

至少分解：
```text
Intent
Emotion
Risk
State
Policy
Memory
RAG
Recommendation
LLM Response
Total
```

LLM 调用记录：
```text
count
tokens
fallback reason
```

## 8. 目标

```text
Trace coverage = 100%
Replay success >= 99%
Version provenance = 100%
Missing critical field <= 1%
```

## 9. 交付物

```text
docs/
├── agent_trace_schema_v1.md
├── replay_design.md
├── error_attribution_taxonomy.md
└── phase7_final_review.md

backend/tracing/
├── schema.py
├── logger.py
├── context.py
├── replay.py
└── attribution.py
```

## 10. Codex 启动指令

```text
Phase 7 不改变任何模型与业务决策。
它必须是 observability-only。

所有已有 Phase 的版本号必须进入 trace：
Intent / Risk / AgentState / Policy / Memory / Recommendation。

最终要求任何 Phase 8 failure case 都能从 trace 回放并定位到具体层。
```
