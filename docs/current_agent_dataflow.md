# 当前 Agent 系统数据流图（Baseline）

> 对应开发计划 Task 0.1（§5.2）
> 日期：2026-08-18
> 本文件用文本图 + 表描述一轮完整请求中**每个模块的输入字段 → 输出字段**及其流动方向。

---

## 1. 单轮完整数据流

```text
用户输入 request.text (AgentRunRequest)
   │
   ├─▶ conversation_manager.get_or_create_session_async(user_id, session_id)
   │     └─▶ SQLite: sessions 表（内存 dict: history/emotion_timeline/key_concerns/stage/...）
   │
   ├─▶ conversation_manager.get_conversation_summary_async(user_id, session_id)
   │     └─▶ dict: {conversation_stage, primary_emotion, emotion_trend, key_concerns,
   │               current_topic, stress_sources, recent_intents, turn_count,
   │               compressed_context, accepted/rejected_recommendations, ...}
   │
   ├─▶ emotion_analyzer.analyze_with_context_async(text, summary)
   │     ├─▶ BERT(bert_emotion_predictor) 表层 → 低置信度+多轮 → LLM 深层
   │     └─▶ (current_emotion, context_emotion, confidence)   ← 中文标签
   │
   ├─▶ emotion_analyzer.build_emotion_state_payload(...)
   │     └─▶ preliminary_emotion_state = {current_emotion, emotion_type, context_emotion,
   │             emotion_intensity, stress_source, user_intent, negative_trend,
   │             confidence, emotion_trend}
   │
   ├─▶ UserProfileTool.get_profile(user_id)
   │     └─▶ user_profile = {risk_level, preferred_support_style, avoid_style,
   │             main_stress_sources, recommendation_feedback, ...}
   │
   ├─▶ MoodTrackingTool.get_recent_trend(user_id, limit=20)
   │     └─▶ historical_high_risk_count
   │
   ├─▶ RiskMemoryReader.get_baseline(user_id)
   │     └─▶ risk_baseline (low/medium/high) + common_triggers
   │
   ├─▶ risk_evaluator.evaluate(text, preliminary_emotion_state, summary,
   │         long_term_risk_level, historical_high_risk_count, risk_baseline)
   │     ├─▶ bert_risk_predictor.predict(text) → utterance level + rule_override
   │     ├─▶ session_risk_aggregator.aggregate(...) → session_level + risk_trend
   │     ├─▶ _analyze_context → risk_context {subject, is_discussion, is_third_party, ...}
   │     └─▶ urgent_issue = {level, legacy_level, level_index, level_label, message,
   │             suggestions, triggers, risk_score, raw_score, risk_evidence,
   │             risk_trend, session_aggregation, escalation_reasons, risk_context}
   │
   ├─▶ recommend_gate.decide(preliminary_emotion_state, urgent_issue, summary, user_profile)
   │     └─▶ recommendation_decision = {should_recommend, recommend_type(none/soft/hard/
   │             safety_only), score, threshold, reason_codes, cooldown_remaining}
   │
   ├─▶ 路由（if-else，无显式 Action Space）:
   │     ├─ level_3            → urgent_detector.generate_crisis_response_async → final_response
   │     ├─ 第三方危机求助      → generate_third_party_support_response_async → final_response
   │     ├─ level_2            → generate_crisis_response_async（高危支持）→ final_response
   │     └─ 正常                → ReAct Agent Loop
   │           └─ _build_initial_messages(text, history, summary, urgent_issue, ...)
   │                ├─ SYSTEM_PROMPT_TEMPLATE（阶段策略注入）
   │                ├─ MemoryContextBuilder.build → 记忆注入文本（BM25 检索 + token 预算）
   │                │     └─ memory_retriever.retrieve → memory_injector.build_context
   │                └─ _inject_knowledge_context → 知识库 RAG 注入（top3 chunks）
   │           └─ LLM Function Calling 循环（≤5 轮）:
   │                ├─ search_knowledge_base / get_user_profile / update_user_profile
   │                ├─ log_mood_event / get_recent_mood_trend
   │                └─ recommend_content（受 gate 控制：非 hard 则跳过）
   │
   ├─▶ conversation_manager.add_interaction(...) → 持久化 4 类：
   │     ├─ conversation_history + session 元数据
   │     ├─ mood_events
   │     ├─ user_profile（长期信号累积）
   │     └─ fire-and-forget: memory_writer → MemoryPolicy 门控 → memory_store.batch_write
   │
   ├─▶ rejection_detector.detect_rejection(text, ids) → has_rejected_recommendation
   ├─▶ urgent_logger.log_interaction_async（非低风险时）
   │
   ├─▶ _extract_recommendations(messages) → final_recommendations + rationale
   │     └─ gate=hard 且 LLM 未推荐 → ContentRecommendTool.recommend 兜底
   │
   ├─▶ output_safety_checker.review(response_text, risk_state, decision, recs, rationale)
   │     └─ 可覆盖: response_text / recommendations / rationale / decision
   │
   ├─▶ conversation_manager.mark_recommendation(...) → recommendation_events
   ├─▶ write_trace(TraceEvent) → logs/recommendation_trace.jsonl
   └─▶ RiskMemoryWriter.update(...) → risk_events / risk_baselines / risk_triggers
```

---

## 2. 持久化表 → 写入方 → 读取方

| SQLite 表 | 写入方 | 读取方 |
|---|---|---|
| `sessions` | conversation_manager | conversation_manager / API |
| `conversation_history` | conversation_manager.add_interaction | 摘要构建 / history 窗口 |
| `user_profile` | conversation_manager / UserProfileTool / RiskMemoryWriter(兼容) | orchestrator / recommend_gate |
| `mood_events` | MoodTrackingTool / add_interaction | 趋势 / 历史风险计数 / 记忆检索 |
| `recommendation_events` | mark_recommendation | 统计 / 冷却 |
| `recommendation_feedback` | record_recommendation_feedback | Gate 偏好分 |
| `memory_items` | memory_store（经 memory_writer） | memory_retriever |
| `risk_events` / `risk_baselines` / `risk_triggers` / `protective_factors` | RiskMemoryWriter | RiskMemoryReader / memory_context_builder |
| `memory_embeddings` / `memory_outbox` | **无写入方**（死表） | - |

---

## 3. 数据依赖与循环

- **循环 A（跨会话风险闭环）**：每轮开头 `RiskMemoryReader.get_baseline` → 注入 `risk_evaluator`（Rule5 Safety Gate 基线修正）→ 每轮末尾 `RiskMemoryWriter.update` → 下轮读取。闭环已形成。
- **循环 B（推荐反馈）**：`/content/feedback` → user_profile.recommendation_feedback → Gate 偏好分与 prompt → 下次推荐。半闭环（排序权重未消费）。
- **循环 C（记忆）**：add_interaction 写入 → 下轮 MemoryContextBuilder 检索 → prompt 注入。闭环已形成（但无去重/冲突治理）。

---

## 4. LLM 调用点清单

| 调用点 | 模块/函数 | 用途 | 失败兜底 |
|---|---|---|---|
| emotion 深层分析 | emotion_analyzer._call_llm | 上下文情绪 / 多轮情绪 | 返回 中性/中性/0.5 |
| 危机回复生成 | urgent_detector.generate_*_response_async | Level 2/3 支持回复 | 默认回复文本 |
| ReAct Agent Loop | agent_orchestrator.run_agent | 主对话（Function Calling） | "技术问题，请稍后再试" |
| 内容 AI Rerank | content_recommender._ai_based_recommendation | 推荐排序 | 回退规则排序 |
| 知识结构化提取 | knowledge_ingest（离线 CLI） | RAG 语料构建 | 3 次重试 |

---

## 5. 规则层清单（确定性逻辑）

| 规则 | 位置 | 作用 |
|---|---|---|
| 20 条高危正则 | bert_risk_predictor | rule_override → level_3（最高优先级） |
| 讨论/第三方/安全否认上下文 | risk_evaluator._analyze_context | 修正语境（讨论→L0、第三方非求助→L0、安全否认→降级） |
| 会话级风险状态机 5 条 | session_risk_aggregator | 当前轮优先 / 惯性 / 连续升级 / 安全确认降级 / 基线地板 |
| 紧急/警告关键词 | urgent_detector（仅端点） | 关键词紧急度 |
| 推荐门控加权 + 冷却 + 安全锁 | recommend_gate | hard/soft/none/safety_only |
| 风险内容过滤 | content_recommender._is_item_allowed_for_risk | L2 仅 soft/beginner、L3 禁推 |
| 记忆写入门控 | memory_policy | 按类型置信度/强度阈值 |
| 输出安全检查 | output_safety_checker | 无效化/危险细节/过度保证等模式 |
| 推荐拒绝检测 | rejection_detector | 13 组正则 |
| 意图关键词（4 类） | emotion_analyzer._detect_user_intent | sharing/seeking_help/planning/seeking_relief |

---

## 6. 从一轮输入到最终回复的可追踪性结论

**可以追踪**：单轮内 `input → emotion → risk → gate → route → response → persist` 全程有明确字段流动，`AgentRunResponse.steps` 与 `recommendation_trace.jsonl` 可回放。

**不可追踪**：
- 没有统一的 `trace_id` 贯穿感知层中间结果（emotion/risk/intent 各阶段结果未全量落 trace）
- 记忆检索的命中/丢弃明细只在 prompt 文本中出现，无结构化记录
- 无 AgentAction 语义标注（steps 只有流程名，无"状态→动作"决策记录）

> 这正是 Phase 7（Observability）需要补的：统一 Trace Schema（§12.1）把上述中间决策全部结构化落盘。
