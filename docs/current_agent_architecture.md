# 当前 Agent 系统架构审计报告（Baseline）

> 对应开发计划 Task 0.1（§5.2）
> 日期：2026-08-18
> 审计方式：代码静态审计（核心链路 + 感知层 / 记忆层 / 推荐检索层 交叉核对）

---

## 1. 总体结论

当前系统是**功能完备的"多模块功能集合"**，但**缺少统一的 Agent State 与 Agent Policy 层**：

- 单轮决策链清晰：`情绪 → 风险 → 推荐门控 → 路由 → 对话/干预`
- 记忆系统已有完整的 写入/检索/注入 管线（纯规则，无 LLM）
- 推荐链路完整（Gate → 混合检索 → 规则排序 → LLM Rerank → 安全审查）
- 但 **intent 仅是一层嵌入在情绪分析中的浅层关键词规则**，**不存在独立 intent 分类器**
- **不存在 AgentAction / memory_needed / AgentState 等统一状态字段**
- 会话状态分散在 `conversation_manager` 的裸 dict 与 `agent_orchestrator` 的十余个局部变量中

---

## 2. 模块清单与决策链

### 2.1 请求生命周期（run_agent，backend/agent_orchestrator.py）

```
用户输入
  ↓ [1] conversation_manager.get_or_create_session_async    加载会话（内存/SQLite）
  ↓ [2] emotion_analyzer.analyze_with_context_async         情绪分析（BERT + LLM 双通道）
  ↓ [3] UserProfileTool.get_profile                         用户长期画像（SQLite user_profile）
  ↓ [4] MoodTrackingTool.get_recent_trend                   近期情绪事件（统计历史高风险次数）
  ↓ [5] emotion_analyzer.build_emotion_state_payload        结构化情绪状态（含 user_intent）
  ↓ [6] RiskMemoryReader.get_baseline                       长期风险基线（SQLite risk_baselines）
  ↓ [7] risk_evaluator.evaluate                             风险评估（BERT v4.2 + 会话状态机）
  ↓ [8] recommend_gate.decide                               推荐门控（加权多因子）
  ↓ [9] 路由决策
  │       level_3        → urgent_detector.generate_crisis_response_async（紧急干预）
  │       第三方危机求助  → generate_third_party_support_response_async（第三方支持）
  │       level_2        → generate_crisis_response_async（高风险支持模式）
  │       正常            → ReAct Agent Loop（≤5 轮，LLM Function Calling + 6 工具）
  ↓ [10] conversation_manager.add_interaction               持久化（含 fire-and-forget 记忆写入）
  ↓ [11] rejection_detector.detect_rejection                文本级推荐拒绝检测
  ↓ [12] output_safety_checker.review                       出站安全检查（规则可覆盖回复）
  ↓ [13] RiskMemoryWriter.update                            长期风险记忆落库
  ↓ [14] write_trace                                        推荐全链路追踪（JSONL）
```

### 2.2 各模块 I/O 与特性

#### 感知层

| 模块 | 输入 | 输出 | LLM | 规则 | 持久状态 |
|---|---|---|---|---|---|
| `emotion_analyzer` | text, summary | current/context_emotion, confidence | ✅ DeepSeek | 强度/压力源/意图关键词 | 内存 LRU 缓存(1000, 1h) |
| `bert_emotion_predictor` | text | (label, confidence) 5 类 | ❌ | 标签合并映射 | 懒加载单例 |
| `risk_evaluator` | text, emotion_state, summary, risk_baseline | level + risk_state(dict) | ❌ | 讨论/第三方/安全否认上下文规则 | 懒加载单例 |
| `bert_risk_predictor` | text | level + 4 分类 + 规则兜底 | ❌ | 20 条高危正则(rule_override) | 懒加载单例 |
| `session_risk_aggregator` | 单轮 level + summary 窗口 | session_level, risk_trend | ❌ | 5 条规则状态机 | 无（窗口来自 summary） |
| `urgent_detector` | text, emotion | level + 危机回复 | ✅(仅回复生成) | 16 紧急/15 警告关键词 | JSON 日志 |
| `rejection_detector` | text, active_ids | has_rejected + confidence | ❌ | 13 组正则 | 无 |

**关键发现**：
- `urgent_detector.detect()`（关键词紧急度）**不在 agent 主链路**，主链路紧急度完全来自 `risk_evaluator`（BERT+状态机）。两套紧急度体系并存，存在漂移风险。
- `bert_risk_predictor` 的 `binary_threshold`（二分类阈值融合）**参数存储但未实现**。
- intent 信号 = `emotion_analyzer._detect_user_intent()` 关键词规则，仅产出 `sharing / seeking_help / planning / seeking_relief` 四类，其余一律归为 `sharing`，无置信度校准。

#### 记忆层

| 模块 | 输入 | 输出 | LLM | 规则 | 持久状态 |
|---|---|---|---|---|---|
| `memory_store` | MemoryItem/MemoryQuery | id/SearchResult | ❌ | SQL 过滤 + importance 排序 | SQLite `memory_items` |
| `memory_writer` | TurnContext | memory_id 列表 | ❌ | 3 类规则化提取 + MemoryPolicy 门控 | 经 store 写库 |
| `memory_retriever` | user_id, MemoryQuery | MemorySearchResult | ❌ | SQL→BM25→规则分→RRF→多因子重排 | 只读 |
| `memory_injector` | results, risk_context, budget | InjectedMemoryContext | ❌ | 优先级注入 + token 裁剪 | 无 |
| `memory_policy` | MemoryCandidate, TurnContext | should_write/importance/ttl | ❌ | 按 memory_type 门控 | 无 |
| `memory_context_builder` | user_id, text, states | InjectedMemoryContext | ❌ | 组查询 + token 预算(0.15×2400) | 无 |
| `conversation_manager` | 交互数据 | session dict / summary | ❌ | 压缩阈值 + 阶段推断 + 关键词关切 | 内存 + SQLite |

**记忆类型枚举**：`support_preference / avoid_preference / stress_source / mood_event / coping_strategy / recommendation_feedback / conversation_summary / personal_fact / risk_shadow`
**实际 extractor 只产出 4 类**：stress_source / support_preference / avoid_preference / mood_event

**关键缺陷**：
- `memory_items` **无去重/无冲突消解**，同压力源每轮重复 INSERT
- `MemoryPolicy._appears_at_least`（声称 6 轮窗口频次门控）是**桩实现**
- `recommendation_feedback / coping_strategy` 类型有策略但无 extractor（死代码路径）
- `memory_embeddings / memory_outbox` 表已建但无写入方；语义检索未实现
- 同一 `memory_items` DDL 在 memory_store / database(同步+异步) 三处重复

#### 推荐 / 检索层

| 模块 | 输入 | 输出 | LLM | 规则 | 持久状态 |
|---|---|---|---|---|---|
| `recommend_gate` | emotion/risk/summary/profile | {should_recommend, recommend_type, score, ...} | ❌ | 加权(0.30/0.23/0.23/0.14/0.10) + 冷却 + 3 级安全锁 | 无 |
| `content_recommender` | user_input, emotion, summary, profile | (items, rationale, scores) | ✅ rerank | 5 查询变体 + 规则排序 + 风险过滤 | 无 |
| `hybrid_retriever` | query, items | [(item, score)] | ❌ | RRF(k=60) 融合 BM25+vector+dense | 无 |
| `bm25_retriever` | query, items | [(item, score)] | ❌ | BM25 k1=1.5 b=0.75，中文 2-gram | 无 |
| `vector_retriever` | query, items | [(item, score)] | ❌ | 同义词扩展 + 稀疏余弦 | 无 |
| `dense_retriever` | query, items | [(item, score)] | ❌ | bge-small-zh 均值池化 | 模型懒加载 |
| `knowledge_store` | KnowledgeQuery | chunks | ❌ | RRF(BM25×3 + FAISS-bge) | 内存 + FAISS 索引 |
| `knowledge_ingest` | PDF/TXT | corpus JSON | ✅(提取) | 分块 + 去重 | JSON 落盘 |
| `content_db` | - | ContentItem | ❌ | 关键词打分兜底 | JSON 文件 |

**推荐链路**：
```
Gate 判定 → 候选检索(5 变体 × RRF 混合) → 规则排序(final_score 加权)
  → LLM Rerank(top10 选择) → 生成理由/分数 → 安全审查 → mark_recommendation → trace
```

**RecommendGate 因子**（可 .env 覆盖）：

| 因子 | 权重 | 备注 |
|---|---|---|
| emotion_intensity | 0.30 | 原值 |
| risk_score | 0.23 | level_0→0 / 1→0.4 / 2→0.75 / 3→1.0 |
| intent_score | 0.23 | sharing→0 / seeking_relief→0.45 / planning→0.55 / seeking_help→0.75 |
| trend_score | 0.14 | negative_trend→0.35 |
| preference_score | 0.10 | 偏好 +0.45 / 压力源命中 +0.35 / 拒绝 -0.25 |

阈值：hard=0.58、seeking_help 专用 hard=0.38、soft=0.20；LEVEL_2→safety_only、LEVEL_3→none（强制）。

**反馈闭环（部分）**：
- ✅ 记录：`/content/feedback` API 写 user_profile.recommendation_feedback；文本级拒绝经 detect_rejection
- ✅ 参与：Gate 偏好分扣减（-0.25）、prompt 上下文注入
- ❌ 未参与：feedback 具体值不直接调整内容排序权重（排序侧仅 recent ids 去重 -0.03）

---

## 3. 信号 → 模块 → 字段映射

| 信号 | 产出模块 | 字段 | 说明 |
|---|---|---|---|
| 情绪 | emotion_analyzer | current_emotion / emotion_type / emotion_intensity / emotion_trend | BERT 表层 + LLM 深层 |
| **意图** | emotion_analyzer 内嵌 | user_intent (sharing/seeking_help/planning/seeking_relief) | **浅层关键词规则，非独立分类器** |
| 风险等级 | risk_evaluator(+aggregator) | level / session_level / risk_score | 主链路唯一风险来源 |
| 风险趋势 | session_risk_aggregator | risk_trend (rising/stable/falling) | 窗口均值比较 |
| 风险基线 | RiskMemory | baseline (low/medium/high) | SQLite 跨会话 |
| 紧急度 | risk_evaluator（主）/ urgent_detector（仅端点） | level / triggers / resources | **双体系并存** |
| 拒绝检测 | rejection_detector | has_rejected / rejected_ids | 影响 Gate 偏好分 |
| 推荐决策 | recommend_gate | should_recommend / recommend_type / score | hard/soft/none/safety_only |

---

## 4. 与目标架构（开发计划 §2）的差距

| 目标能力 | 当前状态 | 差距 |
|---|---|---|
| Perception Layer：Intent | ❌ 无独立 intent 分类器 | 仅有 4 类关键词规则，无置信度/open-set |
| Conversation State | ❌ 无 AgentState | 分散裸 dict，无瞬时/持久信号区分 |
| Agent Policy / Router | ❌ 无 Action Space | 路由是 if-else（紧急→干预，否则 ReAct），无状态→动作映射 |
| Memory 生命周期 | ⚠️ 部分 | 有 Write Gate/注入，缺冲突消解/去重/过期治理 |
| Risk 多轮状态 | ⚠️ 部分 | 有 SessionRiskAggregator 状态机 + 趋势，缺 Early Warning |
| Recommendation 反馈闭环 | ⚠️ 部分 | 反馈影响 Gate 与 prompt，不直接影响排序权重 |
| Observability | ⚠️ 部分 | 有推荐 trace，缺统一 Agent trace / replay / error taxonomy |
| Benchmark | ⚠️ 部分 | 有情绪/风险/门控专项评测，缺 Agent 级决策 Benchmark |

---

## 5. 最值得优先改造的 3 处（Phase 优先级依据）

1. **Intent 独立化（Phase 1）**：把 user_intent 从情绪分析中拆出为独立 Intent Recognition 层，建立置信度与 fallback，为 Policy 提供信号。
2. **AgentState（Phase 2）**：将 orchestrator 的十余个局部变量收敛为统一 AgentState，统一瞬时/持久信号。
3. **Agent Policy（Phase 3）**：在现有 if-else 路由基础上形式化 Action Space 与 Hybrid Policy（安全规则 → 确定性规则 → classifier → LLM fallback）。
