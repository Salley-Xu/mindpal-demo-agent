# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概览

**MindPal Pro** 是一个基于 LLM 函数调用的上下文感知心理健康支持系统，采用 ReAct（推理+行动）Agent 架构。系统提供共情对话、情绪分析、紧急情况检测和个性化内容推荐。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动后端服务
cd backend
python main.py
# API地址: http://localhost:8000
# 接口文档: http://localhost:8000/docs

# 启动前端
cd frontend
streamlit run frontend.py
# 界面: http://localhost:8501
```

## 环境配置

在 `backend/.env` 文件中配置：
- `DEEPSEEK_API_KEY` - LLM 调用必需
- `API_BASE_URL` - 默认: `https://api.deepseek.com/v1`
- `CHAT_MODEL` - 默认: `deepseek-chat`

## 系统架构

### Agent 编排模式

核心是 `AgentOrchestrator`（`backend/agent_orchestrator.py`），实现 **ReAct 循环**：

1. **构建消息**：组装系统提示词（包含对话阶段、策略指导、关键关切点、风险等级）
2. **LLM 调用**：发送到 DeepSeek，携带工具定义（`agent_tools.py` 中的 `TOOL_DEFINITIONS`）
3. **工具执行**：若 LLM 返回 `tool_calls`，通过 `asyncio.gather()` 并行执行工具
4. **观察**：将工具结果以 `role: "tool"` 形式追加到消息列表
5. **迭代**：重复直到 LLM 不再调用工具或达到最大轮数（5轮）

### 对话阶段

系统在 `conversation_manager.py` 中跟踪 4 个阶段：
- **initial**（≤2轮）：建立信任，开放式探索
- **exploring**（≤5轮）：帮助识别模式
- **deepening**（≤10轮）：深度分析，连接想法
- **resolving**（>10轮）：具体解决方案，行动计划

各阶段特定提示词在 `agent_prompts.py` 的 `STRATEGY_PROMPTS` 中定义。

### 分层记忆架构

| 层级 | 存储位置 | 生命周期 | 用途 |
|------|----------|----------|------|
| 短期记忆 | `ConversationManager.sessions`（内存） | 会话期间 | 最近20轮对话，LRU淘汰 |
| 工作记忆 | 动态 Prompt | 单次推理 | 最近10轮完整对话 + 摘要 |
| 长期记忆 | SQLite (`data/sessions.db`) | 持久化 | 用户画像、情绪时间线、会话历史 |

**上下文压缩**：当对话超过 5 轮时，`_compress_conversation_context()` 创建语义摘要（早期对话压缩，最近3轮保持完整）。

### 数据库结构（SQLite）

- `sessions`: user_id, session_id, conversation_stage, key_concerns, timestamps
- `conversation_history`: turn_number, user_input, detected_emotion, ai_response
- `emotion_timeline`: 每会话的情绪记录
- `user_profile`: user_id (主键), risk_level, preferences (JSON)
- `mood_events`: 跨会话的情绪追踪

同时存在同步（`db_manager`）和异步（`adb_manager`）管理器，Agent 操作使用异步版本。

### 可用工具（函数调用）

定义在 `agent_tools.py`：

1. **search_knowledge_base**: 查询 `content_db.json` 获取心理学内容
2. **get_user_profile** / **update_user_profile**: 读写用户偏好和风险等级
3. **log_mood_event** / **get_recent_mood_trend**: 记录和分析情绪模式
4. **recommend_content**: 内容推荐引擎

工具自动注入：`user_id`、`session_id`、`conversation_summary`、`current_emotion` 由 `_execute_single_tool()` 自动添加。

## 关键 API 接口

- `POST /chat/intelligent` - 主对话入口（封装 `agent_orchestrator.run_agent`）
- `POST /agent/run` - 返回完整响应及可选的工具执行步骤
- `POST /emotion/analyze` - 仅情绪分析
- `POST /content/recommend` - 内容推荐
- `GET /session/{user_id}/{session_id}/summary` - 会话摘要
- `GET /health` - 健康检查及统计

## 风险与紧急情况处理

`urgent_detector.py` 在 LLM 调用前进行关键词+情绪强度分析：
- `urgent`：紧急危机（自杀/自伤）
- `warning_high`：高风险语言
- `warning_low`：值得关注的模式
- `normal`：无风险

紧急情况异步记录到 `backend/logs/urgent_cases_YYYYMMDD.json`（fire-and-forget）。

## 前端说明

Streamlit 应用（`frontend/frontend.py`）调用后端 API：
- 自动生成 `user_id` 和 `session_id`
- 响应缓存（1小时TTL，最多100条）
- 侧边栏显示对话阶段、情绪趋势、关键关切点
- 推荐内容在 AI 回复后以卡片形式展示

## 定时任务

后端启动时自动配置 APScheduler，每天凌晨3点清理30天前的过期会话（`SESSION_CLEANUP_DAYS` 可配置）。
