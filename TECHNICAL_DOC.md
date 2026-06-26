# MindPal Pro 技术文档

## 1. 项目概况 (Project Overview)

**MindPal Pro** 是一个基于大语言模型（LLM）的上下文感知心理健康支持与个性化推荐系统。它旨在通过智能对话为用户提供情感支持，并根据用户的情绪状态和对话内容推荐相关的心理健康资源。

### 核心功能
*   **上下文感知对话**: 基于 `AgentOrchestrator` 和 DeepSeek 模型，能够记住用户历史对话，提供连贯、共情的互动。
*   **情绪分析**: 实时分析用户输入的文本，识别情绪状态（如焦虑、压力、平静等）及变化趋势。
*   **紧急情况检测**: 自动识别潜在的危机信号（如自伤倾向），并触发紧急干预流程。
*   **个性化内容推荐**: 根据对话阶段和用户情绪，推荐文章、音频、视频或心理练习工具。
*   **会话持久化**: 使用 SQLite 数据库存储会话历史，支持跨会话记忆。

---

## 2. 技术栈 (Technology Stack)

### 后端 (Backend)
*   **语言**: Python 3.13
*   **框架**: FastAPI (高性能 Web 框架)
*   **服务器**: Uvicorn (ASGI 服务器)
*   **AI 引擎**: OpenAI SDK (调用 DeepSeek 模型), 支持 Function Calling
*   **数据库**: SQLite (通过 `aiosqlite` 异步访问)
*   **任务调度**: APScheduler (用于定期清理过期会话)

### 前端 (Frontend)
*   **框架**: Streamlit (快速构建数据应用)
*   **通信**: HTTP Requests (与后端 API 交互)

---

## 3. 系统架构与核心模块 (System Architecture)

项目采用前后端分离架构，通过 RESTful API 进行通信。

### 目录结构
```
mindpal_demo/agent_version/
├── backend/                # 后端核心代码
│   ├── main.py             # FastAPI 应用入口
│   ├── api_endpoints.py    # API 路由定义
│   ├── agent_orchestrator.py # Agent 编排器 (核心)
│   ├── agent_prompts.py    # Agent 提示词模板 (Prompt Engineering)
│   ├── agent_tools.py      # Agent 可用工具集
│   ├── emotion_analyzer.py # 情绪分析模块
│   ├── urgent_detector.py  # 紧急情况检测模块
│   ├── content_recommender.py # 内容推荐引擎
│   ├── conversation_manager.py # 会话状态管理
│   ├── database.py         # 数据库操作封装
│   └── models.py           # Pydantic 数据模型
├── frontend/               # 前端代码
│   └── frontend.py         # Streamlit 界面逻辑
├── data/                   # 数据存储
│   ├── sessions.db         # SQLite 数据库文件
│   └── content_db.json     # 推荐内容库
└── requirements.txt        # 项目依赖
```

### 核心模块解析

1.  **Agent Orchestrator (`agent_orchestrator.py`)**:
    *   系统的"大脑"。采用 ReAct (Reasoning + Acting) 模式。
    *   根据当前对话阶段（Initial, Exploring, Deepening, Resolving）动态选择 Prompt 策略。
    *   利用 LLM Function Calling 机制自主决定是否调用外部工具（如查询知识库、记录心情、推荐内容）。

2.  **Emotion Analyzer (`emotion_analyzer.py`)**:
    *   结合当前输入和历史上下文，分析用户情绪。
    *   输出：当前情绪、上下文情绪背景、置信度。

3.  **Urgent Detector (`urgent_detector.py`)**:
    *   基于关键词匹配和情绪强度，检测高风险内容。
    *   分级预警：Urgent (紧急), Warning High (高风险), Warning Low (低风险)。

4.  **Content Recommender (`content_recommender.py`)**:
    *   基于规则和语义匹配的推荐引擎。
    *   根据用户情绪标签和关注点，从 `content_db.json` 中检索最匹配的资源。

---

## 4. 记忆与上下文管理系统 (Memory & Context Management)

本项目采用分层记忆架构，模拟人类的短期记忆（工作记忆）和长期记忆机制，以解决大模型上下文窗口有限和跨会话连贯性问题。

### 4.1 记忆分层架构 (Layered Memory Architecture)

| 记忆类型 | 存储位置 | 生命周期 | 关键用途 |
| :--- | :--- | :--- | :--- |
| **短期记忆** (Short-term) | 内存 (`ConversationManager.sessions`) | 会话期间有效 (LRU 策略) | 维持当前多轮对话的连贯性，提供毫秒级读写。 |
| **工作记忆** (Working) | 动态 Prompt | 单次推理 | 提供给 LLM 的当前上下文窗口，包含最近对话和历史摘要。 |
| **长期记忆** (Long-term) | SQLite 数据库 (`sessions.db`) | 永久存储 | 跨会话的用户画像 (Profile)、历史会话恢复、情绪演变记录。 |

### 4.2 上下文压缩策略 (Context Compression Strategy)
为了在有限的 Token 限制内保持长对话能力，系统实现了 **"滑动窗口 + 语义压缩"** 混合策略（详见 `conversation_manager.py`）：

1.  **最近 N 轮 (Full Fidelity)**:
    -   保留最近 **3轮** 完整对话（用户原文 + AI 回复）。
    -   确保 AI 对用户最新的指令能做出精确响应。
2.  **历史摘要 (Semantic Compression)**:
    -   当对话超过 5 轮时触发 `_compress_conversation_context`。
    -   对早期对话进行有损压缩（目前采用提取关键句策略）。
    -   **压缩示例**: `"历史对话摘要: 用户提到: 考试压力大... 用户提到: 昨晚失眠... 最近对话: [完整内容]..."`
3.  **滑动窗口 (Sliding Window)**:
    -   物理内存中仅保留最近 **20轮** (`max_history`)，旧数据自动移出内存（但已异步持久化到数据库）。

### 4.3 长期记忆实现细节
1.  **用户画像 (User Profile)**:
    -   通过 `UserProfileTool` 管理，存储在 SQLite 中。
    -   **结构化字段**: 风险等级 (`risk_level`)、内容偏好 (`preferred_types`: 文章/音频)、难度偏好 (`preferred_difficulty`)。
    -   **动态更新**: Agent 会在对话中自动捕捉并更新这些信息（例如用户说“我不喜欢太难的文章” -> 更新偏好为 "beginner"）。
2.  **情绪时间线 (Emotion Timeline)**:
    -   记录每次交互的情绪标签、时间戳和触发文本。
    -   用于计算情绪趋势（Escalating/Improving/Stable），辅助风险控制。

### 4.4 数据一致性与持久化
-   **读写策略**: 优先读取内存缓存 (Hot Data)，未命中则懒加载数据库 (Cold Data)。
-   **异步写入**: `add_interaction` 中使用 `asyncio.create_task` 异步写入 SQLite，确保 API 响应速度不受 I/O 影响。
-   **自动清理**: 后台调度器 (`APScheduler`) 每天凌晨自动清理 30 天前的过期匿名会话，防止数据库无限膨胀。

---

## 5. 工作流程 (Workflow)

以下是用户发起一次对话的完整数据流转过程：

1.  **用户输入**: 用户在 Streamlit 前端输入文本（例如："我最近工作压力很大"）。
2.  **API 请求**: 前端发送 POST 请求到后端 `/chat/intelligent` 接口。
3.  **后端处理**:
    *   **校验**: `validate_user_input` 检查输入合法性。
    *   **上下文获取**: `conversation_manager` 加载该用户的历史会话摘要。
    *   **Agent 执行 (`agent_orchestrator.run_agent`)**:
        *   **Step 1**: 构建包含 System Prompt（含当前策略）、历史摘要和用户输入的 Prompt。
        *   **Step 2**: 调用 DeepSeek 模型。
        *   **Step 3 (思考)**: 模型判断是否需要调用工具（例如 `recommend_content`）。
        *   **Step 4 (行动)**: 如果需要，后端执行对应 Python 函数，获取结果。
        *   **Step 5 (响应)**: 模型根据工具结果和上下文，生成最终回复文本。
    *   **并行处理**: 同时触发 `emotion_analyzer` 和 `urgent_detector` 更新状态（异步或同步依赖具体实现）。
4.  **结果返回**: 后端返回 JSON 响应，包含 AI 回复、推荐内容列表、情绪分析结果。
5.  **前端展示**: Streamlit 渲染 AI 回复气泡，并以卡片形式展示推荐内容。

## 6. Agent 核心机制详解 (Deep Dive: Agent Architecture)

**AgentOrchestrator** 是系统最复杂的组件，它负责将大模型 (LLM) 转化为具备行动能力的智能体。以下是其核心设计：

### 5.1 ReAct 循环 (ReAct Loop)
项目实现了标准的 ReAct (Reasoning + Acting) 模式，通过 `while` 循环支持多轮工具调用：
1.  **思考 (Thought)**: 将 System Prompt、历史记录、用户输入发送给 LLM。
2.  **决策 (Decision)**: LLM 决定是否需要调用工具（通过 `tool_calls` 字段返回）。
3.  **行动 (Action)**: `_execute_single_tool` 解析参数并执行本地 Python 函数。
4.  **观察 (Observation)**: 将工具执行结果以 `role: tool` 的形式追加到消息列表。
5.  **迭代 (Iteration)**: 带着新的上下文再次请求 LLM，直到 LLM 决定不再调用工具或达到最大轮数 (Max Turns = 5)。

### 5.2 动态 Prompt 策略与工程化 (Prompt Engineering & Strategy)
系统将所有提示词逻辑解耦到了 `agent_prompts.py`，实现了更清晰的 Prompt 工程管理：

1.  **策略分离**: `STRATEGY_PROMPTS` 字典定义了不同对话阶段（Initial, Exploring, Deepening, Resolving）的特定指导原则，这使得调整 AI 人设和策略无需修改核心代码。
2.  **模板化**: 使用 `SYSTEM_PROMPT_TEMPLATE` 进行结构化组装。Agent 在运行时动态注入：
    *   `stage`: 当前对话阶段。
    *   `key_concerns`: 用户关注的核心问题（自动格式化处理）。
    *   `risk_level`: 紧急情况等级。
    *   `strategy_guidance`: 对应阶段的策略文本。

这种设计（Logic/Prompt Separation）提高了系统的可维护性，便于非开发人员（如 Prompt 工程师）独立调优 AI 表现。

### 5.3 Function Calling 实现
项目使用了 OpenAI SDK 的原生 `tools` 参数定义工具集 (`TOOL_DEFINITIONS`)，包括：
*   `search_knowledge_base`: RAG (检索增强生成)，查询专业心理学知识。
*   `recommend_content`: 个性化推荐引擎接口。
*   `log_mood_event` / `get_recent_mood_trend`: 情绪追踪与记忆。
*   `get_user_profile` / `update_user_profile`: 长期用户画像管理。

### 5.4 上下文注入 (Context Injection)
在调用工具时，Agent 会自动注入隐式上下文（用户不需要在 Prompt 里说 "我是用户123"）：
*   **显式参数**: LLM 生成的参数（如搜索关键词）。
*   **隐式参数**: 代码自动注入 `user_id`, `session_id`, `conversation_summary`, `current_emotion`，确保工具能获取完整的环境信息。

### 5.5 LLM 交互详情 (Prompt Structure)
每次与 DeepSeek API 交互时，系统会构建一个标准的消息列表 (`messages`)，其结构如下：

1.  **System Message (系统指令)**:
    -   **身份定义**: "你是一个富有同理心的心理健康助手..."
    -   **动态状态**: 注入当前 `conversation_stage` (如 Exploring) 和 `risk_level`。
    -   **策略指导**: 从 `STRATEGY_PROMPTS` 中选取的特定阶段话术指导。
    -   **关键信息**: 用户的 `key_concerns` (如 "学业压力")。

2.  **Conversation History (对话历史)**:
    -   最近 10 轮的完整对话记录 (User Input + AI Response)。
    -   *注：更早的历史通过 "Summary" 形式被压缩在 System Message 或上下文摘要中（如有）。*

3.  **User Message (当前输入)**:
    -   用户的最新文本输入。

4.  **Tool Interactions (工具交互 - 可选)**:
    -   如果触发工具调用，后续会追加 `assistant` (tool_calls) 和 `tool` (execution result) 类型的消息，形成完整的 ReAct 思考链。

---

## 7. 关键 API 接口 (Key API Endpoints)

| 方法 | 路径 | 描述 |
| :--- | :--- | :--- |
| `POST` | `/chat/intelligent` | **核心接口**。接收用户文本，返回智能对话回复。 |
| `POST` | `/emotion/analyze` | 仅进行情绪分析，返回情绪标签和趋势。 |
| `POST` | `/content/recommend` | 根据输入参数获取推荐内容。 |
| `GET` | `/session/{user_id}/list` | 获取用户的历史会话列表。 |
| `GET` | `/health` | 健康检查，返回系统状态和统计信息。 |

## 8. 面试重点 (Interview Highlights)

在面试中，你可以重点提及以下技术亮点：

1.  **Agent 架构设计**: 如何通过 Function Calling 让 LLM 具备"行动"能力，而不仅仅是聊天。
2.  **上下文管理**: 如何在多轮对话中通过"摘要"机制（Conversation Summary）解决 LLM 上下文窗口限制问题，同时保持长期记忆。
3.  **分层处理机制**: 紧急情况检测（规则引擎）优先于生成式模型，确保安全性。
4.  **异步编程**: 使用 FastAPI 和 `async/await` 处理高并发 I/O 操作（数据库读写、LLM API 调用）。

---

## 9. 快速启动 (Quick Start)

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **配置环境**:
    在 `backend` 目录下创建 `.env` 文件，填入 `DEEPSEEK_API_KEY` 等配置。

3.  **启动后端**:
    ```bash
    cd backend
    python main.py
    ```

4.  **启动前端**:
    ```bash
    cd frontend
    streamlit run frontend.py
    ```
