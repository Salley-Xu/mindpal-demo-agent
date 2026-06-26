# MindPal Pro 面试演示文稿大纲 & 问答准备

这份文档旨在帮助你进行 **MindPal Pro** 项目的面试演示。它包含两部分：
1.  **PPT 演示大纲**: 可直接用于制作幻灯片或作为口述脚本。
2.  **面试核心 Q&A**: 针对面试官可能提出的技术深挖问题的标准回答。

---

## 第一部分：PPT 演示大纲 (Presentation Outline)

### Slide 1: 封面 (Cover)
*   **标题**: MindPal Pro - 基于大模型的上下文感知心理健康智能体
*   **副标题**: 结合 CBT（认知行为疗法）理念与 AI Agent 技术的个性化支持系统
*   **演讲者**: [你的名字]

### Slide 2: 项目背景与痛点 (Background & Pain Points)
*   **背景**: 心理健康需求日益增长，但专业资源稀缺、费用高昂。
*   **现有方案痛点**:
    *   传统聊天机器人：基于规则，缺乏共情，无法理解复杂语境。
    *   普通 LLM：没有长期记忆，聊几句就忘了前文；缺乏领域专业性。
*   **目标**: 构建一个**有记忆、懂情绪、能行动**的 AI 心理助手。

### Slide 3: 核心解决方案 (Core Solution)
*   **AI Agent (智能体)**: 不仅仅是聊天，采用 **ReAct** 模式，能主动思考并调用工具（推荐内容、查询知识库）。
*   **长期记忆 (Long-term Memory)**: 记住用户的画像、偏好和过往经历，提供连续的关怀。
*   **情绪风控 (Safety First)**: 实时监控情绪风险，遇到危机自动触发干预机制。

### Slide 4: 系统架构 (System Architecture)
*   **前端**: Streamlit (快速交互界面)
*   **后端**: FastAPI (高性能异步 API)
*   **核心引擎**: DeepSeek LLM (大语言模型) + Agent Orchestrator (编排器)
*   **数据层**: SQLite (异步存储会话与画像) + Vector/JSON (推荐内容库)
*   *(口述提示: 强调前后端分离，异步并发处理能力)*

### Slide 5: 核心技术 1 - Agent 编排 (ReAct Engine)
*   **大脑**: `AgentOrchestrator`
*   **机制**: **Reasoning + Acting** 循环
    1.  **观察**: 接收用户输入。
    2.  **思考**: 结合当前 Prompt 策略，决定是否需要工具支持。
    3.  **行动**: 自动调用 `search_knowledge_base` 或 `recommend_content`。
    4.  **响应**: 综合工具结果生成最终回复。
*   **亮点**: 动态 Prompt 策略（根据 Initial/Exploring/Deepening 不同阶段切换话术）。

### Slide 6: 核心技术 2 - 动态上下文管理 (Context Management)
*   **挑战**: LLM Context Window 有限，且 Token 费用昂贵。
*   **策略**: **分层记忆系统**
    *   **短期记忆**: 内存中保留最近 3-5 轮完整对话 (Full Fidelity)。
    *   **历史摘要**: 对早期对话进行语义压缩 (Semantic Compression)，提取关键信息。
    *   **滑动窗口**: 自动移出过旧的对话，保持上下文精简且高效。

### Slide 7: 核心技术 3 - 情绪感知与安全 (Emotion & Safety)
*   **双轨分析**:
    *   **Emotion Analyzer**: 分析表层情绪（如"愤怒"）和深层情绪（如"被忽视感"）。
    *   **Urgent Detector**: 独立的规则引擎，实时扫描高危关键词（自伤、暴力）。
*   **响应机制**:
    *   普通情绪 -> 给予共情回应。
    *   **高风险** -> 强制打断，输出危机干预话术，推荐求助热线。

### Slide 8: 挑战与解决方案 (Challenges & Solutions)
*   **挑战1**: 响应延迟 (Latency)
    *   *解法*: 全链路异步 (`asyncio`), 数据库读写与情绪分析并行处理。
*   **挑战2**: 幻觉问题 (Hallucination)
    *   *解法*: RAG (检索增强生成)，强制模型基于知识库回答专业问题；Prompt 约束。
*   **挑战3**: 状态一致性
    *   *解法*: 使用 SQLite 事务管理，确保会话状态和用户画像更新的一致性。

### Slide 9: 总结与展望 (Conclusion & Future)
*   **成果**: 完成了一个具备完整 Agent 能力的 MVP，验证了 "记忆+情绪+行动" 的技术路径。
*   **未来计划**:
    *   引入向量数据库 (Vector DB) 提升 RAG 检索精度。
    *   支持多模态输入（语音情感识别）。
    *   微调 (Fine-tuning) 垂直领域的心理模型。

---

## 第二部分：面试核心 Q&A (Core Interview Questions)

### Q1: 你的 Agent 是如何工作的？请详细描述 ReAct 流程。
**参考回答**:
我的 Agent 基于 `AgentOrchestrator` 类实现。它不仅仅是把 Prompt 发给 LLM，而是维护了一个 `while` 循环。
每次用户输入时：
1.  首先构建 Prompt，包含系统指令、对话历史和当前阶段策略。
2.  发送给 DeepSeek API，并开启 `tool_choice="auto"`。
3.  如果 LLM 返回 `tool_calls`（比如需要推荐文章），我的代码会解析这个请求，在本地执行对应的 Python 函数（如 `content_recommender`）。
4.  将函数的执行结果（Observation）作为新消息追加回历史记录。
5.  再次请求 LLM，让它根据这个结果生成最终回复。
这就是标准的 **Reasoning (推理) -> Acting (行动) -> Observation (观察)** 闭环。

### Q2: 你提到的"分层记忆"具体是怎么实现的？
**参考回答**:
为了解决 Token 限制和长期记忆问题，我设计了三层结构：
1.  **短期记忆 (Short-term)**: 在内存中保留最近 **20轮** 对话，但发给 LLM 的 Prompt 只包含最近 **10轮**，保证对最新指令的精确响应。
2.  **摘要记忆 (Summary)**: 当对话轮数过多时，我会触发一个后台任务，让 LLM 把旧的对话压缩成一段简短的 Summary（包含用户核心关切和情绪状态），下次交互时把这个 Summary 放在 System Prompt 里。
3.  **长期记忆 (Long-term)**: 使用 SQLite 数据库存储用户的 Profile（画像），比如"风险等级"、"内容偏好"。这部分信息是跨会话持久化的，Agent 可以随时通过工具调用来读取或更新。

### Q3: 情绪分析是实时的吗？会影响对话延迟吗？
**参考回答**:
是实时的，而且为了保证**回复的共情质量和安全性**，我在设计上将其作为**前置阻塞步骤**。
1.  **为什么串行等待**: 我必须先拿到用户的情绪状态（比如“极度绝望”），才能决定是进入“危机干预模式”还是“普通对话模式”，并将情绪注入到 System Prompt 中让 AI 调整语气。
2.  **如何优化延迟**:
    *   **缓存**: `EmotionAnalyzer` 内部实现了 `LRU Cache`，对于相似或重复的文本直接返回缓存结果，毫秒级响应。
    *   **异步持久化**: 虽然分析过程是 await 的，但**写入数据库**的操作（在 `add_interaction` 中）是完全异步的 (`asyncio.create_task`)，不会阻塞给用户的最终响应。

### Q4: 如果用户在对话中表现出强烈的自杀倾向，系统怎么处理？
**参考回答**:
安全是心理健康应用的第一优先级。我实现了一个独立的 `UrgentDetector` 模块。
它不完全依赖 LLM（因为 LLM 可能会产生幻觉或被绕过），而是结合了**关键词匹配**和**规则引擎**。
一旦检测到 "自杀"、"想死" 等高危词汇，系统会：
1.  **覆盖 (Override)** Agent 的正常回复逻辑。
2.  强制输出预设的危机干预文本（包含求助热线）。
3.  在后台将该会话标记为 "High Risk"，并可以触发报警或通知管理员（模拟功能）。

### Q5: 为什么选择 FastAPI 而不是 Flask/Django？
**参考回答**:
主要因为 **异步性能 (Async IO)**。
我的应用是 I/O 密集型的（大量的 LLM API 调用、数据库读写）。FastAPI 原生支持 `async/await`，这意味着在等待 LLM 回复的几秒钟内，服务器线程不会被阻塞，可以同时处理其他用户的请求，这对于构建高并发的 AI 应用至关重要。此外，FastAPI 自动生成 Swagger 文档也非常方便调试。

### Q6: 每次对话发送给 LLM 的 Prompt 结构是怎样的？
**参考回答**:
Prompt 包含四个部分：
1.  **System Message**: 定义人设，并动态注入当前的 `conversation_stage` (如 Exploring) 和 `strategy_guidance` (策略指导)。
2.  **Summary**: 之前对话的压缩摘要。
3.  **History**: 最近 10 轮的完整对话记录。
4.  **User Input**: 用户当前的输入。
这种结构确保了 LLM 既能看到宏观背景，又能精准捕捉当前细节。

### Q7: 项目中有用到 RAG (检索增强生成) 吗？
**参考回答**:
用到了基础版的 RAG。
在 `agent_tools.py` 中定义了 `search_knowledge_base` 工具。当用户问专业心理学术语（如"什么是 CBT"）时，Agent 会调用这个工具，从本地的 JSON/文本库中检索相关定义，然后作为上下文喂给 LLM。这避免了模型一本正经地胡说八道，保证了专业性。未来计划升级为向量数据库检索。
