# CLAUDE.md

本文件为 Claude Code（claude.ai/code）操作此仓库时提供指导和上下文。

## 项目概述

**MindPal Pro 后端 v3.2** — 基于 DeepSeek Chat LLM 的上下文感知心理支持对话系统，具备情绪分析（LLM + BERT 双通道）、四级风险评估（Level 0-3，BERT v4.2 模型）、会话级风险聚合、个性化内容推荐、长期记忆系统和会话持久化能力。

## 技术栈

- **后端**: Python FastAPI + DeepSeek Chat（OpenAI 兼容 API）+ SQLite（aiosqlite）
- **前端**: Streamlit 单页应用
- **测试**: pytest（同步 + `@pytest.mark.asyncio` 异步）
- **BERT 模型**: `hfl/chinese-macbert-base` 微调（风险分类 + 情绪分类）
- **检索**: BM25（自实现）+ 同义词向量相似度混合检索（RRF 融合）
- **部署**: Docker Compose（双容器），支持 NVIDIA GPU 加速 BERT 推理

## 核心架构

```
backend/
├── main.py                     # FastAPI 入口，CORS，APScheduler 定时任务
├── config.py                   # .env 配置加载（DEEPSEEK_API_KEY 等）
├── api_endpoints.py            # REST 路由（/chat/intelligent, /emotion/analyze, /agent/run, /content/*, /session/*）
├── models.py                   # Pydantic 模型定义（含 Memory/Risk v2.0 数据模型）
├── database.py                 # SQLite 持久化（同步 DatabaseManager + 异步 AsyncDatabaseManager）
│
├── agent_orchestrator.py       # 核心编排器：ReAct Agent 循环（LLM Function Calling）
├── agent_tools.py              # 工具定义（OpenAI function-calling 格式）+ 工具实现
├── agent_prompts.py            # 系统提示模板 + 对话阶段策略提示
│
├── risk_evaluator.py           # 风险评估器：委托 BERT v4.2 模型进行 4 级分类
├── bert_risk_predictor.py      # BERT 风险模型封装（MultiTaskBERT + 规则兜底 + 阈值融合）
├── risk_levels.py              # 风险等级常量、别名归一化、工具函数
├── session_risk_aggregator.py  # 会话级风险聚合器（v5.0 状态机：惯性、连续升级、趋势判定）
├── risk_memory.py              # 风险记忆管理模块
├── risk_memory_store.py        # 风险记忆持久化层（SQLite 存储 RiskEvent / RiskBaseline）
│
├── emotion_analyzer.py         # 情绪分析器（LLM 分析 + 可选 BERT 双通道，带 LRU 缓存）
├── bert_emotion_predictor.py   # BERT 情绪分类模型（5 类：中性/快乐/焦虑/抑郁/愤怒）
│
├── urgent_detector.py          # 危机关键词检测 + LLM 生成危机/第三方求助回复
├── output_safety_checker.py    # 出站安全检查（无效化语言、危险细节、过度保证等模式匹配）
├── recommend_gate.py           # 推荐门控（加权多因子打分，决定 hard/soft/none）
├── recommendation_trace.py     # 推荐全链路追踪（JSONL 日志，支持离线分析评估）
├── content_recommender.py      # 个性化推荐引擎（混合检索 + 规则排序 + 可选 AI rerank）
├── content_db.py               # 内容数据库（JSON 文件，搜索/CRUD）
├── hybrid_retriever.py         # BM25 + 向量混合检索（RRF 融合）
├── bm25_retriever.py           # 纯 Python BM25 实现
├── vector_retriever.py         # 同义词扩展 + 稀疏向量余弦相似度检索
│
├── memory_store.py             # Memory System v2.0：统一记忆存储（CRUD + 检索入口）
├── conversation_manager.py     # 会话管理（内存缓存 + SQLite 持久化，Token 感知上下文压缩）
├── middleware.py                # CORS 中间件
├── error_handler.py            # 全局异常处理器
└── utils.py                    # 输入验证、日志配置、ID 生成等工具函数

frontend/
├── frontend.py                 # Streamlit 聊天 UI（单文件 ~850 行）
└── debug_panel.py              # 调试面板（情绪/风险/推荐状态快照）

tests/                          # pytest 测试用例
├── test_p0_*                   # P0 关键安全测试（safety_route, output_safety 等）
├── test_p1_*                   # P1 核心功能测试（memory_gate, memory_retrieval 等）
├── test_p2_*                   # P2 评估测试（eval_safety）
├── test_m1_*                   # 契约/模式测试（Pydantic 模型验证、OpenAPI schema）
├── test_m2_*                   # 模块级单元测试（risk_evaluator, recommend_gate, emotion_analyzer 等）
├── test_m3_*                   # 集成/回归测试（integration_regression, feedback_loop, debug_panel）
├── test_risk_memory.py         # 风险记忆单元测试
└── test_session_risk_aggregator.py  # 会话风险聚合器单元测试

agent_test_data/                # 离线评测数据集（JSONL）+ 评测框架
├── eval_skeleton.py            # 评测框架（--task all|emotion_risk|recommendation_gate|...）
├── eval_emotion_risk.py        # 情绪+风险联合评测
├── eval_recommend_gate.py      # 推荐门控评测
├── eval_final.py               # 综合评测入口
├── eval_datasets.py            # 数据集加载工具
├── eval_metrics.py             # 评测指标（准确率、召回率、F1 等）
├── expand_eval_set.py          # 数据集扩充工具
├── build_faiss_index.py        # FAISS 索引构建
├── faiss_utils.py              # FAISS 检索工具（含 bge-small-zh-v1.5 嵌入）
└── hf_models/                  # HuggingFace 模型下载（bge-small-zh-v1.5）

docs/                           # 设计文档
├── EVALUATION_DESIGN.md        # 评测设计文档
├── FAISS 向量检索接入方案.md     # FAISS 检索方案
├── 风险安全模块技术文档.md        # 风险安全模块技术文档
├── 推荐模块技术文档.md           # 推荐模块技术文档
├── 压力管理Agent风险安全模块第二三阶段设计文档.md
├── 压力管理Agent_情绪识别模块系统设计.md
├── 技术文档-长期记忆模块现状.md
├── 长期记忆模块现状与问题分析.md
└── 长期记忆模块v2.0重构设计方案.md

bert_data/                      # BERT 模型训练数据与脚本
├── models/                     # 训练好的模型权重（v4_2_domain_only_v2, emotion_v1 等）
├── raw/                        # 原始数据
├── processed_v*                # 各版本处理后数据
└── 报告：v4.2_Baseline_技术报告.md
```

## 请求生命周期

`agent_orchestrator.py` 中 `run_agent()` 的执行流程：

1. **加载会话** — 从 ConversationManager 获取/创建会话（SQLite 或内存）
2. **情绪分析** — LLM 调用分析当前情绪 + 上下文情绪（可选 BERT 双通道加速）
3. **用户画像** — 从 SQLite 加载长期风险等级和偏好
4. **风险评估** — `risk_evaluator.evaluate()` 委托 BERT v4.2 模型进行 4 级分类
5. **会话级风险聚合** — `session_risk_aggregator.aggregate()` 状态机：考虑单轮预测 + 历史风险 + 惯性 + 连续升级 + 安全确认，输出聚合等级和趋势
6. **推荐门控** — `recommend_gate.decide()` → hard / soft / none（加权多因子：情绪、风险、意图、趋势、偏好）
7. **推荐全链路追踪** — `recommendation_trace.write_trace()` 记录每次推荐决策到 JSONL
8. **路由决策**：
   - **Level 3（紧急）** → `urgent_detector.generate_crisis_response_async()`，绕过 LLM
   - **Level 2（高风险）** → `urgent_detector.generate_crisis_response_async()` 专用支持模式
   - **第三方危机求助** → 专用第三方支持回复
   - **正常对话** → ReAct Agent Loop（最多 5 轮）：LLM Function Calling + 6 个工具
9. **输出安全检查** — `output_safety_checker.review()` 应用规则，可覆盖回复
10. **记忆写入** — Memory System v2.0：提取关键信息写入 MemoryStore / RiskMemoryStore
11. **持久化** — 对话历史、风险状态、情绪事件写入 SQLite

## 四级风险体系

定义在 `risk_levels.py`，基于 BERT v4.2 模型预测 + SessionRiskAggregator v5.0 状态机聚合：

| 等级 | 常量 | 含义 | 行为 |
|------|------|------|------|
| Level 0 | `LEVEL_0` | 低风险，正常对话 | 完整 Agent 循环 |
| Level 1 | `LEVEL_1` | 中风险，持续负面情绪 | 加强安抚和支持建议 |
| Level 2 | `LEVEL_2` | 中高风险，需结构化支持 | 专用高风险回复 + 仅推荐 soft/beginner 内容 |
| Level 3 | `LEVEL_3` | 紧急，立即干预 | 跳过 LLM，直接危机干预回复 |

关键函数：`normalize_risk_level()` 处理别名映射，`risk_level_index()` 获取排序索引，`is_emergency_risk()` / `is_high_support_risk()` / `is_non_low_risk()` 谓词判断。

聚合策略（`session_risk_aggregator.py`）：
- **当前轮优先**：BERT 单轮预测为主
- **风险惯性**：不因一轮好转而断崖降级
- **连续升级检测**：多轮轻度信号累积触发更高等级
- **安全确认参与降级**：`SAFE_DENIAL_PATTERNS` 匹配时降低等级
- **趋势判定**：输出 rising / stable / falling

## BERT 双通道推理

| 模型 | 文件 | 架构 | 标签 |
|------|------|------|------|
| 风险分类 v4.2 | `bert_risk_predictor.py` | MultiTaskBERT（4 分类 + 二分类） | level_0 ~ level_3 |
| 情绪分类 v1 | `bert_emotion_predictor.py` | EmotionBERT（5 分类头） | 中性/快乐/焦虑/抑郁/愤怒 |

- 首次请求时延迟加载（约 400MB），可通过 `BERT_DEVICE=cuda` / `EMOTION_DEVICE=cuda` 启用 GPU
- 风险模型推理时应用三级融合：规则兜底 → binary 升级 → 4 分类预测
- BERT 情绪分析通过 `.env` 中 `USE_BERT_EMOTION=true/false` 控制开关
- 情绪分析器（`emotion_analyzer.py`）LLM 通道带有 LRU 缓存（1000 条，1h TTL）

## Agent 工具（OpenAI Function Calling）

定义在 `agent_tools.py` 的 `TOOL_DEFINITIONS` 中：

| 工具 | 功能 |
|------|------|
| `search_knowledge_base` | 搜索心理学知识库 |
| `get_user_profile` | 获取用户长期画像 |
| `update_user_profile` | 更新用户画像 |
| `log_mood_event` | 记录情绪事件 |
| `get_recent_mood_trend` | 获取近期情绪趋势 |
| `recommend_content` | 推荐内容（受 RecommendGate 控制） |

## 推荐模块全链路

1. **推荐门控** (`recommend_gate.py`): 加权多因子（情绪 0.30 + 风险 0.23 + 意图 0.23 + 趋势 0.14 + 偏好 0.10）→ hard/soft/none
2. **内容检索** (`content_recommender.py`): 混合检索（BM25 + 向量 RRF 融合）+ 规则排序 + 可选 AI rerank
3. **全链路追踪** (`recommendation_trace.py`): 每次推荐请求的结构化日志（JSONL），含门控输入/输出、候选数、rerank 状态、安全覆盖等

## Memory System v2.0

四层架构：
- **MemoryStore** (`memory_store.py`): 统一记忆存储，基于 SQLite `memory_items` 表 CRUD
- **RiskMemoryStore** (`risk_memory_store.py`): 风险事件/基线/触发词/保护因素的持久化
- **RiskMemory** (`risk_memory.py`): 风险记忆高层管理（读写聚合 + 衰减）
- **SessionRiskAggregator** (`session_risk_aggregator.py`): 会话级风险聚合状态机

`models.py` 中定义的数据模型：`MemoryItem`, `MemoryCandidate`, `MemoryQuery`, `MemorySearchResult`, `TurnContext`, `InjectedMemoryContext`, `RiskEvent`, `RiskBaseline`, `RiskTrigger`, `ProtectiveFactor`

## API 端点一览

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 首页 + 服务状态 |
| `/health` | GET | 健康检查 |
| `/emotion/analyze` | POST | 情绪分析 + 紧急检测 |
| `/chat/intelligent` | POST | 智能对话（轻量封装） |
| `/agent/run` | POST | Agent 完整入口（含步骤摘要） |
| `/session/{uid}/{sid}/summary` | GET | 会话摘要 |
| `/session/{uid}/{sid}/history` | GET | 会话历史 |
| `/session/{uid}/list` | GET | 用户会话列表 |
| `/session/{uid}/{sid}` | DELETE | 清除会话 |
| `/session/cleanup` | POST | 批量清理过期会话 |
| `/session/statistics` | GET | 会话统计 |
| `/content/recommend` | POST | 个性化内容推荐 |
| `/content/feedback` | POST | 推荐反馈 |
| `/content/search` | GET | 搜索内容 |
| `/content/{id}` | GET | 内容详情 |
| `/content/stats` | GET | 内容统计 |
| `/urgent/cases` | GET | 紧急情况记录 |
| `/resources/emergency` | GET | 紧急求助资源 |

## 常用命令

```bash
# 环境配置
cp .env.example backend/.env    # 复制环境变量模板
# 编辑 backend/.env 填入 DEEPSEEK_API_KEY

# 安装依赖
pip install -r requirements.txt

# 运行后端
cd backend && python main.py
# 或 .\run_backend.ps1
# 访问 http://localhost:8000/docs 查看 API 文档

# 运行前端（需先启动后端）
cd frontend && streamlit run frontend.py
# 或 .\run_frontend.ps1
# 访问 http://localhost:8501

# Docker 部署
docker-compose up -d                       # 启动所有服务（含 GPU）
docker-compose up -d --build               # 重建镜像后启动
docker-compose logs -f backend             # 查看后端日志
docker-compose logs -f frontend            # 查看前端日志

# 运行所有测试
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_m2_risk_evaluator.py -v

# 运行单个测试函数
python -m pytest tests/test_p0_safety_route.py::test_high_risk_routes_to_dedicated_safety_response -v

# 按优先级运行
python -m pytest tests/test_p0_* -v    # P0 安全测试
python -m pytest tests/test_m2_* -v    # 模块单元测试
python -m pytest tests/test_m3_* -v    # 集成测试

# 直接运行测试文件（含 if __name__ == "__main__": main()）
python tests/test_m2_risk_evaluator.py

# 离线评测
python agent_test_data/eval_skeleton.py --task all
python agent_test_data/eval_skeleton.py --task emotion_risk
python agent_test_data/eval_skeleton.py --task recommendation_gate
python agent_test_data/eval_final.py
```

## 测试模式

所有测试文件共享以下模式：

```python
import os, sys
import pytest

# 环境设置（模块级）
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.extend([PROJECT_ROOT, BACKEND_DIR])

# 同步测试
def test_something():
    ...

# 异步测试
@pytest.mark.asyncio
async def test_async_something():
    ...
```

- 每文件还可通过 `python tests/test_xxx.py` 直接运行（有 `if __name__ == "__main__": main()`）
- 不使用 pytest fixture，手动 save/restore 单例方法
- 测试用 UUID 隔离 user_id/session_id

## `.env` 关键配置区域

| 配置段 | 文件 | 关键项 |
|--------|------|--------|
| API 密钥 | `.env.example` | `DEEPSEEK_API_KEY`, `CHAT_MODEL`, `API_BASE_URL` |
| BERT 风险模型 | `.env.example:46-49` | `BERT_MODEL_PATH`, `BERT_DEVICE`, `BERT_BINARY_THRESHOLD` |
| BERT 情绪模型 | `.env.example:51-55` | `USE_BERT_EMOTION`, `EMOTION_MODEL_PATH` |
| 推荐门控权重 | `.env.example:57-65` | `RECOMMEND_GATE_*_WEIGHT`, `RECOMMEND_GATE_*_THRESHOLD` |
| 记忆系统 | `.env.example:67-75` | `MEMORY_ENABLED`, `MEMORY_*_RATIO`, `MEMORY_DECAY_DAYS_*` |
| 风险记忆 | `.env.example:77-80` | `RISK_BASELINE_*_WINDOW_DAYS`, `RISK_EVENT_DECAY_DAYS` |
| 会话持久化 | `.env.example:31-34` | `SESSION_PERSISTENCE_ENABLED`, `SESSION_DB_PATH` |
| 定时任务 | `.env.example:83-84` | `SESSION_CLEANUP_HOUR`, `SESSION_CLEANUP_MINUTE` |

## 重要注意事项

- 所有 LLM 调用走 DeepSeek Chat（OpenAI 兼容 API），通过 `config.py` 读取 `.env` 配置
- `.env` 文件含敏感 API 密钥，禁止提交
- 测试设置 `DEEPSEEK_API_KEY=test-key` 绕过真实 API，通过 mock/stub 隔离 LLM
- `tests/` 中每个文件独立添加 `sys.path`，运行测试时需在项目根目录下执行
- `agent_test_data/` 含离线评测数据集和评测框架（FAISS + bge-small-zh-v1.5）
- 数据库迁移是附加式的（`_ensure_column_exists`），不会破坏已有数据
- 风险评估级联：BERT v4.2 模型 → SessionRiskAggregator v5.0 状态机 → RiskMemoryStore 持久化
- 情绪分析级联：可选 BERT v1 预分类（高置信度直接返回）→ LLM 深度分析（兜底）
- BERT 模型存储在 `bert_data/models/` 下（约 400MB），首次请求时延迟加载
- Docker 部署时需注意容器内路径：`WORKDIR=/app`，数据路径为 `backend/data/`
- `docs/` 目录包含详细的设计文档（风险安全、情绪识别、长期记忆、推荐模块等）
