# 🧠 MindPal Pro — AI 心理支持对话系统

**MindPal Pro** 是一个基于大语言模型的上下文感知心理支持对话系统，具备**情绪分析**、**四级风险评估**、**个性化内容推荐**、**长期记忆**和**会话持久化**能力。

系统使用 **DeepSeek Chat** 作为底层 LLM，结合 **BERT 双通道推理**（风险分类 v4.2 + 情绪分类 v1）实现离线快速预判，采用 **ReAct Agent 循环**进行智能对话，并通过 **SQLite** 实现数据结构化持久化。

---

## 📚 技术文档索引

> 面向 **Agent 算法 / 搜推算法求职** 的技术叙事（Stateful Adaptive Agent 升级全过程）：

- **[docs/README.md](docs/README.md)** — 全部技术文档的整合索引（计划 → 架构 → 评测 → Phase 评审 → Intent → 进度）
- **评测体系**：Benchmark v1.1 冻结（362 条）· Risk Ablation 根因审计 · Baseline 报告
- **Intent Recognition**：独立感知层（Macro 0.844 / high_risk Recall 0.873，已冻结）
- **[docs/execution_progress.md](docs/execution_progress.md)** — 每阶段五要素执行日志（任务/完成/卡点/计划/踩坑）

---

## ✨ 核心功能

- **💬 智能对话** — ReAct Agent 编排器，LLM Function Calling，6 个心理支持工具
- **🔬 情绪分析** — LLM 深度分析 + 可选 BERT 预分类双通道（带 LRU 缓存）
- **⚠️ 四级风险评估** — BERT v4.2 模型 → 会话级风险聚合 → 风险记忆衰减
- **🚨 危机干预** — 紧急/高风险时自动绕过 LLM，直接生成安全回复
- **📚 个性化推荐** — BM25 + 向量混合检索（RRF 融合）+ 规则排序 + 可选 AI rerank
- **🧠 长期记忆** — Memory System v2.0：用户画像、情绪事件、风险记忆的持久化与检索
- **🔒 输出安全** — 出站安全检查（无效化语言、危险细节、过度保证等）
- **📱 Web 界面** — Streamlit 单页应用 + 调试面板
- **🐳 Docker 部署** — 双容器部署，支持 NVIDIA GPU 加速 BERT 推理

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                       │
│                    (frontend/frontend.py)                       │
│                    + debug_panel.py                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP REST API
┌───────────────────────────▼─────────────────────────────────────┐
│                     FastAPI Backend (:8000)                     │
│  ┌─────────────┐  ┌────────────────┐  ┌────────────────────┐   │
│  │  Emotion     │  │  Risk           │  │  Content           │   │
│  │  Analyzer    │  │  Evaluator      │  │  Recommender       │   │
│  │  (LLM+BERT)  │  │  (BERT v4.2)   │  │  (BM25+Vector)     │   │
│  └──────┬───────┘  └───────┬─────────┘  └────────┬───────────┘   │
│         │                  │                      │              │
│         └──────────────────┼──────────────────────┘              │
│                            │                                     │
│                    ┌───────▼────────┐                            │
│                    │ AgentOrchestrator│                           │
│                    │ (ReAct Loop)   │                            │
│                    └───────┬────────┘                            │
│                            │                                     │
│  ┌─────────────────────────┼─────────────────────────────┐      │
│  │    Session Risk Aggregator  │  Recommend Gate         │      │
│  │    Memory Store / Risk Mem  │  Safety Checker         │      │
│  │    Conversation Manager     │  Urgent Detector        │      │
│  └─────────────────────────────┴─────────────────────────┘      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                        SQLite 持久化                             │
│    sessions.db  │  memory_items  │  risk_events  │  profiles   │
└─────────────────────────────────────────────────────────────────┘
```

### 模块组织

```
backend/
├── 入口层        main.py, config.py, api_endpoints.py, models.py
├── Agent 层      agent_orchestrator.py, agent_tools.py, agent_prompts.py
├── 风险层        risk_evaluator.py, bert_risk_predictor.py, risk_levels.py
│                 session_risk_aggregator.py, risk_memory*.py
├── 情绪层        emotion_analyzer.py, bert_emotion_predictor.py
├── 推荐层        content_recommender.py, content_db.py, recommend_gate.py
│                 recommendation_trace.py, hybrid_retriever.py
│                 bm25_retriever.py, vector_retriever.py
├── 安全层        urgent_detector.py, output_safety_checker.py
├── 记忆层        memory_store.py, conversation_manager.py
└── 基础设施      database.py, middleware.py, error_handler.py, utils.py
```

---

## 🚀 快速开始

### 前置要求

- Python 3.9+
- DeepSeek API Key（[申请地址](https://platform.deepseek.com/)）

### 1. 安装

```bash
# 克隆仓库
git clone <repo-url>
cd mindpal_demo/agent_version

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example backend/.env
```

编辑 `backend/.env`，填入你的 DeepSeek API Key：

```ini
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
```

### 2. 启动后端

```bash
cd backend
python main.py
```

访问 http://localhost:8000/docs 查看交互式 API 文档（Swagger UI）。

首次请求时，BERT 风险模型和情绪模型会自动加载（约 400MB，约 10-30 秒）。

### 3. 启动前端

```bash
# 新开终端，确保后端已启动
cd frontend
streamlit run frontend.py
```

访问 http://localhost:8501 打开聊天界面。

### 4. 快速验证

```bash
# 健康检查
curl http://localhost:8000/health

# 情绪分析
curl -X POST http://localhost:8000/emotion/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "最近压力很大，睡不着觉", "user_id": "test_user"}'

# 智能对话
curl -X POST http://localhost:8000/chat/intelligent \
  -H "Content-Type: application/json" \
  -d '{"text": "最近工作压力好大", "user_id": "test_user", "session_id": "test_session"}'
```

---

## 📖 API 文档

启动后端后访问 http://localhost:8000/docs 可查看完整 API 文档。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 首页 + 服务状态 |
| `/health` | GET | 健康检查 |
| `/emotion/analyze` | POST | 情绪分析 |
| `/chat/intelligent` | POST | 智能对话 |
| `/agent/run` | POST | Agent 完整入口 |
| `/session/{user_id}/{session_id}/summary` | GET | 会话摘要 |
| `/session/{user_id}/{session_id}/history` | GET | 会话历史 |
| `/session/{user_id}/list` | GET | 用户会话列表 |
| `/session/{user_id}/{session_id}` | DELETE | 清除会话 |
| `/session/cleanup` | POST | 清理过期会话 |
| `/session/statistics` | GET | 会话统计 |
| `/content/recommend` | POST | 个性化推荐 |
| `/content/feedback` | POST | 推荐反馈 |
| `/content/search` | GET | 搜索内容 |
| `/content/{id}` | GET | 内容详情 |
| `/content/stats` | GET | 内容统计 |
| `/urgent/cases` | GET | 紧急情况记录 |
| `/resources/emergency` | GET | 紧急求助资源 |

---

## ⚙️ 配置

系统通过 `backend/.env` 文件加载配置，完整配置项见 `.env.example`。关键配置段：

| 配置段 | 关键项 | 说明 |
|--------|--------|------|
| **API 密钥** | `DEEPSEEK_API_KEY` | DeepSeek Chat API Key（必填） |
| **LLM 模型** | `CHAT_MODEL`, `API_BASE_URL` | 模型名称和 API 地址 |
| **BERT 风险** | `BERT_MODEL_PATH`, `BERT_DEVICE` | 风险模型路径，设为 `cuda` 启用 GPU |
| **BERT 情绪** | `USE_BERT_EMOTION`, `EMOTION_MODEL_PATH` | 情绪 BERT 开关和路径 |
| **推荐门控** | `RECOMMEND_GATE_*_WEIGHT` | 推荐门控加权因子调参 |
| **记忆系统** | `MEMORY_ENABLED`, `MEMORY_DECAY_DAYS_*` | 记忆功能开关和衰减周期 |
| **会话持久化** | `SESSION_PERSISTENCE_ENABLED`, `SESSION_DB_PATH` | 数据库持久化配置 |

---

## 🐳 Docker 部署

支持双容器部署（FastAPI 后端 + Streamlit 前端），含 NVIDIA GPU 加速支持。

```bash
# 确保已配置 backend/.env
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f backend
docker-compose logs -f frontend

# 重建镜像后启动
docker-compose up -d --build

# 停止
docker-compose down
```

> **注意**：容器内工作目录为 `/app`，数据文件路径为 `backend/data/`，`SESSION_DB_PATH` 和 `CONTENT_DB_FILE` 需相应调整（docker-compose.yml 已自动处理）。

---

## 🧪 测试

### 运行测试

```bash
# 全部测试（推荐在项目根目录执行）
python -m pytest tests/ -v

# 按优先级运行
python -m pytest tests/test_p0_* -v    # P0 安全测试
python -m pytest tests/test_m2_* -v    # 模块单元测试
python -m pytest tests/test_m3_* -v    # 集成测试

# 单个测试文件
python -m pytest tests/test_m2_risk_evaluator.py -v

# 单个测试函数
python -m pytest tests/test_p0_safety_route.py::test_high_risk_routes_to_dedicated_safety_response -v
```

### 测试优先级体系

| 等级 | 说明 | 数量 |
|------|------|------|
| **P0** | 关键安全测试（安全路由、输出安全、风险并发） | 5 文件 |
| **P1** | 核心功能测试（记忆门控、记忆检索） | 3 文件 |
| **P2** | 评估测试（安全评估） | 1 文件 |
| **M1** | 契约/模式测试（Pydantic 模型、OpenAPI schema） | 1 文件 |
| **M2** | 模块级单元测试 | 14 文件 |
| **M3** | 集成/回归测试 | 3 文件 |

### 离线评测

```bash
# 运行全部离线评测
python agent_test_data/eval_skeleton.py --task all

# 按任务评测
python agent_test_data/eval_skeleton.py --task emotion_risk
python agent_test_data/eval_skeleton.py --task recommendation_gate

# 综合评测
python agent_test_data/eval_final.py
```

---

## 🧩 风险评估四级体系

| 等级 | 含义 | 行为 |
|------|------|------|
| **Level 0** | 低风险，正常对话 | 完整 Agent 循环 |
| **Level 1** | 中风险，持续负面情绪 | 加强安抚和支持建议 |
| **Level 2** | 中高风险，需结构化支持 | 专用高风险回复，仅推荐 soft/beginner 内容 |
| **Level 3** | 紧急，立即干预 | 跳过 LLM，直接危机干预回复 |

评估链路：**BERT v4.2 单轮预测** → **SessionRiskAggregator v5.0 状态机**（惯性/连续升级/安全确认/趋势判定）→ **RiskMemoryStore 持久化**

---

## 🧠 推荐模块

推荐流程：**推荐门控**（加权多因子决策）→ **混合检索**（BM25 + 向量 RRF 融合）→ **规则排序** → **可选 AI rerank**

门控加权因子（可通过 `.env` 调参）：

| 因子 | 默认权重 | 说明 |
|------|----------|------|
| 情绪状态 | 0.30 | 当前情绪强度和类型 |
| 风险等级 | 0.23 | 当前风险评估结果 |
| 用户意图 | 0.23 | 用户是否主动寻求帮助 |
| 情绪趋势 | 0.14 | 情绪恶化/好转趋势 |
| 用户偏好 | 0.10 | 历史偏好和反馈 |

每次推荐决策会记录全链路追踪日志（`logs/recommendation_trace.jsonl`），支持离线分析评估。

---

## 📁 项目结构

```
agent_version/
├── backend/                    # Python FastAPI 后端
│   ├── main.py                # 入口 + 定时任务
│   ├── config.py              # 环境配置
│   ├── api_endpoints.py       # REST API 路由
│   ├── models.py              # Pydantic 数据模型
│   ├── database.py            # SQLite 持久化
│   ├── agent_orchestrator.py  # ReAct Agent 编排器
│   ├── agent_tools.py         # Function Calling 工具
│   ├── agent_prompts.py       # Agent 提示词
│   ├── risk_evaluator.py      # BERT 风险评估
│   ├── bert_risk_predictor.py # BERT 风险推理封装
│   ├── risk_levels.py         # 风险等级常量
│   ├── risk_memory.py         # 风险记忆管理
│   ├── risk_memory_store.py   # 风险记忆持久化
│   ├── session_risk_aggregator.py  # 风险聚合状态机
│   ├── emotion_analyzer.py    # LLM 情绪分析
│   ├── bert_emotion_predictor.py  # BERT 情绪预测
│   ├── recommend_gate.py      # 推荐门控决策
│   ├── recommendation_trace.py # 推荐全链路追踪
│   ├── content_recommender.py # 个性化推荐引擎
│   ├── content_db.py          # 内容数据库
│   ├── hybrid_retriever.py    # 混合检索（RRF 融合）
│   ├── bm25_retriever.py      # BM25 检索
│   ├── vector_retriever.py    # 向量检索
│   ├── memory_store.py        # 记忆存储
│   ├── conversation_manager.py # 会话管理
│   ├── urgent_detector.py     # 危机检测与回复
│   ├── output_safety_checker.py # 输出安全检查
│   ├── middleware.py           # CORS 中间件
│   ├── error_handler.py       # 异常处理器
│   └── utils.py               # 工具函数
├── frontend/                   # Streamlit 前端
│   ├── frontend.py            # 聊天 UI
│   └── debug_panel.py         # 调试面板
├── tests/                      # 测试套件（30 文件）
│   ├── test_p0_*              # P0 安全测试
│   ├── test_m2_*              # 模块单元测试
│   └── ...
├── agent_test_data/            # 离线评测框架
│   ├── eval_skeleton.py       # 主评测脚本
│   ├── eval_emotion_risk.py   # 情绪+风险评测
│   ├── eval_recommend_gate.py # 推荐门控评测
│   ├── emotion_risk_turns.jsonl # 评测数据集
│   └── ...
├── bert_data/                  # BERT 模型训练
│   ├── models/                # 模型权重
│   └── scripts/               # 训练脚本
├── docs/                       # 设计文档
├── data/                       # 运行时数据
├── .env.example                # 环境变量模板
├── docker-compose.yml          # Docker 编排
├── Dockerfile                  # 后端 Dockerfile
├── Dockerfile.frontend         # 前端 Dockerfile
└── requirements.txt            # Python 依赖
```

---

## 📚 设计文档

详细的系统设计文档位于 `docs/` 目录：

| 文档 | 说明 |
|------|------|
| [风险安全模块技术文档](docs/风险安全模块技术文档.md) | 风险评估架构、四级体系、BERT 模型 |
| [情绪识别模块系统设计](docs/压力管理Agent_情绪识别模块系统设计.md) | 情绪分析双通道设计 |
| [推荐模块技术文档](docs/推荐模块技术文档.md) | 推荐引擎架构、门控策略、混合检索 |
| [长期记忆模块 v2.0 重构设计](docs/长期记忆模块v2.0重构设计方案.md) | Memory System v2.0 重构方案 |
| [FAISS 向量检索接入方案](docs/FAISS%20向量检索接入方案.md) | 向量检索集成方案 |
| [评测设计](docs/EVALUATION_DESIGN.md) | 测试与离线评测设计 |

---

## ⚡ 性能与部署注意事项

- **BERT 模型延迟加载**：首次请求时加载约 400MB 模型，约 10-30 秒。后续请求 CPU 推理约 200-500ms/次
- **GPU 加速**：设置 `BERT_DEVICE=cuda` 和 `EMOTION_DEVICE=cuda` 启用 GPU，推理延迟降至 20-50ms/次
- **情绪分析缓存**：LRU 缓存（默认 1000 条，1 小时 TTL），重复文本命中缓存跳过 LLM 调用
- **数据库迁移**：所有变更使用 `_ensure_column_exists()` 附加式迁移，不会破坏已有数据
- **Docker GPU**：需安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## 🔧 开发

```bash
# 运行单个测试文件（直接执行，不通过 pytest）
python tests/test_m2_risk_evaluator.py

# BERT 模型训练（见 bert_data/scripts/）
python bert_data/scripts/train_v4_2.py

# FAISS 索引构建
python agent_test_data/build_faiss_index.py
```

---

## 📄 许可证

本项目为内部项目，详情请联系项目维护者。
