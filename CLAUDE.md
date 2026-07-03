# CLAUDE.md

本文件为 Claude Code（claude.ai/code）操作此仓库时提供指导和上下文。

## 项目概述

**MindPal Pro 后端 v3.2** — 基于 DeepSeek Chat LLM 的上下文感知心理支持对话系统，具备情绪分析、四级风险评估（Level 0-3）、个性化内容推荐和会话持久化能力。

## 技术栈

- **后端**: Python FastAPI + DeepSeek Chat（OpenAI 兼容 API）+ SQLite（aiosqlite）
- **前端**: Streamlit 单页应用
- **测试**: pytest（同步 + `@pytest.mark.asyncio` 异步）
- **检索**: BM25（自实现）+ 同义词向量相似度混合检索（RRF 融合）

## 核心架构

```
backend/
├── main.py                     # FastAPI 入口，CORS，APScheduler 定时任务
├── config.py                   # .env 配置加载（DEEPSEEK_API_KEY 等）
├── api_endpoints.py            # REST 路由（/chat/intelligent, /emotion/analyze, /agent/run, /content/*）
├── models.py                   # Pydantic 模型定义
├── database.py                 # SQLite 持久化（同步 DatabaseManager + 异步 AsyncDatabaseManager）
│
├── agent_orchestrator.py       # 核心编排器：ReAct Agent 循环（LLM Function Calling）
├── agent_tools.py              # 工具定义（OpenAI function-calling 格式）+ 工具实现
├── agent_prompts.py            # 系统提示模板 + 对话阶段策略提示
│
├── risk_evaluator.py           # 风险评估器：委托 BERT v4.2 模型进行 4 级风险分类
├── bert_risk_predictor.py      # BERT 模型封装（MultiTaskBERT + 规则兜底 + 阈值融合）
├── risk_levels.py              # 风险等级常量、别名归一化、工具函数
├── urgent_detector.py          # 危机关键词检测 + LLM 生成危机/第三方求助回复
├── recommend_gate.py           # 推荐门控（加权多因子打分，决定 hard/soft/none）
│
├── emotion_analyzer.py         # 基于 LLM 的情绪分析（带 LRU 缓存）
├── output_safety_checker.py    # 出站安全检查（无效化语言、危险细节、过度保证等模式匹配）
├── conversation_manager.py     # 会话管理（内存缓存 + SQLite 持久化，Token 感知上下文压缩）
├── content_recommender.py      # 个性化推荐引擎（混合检索 + 规则排序 + 可选 AI rerank）
├── content_db.py               # 内容数据库（JSON 文件，搜索/CRUD）
├── hybrid_retriever.py         # BM25 + 向量混合检索（RRF 融合）
├── bm25_retriever.py           # 纯 Python BM25 实现
├── vector_retriever.py         # 同义词扩展 + 稀疏向量余弦相似度检索
├── middleware.py                # CORS 中间件
├── error_handler.py            # 全局异常处理器
└── utils.py                    # 输入验证、日志工具等

frontend/
├── frontend.py                 # Streamlit 聊天 UI（单文件 ~850 行）
└── debug_panel.py              # 调试面板（情绪/风险/推荐状态快照）

tests/                          # pytest 测试用例
├── test_p0_*                   # P0 关键安全测试（safety_route, output_safety 等）
├── test_p1_*                   # P1 核心功能测试（memory_gate, memory_retrieval 等）
├── test_p2_*                   # P2 评估测试（eval_safety）
├── test_m1_*                   # 契约/模式测试（Pydantic 模型验证、OpenAPI schema）
├── test_m2_*                   # 模块级单元测试（risk_evaluator, recommend_gate 等）
└── test_m3_*                   # 集成/回归测试（integration_regression, feedback_loop 等）

agent_test_data/                # 离线评测数据集（JSONL）
├── eval_skeleton.py            # 评测框架（--task all|emotion_risk|recommendation_gate|...）
├── build_faiss_index.py        # FAISS 索引构建
└── hf_models/                  # HuggingFace 模型下载（bge-small-zh-v1.5）
```

## 请求生命周期

`agent_orchestrator.py` 中 `run_agent()` 的执行流程：

1. **加载会话** — 从 ConversationManager 获取/创建会话（SQLite 或内存）
2. **情绪分析** — LLM 调用分析当前情绪 + 上下文情绪
3. **用户画像** — 从 SQLite 加载长期风险等级和偏好
4. **风险评估** — `risk_evaluator.evaluate()` 委托 BERT v4.2 模型进行 4 级分类
5. **推荐门控** — `recommend_gate.decide()` → hard / soft / none
6. **路由决策**：
   - **Level 3（紧急）** → `urgent_detector.generate_crisis_response_async()`，绕过 LLM
   - **Level 2（高风险）** → `urgent_detector.generate_crisis_response_async()` 专用支持模式
   - **第三方危机求助** → 专用第三方支持回复
   - **正常对话** → ReAct Agent Loop（最多 5 轮）：LLM Function Calling + 6 个工具
7. **输出安全检查** — `output_safety_checker.review()` 应用规则，可覆盖回复
8. **持久化** — 对话历史、风险状态、情绪事件写入 SQLite

## 四级风险体系

定义在 `risk_levels.py`，基于 BERT v4.2 模型预测：

| 等级 | 常量 | 含义 | 行为 |
|------|------|------|------|
| Level 0 | `LEVEL_0` | 低风险，正常对话 | 完整 Agent 循环 |
| Level 1 | `LEVEL_1` | 中风险，持续负面情绪 | 加强安抚和支持建议 |
| Level 2 | `LEVEL_2` | 中高风险，需结构化支持 | 专用高风险回复 + 仅推荐 soft/beginner 内容 |
| Level 3 | `LEVEL_3` | 紧急，立即干预 | 跳过 LLM，直接危机干预回复 |

关键函数：`normalize_risk_level()` 处理别名映射，`risk_level_index()` 获取排序索引，`is_emergency_risk()` / `is_non_low_risk()` 谓词判断。

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

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行后端
cd backend && python main.py
# 或 .\run_backend.ps1
# 访问 http://localhost:8000/docs 查看 API 文档

# 运行前端
cd frontend && streamlit run frontend.py
# 或 .\run_frontend.ps1
# 访问 http://localhost:8501

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

# 离线评测
python agent_test_data/eval_skeleton.py --task all
```

## 重要注意事项

- 所有 LLM 调用走 DeepSeek Chat（OpenAI 兼容 API）
- `.env` 文件含敏感 API 密钥，禁止提交
- 测试设置 `DEEPSEEK_API_KEY=test-key` 绕过真实 API，通过 mock/stub 隔离 LLM
- `tests/` 中每个文件独立添加 `sys.path`，运行测试时需在项目根目录下执行
- `agent_test_data/` 含离线评测数据集和 FAISS 构建工具
- 数据库迁移是附加式的（`_ensure_column_exists`），不会破坏已有数据
- **风险评估已完全基于 BERT v4.2 模型**（`bert_data/models/v4_2_multitask_calibrated/best_model/`），旧的规则系统（7 维度关键词打分）已移除。`risk_evaluator.py` 委托 `bert_risk_predictor.py` 进行推理，输出兼容的 4 级风险字典
- BERT 模型使用 `hfl/chinese-macbert-base` + MultiTaskBERT（4 分类 + 二分类），在 SOS-1K 数据集上微调，推理时应用三级融合（规则兜底 → binary 升级 → 4 分类预测）
- 首次请求时延迟加载模型（约 400MB），可通过 `BERT_DEVICE=cuda` 启用 GPU 推理
