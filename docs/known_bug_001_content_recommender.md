# Known Bug 001 — content_recommender missing import

> 对应 Phase 3 Preflight P3-0.2
> 日期：2026-08-19

## 1. 触发条件

- `backend/agent_orchestrator.py` 第 537 行引用 `content_recommender.enable_ai_rerank`
- 但文件头部未 import `content_recommender`
- **仅在特定路径触发**：Agent ReAct 循环返回错误（LLM 失败/异常）后，代码走到 `_extract_recommendations` 后仍需读取 `enable_ai_rerank` 时，抛出 `NameError: name 'content_recommender' is not defined`

## 2. 根因

`agent_orchestrator.py` 使用了 `content_recommender` 单例，但遗漏 import 语句（Phase 0 之前的既有缺陷，非 Phase 2 引入）。

## 3. 修复

```python
# agent_orchestrator.py imports
from recommend_gate import recommend_gate
+ from content_recommender import content_recommender
```

**独立 commit**：`fix(orchestrator): import content recommender fallback dependency`（未与 Policy 重构混合）。

## 4. before / after

| 场景 | Before | After |
|---|---|---|
| LLM 失败路径（读 enable_ai_rerank） | `NameError` | 正常返回配置值 |
| 正常路径（LLM 成功） | 不触发（模块逻辑路径不同） | 不触发 |
| 模块导入 | orchestrator 导入正常 | 导入正常 |

## 5. 测试结果

- ✅ fallback path smoke：`getattr(content_recommender, 'enable_ai_rerank', False)` 正常
- ✅ LLM failure 路径：orchestrator 导入 + 模块加载正常
- ✅ normal path regression：模块可加载、单例可用

## 6. 影响面

- 仅修复缺失 import，不改变任何模块逻辑
- 冻结 Benchmark 指标不受影响（该路径不进入评测模块调用链）
