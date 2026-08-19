# Agent Benchmark Schema v1.1

> 对应 Phase 0.5 Task 0.5.1（§4）
> 日期：2026-08-18
> 代码定义：`evaluation/benchmark_schema.py`
> 状态：**冻结**（Phase 1–7 统一使用，不再修改语义）

---

## 1. v1 → v1.1 修正内容

| 项 | v1 | v1.1 | 原因 |
|---|---|---|---|
| Action 语义 | 单一 `agent_action`（混合响应+工具） | `primary_action` + `tool_actions` + `safety_target` | 真实 Agent 一轮可同时 retrieve + recommend + continue |
| 第三方危机 | `recommendation_action=third_party_support` | `recommendation_action=safety_only` + `safety_target=third_party` | 语义归属错误：第三方是"对象"而非"推荐模式" |
| 知识检索 | `should_retrieve_knowledge: bool` | `tool_actions: [retrieve_knowledge]` | 统一为工具动作 |
| 兼容 | 无 | `migrate_v1_case_to_v1_1()` | v1 数据可自动迁移 |

## 2. 枚举定义

### PrimaryAction（最终响应，互斥）

```python
CONTINUE_CHAT = "continue_chat"
ASK_CLARIFICATION = "ask_clarification"
INFORMATION_RESPONSE = "information_response"
SAFETY_INTERVENTION = "safety_intervention"
```

### ToolAction（内部工具，可多选）

```python
RETRIEVE_MEMORY = "retrieve_memory"
RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
RECOMMEND_RESOURCE = "recommend_resource"
```

### SafetyTarget（安全干预对象）

```python
NONE = "none"
SELF = "self"
THIRD_PARTY = "third_party"
```

### RecommendationAction（对齐 RecommendGate）

```python
HARD = "hard"
SOFT = "soft"
NONE = "none"
SAFETY_ONLY = "safety_only"
```

> ⚠️ `third_party_support` 已移除，由 `safety_only` + `safety_target=third_party` 表达。

## 3. ExpectedOutcome（v1.1）

```python
class ExpectedOutcome(BaseModel):
    intent: List[IntentLabel]          # multi-label
    emotion: Optional[str]
    emotion_intensity: Optional[float]

    risk_level: RiskLevel              # 0-3
    risk_trend: RiskTrend              # new/rising/stable/falling/fluctuating

    memory_needed: bool
    memory_refs: List[str]

    recommendation_action: RecommendationAction

    primary_action: PrimaryAction      # 互斥
    tool_actions: List[ToolAction]     # 多选
    safety_target: SafetyTarget

    conversation_stage: Optional[ConversationStage]
```

## 4. 迁移 helper

```python
from evaluation.benchmark_schema import migrate_v1_case_to_v1_1, migrate_v1_cases_to_v1_1

v1_case = {"case_id": "...", "expected": {"agent_action": "retrieve_memory", ...}}
v1_1_case = migrate_v1_case_to_v1_1(v1_case)
```

规则：
- `agent_action=retrieve_memory` → `primary=continue_chat` + `tool=[retrieve_memory]`
- `agent_action=recommend_resource` → `primary=continue_chat` + `tool=[recommend_resource]`
- `agent_action=safety_intervention` → `primary=safety_intervention` + `safety_target=self`
- `recommendation_action=third_party_support` → `safety_only` + `safety_target=third_party`
- `should_retrieve_knowledge=true` → 并入 `tool_actions`

**已验证**：v1 的 100 条全部可迁移，迁移结果与 v1.1 种子逐条一致，且通过 Pydantic 校验（见 `evaluation/tests/test_v1_to_v1_1_migration.py`）。

## 5. 数据集

- `evaluation/datasets/agent_benchmark_v1.jsonl`（100 条）— **历史冻结**，不覆盖
- `evaluation/datasets/agent_benchmark_v1_1.jsonl`（362 条）— **正式使用**
- 生成：`generate_benchmark_v1_1_expand.py`（种子 = v1 迁移 100 条 + 新增 262 条）

## 6. 校验

```bash
python -m evaluation.validate_dataset evaluation/datasets/agent_benchmark_v1_1.jsonl
# 结果：362 条，0 errors
```

## 7. 指标口径（v1.1）

| 模块 | 主指标 | 说明 |
|---|---|---|
| Risk | Macro F1 / High-risk Recall / FPR / FNR | |
| Emotion | Coarse 5-class Accuracy / Macro F1 | 细粒度作 Coverage |
| Intent | **Legacy Intent Coverage** | 当前系统=关键词规则，不当作正式模型指标 |
| Recommendation | Trigger P/R / Mode Accuracy | 注明受上游信号影响 |
| Primary Action | Accuracy / Macro F1 / Safety Recall | |
| Tool Action | Micro/Macro F1 / Exact Match / 每工具 P/R | multi-label |
| Memory | **Current Memory Retrieval Behavior** | 行为观测非门控质量 |
| Risk Trend | 多轮 Accuracy | |
