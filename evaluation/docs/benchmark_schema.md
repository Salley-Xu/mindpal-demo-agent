# Agent Benchmark Schema v1

> 对应开发计划 §5.3 / §5.4 / §6.2 / §7.2 / §8.2
> 代码定义见 `evaluation/benchmark_schema.py`（Pydantic 模型 + 枚举）

## 1. 文件格式

统一为 **JSONL**（每行一个 JSON case），UTF-8 编码，一行一个 `BenchmarkCase`。

## 2. Case 结构

```json
{
  "case_id": "agent_0001",
  "conversation": [
    {"role": "user", "content": "最近连续加班，感觉很累"}
  ],
  "expected": {
    "intent": ["emotional_expression"],
    "emotion": "fatigue",
    "emotion_intensity": 0.6,
    "risk_level": 0,
    "risk_trend": "new",
    "memory_needed": false,
    "memory_refs": [],
    "recommendation_action": "none",
    "agent_action": "continue_chat",
    "should_retrieve_knowledge": false
  },
  "precondition": null,
  "tags": ["single_turn", "low_risk", "emotional"],
  "source": "template"
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `case_id` | str | ✅ | 唯一 ID，如 `agent_0001` |
| `conversation` | list[Turn] | ✅ | ≥1 轮；`role ∈ {user, assistant}`；`recommendation_ids` 仅 feedback case 使用 |
| `expected.intent` | list[IntentLabel] | ✅ | multi-label，来自 Intent Taxonomy v1 |
| `expected.emotion` | str | ✅ | 归一化情绪标签（见 §4） |
| `expected.emotion_intensity` | float | ⭕ | 0~1 |
| `expected.risk_level` | int | ✅ | 0~3 |
| `expected.risk_trend` | str | ⭕ | new/rising/stable/falling/fluctuating |
| `expected.memory_needed` | bool | ✅ | 是否应检索长期记忆 |
| `expected.memory_refs` | list[str] | ⭕ | 应被召回的长期记忆要点（模糊匹配） |
| `expected.recommendation_action` | str | ✅ | hard/soft/none/third_party_support |
| `expected.agent_action` | str | ✅ | Action Space v1 的 6 个动作之一 |
| `expected.should_retrieve_knowledge` | bool | ⭕ | 是否应触发知识库 RAG |
| `precondition` | object | ⭕ | 用户画像 / 长期记忆前置（见 §5） |
| `tags` | list[str] | ⭕ | 场景标签，可多标 |
| `source` | str | ✅ | template / llm / human / reused |
| `notes` | str | ⭕ | 备注 |

## 3. Intent Taxonomy v1（multi-label）

| 标签 | 含义 | 示例 |
|---|---|---|
| `casual_chat` | 普通闲聊 | "今天天气不错" |
| `emotional_expression` | 情绪表达 | "我最近特别焦虑" |
| `explicit_help_request` | 显式求助 | "有没有什么办法能让我平静下来？" |
| `information_request` | 信息请求 | "什么是正念呼吸？" |
| `resource_request` | 资源/内容请求 | "能推荐我一些缓解压力的内容吗？" |
| `feedback` | 推荐/建议反馈 | "那个冥想音频试了，对我没用" |
| `follow_up` | 对上一轮的追问 | "你刚才说的呼吸法具体怎么练？" |
| `memory_reference` | 引用历史/长期记忆 | "我上次跟你提过的那个项目黄了" |
| `high_risk_expression` | 高风险表达 | "活着真没意思" |
| `meta_question` | 关于 Agent 本身 | "你是 AI 吗？" |

> ⚠️ 与现有系统的区别：当前 `emotion_analyzer` 的 `user_intent` 字段值
> （sharing/seeking_relief/planning/seeking_help/venting）是**压力情境下的表达意图**，
> 不是本 Taxonomy 的 Agent 决策意图。两者在 runner 中需区分对待。

## 4. 情绪标签全集

BERT 5 类：`中性 / 快乐 / 焦虑 / 抑郁 / 愤怒`
扩展标签：`stress / fatigue / panic / fear / sadness / hopelessness / grief / guilt / shame / loneliness / relief / hope / calm / gratitude / neutral`

## 5. Agent Action Space v1

| 动作 | 含义 | 典型触发 |
|---|---|---|
| `continue_chat` | 继续共情对话 | 普通情绪表达 |
| `ask_clarification` | 澄清追问 | 信息不足、歧义 |
| `retrieve_memory` | 检索长期记忆 | 引用历史/个性化需要 |
| `recommend_resource` | 推荐资源/内容 | 显式求助、资源请求 |
| `information_response` | 信息性回复 | 信息请求、meta 问题 |
| `safety_intervention` | 安全干预 | 高风险（Level 2/3） |

## 6. Precondition（可选）

```json
{
  "user_profile": {
    "risk_level": "low",
    "preferred_support_style": "direct_actionable",
    "main_stress_sources": ["work_pressure"],
    "recommendation_feedback": {"c_001": "tried_ineffective"}
  },
  "memory": {
    "items": ["用户膝盖受伤，近期无法跑步"],
    "conflicts": [{"old": "用户喜欢跑步", "new": "膝盖受伤无法跑步"}]
  }
}
```

用于 memory_reference / 冲突消解 / 过期记忆 / 偏好更新等 case。

## 7. 校验

```python
from evaluation.benchmark_schema import BenchmarkCase
import json

with open("evaluation/datasets/agent_benchmark_v1.jsonl", encoding="utf-8") as f:
    for line in f:
        BenchmarkCase.model_validate(json.loads(line))
```

## 8. 与既有评测数据的换算关系

| 既有数据集 | 字段 | 本 Schema 对应 |
|---|---|---|
| `emotion_risk_turns.jsonl` | `expected.emotion_type` | `expected.emotion` |
| `emotion_risk_turns.jsonl` | `expected.risk_level`(str) | `expected.risk_level`(int, 经 normalize_risk) |
| `emotion_risk_turns.jsonl` | `expected.user_intent` | ⚠️ 语义不同（表达意图 vs 决策意图） |
| `recommendation_gate_cases.jsonl` | `expected.recommend_mode` | `expected.recommendation_action` |
| `end_to_end_dialogues.jsonl` | `turns[].expected_state` | `expected`（需拆解） |
| `rag_retrieval_cases.jsonl` | `expected_top_doc_ids` | 独立检索评测，不在本 Schema 内 |
