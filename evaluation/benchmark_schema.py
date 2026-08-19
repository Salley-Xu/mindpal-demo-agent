# -*- coding: utf-8 -*-
"""
Agent Benchmark Schema v1.1 — MindPal Stateful Adaptive Agent 评测集 Schema。

v1.1 相对 v1 的关键修正（对齐 Phase 0.5 执行文档 §4）：
  1. Action 语义拆分：
       AgentAction(v1, 混合)  →  PrimaryAction(最终响应) + ToolAction(内部工具) + SafetyTarget
  2. recommendation_action 移除 third_party_support（改由 SafetyTarget.third_party 表达）
  3. should_retrieve_knowledge 并入 tool_actions: ["retrieve_knowledge"]
  4. 提供 migrate_v1_case_to_v1_1 迁移 helper，v1 数据可自动迁移

数据结构用于：
  1. 校验评测数据集
  2. Evaluation Runner 读取与归一化预测结果
  3. Phase 1 Intent Taxonomy 标签来源
  4. Phase 3 Agent Policy 的 ActionPlan 设计基础

对齐文档：
  - MindPal_Agent_开发任务计划_v1.0.md §5.3 / §6.2 / §7.2 / §8.2
  - MindPal_Agent_Phase0.5_执行文档.md §4
"""
from __future__ import annotations

import json
import copy
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """对话角色。"""
    USER = "user"
    ASSISTANT = "assistant"


class IntentLabel(str, Enum):
    """
    Intent Taxonomy v1（支持 multi-label）。
    对齐开发计划 §6.2。
    """
    CASUAL_CHAT = "casual_chat"                       # 普通闲聊
    EMOTIONAL_EXPRESSION = "emotional_expression"     # 情绪表达
    EXPLICIT_HELP_REQUEST = "explicit_help_request"   # 显式求助
    INFORMATION_REQUEST = "information_request"       # 信息请求
    RESOURCE_REQUEST = "resource_request"             # 资源/内容请求
    FEEDBACK = "feedback"                             # 推荐/建议反馈
    FOLLOW_UP = "follow_up"                           # 对上一轮的追问
    MEMORY_REFERENCE = "memory_reference"             # 引用历史/长期记忆
    HIGH_RISK_EXPRESSION = "high_risk_expression"     # 高风险表达（自伤等）
    META_QUESTION = "meta_question"                   # 关于 Agent 本身的问题


# 已废弃（v1）。v1.1 拆分为 PrimaryAction + ToolAction。
# 保留仅为迁移与兼容，不得在新数据中使用。
class AgentAction(str, Enum):
    """[DEPRECATED v1] 混合语义的 Action。"""
    CONTINUE_CHAT = "continue_chat"
    ASK_CLARIFICATION = "ask_clarification"
    RETRIEVE_MEMORY = "retrieve_memory"
    RECOMMEND_RESOURCE = "recommend_resource"
    INFORMATION_RESPONSE = "information_response"
    SAFETY_INTERVENTION = "safety_intervention"


class PrimaryAction(str, Enum):
    """
    最终响应动作（Agent 本轮的主要响应类型）。
    对齐 Phase 0.5 §4.2。
    """
    CONTINUE_CHAT = "continue_chat"                    # 继续对话（共情回复）
    ASK_CLARIFICATION = "ask_clarification"            # 澄清追问
    INFORMATION_RESPONSE = "information_response"      # 信息性回复
    SAFETY_INTERVENTION = "safety_intervention"        # 安全干预（高风险路由）


class ToolAction(str, Enum):
    """
    内部 Tool / Capability 动作（可多选，与 primary_action 正交）。
    对齐 Phase 0.5 §4.2。
    """
    RETRIEVE_MEMORY = "retrieve_memory"                # 检索长期记忆
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"          # 检索专业知识库（RAG）
    RECOMMEND_RESOURCE = "recommend_resource"          # 推荐资源/内容


class RecommendationAction(str, Enum):
    """推荐动作（对齐 RecommendGate 输出 hard/soft/none/safety_only）。"""
    HARD = "hard"
    SOFT = "soft"
    NONE = "none"
    SAFETY_ONLY = "safety_only"


class SafetyTarget(str, Enum):
    """安全干预的对象（对齐 Phase 0.5 §4.3）。"""
    NONE = "none"                # 非安全场景
    SELF = "self"                # 用户自身高风险
    THIRD_PARTY = "third_party"  # 第三方（朋友/家人）高风险


class RiskTrend(str, Enum):
    """风险趋势（对齐 SessionRiskAggregator / 开发计划 §10.3）。"""
    RISING = "rising"
    STABLE = "stable"
    FALLING = "falling"
    FLUCTUATING = "fluctuating"
    NEW = "new"            # 首轮无历史


class RiskLevel(int, Enum):
    """四级风险等级（对齐 risk_levels.py LEVEL_0 ~ LEVEL_3）。"""
    LEVEL_0 = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3


class ConversationStage(str, Enum):
    """对话阶段（对齐 agent_prompts.py STRATEGY_PROMPTS）。"""
    INITIAL = "initial"
    EXPLORING = "exploring"
    DEEPENING = "deepening"
    RESOLVING = "resolving"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    """一轮对话（benchmark 的输入）。"""
    role: Role
    content: str
    # 仅 feedback 类 case 使用：本轮之前被推荐的内容 id
    recommendation_ids: List[str] = Field(default_factory=list)


class UserProfilePrecondition(BaseModel):
    """case 的前置画像（可选，模拟用户长期画像）。"""
    user_id: Optional[str] = None
    risk_level: Optional[str] = "low"
    preferred_support_style: Optional[str] = None
    avoid_style: List[str] = Field(default_factory=list)
    main_stress_sources: List[str] = Field(default_factory=list)
    recommendation_feedback: dict = Field(default_factory=dict)


class MemoryPrecondition(BaseModel):
    """
    case 的前置长期记忆（可选）。
    用于 memory_reference / 冲突 / 过期记忆 等 case。
    每条是一个"用户已经告诉过 Agent 的事实"。
    """
    items: List[str] = Field(default_factory=list)
    # 可选：已知的偏好冲突对（旧偏好 vs 新事实）
    conflicts: List[dict] = Field(default_factory=list)


class Precondition(BaseModel):
    """case 前置条件容器（可选字段，用于多轮/记忆/画像类 case）。"""
    user_profile: Optional[UserProfilePrecondition] = None
    memory: Optional[MemoryPrecondition] = None
    recent_turns: List[ConversationTurn] = Field(default_factory=list)


class ExpectedOutcome(BaseModel):
    """
    Benchmark 期望输出 v1.1（结构化）。
    对齐 Phase 0.5 §4.2。
    """
    # --- 感知层 ---
    intent: List[IntentLabel] = Field(default_factory=list)
    emotion: Optional[str] = None                 # 归一化情绪标签（见 EMOTION_LABELS）
    emotion_intensity: Optional[float] = Field(None, ge=0.0, le=1.0)

    # --- 风险层 ---
    risk_level: RiskLevel = RiskLevel.LEVEL_0
    risk_trend: RiskTrend = RiskTrend.NEW

    # --- 记忆层 ---
    memory_needed: bool = False                   # 是否应检索记忆
    memory_refs: List[str] = Field(default_factory=list)  # 应被召回的记忆要点（模糊匹配）

    # --- 推荐层 ---
    recommendation_action: RecommendationAction = RecommendationAction.NONE

    # --- 策略层（v1.1 拆分）---
    primary_action: PrimaryAction = PrimaryAction.CONTINUE_CHAT
    tool_actions: List[ToolAction] = Field(default_factory=list)   # 可多选
    safety_target: SafetyTarget = SafetyTarget.NONE

    # --- 会话层 ---
    conversation_stage: Optional[ConversationStage] = None


class BenchmarkCase(BaseModel):
    """一条完整 Agent Benchmark case。"""
    case_id: str
    conversation: List[ConversationTurn]                 # 多轮对话（>=1 轮）
    expected: ExpectedOutcome
    precondition: Optional[Precondition] = None
    tags: List[str] = Field(default_factory=list)        # 场景标签（multi-tag）
    source: str = "template"                             # template | llm | human | reused
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# v1 → v1.1 迁移 helper（对齐 Phase 0.5 §4.5）
# ---------------------------------------------------------------------------

# 旧 AgentAction -> (primary_action, [tool_actions])
_AGENT_ACTION_TO_V1_1 = {
    "continue_chat": ("continue_chat", []),
    "ask_clarification": ("ask_clarification", []),
    "information_response": ("information_response", []),
    "safety_intervention": ("safety_intervention", []),
    "retrieve_memory": ("continue_chat", ["retrieve_memory"]),
    "recommend_resource": ("continue_chat", ["recommend_resource"]),
}


def migrate_v1_case_to_v1_1(case_dict: dict) -> dict:
    """
    把 v1 case（含 agent_action / should_retrieve_knowledge / third_party_support）迁移到 v1.1。

    - 不修改输入；返回新 dict。
    - v1.1 数据不应再包含 agent_action / should_retrieve_knowledge。
    """
    new = copy.deepcopy(case_dict)
    exp = new.get("expected", {})
    if not isinstance(exp, dict):
        raise ValueError("expected 必须是 dict")

    agent_action = exp.pop("agent_action", "continue_chat")
    retrieve_knowledge = bool(exp.pop("should_retrieve_knowledge", False))

    primary, tools = _AGENT_ACTION_TO_V1_1.get(str(agent_action), ("continue_chat", []))
    tools = list(tools)
    if retrieve_knowledge and "retrieve_knowledge" not in tools:
        tools.append("retrieve_knowledge")

    # safety target
    rec_action = str(exp.get("recommendation_action", "none"))
    if rec_action == "third_party_support":
        exp["recommendation_action"] = "safety_only"
        safety_target = "third_party"
    elif primary == "safety_intervention":
        safety_target = "self"
    else:
        safety_target = "none"

    exp["primary_action"] = primary
    exp["tool_actions"] = tools
    exp["safety_target"] = safety_target
    return new


def migrate_v1_cases_to_v1_1(cases: List[dict]) -> List[dict]:
    """批量迁移 v1 cases。"""
    return [migrate_v1_case_to_v1_1(c) for c in cases]


# ---------------------------------------------------------------------------
# 常量（供数据集生成与 runner 复用）
# ---------------------------------------------------------------------------

# 归一化情绪标签全集（规范英文集，对齐 BERT 情绪 5 类 + LLM 常见输出）
# 备注：系统 emotion_analyzer 返回中文标签（中性/快乐/焦虑/抑郁/愤怒 等），
# runner 中经 _normalize_emotion 映射到本规范集再比对。
EMOTION_LABELS = [
    "neutral", "happy", "anxiety", "sadness", "anger",
    "stress", "fatigue", "panic", "fear", "hopelessness",
    "grief", "guilt", "shame", "loneliness",
    "relief", "hope", "calm", "gratitude",
]

# 场景标签（对齐开发计划 §5.4 数据分布建议）
SCENE_TAGS = [
    "single_turn", "multi_turn",
    "casual_chat", "emotional", "explicit_help", "information",
    "memory_reference", "recommendation", "recommendation_feedback",
    "low_risk", "medium_risk", "high_risk",
    "ambiguous_intent", "multi_label", "long_multi_turn",
    "knowledge_rag",
]

# 风险等级字符串 <-> 数字映射（兼容 normalize_risk_level 的别名）
RISK_LEVEL_STR = {
    "level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3,
    "low": 0, "medium": 2, "high": 3,
    "none": 0, "moderate": 1,
}

RISK_LEVEL_INT_TO_STR = {
    0: "level_0", 1: "level_1", 2: "level_2", 3: "level_3",
}


def normalize_risk(value) -> int:
    """把字符串/数字风险等级归一化为 0-3 整数。"""
    if isinstance(value, bool):
        return int(value)  # 防御：bool 是 int 子类
    if isinstance(value, int):
        return max(0, min(int(value), 3))
    if isinstance(value, RiskLevel):
        return int(value)
    key = str(value).strip().lower()
    return RISK_LEVEL_STR.get(key, 0)


def case_to_dict(case: BenchmarkCase) -> dict:
    """序列化（供 JSONL 写入）。"""
    return json.loads(case.model_dump_json())
