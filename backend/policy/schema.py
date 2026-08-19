# -*- coding: utf-8 -*-
"""
Policy Contract v1（Phase 3 Preflight P3-0.3，冻结）。

输入：AgentState（唯一正式输入，禁止绕过）
输出：ActionPlan + PolicyResult

对齐 Benchmark Schema 的 ActionPlan 冻结接口。
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class PrimaryAction(str, Enum):
    CONTINUE_CHAT = "continue_chat"
    ASK_CLARIFICATION = "ask_clarification"
    INFORMATION_RESPONSE = "information_response"
    SAFETY_INTERVENTION = "safety_intervention"


class ToolAction(str, Enum):
    RETRIEVE_MEMORY = "retrieve_memory"
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
    RECOMMEND_RESOURCE = "recommend_resource"


class SafetyTarget(str, Enum):
    NONE = "none"
    SELF = "self"
    THIRD_PARTY = "third_party"


class RecommendationMode(str, Enum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"
    SAFETY_ONLY = "safety_only"


class ActionPlan(BaseModel):
    """Policy 输出（冻结）。"""
    primary_action: PrimaryAction
    tool_actions: List[ToolAction] = Field(default_factory=list)
    safety_target: SafetyTarget = SafetyTarget.NONE
    recommendation_mode: RecommendationMode = RecommendationMode.NONE


class PolicyResult(BaseModel):
    """Policy 决策运行时包装（可 Trace）。"""
    action_plan: ActionPlan
    confidence: float = 0.0
    source: str = "rule"                       # rule | classifier | llm | hybrid
    matched_rules: List[str] = Field(default_factory=list)
    fallback_reason: Optional[str] = None
    policy_version: str = "1.0"

    @property
    def primary_action(self) -> str:
        return self.action_plan.primary_action.value

    @property
    def tool_actions(self) -> List[str]:
        return [t.value for t in self.action_plan.tool_actions]

    @property
    def safety_target(self) -> str:
        return self.action_plan.safety_target.value

    @property
    def recommendation_mode(self) -> str:
        return self.action_plan.recommendation_mode.value
