# -*- coding: utf-8 -*-
"""
Policy Validator（Phase 3 Task 3.9 附属，对齐 docs/policy_invariants_v1.md §4）。

每个 PolicyResult 必须通过：
    1. Schema validation（枚举合法）
    2. Policy invariant validation（INV-01 ~ INV-11）
    3. Safety override validation（LLM / 低层结果不得覆盖 Safety）

返回 ValidationResult：valid + 违规不变量列表。违规记录同时可作为
Safety invariant = 100% 的评测依据。
"""
from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, Field

from policy.schema import ActionPlan, PolicyResult, PrimaryAction, RecommendationMode, SafetyTarget
from state.schema import AgentState


class ValidationResult(BaseModel):
    valid: bool = True
    violated: List[str] = Field(default_factory=list)   # 违规的 INV ID
    messages: List[str] = Field(default_factory=list)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        self.valid = self.valid and other.valid
        self.violated += other.violated
        self.messages += other.messages
        return self


class PolicyValidator:
    """不变量校验器。"""

    def validate_plan(self, plan: ActionPlan) -> ValidationResult:
        """Schema 校验 + 纯 ActionPlan 内部不变量（INV-04/05/06/07/08）。"""
        result = ValidationResult()
        pa = plan.primary_action
        tools = plan.tool_actions
        rec = plan.recommendation_mode
        st = plan.safety_target

        # INV-04 Primary 互斥：enum 单值天然互斥（无需检查）
        # INV-05 Tool 多选：list 天然支持（无需检查）
        # INV-06/07/08 正交性：schema 分离字段天然满足（无需检查）

        # INV-03 Safety-only 不允许普通推荐
        if rec == RecommendationMode.SAFETY_ONLY and pa != PrimaryAction.SAFETY_INTERVENTION:
            result.valid = False
            result.violated.append("INV-03")
            result.messages.append("safety_only rec 但 primary 非 safety_intervention")

        # INV-01/02 链：safety_intervention 必须 safety_target != none
        if pa == PrimaryAction.SAFETY_INTERVENTION and st == SafetyTarget.NONE:
            result.valid = False
            result.violated.append("INV-01/02")
            result.messages.append("safety_intervention 缺少 safety_target")

        # INV-10 third-party target 正确：safety_target=third_party 必须 primary=safety_intervention
        if st == SafetyTarget.THIRD_PARTY and pa != PrimaryAction.SAFETY_INTERVENTION:
            result.valid = False
            result.violated.append("INV-10")
            result.messages.append("third_party target 但 primary 非 safety_intervention")

        return result

    def validate_result(self, state: AgentState, result: PolicyResult) -> ValidationResult:
        """完整校验：Schema + 不变量 + Safety override。"""
        out = self.validate_plan(result.action_plan)

        # INV-01/02 Safety > Normal：risk>=2 时必须 safety
        if state.risk.level >= 2 and not state.risk.safe_denial:
            if result.action_plan.primary_action != PrimaryAction.SAFETY_INTERVENTION:
                out.valid = False
                out.violated.append("INV-01")
                out.messages.append(f"risk={state.risk.level} 但 primary={result.action_plan.primary_action.value}")

        # INV-11 discussion / safe_denial 不强行升级
        if state.risk.safe_denial and result.action_plan.primary_action == PrimaryAction.SAFETY_INTERVENTION:
            out.valid = False
            out.violated.append("INV-11")
            out.messages.append("safe_denial 被强行升级为 safety_intervention")

        # INV-10 第三方正确性
        if state.risk.is_third_party and result.action_plan.primary_action == PrimaryAction.SAFETY_INTERVENTION:
            if result.action_plan.safety_target != SafetyTarget.THIRD_PARTY:
                out.valid = False
                out.violated.append("INV-10")
                out.messages.append("third_party risk 但 safety_target 非 third_party")

        return out


policy_validator = PolicyValidator()
