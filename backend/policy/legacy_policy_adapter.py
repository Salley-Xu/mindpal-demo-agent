# -*- coding: utf-8 -*-
"""
LegacyPolicyAdapter（Phase 3 Preflight P3-0.4）。

把当前系统 State→Action 行为编码为 ActionPlan（只解释，不修改当前行为）。
作为 Phase 3 Deterministic Policy 的规则依据和 Current Policy Baseline。
"""
from __future__ import annotations

from typing import Dict

from policy.schema import (
    ActionPlan,
    PolicyResult,
    PrimaryAction,
    RecommendationMode,
    SafetyTarget,
    ToolAction,
)
from state.schema import AgentState


class LegacyPolicyAdapter:
    """编码当前 legacy 路由行为。"""

    def decide(self, state: AgentState) -> PolicyResult:
        matched = []
        # R01: L3 紧急 → safety self + none
        if state.risk.level >= 3:
            matched.append("R01_emergency_L3")
            return self._safety(SafetyTarget.SELF, RecommendationMode.NONE, matched)
        # R02: 第三方危机求助 → safety third_party + safety_only
        if state.risk.is_third_party and state.derived.high_risk_intent_signal:
            matched.append("R02_third_party_crisis")
            return self._safety(SafetyTarget.THIRD_PARTY, RecommendationMode.SAFETY_ONLY, matched)
        # R03: L2 高风险 → safety self + safety_only
        if state.risk.level == 2:
            matched.append("R03_high_risk_L2")
            return self._safety(SafetyTarget.SELF, RecommendationMode.SAFETY_ONLY, matched)

        # 正常路由（ReAct）：R07/R08/R09 legacy 每轮检索 memory + knowledge
        matched.append("R07_normal_react")
        tools = []
        if not state.risk.is_discussion:
            tools.append(ToolAction.RETRIEVE_MEMORY)
            tools.append(ToolAction.RETRIEVE_KNOWLEDGE)
            matched.append("R08_legacy_always_memory")
            matched.append("R09_legacy_always_knowledge")

        # R04: RecommendGate 简化映射（从 state 派生 legacy gate 倾向）
        rec_mode = self._legacy_gate_mode(state)
        if rec_mode == RecommendationMode.HARD:
            tools.append(ToolAction.RECOMMEND_RESOURCE)
            matched.append("R04_gate_hard")

        return PolicyResult(
            action_plan=ActionPlan(
                primary_action=PrimaryAction.CONTINUE_CHAT,
                tool_actions=tools,
                safety_target=SafetyTarget.NONE,
                recommendation_mode=rec_mode,
            ),
            confidence=0.9,
            source="legacy_adapter",
            matched_rules=matched,
            policy_version="1.0",
        )

    def _legacy_gate_mode(self, state: AgentState) -> RecommendationMode:
        """从 AgentState 派生 legacy recommend_gate 倾向（简化加权）。"""
        if state.risk.level >= 3:
            return RecommendationMode.NONE
        if state.risk.level == 2:
            return RecommendationMode.SAFETY_ONLY
        score = 0.0
        score += 0.30 * (state.emotion.intensity or 0.5)
        if state.derived.help_seeking:
            score += 0.23 * 0.75
        elif state.derived.resource_seeking:
            score += 0.23 * 0.55
        if state.derived.negative_emotion:
            score += 0.14 * 0.35
        if state.recommendation.rejection_detected:
            score -= 0.25
        if score >= 0.58:
            return RecommendationMode.HARD
        if score >= 0.20:
            return RecommendationMode.SOFT
        return RecommendationMode.NONE

    def _safety(self, target: SafetyTarget, rec: RecommendationMode, matched) -> PolicyResult:
        return PolicyResult(
            action_plan=ActionPlan(
                primary_action=PrimaryAction.SAFETY_INTERVENTION,
                tool_actions=[],
                safety_target=target,
                recommendation_mode=rec,
            ),
            confidence=1.0,
            source="legacy_adapter",
            matched_rules=matched,
            policy_version="1.0",
        )


legacy_policy_adapter = LegacyPolicyAdapter()
