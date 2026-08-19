# -*- coding: utf-8 -*-
"""
Safety Policy（Phase 3 Task 3.2，P0 硬安全层）。

最高优先级，命中即返回，后续层不得覆盖。

规则（对齐 docs/policy_invariants_v1.md §3 + docs/legacy_policy_audit.md R01/R02/R03/R05/R06）：

    S01  L3 紧急          → safety_intervention, self,      rec=none
    S02  第三方危机        → safety_intervention, third_party, rec=safety_only
    S03  L2 高风险        → safety_intervention, self,      rec=none
    S04  high_risk_intent 兜底 → safety_intervention, self/third_party, rec=safety_only(第三方)/none

INV-11 保护：discussion / safe_denial 不因安全规则强行升级。
    - safe_denial：用户明确否认 → 不触发 Safety（若 risk_evaluator 已据此降级，level 已低于 2）
    - discussion：讨论语境且非求助 → 不触发 Safety（由 Deterministic 正常处理）

设计依据（Frozen Benchmark v1.1 gold 分析）：
    - risk >= 2 ⟺ primary=safety_intervention（双向无例外，362 中 0 反例）
    - third_party 全部带 high_risk_expression 且文本含第三方主体词
    - L2 self 的 gold rec 以 none 为主（24/31），safety_only 仅用于第三方危机（9/10）
    - gold 中 high_risk_intent ⟹ risk>=2（0 反例）→ S04 兜底在 gold 上无 FPR
      （FPR 需在 Independent Safety Slice 上验证，见 C3.3）
"""
from __future__ import annotations

from typing import Optional

from policy.engine import build_policy_result
from policy.schema import ActionPlan, PolicyResult, PrimaryAction, RecommendationMode, SafetyTarget
from state.schema import AgentState


class SafetyPolicy:
    """P0 安全不变量层。"""

    def decide(self, state: AgentState) -> Optional[PolicyResult]:
        r = state.risk

        # INV-11：安全否认 → 不强行升级（交由低层正常处理）
        if r.safe_denial:
            return None

        # S02 第三方危机：第三方主体 + 高风险信号（level>=2 或 high_risk intent）
        if r.is_third_party:
            if state.derived.high_risk_intent_signal or r.level >= 2:
                return build_policy_result(
                    ActionPlan(
                        primary_action=PrimaryAction.SAFETY_INTERVENTION,
                        tool_actions=[],
                        safety_target=SafetyTarget.THIRD_PARTY,
                        recommendation_mode=RecommendationMode.SAFETY_ONLY,
                    ),
                    confidence=1.0,
                    matched_rules=["S02_third_party_crisis"],
                )
            # 第三方但非高风险（讨论/低危）→ 不触发 Safety
            return None

        # S01 L3 紧急
        if r.level >= 3:
            return build_policy_result(
                ActionPlan(
                    primary_action=PrimaryAction.SAFETY_INTERVENTION,
                    tool_actions=[],
                    safety_target=SafetyTarget.SELF,
                    recommendation_mode=RecommendationMode.NONE,
                ),
                confidence=1.0,
                matched_rules=["S01_emergency_L3"],
            )

        # S03 L2 高风险（gold：L2 self 的 rec 以 none 为主）
        if r.level == 2:
            return build_policy_result(
                ActionPlan(
                    primary_action=PrimaryAction.SAFETY_INTERVENTION,
                    tool_actions=[],
                    safety_target=SafetyTarget.SELF,
                    recommendation_mode=RecommendationMode.NONE,
                ),
                confidence=1.0,
                matched_rules=["S03_high_risk_L2"],
            )

        # S04 high_risk_intent 兜底（对齐 Policy Invariants §3：L2/L3 或 high-risk intent 均可触发）
        # 守卫：safe_denial / discussion 不强行升级（INV-11）
        if (state.derived.high_risk_intent_signal
                and not r.safe_denial
                and not r.is_discussion):
            if r.is_third_party:
                return build_policy_result(
                    ActionPlan(
                        primary_action=PrimaryAction.SAFETY_INTERVENTION,
                        tool_actions=[],
                        safety_target=SafetyTarget.THIRD_PARTY,
                        recommendation_mode=RecommendationMode.SAFETY_ONLY,
                    ),
                    confidence=0.9,
                    matched_rules=["S04_high_risk_intent_third_party"],
                )
            return build_policy_result(
                ActionPlan(
                    primary_action=PrimaryAction.SAFETY_INTERVENTION,
                    tool_actions=[],
                    safety_target=SafetyTarget.SELF,
                    recommendation_mode=RecommendationMode.NONE,
                ),
                confidence=0.9,
                matched_rules=["S04_high_risk_intent"],
            )

        return None


safety_policy = SafetyPolicy()
