# -*- coding: utf-8 -*-
"""
Hybrid Policy Orchestrator（Phase 3 Task 3.9）。

分层优先级（docs/policy_invariants_v1.md §2）：
    P0 Safety Invariants   → 命中即返回，后续层不得覆盖
    P1 Deterministic Rules → confident 则返回
    P2 Learned Policy      → （可选，Deterministic 达标则关闭）
    P3 Ambiguity Gate      → 判定是否降级到 P4
    P4 LLM Fallback        → 仅 ambiguous case

每次决策记录：source / confidence / matched_rules / fallback_reason / policy_version。
"""
from __future__ import annotations

from typing import Optional

from policy.ambiguity import AmbiguityDetector, PolicyAmbiguity
from policy.engine import DeterministicDecision, PolicyConfig
from policy.llm_fallback import LLMFallbackPolicy
from policy.safety import SafetyPolicy
from policy.schema import PolicyResult
from policy.validator import policy_validator
from state.schema import AgentState

# P2 Learned（可选）：由 classifier.py 提供，默认关闭
try:
    from policy.classifier import learned_policy  # type: ignore
    _HAS_LEARNED = True
except ImportError:  # pragma: no cover
    _HAS_LEARNED = False


class HybridPolicy:
    """Phase 3 最终决策引擎：AgentState → PolicyResult。"""

    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self.safety = SafetyPolicy()
        self.deterministic = None  # lazy import（避免循环）
        self.ambiguity = AmbiguityDetector()
        self.llm_fallback = LLMFallbackPolicy(enabled=self.config.enable_llm_fallback)

    def _get_deterministic(self):
        if self.deterministic is None:
            from policy.deterministic import deterministic_policy
            self.deterministic = deterministic_policy
        return self.deterministic

    def decide(self, state: AgentState, *, use_llm: bool = True) -> PolicyResult:
        """主入口（同步）。"""
        if self.config.mode == "legacy":
            from policy.legacy_policy_adapter import legacy_policy_adapter
            return legacy_policy_adapter.decide(state)

        # ---- P0 Safety（最高优先级） ----
        if self.config.enable_safety_rules:
            safety_result = self.safety.decide(state)
            if safety_result is not None:
                # Safety 结果自身必须通过不变量校验
                policy_validator.validate_result(state, safety_result)
                return safety_result

        # ---- P1 Deterministic ----
        det = self._get_deterministic().decide(state)

        # ---- P3 Ambiguity Gate ----
        amb = self.ambiguity.detect(state, det)

        # 无歧义 → 直接返回 Deterministic
        if not amb.is_ambiguous and det.is_confident:
            return det.to_policy_result(source="rule")

        # ---- P2 Learned（可选，仅当启用且 Deterministic 不自信） ----
        if self.config.enable_learned_policy and _HAS_LEARNED and not amb.is_ambiguous:
            learned_result = self._learned_decide(state, det)
            if learned_result is not None:
                return learned_result

        # ---- P4 LLM Fallback ----
        if use_llm and self.config.enable_llm_fallback:
            llm_result = self.llm_fallback.decide(state, amb)
            if llm_result is not None:
                return llm_result

        # LLM 关闭/失败 → 回退 Deterministic（带 fallback_reason 标记）
        return det.to_policy_result(
            source="deterministic_fallback",
            fallback_reason="llm_disabled_or_failed",
        )

    def _learned_decide(self, state: AgentState, det: DeterministicDecision) -> Optional[PolicyResult]:
        """P2 Learned 层（仅当需要时实现，见 classifier.py）。"""
        learned_result = learned_policy.decide(state)  # type: ignore
        if learned_result is not None and learned_result.confidence >= self.config.learned_confidence_threshold:
            return learned_result
        return None

    # ---- 统计 / Trace ----

    def classify_decision(self, result: PolicyResult) -> str:
        """决策来源分类：safety | deterministic | learned | llm | deterministic_fallback。"""
        if result.source == "rule":
            return "safety" if "S0" in "".join(result.matched_rules) or \
                result.action_plan.primary_action.value == "safety_intervention" else "deterministic"
        return result.source


hybrid_policy = HybridPolicy()
