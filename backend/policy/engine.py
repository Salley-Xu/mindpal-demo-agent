# -*- coding: utf-8 -*-
"""
Policy Core（Phase 3 Task 3.1）。

分层决策引擎骨架：
    P0 Safety Invariants
    P1 Deterministic Rules
    P2 Learned Policy（可选）
    P3 Confidence / Ambiguity Gate
    P4 LLM Fallback

原则：高优先级决策不可被低优先级覆盖（见 docs/policy_invariants_v1.md）。
本模块只提供：配置、层协议、决策容器、共享 helper。
实际的层编排在 hybrid.py（Task 3.9）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from policy.schema import ActionPlan, PolicyResult
from state.schema import AgentState


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    """Policy 运行配置（对齐 Phase 3 §24）。"""
    version: str = "1.0"
    mode: str = "shadow"                    # legacy | shadow | hybrid

    enable_safety_rules: bool = True
    enable_deterministic: bool = True
    enable_learned_policy: bool = False
    enable_llm_fallback: bool = True

    deterministic_confidence_threshold: float = 0.85
    learned_confidence_threshold: float = 0.80
    max_llm_fallback_rate: float = 0.20

    # 意图不确定性阈值（置信度低于该值视为 uncertain）
    intent_uncertainty_threshold: float = 0.55


# ---------------------------------------------------------------------------
# 层协议
# ---------------------------------------------------------------------------

class PolicyLayer(Protocol):
    """策略层协议：decide 返回 PolicyResult 或 None（未命中）。"""

    def decide(self, state: AgentState) -> Optional[PolicyResult]: ...


# ---------------------------------------------------------------------------
# 决策容器（Deterministic 层输出）
# ---------------------------------------------------------------------------

class DeterministicDecision:
    """
    Deterministic Policy 的决策容器（可 Trace）。

    与 PolicyResult 不同：携带 `is_confident` 供 Ambiguity Gate 判断是否
    需要降级到 P3/P4。matched_rules 记录命中的规则 ID。
    """

    __slots__ = ("action_plan", "confidence", "matched_rules", "reasons", "conflicts")

    def __init__(
        self,
        action_plan: ActionPlan,
        confidence: float = 0.0,
        matched_rules: Optional[List[str]] = None,
        reasons: Optional[List[str]] = None,
        conflicts: Optional[List[str]] = None,
    ):
        self.action_plan = action_plan
        self.confidence = float(confidence)
        self.matched_rules = matched_rules or []
        self.reasons = reasons or []
        self.conflicts = conflicts or []

    @property
    def is_confident(self) -> bool:
        return self.confidence >= _DET_CONF_THRESHOLD and not self.conflicts

    def to_policy_result(self, *, source: str = "rule",
                         fallback_reason: Optional[str] = None) -> PolicyResult:
        return PolicyResult(
            action_plan=self.action_plan,
            confidence=self.confidence,
            source=source,
            matched_rules=list(self.matched_rules),
            fallback_reason=fallback_reason,
            policy_version="1.0",
        )


# 全局阈值（可被 PolicyConfig 覆盖；Deterministic 层直接使用）
_DET_CONF_THRESHOLD = 0.85


def set_confidence_threshold(value: float) -> None:
    global _DET_CONF_THRESHOLD
    _DET_CONF_THRESHOLD = float(value)


# ---------------------------------------------------------------------------
# 共享 helper
# ---------------------------------------------------------------------------

def build_policy_result(
    action_plan: ActionPlan,
    *,
    confidence: float = 1.0,
    source: str = "rule",
    matched_rules: Optional[List[str]] = None,
    fallback_reason: Optional[str] = None,
) -> PolicyResult:
    """构造 PolicyResult（统一版本号）。"""
    return PolicyResult(
        action_plan=action_plan,
        confidence=confidence,
        source=source,
        matched_rules=matched_rules or [],
        fallback_reason=fallback_reason,
        policy_version="1.0",
    )
