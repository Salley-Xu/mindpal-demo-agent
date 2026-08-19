# -*- coding: utf-8 -*-
"""
Policy Ambiguity（Phase 3 Task 3.6，P3 置信度/歧义门控层）。

只把真正难决定的 State 送给后续层（P4 LLM Fallback）。

触发条件（对齐 Phase 3 §11）：
    - no rule matched（确定性规则未命中）
    - conflicting rules（规则冲突）
    - intent uncertain（低置信 / open-set）
    - primary action tie（主动作置信过低）
    - tool action uncertainty
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from policy.engine import DeterministicDecision
from state.schema import AgentState


class PolicyAmbiguity(BaseModel):
    """歧义判定结果。"""
    is_ambiguous: bool = False
    reasons: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class AmbiguityDetector:
    """P3 歧义检测：决定 Deterministic 决策是否足够可信。"""

    def detect(self, state: AgentState, decision: DeterministicDecision) -> PolicyAmbiguity:
        reasons: List[str] = []

        # 1. 规则冲突 → 歧义
        if decision.conflicts:
            reasons.append("conflicting_rules")

        # 2. 规则未命中（无任何 matched rule 但仍有输出）→ 低置信
        if not decision.matched_rules:
            reasons.append("no_rule_matched")

        # 3. 主动作置信过低 → 歧义
        if decision.confidence < 0.72:
            reasons.append("primary_action_low_confidence")

        # 4. 意图不确定（open-set / 低置信 / 无标签）→ 歧义
        if state.derived.intent_uncertain:
            reasons.append("intent_uncertain")
        if state.intent.is_open_set:
            reasons.append("intent_open_set")

        # 5. 信号冲突：多意图且无主导（多个互斥 primary 信号同时出现）
        primary_signals = [
            s for s in ["information_request", "meta_question"] if s in state.intent.labels
        ]
        if len(primary_signals) > 1:
            reasons.append("primary_action_tie")

        is_ambiguous = len(reasons) > 0
        return PolicyAmbiguity(
            is_ambiguous=is_ambiguous,
            reasons=reasons,
            confidence=decision.confidence,
        )


ambiguity_detector = AmbiguityDetector()
