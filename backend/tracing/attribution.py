# -*- coding: utf-8 -*-
"""
Error Attribution（Phase 7 Task 7.7）。

每个失败 case 定位到最早错误层。
"""
from __future__ import annotations

from typing import Dict, List

from tracing.schema import AttributionResult, TurnTrace

# 各层误差信号检查器（规则启发式）
def _perception_error(trace: TurnTrace) -> str:
    """检查 perception 层内部不一致。"""
    p = trace.perception or {}
    risk = p.get("urgent_issue", {})
    intent = p.get("intent_result", {})
    # 高风险意图但 risk 极低 → 感知不一致（risk 或 intent 其一错）
    if "high_risk_expression" in (intent.get("labels") or []) and str(risk.get("level", "level_0")) == "level_0":
        return "perception_intent_risk_conflict"
    return ""


def _policy_error(trace: TurnTrace) -> str:
    pol = trace.policy or {}
    state = trace.state or {}
    risk_level = (state.get("risk") or {}).get("level", 0)
    primary = (pol.get("action_plan") or {}).get("primary_action", "")
    if risk_level >= 2 and primary != "safety_intervention":
        return "policy_safety_miss"
    return ""


def _safety_error(trace: TurnTrace) -> str:
    rec = trace.recommendation or {}
    risk = (trace.state or {}).get("risk", {})
    mode = rec.get("mode", "")
    if risk.get("level", 0) >= 2 and mode in ("hard", "soft"):
        return "safety_rec_violation"
    return ""


class ErrorAttributor:
    """错误归因：从最早错误层开始定位。"""

    LAYERS = ["perception", "state", "policy", "memory", "recommendation", "execution", "llm", "safety"]

    def attribute(self, trace: TurnTrace) -> AttributionResult:
        checks = [
            ("perception", _perception_error(trace)),
            ("policy", _policy_error(trace)),
            ("safety", _safety_error(trace)),
        ]
        for layer, reason in checks:
            if reason:
                return AttributionResult(trace_id=trace.trace_id,
                                         first_error_layer=layer, reason=reason)
        return AttributionResult(trace_id=trace.trace_id, first_error_layer="none",
                                 reason="no_error_detected")


error_attributor = ErrorAttributor()
