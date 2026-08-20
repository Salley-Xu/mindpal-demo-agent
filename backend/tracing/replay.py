# -*- coding: utf-8 -*-
"""
Replay Runner（Phase 7 Task 7.6）。

trace → 重建 AgentState → 重跑 Policy / Tool 选择。
支持 same-version 与 new-version counterfactual replay。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from state.builder import build_agent_state
from tracing.logger import trace_logger
from tracing.schema import TurnTrace


class ReplayRunner:
    """从 trace 重放决策链路。"""

    def reconstruct_state(self, trace: TurnTrace):
        """从 trace.perception 重建 AgentState。"""
        p = trace.perception or {}
        return build_agent_state(
            user_id="u_replay",
            session_id=trace.session_id,
            request_text=(trace.input or {}).get("text", ""),
            turn_index=trace.turn_id,
            intent_result=p.get("intent_result"),
            emotion_state=p.get("emotion_state"),
            urgent_issue=p.get("urgent_issue"),
            conversation_summary=p.get("conversation_summary"),
            user_profile=p.get("user_profile"),
        )

    def replay_policy(self, trace: TurnTrace, policy_mode: str = "hybrid") -> dict:
        """重跑 Policy 决策（counterfactual 可选）。"""
        from policy.hybrid import hybrid_policy
        state = self.reconstruct_state(trace)
        hybrid_policy.config.mode = policy_mode
        result = hybrid_policy.decide(state, use_llm=False)
        return {
            "trace_id": trace.trace_id,
            "policy_mode": policy_mode,
            "action_plan": result.action_plan.model_dump(),
            "source": result.source,
            "matched_rules": result.matched_rules,
        }

    def compare_versions(self, trace: TurnTrace) -> Dict[str, dict]:
        """same-version vs 假设 legacy 的对比。"""
        same = self.replay_policy(trace, "hybrid")
        legacy = self.replay_policy(trace, "legacy")
        return {"same_version": same, "counterfactual_legacy": legacy}


replay_runner = ReplayRunner()
