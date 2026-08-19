# -*- coding: utf-8 -*-
"""
Policy Shadow-mode 集成（Phase 3 Task 3.10）。

生产行为由 legacy 流程驱动；New Policy 只旁路计算 ActionPlan 并记录差异。
绝不抛出异常（与 Phase 2 Shadow 相同的"行为保持"原则）。

记录（Phase 3 §16）：
    legacy_action_plan      生产（legacy 适配）决策
    new_action_plan         New Policy 决策
    gold_action_plan        仅 benchmark 时有

差异分类：same / different / semantic_equivalent
    - "better/worse" 需要 gold 才能判定（benchmark 场景，见 run_policy_eval.py 的 diff 分类）

未通过 Benchmark 前禁止 cutover；可通过 config.agent_policy.mode 切换 legacy|shadow|hybrid。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from config import config
from policy.schema import ActionPlan
from state.schema import AgentState

logger = logging.getLogger(__name__)

_TRACE_LOG_PATH = os.path.join(getattr(config, "LOG_DIR", "logs"), "policy_shadow_trace.jsonl")


class PolicyShadowRunner:
    """旁路运行 New Policy 并记录与 legacy 的差异。"""

    def run(self, state: AgentState, *, legacy_plan: Optional[ActionPlan] = None) -> Optional[dict]:
        """返回 trace dict；任何异常静默返回 None（绝不干扰生产）。"""
        try:
            if state is None:
                return None
            from policy.hybrid import hybrid_policy
            new_result = hybrid_policy.decide(state, use_llm=False)

            if legacy_plan is None:
                from policy.legacy_policy_adapter import legacy_policy_adapter
                legacy_plan = legacy_policy_adapter.decide(state).action_plan

            diff = classify_diff(legacy_plan, new_result.action_plan)
            trace = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": state.identity.user_id,
                "session_id": state.identity.session_id,
                "turn_index": state.turn.turn_index,
                "trace_id": state.meta.trace_id,
                "legacy_action_plan": legacy_plan.model_dump(),
                "new_action_plan": new_result.action_plan.model_dump(),
                "diff": diff,
                "source": new_result.source,
                "confidence": round(new_result.confidence, 4),
                "matched_rules": new_result.matched_rules,
                "fallback_reason": new_result.fallback_reason,
                "state_version": state.meta.state_version,
            }
            _write_trace(trace)
            return trace
        except Exception as e:  # noqa: BLE001
            logger.warning("policy shadow 失败（不影响生产）: %s", e)
            return None


def classify_diff(legacy: ActionPlan, new: ActionPlan) -> str:
    """差异分类：same / different / semantic_equivalent。"""
    if legacy == new:
        return "same"
    # semantic_equivalent：primary 相同且 safety 方向一致，仅 tool/rec 细节差异
    if (legacy.primary_action == new.primary_action
            and legacy.safety_target == new.safety_target):
        return "semantic_equivalent"
    return "different"


def _write_trace(trace: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_TRACE_LOG_PATH), exist_ok=True)
        with open(_TRACE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
    except Exception as e:  # noqa: BLE001
        logger.warning("写入 policy shadow trace 失败: %s", e)


policy_shadow_runner = PolicyShadowRunner()
