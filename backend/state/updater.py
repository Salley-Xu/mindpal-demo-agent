# -*- coding: utf-8 -*-
"""
StateUpdater（Phase 2 Task 2.4 §20-23）。

职责：把上一轮 state 与当前轮 state 合并，应用确定性更新规则。
规则类型：REPLACE / APPEND / ROLLING_WINDOW / ACCUMULATE / DERIVE / RESET。
不负责业务决策。

更新规则（对齐 docs/agent_state_update_rules.md）：
  intent.labels               REPLACE（builder 已处理）
  conversation.recent_intents ROLLING_WINDOW(5)
  risk.level                  REPLACE
  risk.recent_risk_levels     ROLLING_WINDOW(5)
  risk.persistence            ACCUMULATE conditionally（L2+ → +1，否则 RESET 0）
  recommendation.turns_since  ACCUMULATE（本轮有推荐则 0，否则 +1）
  emotion.trend               DERIVE（由 summary 提供，builder 已处理）
  meta.state_version          ACCUMULATE
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from state.schema import AgentState

RISK_HISTORY_WINDOW = 5
RECENT_INTENTS_WINDOW = 5


def update_state(previous_state: Optional[AgentState], current_state: AgentState) -> AgentState:
    """合并上一轮与当前轮状态，返回更新后的 AgentState。"""
    state = current_state.model_copy(deep=True)

    # Risk persistence：条件累积
    if state.risk.level >= 2:
        state.risk.persistence = (previous_state.risk.persistence if previous_state else 0) + 1
    else:
        state.risk.persistence = 0

    # Risk history：滚动窗口
    prev_levels = list(previous_state.risk.recent_risk_levels) if previous_state else []
    state.risk.recent_risk_levels = (prev_levels + [state.risk.level])[-RISK_HISTORY_WINDOW:]

    # Recommendation turns_since_last：累积
    if state.recommendation.recent_recommendation_ids:
        state.recommendation.turns_since_last_recommendation = 0
    else:
        prev_turns = previous_state.recommendation.turns_since_last_recommendation if previous_state else None
        state.recommendation.turns_since_last_recommendation = (prev_turns or 0) + 1

    # 版本与时间
    state.meta.state_version = (previous_state.meta.state_version if previous_state else 0) + 1
    state.meta.updated_at = datetime.now(timezone.utc).isoformat()

    return state
