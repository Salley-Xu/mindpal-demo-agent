# -*- coding: utf-8 -*-
"""
DynamicRiskState（Phase 5 Task 5.6，RiskResult v2 核心）。

单轮 Risk 模型负责 current risk；多轮状态机负责 persistence / trend / escalation / recovery。

输入：单轮预测（level, confidence, probs）+ 会话历史 + 上下文分析
输出：DynamicRiskState（可 Trace，直接对齐 AgentState.risk 消费）

对齐 Phase 5 §9：
    level / confidence / trend / persistence / escalation / subject / discussion /
    safe_denial / recent_levels / reasons
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DynamicRiskState(BaseModel):
    """RiskResult v2：动态风险状态。"""
    level: int = 0
    confidence: float = 0.0
    trend: str = "new"                     # rising | stable | falling | fluctuating | new
    persistence: int = 0                   # 连续满足 L2+ 的 turn 数
    escalation: bool = False               # 相对上轮是否升级
    subject: str = "self"                  # self | third_party | discussion
    discussion: bool = False
    safe_denial: bool = False
    recent_levels: List[int] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    source: str = "dynamic_v2"

    @property
    def is_high_risk(self) -> bool:
        return self.level >= 2


class DynamicRiskTracker:
    """
    多轮风险跟踪器：逐轮更新 DynamicRiskState。

    委托 SessionRiskAggregator v5.0 做聚合（惯性/连续升级/安全确认/趋势），
    额外补充 subject / discussion / safe_denial / persistence / escalation。
    """

    WINDOW_SIZE = 5

    def __init__(self, baseline: str = "low"):
        self.baseline = baseline
        self._history: List[int] = []

    # ---- 更新 ----

    def update(
        self,
        *,
        level: int,
        confidence: float = 0.0,
        probs: Optional[List[float]] = None,
        text: str = "",
        context: Optional[dict] = None,
    ) -> DynamicRiskState:
        """处理一轮单字面预测，返回更新后的动态状态。"""
        context = context or {}
        self._history.append(level)
        self._history = self._history[-self.WINDOW_SIZE:]

        # 聚合（复用 v5.0 状态机）
        from session_risk_aggregator import SessionRiskAggregator
        summary = {"recent_risk_levels": [f"level_{l}" for l in self._history[:-1]]}
        agg = SessionRiskAggregator().aggregate(
            utterance_level=f"level_{level}",
            utterance_binary_prob=confidence,
            utterance_rule_hit=False,
            text=text,
            conversation_summary=summary,
            baseline=self.baseline,
        )
        session_level = int(str(agg["session_level"]).split("_")[1])

        # persistence：连续 L2+ 计数
        persistence = 0
        for l in reversed(self._history):
            if l >= 2:
                persistence += 1
            else:
                break

        # escalation：相对上一轮
        prev = self._history[-2] if len(self._history) >= 2 else level
        escalation = session_level > prev

        # 上下文（subject / discussion / safe_denial）
        subject = context.get("subject", "self")
        discussion = bool(context.get("is_discussion_context", False))
        safe_denial = bool(context.get("is_safe_denial", False))

        state = DynamicRiskState(
            level=session_level,
            confidence=round(confidence, 4),
            trend=agg["risk_trend"],
            persistence=persistence,
            escalation=escalation,
            subject=subject,
            discussion=discussion,
            safe_denial=safe_denial,
            recent_levels=list(self._history),
            reasons=list(agg["session_aggregation"].get("active_rules", [])),
        )
        return state

    # ---- 生命周期 ----

    def reset(self) -> None:
        self._history = []

    @property
    def history(self) -> List[int]:
        return list(self._history)


dynamic_risk_tracker = DynamicRiskTracker()
