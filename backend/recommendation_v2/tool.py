# -*- coding: utf-8 -*-
"""
RecommendationTool（Phase 6 Task 6.1 / 6.10 orchestrator）。

流程：Policy recommendation_mode → 候选检索 → Feature Ranking → Cooldown → Safety Review → 输出。
"""
from __future__ import annotations

from typing import List

from recommendation_v2.feedback import feedback_tracker
from recommendation_v2.ranker import feature_ranker
from recommendation_v2.safety import recommendation_safety
from recommendation_v2.schema import CandidateFeatures
from state.schema import AgentState


class RecommendationTool:
    """推荐工具（Phase 6 统一入口）。"""

    def recommend(self, state: AgentState, candidates: List[CandidateFeatures],
                  rec_mode: str = "none", top_k: int = 3) -> List[CandidateFeatures]:
        # 0. Policy 未授权（none）→ 不推荐
        if rec_mode in ("none",):
            return []

        # 1. Safety Review
        if recommendation_safety.is_blocked(state, rec_mode):
            return []

        # 2. 过滤被拒条目 + 不兼容内容 + cooldown 内条目
        candidates = [c for c in candidates
                      if not feedback_tracker.is_item_rejected(c.item_id)
                      and recommendation_safety.item_allowed(c, state.risk.level)
                      and not feature_ranker.in_cooldown(c.item_id)]
        if not candidates:
            return []

        # 3. Cooldown + Ranking
        scored = []
        feedback_weights = feedback_tracker.get_weights()
        for c in candidates:
            c = feature_ranker.apply_cooldown(c, state.turn.turn_index)
            score = feature_ranker.score(c, feedback_weights)
            scored.append((score, c))

        scored.sort(key=lambda x: -x[0])
        top = [c for _, c in scored[:top_k]]

        # 4. 记录推荐（cooldown）
        for c in top:
            feature_ranker.record_recommendation(c.item_id, c.category)
            feedback_tracker._seen.add(c.item_id)  # noqa: SLF001
        return top


recommendation_tool = RecommendationTool()
