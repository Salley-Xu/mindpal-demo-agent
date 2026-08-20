# -*- coding: utf-8 -*-
"""
Recommendation Safety（Phase 6 Task 6.x）。

safety_only / 高风险时禁止普通推荐。消费 Phase 5 Risk v2 信号。
"""
from __future__ import annotations

from recommendation_v2.schema import CandidateFeatures
from state.schema import AgentState


class RecommendationSafety:
    """推荐安全审核。"""

    def is_blocked(self, state: AgentState, rec_mode: str) -> bool:
        """是否禁止普通推荐。"""
        # Policy 已输出 safety_only → 只允许安全资源
        if rec_mode == "safety_only":
            return True  # tool 层只放行 safety 标记内容
        if state.risk.level >= 2:
            return True  # 高风险禁普通推荐（INV-02）
        return False

    def item_allowed(self, feats: CandidateFeatures, risk_level: int) -> bool:
        """内容与风险兼容性。"""
        return feats.risk_compatibility >= 0.5 or risk_level < 2


recommendation_safety = RecommendationSafety()
