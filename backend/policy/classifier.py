# -*- coding: utf-8 -*-
"""
Learned Policy（Phase 3 Task 3.7，P2 层，可选）。

决策准则（Phase 3 §29）：
    - 若 Deterministic Policy 满足推荐目标（Primary Macro F1 >= 0.90、Safety Recall 达标），
      跳过 Learned classifier，避免为"看起来高级"强行加复杂度。
    - 只有 Deterministic 明显不足时才启用，优先 Logistic Regression / LightGBM / 小 MLP。

输入必须是 AgentState structured features（不直接输入 raw conversation）。
本文件为可选层接口 + 禁用占位：默认 enable_learned_policy=false。
"""
from __future__ import annotations

from typing import Optional

from policy.schema import PolicyResult
from state.schema import AgentState


class LearnedPolicy:
    """P2 可学习层接口（默认禁用）。"""

    enabled: bool = False

    def decide(self, state: AgentState) -> Optional[PolicyResult]:
        """返回 PolicyResult 或 None（未命中 / 未启用）。"""
        if not self.enabled:
            return None
        # 预留：当 Phase 3.7 评估需要时在此接入训练好的分类器。
        # 特征建议（Phase 3 §13）：intent scores / confidence / emotion / risk level /
        #   risk trend / stage / help seeking / memory reference / rejection / turn count。
        return None


learned_policy = LearnedPolicy()
