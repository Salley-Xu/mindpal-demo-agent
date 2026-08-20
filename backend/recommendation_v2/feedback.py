# -*- coding: utf-8 -*-
"""
Feedback Closed-loop（Phase 6 Task 6.5）。

反馈必须直接影响未来 ranking（不仅影响 prompt）。
"""
from __future__ import annotations

from typing import Dict

from recommendation_v2.schema import FeedbackEvent, FeedbackWeights, FeedbackType


class FeedbackTracker:
    """反馈 → 类别/条目权重更新。"""

    def __init__(self):
        self._weights: Dict[str, FeedbackWeights] = {}
        self._seen: set = set()          # 已推荐过的 item_id
        self._rejected_items: set = set()

    def apply(self, event: FeedbackEvent) -> FeedbackWeights:
        """应用反馈，返回该类别更新后的权重。"""
        cat = event.item_category or "default"
        w = self._weights.get(cat, FeedbackWeights())
        self._seen.add(event.item_id)

        if event.feedback in (FeedbackType.ACCEPT, FeedbackType.TRIED_EFFECTIVE):
            w.category_weight = min(2.0, w.category_weight + 0.2)
            w.effective_boost = min(0.5, w.effective_boost + 0.1)
        elif event.feedback in (FeedbackType.REJECT, FeedbackType.NOT_INTERESTED,
                                FeedbackType.TRIED_INEFFECTIVE):
            w.category_weight = max(0.2, w.category_weight - 0.25)
            w.category_penalty = min(1.0, w.category_penalty + 0.3)
            self._rejected_items.add(event.item_id)
        elif event.feedback == FeedbackType.ALREADY_SEEN:
            w.item_penalty = min(1.0, w.item_penalty + 0.5)

        self._weights[cat] = w
        return w

    def get_weights(self) -> Dict[str, FeedbackWeights]:
        return dict(self._weights)

    def is_item_rejected(self, item_id: str) -> bool:
        return item_id in self._rejected_items


feedback_tracker = FeedbackTracker()
