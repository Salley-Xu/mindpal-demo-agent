# -*- coding: utf-8 -*-
"""
Cooldown / Repeat / Diversity Control（Phase 6 Task 6.6）。

避免连续重复、已拒绝仍推荐、同类过密。
"""
from __future__ import annotations

from typing import List, Optional

# 冷却参数（Phase 6 §6）
SAME_ITEM_COOLDOWN_TURNS = 8      # 同 item 冷却
SAME_CATEGORY_COOLDOWN_TURNS = 3  # 同类冷却


class CooldownController:
    """同 item 硬阻断 + 同类抑制。"""

    def __init__(self):
        self._recent_items: List[str] = []
        self._recent_categories: List[str] = []

    def is_in_cooldown(self, item_id: str) -> bool:
        """同 item 冷却期内 → 硬阻断。"""
        return item_id in self._recent_items

    def category_penalty(self, category: str) -> float:
        """同类抑制：近 3 轮同类 → 惩罚。"""
        count = self._recent_categories.count(category)
        return min(1.0, count * 0.5)

    def record(self, item_id: str, category: str) -> None:
        """记录本轮推荐。"""
        self._recent_items.append(item_id)
        self._recent_items = self._recent_items[-SAME_ITEM_COOLDOWN_TURNS:]
        self._recent_categories.append(category)
        self._recent_categories = self._recent_categories[-SAME_CATEGORY_COOLDOWN_TURNS:]

    def reset(self) -> None:
        self._recent_items = []
        self._recent_categories = []


cooldown_controller = CooldownController()
