# -*- coding: utf-8 -*-
"""
Dedup / Conflict Resolution（Phase 4 Task 4.4）。

处理：同义重复 / 偏好更新 / 旧事实被新事实替代 / 相互矛盾 / 时间变化。

关键原则（Phase 4 §7）：
  - 不能简单两条都永久保留
  - 偏好更新 → SUPERSEDE（旧条目归档）
  - 时间变化 → 新事实优先，旧事实标记过时
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from memory_v2.schema import MemoryOperation, MemoryWriteEvent
from models import MemoryCandidate, MemoryItem

# 否定/变化标记（简单规则，足够 v2）
_NEGATION_MARKERS = ["不", "没", "别", "不再", "不想", "讨厌", "放弃", "戒掉"]
_CHANGE_MARKERS = ["现在", "最近", "现在不", "已经不", "改"]


class ConflictResolver:
    """记忆写入去重/冲突解决。"""

    def resolve(self, candidate: MemoryCandidate, existing: List[MemoryItem]
                ) -> Tuple[MemoryOperation, Optional[str], str]:
        """
        返回 (operation, target_item_id, reason)。
          ADD      无冲突 → 新增
          MERGE    同义相似 → 合并到已有
          SUPERSEDE 偏好/事实更新 → 归档旧条目，新增新条目
        """
        if not existing:
            return MemoryOperation.ADD, None, "no_conflict"

        # 1. 精确重复 → 忽略（不重复写）
        for item in existing:
            if item.content == candidate.content:
                return MemoryOperation.ADD, None, "duplicate_skip"

        # 2. 同义重复（content 高度重叠）→ MERGE
        for item in existing:
            sim = _overlap(item.content, candidate.content)
            if sim > 0.70:
                return MemoryOperation.MERGE, item.id, f"synonym_merge:{sim:.2f}"

        # 2b. 强化/维持表达（"继续坚持/不错"）同类型 → MERGE
        if any(m in candidate.content for m in _REINFORCE_MARKERS):
            for item in existing:
                if item.memory_type == candidate.memory_type and _overlap(item.content, candidate.content) > 0.4:
                    return MemoryOperation.MERGE, item.id, "reinforce_merge"

        # 3. 偏好/策略更新 → SUPERSEDE
        for item in existing:
            if item.memory_type == candidate.memory_type and _is_update(item.content, candidate.content):
                return MemoryOperation.SUPERSEDE, item.id, "preference_supersede"

        # 4. 矛盾事实（同类型、同主题、反向表述）→ SUPERSEDE
        for item in existing:
            if item.memory_type == candidate.memory_type and _is_contradictory(item.content, candidate.content):
                return MemoryOperation.SUPERSEDE, item.id, "contradiction_supersede"

        return MemoryOperation.ADD, None, "no_conflict"


def _overlap(a: str, b: str) -> float:
    """字符重叠度（粗去重）。"""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa), len(sb))


# 强化/维持标记（→ MERGE，不是更新）
_REINFORCE_MARKERS = ["继续", "坚持", "挺好的", "不错", "保持", "确实有效"]
# 明确偏好变化标记（→ SUPERSEDE）
_UPDATE_MARKERS = ["不喝了", "不吃", "戒", "讨厌", "放弃", "不再", "现在不", "不想", "没兴趣", "改"]


def _is_update(old: str, new: str) -> bool:
    """新表述是否包含偏好变化（旧偏好被更新）。排除"不错/挺好的"等强化表达。"""
    if any(m in new for m in _REINFORCE_MARKERS):
        return False
    return any(m in new for m in _UPDATE_MARKERS) or any(m in new for m in _CHANGE_MARKERS)


def _is_contradictory(old: str, new: str) -> bool:
    """同主题反向表述。"""
    # 简单规则：同一动词词根 + 否定方向不同
    negation_new = any(m in new for m in _NEGATION_MARKERS)
    negation_old = any(m in old for m in _NEGATION_MARKERS)
    if negation_new != negation_old:
        # 共享 2+ 字符的动词/名词
        common = set(old) & set(new)
        if len(common) >= 4:
            return True
    return False


conflict_resolver = ConflictResolver()
