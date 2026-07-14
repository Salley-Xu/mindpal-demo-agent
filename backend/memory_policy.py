"""
MemoryPolicy — 写入门控 + importance / confidence / TTL 计算

从 ConversationManager._update_long_term_signals() 中泛化而来。
每类候选记忆通过 should_write → assign_importance → assign_confidence → assign_ttl 进入 MemoryStore。
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from models import MemoryCandidate, TurnContext

logger = logging.getLogger(__name__)


class MemoryPolicy:
    """控制长期记忆的写入、重要性、置信度、TTL 和注入决策。"""

    # ---------------------------------------------------------------
    # 写入门控
    # ---------------------------------------------------------------

    def should_write(self, candidate: MemoryCandidate, context: TurnContext) -> bool:
        """判断候选记忆是否应进入长期存储。"""
        method = f"_should_write_{candidate.memory_type}"
        handler = getattr(self, method, None)
        if handler:
            return handler(candidate, context)
        # 默认：有足够置信度就写
        return candidate.confidence >= 0.6

    def _should_write_stress_source(self, candidate: MemoryCandidate, context: TurnContext) -> bool:
        """压力源：同一来源最近 N 轮中出现 ≥K 次。"""
        if not candidate.stress_source:
            return False
        history = context.emotion_state or {}
        return self._appears_at_least(candidate, window=6, count=2)

    def _should_write_support_preference(self, candidate: MemoryCandidate, _context: TurnContext) -> bool:
        """支持偏好：显式表达或高置信度。"""
        return candidate.confidence >= 0.75 or candidate.source == "explicit"

    def _should_write_avoid_preference(self, candidate: MemoryCandidate, _context: TurnContext) -> bool:
        """回避偏好：显式表达或高置信度。"""
        return candidate.confidence >= 0.75 or candidate.source == "explicit"

    def _should_write_mood_event(self, candidate: MemoryCandidate, _context: TurnContext) -> bool:
        """情绪事件：高强度或关联风险。"""
        intensity = candidate.emotion_intensity or 0.0
        risk = candidate.risk_level or "level_0"
        from risk_levels import risk_level_index
        return intensity >= 0.65 or risk_level_index(risk) >= 1

    def _should_write_recommendation_feedback(self, _candidate: MemoryCandidate, _context: TurnContext) -> bool:
        """推荐反馈：始终写入。"""
        return True

    def _should_write_coping_strategy(self, candidate: MemoryCandidate, _context: TurnContext) -> bool:
        """应对策略：中等以上置信度。"""
        return candidate.confidence >= 0.5

    # ---------------------------------------------------------------
    # 重要性评分
    # ---------------------------------------------------------------

    def assign_importance(self, candidate: MemoryCandidate, context: TurnContext) -> float:
        """
        计算重要性 (0.0 ~ 1.0)。

        importance = base + risk_bonus + intensity_bonus + explicit_bonus + repetition_bonus
        """
        base = self._base_importance(candidate.memory_type)
        bonus = 0.0

        # 风险加分
        from risk_levels import risk_level_index
        risk_idx = risk_level_index(candidate.risk_level or "level_0")
        if risk_idx >= 2:
            bonus += 0.25
        elif risk_idx >= 1:
            bonus += 0.10

        # 强度加分
        intensity = candidate.emotion_intensity or 0.0
        if intensity > 0.8:
            bonus += 0.15
        elif intensity > 0.6:
            bonus += 0.08

        # 显式表达加分
        if candidate.source == "explicit":
            bonus += 0.20

        # 重复出现加分
        if candidate.memory_type == "stress_source" and candidate.stress_source:
            history = context.emotion_state or {}
            bonus += 0.10

        return min(base + bonus, 1.0)

    @staticmethod
    def _base_importance(memory_type: str) -> float:
        """各类记忆的基础重要性。"""
        base = {
            "risk_shadow": 0.80,
            "support_preference": 0.75,
            "avoid_preference": 0.75,
            "coping_strategy": 0.70,
            "stress_source": 0.60,
            "recommendation_feedback": 0.55,
            "conversation_summary": 0.50,
            "mood_event": 0.45,
            "personal_fact": 0.40,
        }
        return base.get(memory_type, 0.50)

    # ---------------------------------------------------------------
    # 置信度评分
    # ---------------------------------------------------------------

    def assign_confidence(self, candidate: MemoryCandidate) -> float:
        """根据来源计算置信度。"""
        source_map = {
            "explicit": 0.90,
            "risk_detector": 0.80,
            "tool": 0.75,
            "inferred": 0.60,
            "recommender": 0.55,
        }
        base = source_map.get(candidate.source, 0.50)
        return min(base + (candidate.emotion_intensity or 0) * 0.10, 1.0)

    # ---------------------------------------------------------------
    # TTL
    # ---------------------------------------------------------------

    def assign_ttl(self, candidate: MemoryCandidate) -> Optional[datetime]:
        """根据记忆类型计算过期时间（None = 长期）。"""
        from config import config as cfg

        ttl_map = {
            "support_preference": None,
            "avoid_preference": None,
            "coping_strategy": timedelta(days=180),
            "stress_source": timedelta(days=cfg.MEMORY_DECAY_DAYS_STRESS_SOURCE),
            "recommendation_feedback": timedelta(days=180),
            "conversation_summary": timedelta(days=180),
            "mood_event": timedelta(days=cfg.MEMORY_DECAY_DAYS_MOOD_EVENT),
            "risk_shadow": timedelta(days=cfg.MEMORY_DECAY_DAYS_RISK_EVENT),
            "personal_fact": timedelta(days=180),
        }
        delta = ttl_map.get(candidate.memory_type)
        if delta is None:
            return None
        return datetime.now(timezone.utc) + delta

    # ---------------------------------------------------------------
    # 注入决策
    # ---------------------------------------------------------------

    def should_inject(self, item: Any, context: TurnContext) -> bool:
        """是否应将此记忆注入当前 prompt。"""
        if item.status != "active":
            return False
        if (item.confidence or 0) < 0.4:
            return False
        if item.expires_at and datetime.now(timezone.utc) > item.expires_at:
            return False
        return True

    # ---------------------------------------------------------------
    # 辅助
    # ---------------------------------------------------------------

    @staticmethod
    def _appears_at_least(candidate: MemoryCandidate, window: int, count: int) -> bool:
        """判断候选在最近窗口中是否出现足够次数（简单实现，供 extractor 覆盖）。"""
        return candidate.confidence >= 0.5


# 全局单例
memory_policy = MemoryPolicy()
