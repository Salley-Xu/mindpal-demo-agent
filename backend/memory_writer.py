"""
MemoryWriter — 长期记忆写入 Pipeline

从每轮对话结果中提取候选记忆，经 MemoryPolicy 门控后写入 MemoryStore。

流程:
  TurnContext → MemoryCandidateExtractor.extract()
    → MemoryPolicy.should_write() / assign_importance() / assign_ttl()
    → MemoryStore.batch_write()
"""

import logging
from typing import Any, Dict, List, Optional

from config import config
from models import MemoryCandidate, MemoryItem, TurnContext
from memory_store import memory_store
from memory_policy import memory_policy
from risk_levels import risk_level_index

logger = logging.getLogger(__name__)

# 偏好关键词检测模式（与 ConversationManager._update_long_term_signals 一致）
_DIRECT_STYLE_KEYWORDS = [
    "直接告诉我怎么做", "最好直接告诉我怎么做", "给我具体方法",
    "给我具体建议", "简单说重点", "直接一点", "我该怎么办",
]
_AVOID_EMPTY_COMFORT_KEYWORDS = [
    "不要空泛安慰", "别安慰我", "不喜欢空泛安慰", "不要讲大道理", "别说大道理",
]


class MemoryCandidateExtractor:
    """从 TurnContext 中提取各类记忆候选。"""

    def extract(self, context: TurnContext) -> List[MemoryCandidate]:
        """提取当前轮所有候选记忆。"""
        candidates: List[MemoryCandidate] = []

        candidates.extend(self._extract_stress_source(context))
        candidates.extend(self._extract_preferences(context))
        candidates.extend(self._extract_mood_event(context))

        return candidates

    # ---------------------------------------------------------------
    # 压力源
    # ---------------------------------------------------------------

    def _extract_stress_source(self, context: TurnContext) -> List[MemoryCandidate]:
        """提取压力源候选。"""
        emotion = context.emotion_state or {}
        stress_source = emotion.get("stress_source")
        if not stress_source:
            return []

        # 检查是否重复出现（通过 conversation_summary 的 stress_sources）
        summary = context.conversation_summary or {}
        stress_sources = summary.get("stress_sources", []) or []
        count = sum(1 for s in stress_sources if s == stress_source)

        # 历史中也出现过（从 emotion_state 中的 count 判断）
        intent = emotion.get("user_intent")
        return [MemoryCandidate(
            user_id=context.user_id,
            session_id=context.session_id,
            memory_type="stress_source",
            content=stress_source,
            summary=f"用户存在{stress_source}相关压力",
            source_text=context.user_input[:200],
            source="inferred" if count < 2 else "explicit",
            stress_source=stress_source,
            user_intent=intent,
            confidence=min(0.5 + 0.1 * count, 0.9),
        )]

    # ---------------------------------------------------------------
    # 支持/回避偏好
    # ---------------------------------------------------------------

    def _extract_preferences(self, context: TurnContext) -> List[MemoryCandidate]:
        """提取支持风格偏好和回避偏好。"""
        candidates: List[MemoryCandidate] = []
        text = context.user_input

        # 显式支持风格偏好
        direct_match = any(kw in text for kw in _DIRECT_STYLE_KEYWORDS)
        if direct_match:
            candidates.append(MemoryCandidate(
                user_id=context.user_id,
                session_id=context.session_id,
                memory_type="support_preference",
                content="用户偏好直接可执行的建议",
                summary="用户明确要求直接、可执行的建议",
                source_text=text[:200],
                source="explicit",
                confidence=0.90,
            ))

        # 隐式推断：repeated planning/seeking_help
        emotion = context.emotion_state or {}
        intent = emotion.get("user_intent")
        summary = context.conversation_summary or {}
        recent_intents = summary.get("recent_intents", []) or []
        high_action_count = sum(
            1 for i in recent_intents if i in {"planning", "seeking_help"}
        )
        if intent in {"planning", "seeking_help"} and high_action_count >= 1:
            candidates.append(MemoryCandidate(
                user_id=context.user_id,
                session_id=context.session_id,
                memory_type="support_preference",
                content="用户可能需要直接、可执行的建议",
                summary="用户多次表现出规划和寻求帮助倾向",
                source_text=text[:200],
                source="inferred",
                confidence=0.65,
            ))

        # 回避偏好
        avoid_match = any(kw in text for kw in _AVOID_EMPTY_COMFORT_KEYWORDS)
        if avoid_match:
            candidates.append(MemoryCandidate(
                user_id=context.user_id,
                session_id=context.session_id,
                memory_type="avoid_preference",
                content="用户不喜欢空泛安慰",
                summary="用户明确表示不喜欢空泛安慰或说教",
                source_text=text[:200],
                source="explicit",
                confidence=0.90,
            ))

        return candidates

    # ---------------------------------------------------------------
    # 情绪事件
    # ---------------------------------------------------------------

    def _extract_mood_event(self, context: TurnContext) -> List[MemoryCandidate]:
        """提取重要情绪事件。"""
        emotion = context.emotion_state or {}
        intensity = emotion.get("emotion_intensity", 0.0)
        risk_level = emotion.get("risk_level", "level_0")

        # 低强度 + 低风险 → 不写入 memory_items（只留 mood_events 日志）
        if intensity < 0.65 and risk_level_index(risk_level) < 1:
            return []

        stress_source = emotion.get("stress_source")
        current_emotion = emotion.get("current_emotion") or emotion.get("emotion_type") or "中性"
        intent = emotion.get("user_intent")

        content = f"用户{stress_source and f'因{stress_source}' or ''}产生{current_emotion}情绪"
        summary = content[:120]

        return [MemoryCandidate(
            user_id=context.user_id,
            session_id=context.session_id,
            memory_type="mood_event",
            content=content,
            summary=summary,
            source_text=context.user_input[:200],
            emotion=current_emotion,
            emotion_intensity=intensity,
            risk_level=risk_level,
            stress_source=stress_source,
            user_intent=intent,
            source="inferred",
            confidence=min(0.5 + intensity * 0.3, 0.9),
        )]


class MemoryWriter:
    """长期记忆写入器：提取 → 门控 → 持久化。"""

    def __init__(self, store=None, policy=None, extractor=None):
        self.store = store or memory_store
        self.policy = policy or memory_policy
        self.extractor = extractor or MemoryCandidateExtractor()

    async def process_turn(self, context: TurnContext) -> List[str]:
        """
        处理一轮对话：提取候选 → 门控 → 写入 memory_items。
        返回写入的 memory_id 列表（空列表表示没有写入）。
        """
        if not config.MEMORY_ENABLED:
            return []

        candidates = self.extractor.extract(context)
        if not candidates:
            return []

        items_to_write: List[MemoryItem] = []
        for candidate in candidates:
            if not self.policy.should_write(candidate, context):
                continue

            item = MemoryItem(
                user_id=candidate.user_id,
                session_id=candidate.session_id,
                turn_id=candidate.turn_id,
                memory_type=candidate.memory_type,
                content=candidate.content,
                summary=candidate.summary,
                source_text=candidate.source_text,
                emotion=candidate.emotion,
                emotion_intensity=candidate.emotion_intensity,
                risk_level=candidate.risk_level,
                stress_source=candidate.stress_source,
                user_intent=candidate.user_intent,
                importance=self.policy.assign_importance(candidate, context),
                confidence=self.policy.assign_confidence(candidate),
                source=candidate.source,
                expires_at=self.policy.assign_ttl(candidate),
            )
            items_to_write.append(item)

        if not items_to_write:
            return []

        return await self.store.batch_write(items_to_write)


# 全局单例
memory_writer = MemoryWriter()
