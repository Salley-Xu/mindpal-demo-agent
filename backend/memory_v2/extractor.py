# -*- coding: utf-8 -*-
"""
Memory Candidate Extractor（Phase 4 Task 4.1 / 4.3）。

从 AgentState 提取可写入的记忆候选。扩展 memory_writer.MemoryCandidateExtractor，
覆盖更多 memory type（coping_strategy / coping_feedback / relationship / goal）。
"""
from __future__ import annotations

import re
from typing import List

from memory_v2.schema import MemoryType
from models import MemoryCandidate
from state.schema import AgentState

# 策略/反馈提取标记
_STRATEGY_MARKERS = ["我试了", "我发现", "这样做", "这个方法", "管用", "有效", "坚持", "练习"]
_FEEDBACK_MARKERS = ["有效", "管用", "没效果", "没用", "有帮助", "改善", "缓解"]
_RELATIONSHIP_MARKERS = ["我朋友", "我家人", "我妈", "我爸", "我同事", "我伴侣", "我孩子", "我室友"]


class MemoryExtractorV2:
    """记忆候选提取 v2。"""

    def extract(self, state: AgentState) -> List[MemoryCandidate]:
        text = state.turn.current_text or ""
        candidates: List[MemoryCandidate] = []

        # 1. 偏好（显式）：含"喜欢/不喜欢/偏好/想要"
        if re.search(r"喜欢|偏好|想要|不喜欢|不想", text):
            candidates.append(self._cand(state, MemoryType.PREFERENCE.value, text, source="explicit", conf=0.8))

        # 2. 应对策略（用户提到尝试的方法）
        if any(m in text for m in _STRATEGY_MARKERS):
            candidates.append(self._cand(state, MemoryType.COPING_STRATEGY.value, text, source="inferred", conf=0.7))

        # 3. 策略反馈
        if any(m in text for m in _FEEDBACK_MARKERS):
            candidates.append(self._cand(state, MemoryType.COPING_FEEDBACK.value, text, source="explicit", conf=0.8))

        # 4. 关系（提到他人）
        if any(m in text for m in _RELATIONSHIP_MARKERS):
            candidates.append(self._cand(state, MemoryType.RELATIONSHIP.value, text, source="inferred", conf=0.6))

        # 5. 目标（"我想/希望/打算"）
        if re.search(r"我想|我希望|打算|目标|要努力", text):
            candidates.append(self._cand(state, MemoryType.GOAL.value, text, source="explicit", conf=0.7))

        return candidates

    @staticmethod
    def _cand(state: AgentState, mtype: str, content: str, source: str, conf: float) -> MemoryCandidate:
        return MemoryCandidate(
            user_id=state.identity.user_id,
            session_id=state.identity.session_id,
            memory_type=mtype,
            content=content,
            source_text=content,
            emotion=state.emotion.current_emotion,
            risk_level=f"level_{state.risk.level}",
            user_intent=",".join(state.intent.labels),
            confidence=conf,
            source=source,
        )


memory_extractor_v2 = MemoryExtractorV2()
