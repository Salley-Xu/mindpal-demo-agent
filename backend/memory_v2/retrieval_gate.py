# -*- coding: utf-8 -*-
"""
Retrieval Gate（Phase 4 Task 4.5，最高优先级）。

从"每轮都检索"变为"需要时才检索"。

输入：AgentState 信号（intent / memory_reference / personalization / risk / stage）
输出：RetrievalDecision（retrieve_memory / scope / memory_types / reason）

设计依据：Frozen Agent Benchmark v1.1 gold 分析：
  - memory_reference intent → 必然检索（40/46）
  - resource/info/help + 非首轮 → 可检索（个性化相关）
  - casual/emotional 单轮 → 不检索
"""
from __future__ import annotations

from typing import List

from memory_v2.schema import MemoryType, RetrievalDecision
from state.schema import AgentState

# 意图 → 检索必要性
_ALWAYS_RETRIEVE_INTENTS = {"memory_reference"}
# information_request 是知识查询，非个性化 → 不触发检索
_MAYBE_RETRIEVE_INTENTS = {"resource_request", "explicit_help_request", "follow_up"}
# 偏好变化标记（"以前...现在不..." → 需检索旧偏好做 conflict 检测）
_PREFERENCE_CHANGE_MARKERS = ["不", "不想", "不再", "现在不", "以前", "以前喜欢", "改", "放弃", "戒", "讨厌"]


class RetrievalGate:
    """决定是否检索记忆。"""

    def decide(self, state: AgentState) -> RetrievalDecision:
        labels = set(state.intent.labels)

        # 1. memory_reference → 必然检索（gold 40/46）
        if "memory_reference" in labels:
            return RetrievalDecision(
                retrieve_memory=True, retrieval_scope="long_term",
                memory_types=[MemoryType.PREFERENCE, MemoryType.EVENT, MemoryType.COPING_STRATEGY],
                reason="explicit_memory_reference")

        # 2. 明确历史引用信号（memory.explicit_memory_reference）
        if state.memory.explicit_memory_reference:
            return RetrievalDecision(
                retrieve_memory=True, retrieval_scope="long_term",
                memory_types=[MemoryType.PREFERENCE, MemoryType.EVENT],
                reason="memory_signal_flag")

        # 3. 个性化相关（resource/help/follow-up）+ 非首轮
        if labels & _MAYBE_RETRIEVE_INTENTS and state.turn.turn_index > 0:
            types = [MemoryType.PREFERENCE]
            if "resource_request" in labels:
                types.append(MemoryType.COPING_FEEDBACK)
            if "follow_up" in labels:
                types.append(MemoryType.EVENT)
            return RetrievalDecision(
                retrieve_memory=True, retrieval_scope="long_term",
                memory_types=types, reason="personalization_relevant")

        # 3b. 偏好变化表达（"以前...现在不..."）→ 检索旧偏好做 conflict 检测
        text = state.turn.current_text or ""
        if labels & {"emotional_expression", "casual_chat"} and _looks_like_preference_change(text):
            return RetrievalDecision(
                retrieve_memory=True, retrieval_scope="long_term",
                memory_types=[MemoryType.PREFERENCE], reason="preference_change_detected")

        # 4. 长期用户有可用记忆 + 高风险 → 风险上下文
        if state.risk.level >= 2 and state.memory.available_memory_count > 0:
            return RetrievalDecision(
                retrieve_memory=True, retrieval_scope="risk",
                memory_types=[MemoryType.PROFILE, MemoryType.EVENT],
                reason="risk_context")

        # 5. 否则不检索
        return RetrievalDecision(
            retrieve_memory=False, retrieval_scope="none", memory_types=[], reason="no_memory_needed")


def _looks_like_preference_change(text: str) -> bool:
    """偏好变化启发式：含"以前/不/现在不"等变化表达。"""
    if "以前" in text or "现在不" in text or "不...了" in text:
        return True
    if any(m in text for m in ["不想", "不再", "放弃了", "戒掉", "不喝", "不吃", "讨厌"]):
        return True
    return False


retrieval_gate = RetrievalGate()
