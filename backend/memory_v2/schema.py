# -*- coding: utf-8 -*-
"""
Memory 2.0 Schema（Phase 4 Task 4.2：Memory Type / Operation Freeze）。

Type 配置：extract condition / stability / TTL / merge key / retrieval priority。
Operation：ADD / UPDATE / MERGE / SUPERSEDE / EXPIRE / DELETE。
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class MemoryType(str, Enum):
    PROFILE = "profile"                          # 长期画像（身份/背景）
    PREFERENCE = "preference"                    # 偏好（支持/回避风格）
    EVENT = "event"                              # 事件（发生了什么）
    GOAL = "goal"                                # 目标/愿望
    COPING_STRATEGY = "coping_strategy"          # 有效应对策略
    COPING_FEEDBACK = "coping_feedback"          # 策略效果反馈
    RELATIONSHIP = "relationship"                # 人际关系
    INTERACTION = "interaction"                  # 交互偏好/模式


class MemoryOperation(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    SUPERSEDE = "SUPERSEDE"
    EXPIRE = "EXPIRE"
    DELETE = "DELETE"


class MemoryTypeConfig(BaseModel):
    """每类记忆的治理配置。"""
    memory_type: MemoryType
    extract_condition: str = "explicit"          # explicit | inferred
    stability: float = 0.5                       # 0-1，稳定度
    default_ttl_days: Optional[int] = None       # None = 长期
    merge_key: str = "content"                   # 去重/合并键
    retrieval_priority: float = 1.0              # 检索加权


# 冻结配置
MEMORY_TYPE_CONFIGS: dict = {
    MemoryType.PROFILE: MemoryTypeConfig(memory_type=MemoryType.PROFILE, extract_condition="explicit",
                                         stability=0.9, default_ttl_days=None, merge_key="topic", retrieval_priority=1.0),
    MemoryType.PREFERENCE: MemoryTypeConfig(memory_type=MemoryType.PREFERENCE, extract_condition="explicit",
                                            stability=0.8, default_ttl_days=None, merge_key="preference", retrieval_priority=1.2),
    MemoryType.EVENT: MemoryTypeConfig(memory_type=MemoryType.EVENT, extract_condition="inferred",
                                       stability=0.5, default_ttl_days=180, merge_key="event", retrieval_priority=0.8),
    MemoryType.GOAL: MemoryTypeConfig(memory_type=MemoryType.GOAL, extract_condition="explicit",
                                      stability=0.7, default_ttl_days=None, merge_key="goal", retrieval_priority=1.1),
    MemoryType.COPING_STRATEGY: MemoryTypeConfig(memory_type=MemoryType.COPING_STRATEGY, extract_condition="inferred",
                                                 stability=0.6, default_ttl_days=90, merge_key="strategy", retrieval_priority=1.0),
    MemoryType.COPING_FEEDBACK: MemoryTypeConfig(memory_type=MemoryType.COPING_FEEDBACK, extract_condition="explicit",
                                                 stability=0.7, default_ttl_days=60, merge_key="feedback", retrieval_priority=0.9),
    MemoryType.RELATIONSHIP: MemoryTypeConfig(memory_type=MemoryType.RELATIONSHIP, extract_condition="explicit",
                                              stability=0.85, default_ttl_days=None, merge_key="person", retrieval_priority=1.1),
    MemoryType.INTERACTION: MemoryTypeConfig(memory_type=MemoryType.INTERACTION, extract_condition="inferred",
                                             stability=0.5, default_ttl_days=30, merge_key="pattern", retrieval_priority=0.7),
}


class MemoryWriteEvent(BaseModel):
    """每次写入必须可 Trace。"""
    operation: MemoryOperation
    reason: str
    source_turn: Optional[str] = None
    confidence: float = 0.0
    candidate_type: MemoryType
    candidate_content: str


class RetrievalDecision(BaseModel):
    """Retrieval Gate 输出。"""
    retrieve_memory: bool
    retrieval_scope: str = "session"             # none | session | long_term | risk
    memory_types: List[MemoryType] = []
    reason: str = ""
