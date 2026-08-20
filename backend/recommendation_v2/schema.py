# -*- coding: utf-8 -*-
"""
Recommendation 2.0 Schema（Phase 6 Task 6.2 / 6.5）。

Feedback 类型统一 + Cooldown 配置。
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class FeedbackType(str, Enum):
    ACCEPT = "accept"                          # 接受/有用
    REJECT = "reject"                          # 拒绝/不想要
    TRIED_EFFECTIVE = "tried_effective"        # 试过且有效
    TRIED_INEFFECTIVE = "tried_ineffective"    # 试过但无效
    NOT_INTERESTED = "not_interested"          # 不感兴趣
    ALREADY_SEEN = "already_seen"              # 已看过


class FeedbackEvent(BaseModel):
    user_id: str
    item_id: str
    item_category: str = ""
    feedback: FeedbackType
    turn_id: Optional[str] = None
    timestamp: Optional[str] = None


class CandidateFeatures(BaseModel):
    """候选内容的特征向量（Phase 6 §4）。"""
    item_id: str
    semantic_similarity: float = 0.0
    bm25: float = 0.0
    intent_match: float = 0.0
    emotion_fit: float = 0.0
    risk_compatibility: float = 1.0
    profile_preference: float = 0.0
    memory_preference: float = 0.0
    recency: float = 0.0
    repeat_penalty: float = 0.0          # 0 = 无重复
    negative_feedback_penalty: float = 0.0
    already_seen: bool = False
    category: str = ""


class FeedbackWeights(BaseModel):
    """反馈 → 未来 ranking 的权重影响。"""
    category_weight: float = 1.0         # 正反馈提升该类
    item_penalty: float = 0.0            # 负反馈压制该 item
    category_penalty: float = 0.0        # 负反馈压制该类
    effective_boost: float = 0.0         # tried_effective 提升该类
