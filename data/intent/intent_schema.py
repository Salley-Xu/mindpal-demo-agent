# -*- coding: utf-8 -*-
"""
Intent Dataset Schema（Phase 1 Task 1.2 §11）。

对齐 IntentResult 冻结接口（labels 来自 Intent Taxonomy v1 的 10 类）。
用于 Seed / Train / Dev / Test / OOD 各数据集。
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

# Intent Taxonomy v1（与 evaluation/benchmark_schema.IntentLabel 一致）
INTENT_LABELS = [
    "casual_chat",
    "emotional_expression",
    "explicit_help_request",
    "information_request",
    "resource_request",
    "feedback",
    "follow_up",
    "memory_reference",
    "high_risk_expression",
    "meta_question",
]


class IntentLabel(str, Enum):
    CASUAL_CHAT = "casual_chat"
    EMOTIONAL_EXPRESSION = "emotional_expression"
    EXPLICIT_HELP_REQUEST = "explicit_help_request"
    INFORMATION_REQUEST = "information_request"
    RESOURCE_REQUEST = "resource_request"
    FEEDBACK = "feedback"
    FOLLOW_UP = "follow_up"
    MEMORY_REFERENCE = "memory_reference"
    HIGH_RISK_EXPRESSION = "high_risk_expression"
    META_QUESTION = "meta_question"


class IntentTurn(BaseModel):
    role: str = "user"          # user | assistant
    content: str


class IntentData(BaseModel):
    id: str
    conversation: List[IntentTurn] = Field(default_factory=lambda: [])
    text: str                    # 被标注的当前话语（通常为最后一条 user 轮）
    labels: List[IntentLabel]
    source: str = "template"     # human | template | llm
    difficulty: str = "medium"   # easy | medium | hard
    is_ood: bool = False         # OOD 数据集标记
    notes: Optional[str] = None


class IntentSplit(BaseModel):
    """划分结果。"""
    train: List[IntentData] = []
    dev: List[IntentData] = []
    test: List[IntentData] = []


def to_intent_result(data: IntentData, confidence: float, is_open_set: bool, source: str) -> dict:
    """从标注数据构造 IntentResult（冻结接口）。"""
    return {
        "labels": [l.value for l in data.labels],
        "confidence": confidence,
        "is_open_set": is_open_set,
        "source": source,
    }
