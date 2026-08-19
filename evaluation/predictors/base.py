# -*- coding: utf-8 -*-
"""
Predictor 抽象 — Agent Benchmark 的预测器接口。

每个 predictor 把一条 BenchmarkCase 转为结构化的 PredictedOutcome，
由 runner 与 expected 比对并计算指标。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from evaluation.benchmark_schema import BenchmarkCase


@dataclass
class PredictedOutcome:
    """一条 case 的结构化预测结果 v1.1（对齐 ExpectedOutcome 的字段）。"""
    intent: List[str] = field(default_factory=list)
    intent_confidence: float = 0.5
    emotion: str = "neutral"
    emotion_intensity: float = 0.5
    risk_level: int = 0
    risk_trend: str = "new"
    memory_needed: bool = False
    recommendation_action: str = "none"
    primary_action: str = "continue_chat"
    tool_actions: List[str] = field(default_factory=list)
    safety_target: str = "none"
    # 原始模块输出（调试 / 报告用）
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def retrieve_knowledge(self) -> bool:
        """兼容字段：knowledge tool 是否被预测。"""
        return "retrieve_knowledge" in self.tool_actions


class BasePredictor(ABC):
    """预测器基类。"""

    name: str = "base"

    @abstractmethod
    def predict(self, case: BenchmarkCase) -> PredictedOutcome:
        ...

    def batch(self, cases):
        return [self.predict(c) for c in cases]
