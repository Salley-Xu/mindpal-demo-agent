# -*- coding: utf-8 -*-
"""
Hybrid Intent（Phase 1 Task 1.11 §31-32）。

流程：Small Model → label_scores → Calibration → Open-set/Confidence Gate
       → high confidence → classifier；uncertain → LLM fallback → IntentResult。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from evaluation.intent.predictors import IntentPredictor, LLMPredictor

logger = logging.getLogger(__name__)


class HybridIntentPredictor(IntentPredictor):
    """Hybrid：Small Model + Open-set + LLM Fallback。"""

    name = "hybrid"

    def __init__(self, classifier: IntentPredictor, open_set=None,
                 fallback_confidence_threshold: float = 0.55,
                 open_set_threshold: float = 0.35,
                 enable_llm_fallback: bool = True,
                 llm: Optional[LLMPredictor] = None):
        self.classifier = classifier
        self.open_set = open_set
        self.open_set_threshold = open_set_threshold
        self.fallback_threshold = fallback_confidence_threshold
        self.enable_llm_fallback = enable_llm_fallback
        self.llm = llm or LLMPredictor()

    def predict(self, text: str, context: Optional[List[Dict]] = None) -> Dict:
        clf_result = self.classifier.predict(text, context)
        scores = clf_result.get("label_scores", {})
        max_score = max(scores.values()) if scores else 0.0

        # Open-set 判定
        is_open = (not scores) or (self.open_set.is_open_set(scores) if self.open_set
                                   else max_score < self.open_set_threshold)

        fallback_reason = None
        if is_open:
            fallback_reason = "open_set"
        elif max_score < self.fallback_threshold:
            fallback_reason = "low_confidence"

        if fallback_reason and self.enable_llm_fallback:
            llm_result = self.llm.predict(text, context)
            return {
                **llm_result,
                "fallback_reason": fallback_reason,
                "classifier_scores": scores,
            }

        return {
            **clf_result,
            "fallback_reason": fallback_reason,
        }

    @property
    def llm_call_rate(self) -> float:
        return 1.0 if self.enable_llm_fallback else 0.0
