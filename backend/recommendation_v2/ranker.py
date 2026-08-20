# -*- coding: utf-8 -*-
"""
Feature Ranker（Phase 6 Task 6.4 / 6.6）。

加权特征排序 + Cooldown / 重复抑制 + 负反馈压制。

特征权重（Phase 6 §4）：semantic/BM25/intent/emotion/risk/profile/memory/recency/repeat/neg-feedback。
"""
from __future__ import annotations

from typing import Dict, List

from recommendation_v2.schema import CandidateFeatures, FeedbackWeights

# 特征权重（可配置）
_FEATURE_WEIGHTS = {
    "semantic_similarity": 0.20,
    "bm25": 0.10,
    "intent_match": 0.15,
    "emotion_fit": 0.15,
    "risk_compatibility": 0.20,
    "profile_preference": 0.10,
    "memory_preference": 0.10,
}

# Cooldown（Phase 6 §6）
_SAME_ITEM_COOLDOWN_TURNS = 8      # 同 item 冷却
_SAME_CATEGORY_COOLDOWN_TURNS = 3  # 同类冷却


class FeatureRanker:
    """特征排序器。"""

    def __init__(self):
        self._recent_items: List[str] = []
        self._recent_categories: List[str] = []

    def score(self, feats: CandidateFeatures, feedback: Dict[str, FeedbackWeights]) -> float:
        """计算候选总分（含 repeat/neg-feedback 惩罚）。"""
        score = 0.0
        score += _FEATURE_WEIGHTS["semantic_similarity"] * feats.semantic_similarity
        score += _FEATURE_WEIGHTS["bm25"] * feats.bm25
        score += _FEATURE_WEIGHTS["intent_match"] * feats.intent_match
        score += _FEATURE_WEIGHTS["emotion_fit"] * feats.emotion_fit
        score += _FEATURE_WEIGHTS["risk_compatibility"] * feats.risk_compatibility
        score += _FEATURE_WEIGHTS["profile_preference"] * feats.profile_preference
        score += _FEATURE_WEIGHTS["memory_preference"] * feats.memory_preference

        # 反馈权重
        w = feedback.get(feats.category, FeedbackWeights())
        score *= w.category_weight
        score += w.item_penalty * feats.negative_feedback_penalty
        score += w.effective_boost

        # 重复惩罚 / already_seen
        if feats.already_seen:
            score -= 0.5
        if feats.repeat_penalty > 0:
            score -= feats.repeat_penalty * 0.4
        return max(0.0, score)

    def apply_cooldown(self, feats: CandidateFeatures, turn_index: int) -> CandidateFeatures:
        """根据冷却状态设置 repeat_penalty / already_seen。"""
        if feats.item_id in self._recent_items:
            feats.repeat_penalty = 1.0
        if feats.category in self._recent_categories:
            feats.repeat_penalty = max(feats.repeat_penalty, 0.5)
        return feats

    def in_cooldown(self, item_id: str) -> bool:
        """同 item 冷却期内 → 硬阻断。"""
        return item_id in self._recent_items

    def record_recommendation(self, item_id: str, category: str) -> None:
        """记录本轮推荐，供后续 cooldown。"""
        self._recent_items.append(item_id)
        self._recent_items = self._recent_items[-_SAME_ITEM_COOLDOWN_TURNS:]
        self._recent_categories.append(category)
        self._recent_categories = self._recent_categories[-_SAME_CATEGORY_COOLDOWN_TURNS:]


feature_ranker = FeatureRanker()
