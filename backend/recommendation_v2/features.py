# -*- coding: utf-8 -*-
"""
Feature Extraction（Phase 6 Task 6.4）。

从 AgentState + 候选内容提取 Ranking 特征（Phase 6 §4）。
"""
from __future__ import annotations

from typing import List

from recommendation_v2.schema import CandidateFeatures
from state.schema import AgentState


class FeatureExtractor:
    """候选内容特征提取。"""

    def extract(self, state: AgentState, candidate: dict, query_terms: List[str]) -> CandidateFeatures:
        """
        从候选 dict（content_db 风格：id/category/title/content/keywords）提取特征。
        特征值 0-1 归一化。
        """
        cid = candidate.get("id") or candidate.get("content_id") or "unknown"
        cat = candidate.get("category", "") or candidate.get("content_type", "")
        title = candidate.get("title", "")
        content = candidate.get("content", "")
        keywords = candidate.get("keywords", []) or []

        # semantic / bm25 近似：关键词 + 词面重叠
        text_blob = title + " " + content + " " + " ".join(keywords)
        overlap = sum(1 for t in query_terms if t and t in text_blob)
        semantic = min(1.0, overlap / max(1, len(query_terms)))
        bm25 = min(1.0, overlap / max(1, len(query_terms))) * 0.8

        # intent match：资源请求与内容类目对齐
        intent_match = 0.7 if state.derived.resource_seeking else 0.3

        # emotion fit：负面情绪偏好安抚类
        emotion_fit = 0.8 if state.derived.negative_emotion and cat in ("relax", "calm") else 0.4

        # risk compatibility：高风险只放行安全资源
        risk_compat = 1.0 if state.risk.level < 2 or cat == "safety" else 0.1

        # profile / memory preference
        profile_pref = 0.6 if state.user_context.preferred_support_style and \
            state.user_context.preferred_support_style in cat else 0.3

        return CandidateFeatures(
            item_id=cid,
            category=cat,
            semantic_similarity=round(semantic, 3),
            bm25=round(bm25, 3),
            intent_match=round(intent_match, 3),
            emotion_fit=round(emotion_fit, 3),
            risk_compatibility=round(risk_compat, 3),
            profile_preference=round(profile_pref, 3),
        )


feature_extractor = FeatureExtractor()
