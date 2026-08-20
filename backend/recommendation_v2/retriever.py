# -*- coding: utf-8 -*-
"""
Candidate Retrieval（Phase 6 Task 6.3）。

复用现有 content_db 内容库 + 关键词检索，返回候选 dict 列表。
Retrieval 本身只负责召回，排序交给 Feature Ranker（6.4）。
"""
from __future__ import annotations

import logging
from typing import List, Optional

from state.schema import AgentState

logger = logging.getLogger(__name__)


class RecommendationRetriever:
    """候选召回（Phase 6 的检索接口）。"""

    def __init__(self):
        self._content_db = None

    def _get_db(self):
        if self._content_db is None:
            try:
                from content_db import ContentDatabase
                self._content_db = ContentDatabase()
            except Exception as e:  # noqa: BLE001
                logger.warning("content_db 加载失败: %s", e)
                self._content_db = False
        return self._content_db or None

    def retrieve(self, state: AgentState, query_terms: Optional[List[str]] = None,
                 top_k: int = 10) -> List[dict]:
        """检索候选内容。失败时返回空列表（不抛异常）。"""
        db = self._get_db()
        if not db:
            return []
        try:
            terms = query_terms or self._extract_terms(state)
            # 简单关键词匹配召回（排序交给 ranker）
            all_items = db.search(keywords=terms, limit=top_k * 3) if hasattr(db, "search") else []
            return all_items[:top_k * 3]
        except Exception as e:  # noqa: BLE001
            logger.warning("候选召回失败: %s", e)
            return []

    @staticmethod
    def _extract_terms(state: AgentState) -> List[str]:
        text = state.turn.current_text or ""
        # 轻量分词：去掉常见停用词
        stop = {"的", "了", "吗", "呢", "啊", "我", "你", "他", "她", "推荐", "有没有", "什么", "怎么"}
        return [t for t in text if t not in stop and len(t) >= 2][:8]


recommendation_retriever = RecommendationRetriever()
