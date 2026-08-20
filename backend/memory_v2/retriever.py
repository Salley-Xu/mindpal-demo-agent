# -*- coding: utf-8 -*-
"""
Memory Retrieval v2（Phase 4 Task 4.6）。

复用现有 BM25 检索 + 增加 type/recency/importance/confidence 加权。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from memory_v2.schema import MemoryType
from models import MemoryItem, MemoryQuery, MemorySearchResult

logger = logging.getLogger(__name__)


class MemoryRetrieverV2:
    """记忆检索 v2。"""

    def __init__(self, store=None):
        self._store = store
        self._bm25 = None

    def _get_components(self):
        if self._bm25 is None:
            from memory_retriever import MemoryBM25Retriever
            self._bm25 = MemoryBM25Retriever()
        return self._bm25

    def retrieve(self, query: MemoryQuery, items: List[MemoryItem], top_k: int = 10) -> List[MemorySearchResult]:
        """在给定 items 上检索（BM25 + type/recency/importance 加权）。"""
        bm25 = self._get_components()
        now = datetime.now()

        scored = []
        for item in items:
            if item.status != "active":
                continue
            # BM25 语义分
            lex = bm25._score(query.text, item.content) if hasattr(bm25, "_score") else 0.0
            # type 匹配
            type_match = 1.0 if (query.memory_types and item.memory_type in
                                 [t.value for t in query.memory_types]) else 0.5
            # recency：越新越高
            age_days = (now - item.created_at).days if item.created_at else 365
            recency = max(0.0, 1.0 - age_days / 365)
            # importance / confidence
            imp = item.importance or 0.5
            conf = item.confidence or 0.5

            score = (0.5 * lex + 0.2 * type_match + 0.15 * recency + 0.15 * imp) * conf
            scored.append((score, item))

        scored.sort(key=lambda x: -x[0])
        return [MemorySearchResult(item=it, score=round(s, 4), rank=i, retrieval_method="memory_v2")
                for i, (s, it) in enumerate(scored[:top_k])]


memory_retriever_v2 = MemoryRetrieverV2()
