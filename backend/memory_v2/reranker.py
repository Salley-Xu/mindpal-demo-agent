# -*- coding: utf-8 -*-
"""
Memory Rerank（Phase 4 Task 4.7）。

对候选记忆做最终排序：intent match + personalization relevance 加权。
第一版不做 LLM rerank（Phase 4 §9）。
"""
from __future__ import annotations

from typing import List

from models import MemorySearchResult


class MemoryRerankerV2:
    """记忆重排 v2。"""

    def rerank(self, results: List[MemorySearchResult], query_text: str) -> List[MemorySearchResult]:
        """按 intent/personalization 相关性小幅重排。"""
        for r in results:
            # 引用词命中（"上次/之前/记得" + 记忆内容重叠）加分
            content = r.item.content
            overlap = sum(1 for w in query_text if w and w in content)
            r.score += overlap * 0.05
        results.sort(key=lambda x: -x.score)
        for i, r in enumerate(results):
            r.rank = i
        return results


memory_reranker_v2 = MemoryRerankerV2()
