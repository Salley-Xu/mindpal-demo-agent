from typing import Dict, List, Tuple

from bm25_retriever import bm25_retriever
from models import ContentItem
from vector_retriever import vector_retriever


class HybridRetriever:
    """混合检索器，使用 BM25 + 轻量向量召回，再以 RRF 融合。"""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        items: List[ContentItem],
        limit: int = 10,
        extra_terms: List[str] | None = None,
    ) -> List[Tuple[ContentItem, float]]:
        bm25_results = bm25_retriever.retrieve(
            query=query,
            items=items,
            limit=max(limit, 10),
            extra_terms=extra_terms,
        )
        vector_results = vector_retriever.retrieve(
            query=query,
            items=items,
            limit=max(limit, 10),
            extra_terms=extra_terms,
        )
        fused_scores = self._reciprocal_rank_fusion([bm25_results, vector_results])
        ranked = sorted(fused_scores.values(), key=lambda pair: pair[1], reverse=True)
        return ranked[:limit]

    def _reciprocal_rank_fusion(
        self,
        result_sets: List[List[Tuple[ContentItem, float]]],
    ) -> Dict[str, Tuple[ContentItem, float]]:
        scores: Dict[str, float] = {}
        items_by_id: Dict[str, ContentItem] = {}
        for result_set in result_sets:
            for rank, (item, _score) in enumerate(result_set, start=1):
                items_by_id[item.id] = item
                scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (self.rrf_k + rank)
        return {
            item_id: (items_by_id[item_id], score)
            for item_id, score in scores.items()
        }


hybrid_retriever = HybridRetriever()
