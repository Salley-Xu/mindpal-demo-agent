from typing import Dict, List, Optional, Tuple

from bm25_retriever import bm25_retriever
from config import config
from dense_retriever import dense_retriever
from models import ContentItem
from vector_retriever import vector_retriever


class HybridRetriever:
    """混合检索器，BM25 + 同义词向量 + 可选 Dense embedding，RRF 融合。"""

    def __init__(self, rrf_k: int = 60, enable_dense: bool = False):
        self.rrf_k = rrf_k
        self.enable_dense = enable_dense

    def retrieve(
        self,
        query: str,
        items: List[ContentItem],
        limit: int = 10,
        extra_terms: Optional[List[str]] = None,
    ) -> List[Tuple[ContentItem, float]]:
        # BM25 和同义词向量（始终启用）
        inner_limit = max(limit, 10)
        bm25_results = bm25_retriever.retrieve(
            query=query, items=items, limit=inner_limit, extra_terms=extra_terms,
        )
        vector_results = vector_retriever.retrieve(
            query=query, items=items, limit=inner_limit, extra_terms=extra_terms,
        )
        result_sets = [bm25_results, vector_results]

        # Dense embedding（可选，通过 ENABLE_DENSE_RETRIEVAL 控制）
        if self.enable_dense:
            try:
                dense_results = dense_retriever.retrieve(
                    query=query, items=items, limit=inner_limit, extra_terms=extra_terms,
                )
                if dense_results:
                    result_sets.append(dense_results)
            except Exception:
                pass  # dense 失败时降级为 2 路 RRF

        fused_scores = self._reciprocal_rank_fusion(result_sets)
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


hybrid_retriever = HybridRetriever(
    enable_dense=config.ENABLE_DENSE_RETRIEVAL if hasattr(config, "ENABLE_DENSE_RETRIEVAL") else False,
)
