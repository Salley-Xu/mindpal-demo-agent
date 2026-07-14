"""
MemoryRetriever — 多阶段记忆检索

流程:
  1. SQL filter recall (user_id + status + memory_type + expires_at)
  2. BM25 lexical recall
  3. Risk recall (always includes baseline + recent triggers)
  4. RRF fusion
  5. Multi-factor reranking
"""

import logging
import re
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from config import config
from memory_store import memory_store
from models import MemoryItem, MemoryQuery, MemorySearchResult
from risk_levels import risk_level_index

logger = logging.getLogger(__name__)


class MemoryBM25Retriever:
    """BM25 检索器（适配 MemoryItem，复用 BM25 算法）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def retrieve(
        self,
        query: str,
        items: List[MemoryItem],
        limit: int = 10,
        extra_terms: Optional[List[str]] = None,
    ) -> List[Tuple[MemoryItem, float]]:
        if not items or not query:
            return []

        query_terms = self._tokenize(query)
        if extra_terms:
            for term in extra_terms:
                query_terms.extend(self._tokenize(term))
        if not query_terms:
            return []

        docs = [self._build_document(item) for item in items]
        doc_tokens = [self._tokenize(doc) for doc in docs]
        doc_lens = [len(tokens) for tokens in doc_tokens]
        avg_doc_len = sum(doc_lens) / max(len(doc_lens), 1)
        doc_freqs = self._document_frequencies(doc_tokens)

        scored: List[Tuple[MemoryItem, float]] = []
        for item, tokens, doc_len in zip(items, doc_tokens, doc_lens):
            term_counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                if term not in term_counts:
                    continue
                idf = self._idf(term, len(items), doc_freqs)
                tf = term_counts[term]
                denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(avg_doc_len, 1)))
                score += idf * (tf * (self.k1 + 1)) / max(denom, 1e-9)
            if score > 0:
                scored.append((item, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def _build_document(self, item: MemoryItem) -> str:
        parts = [item.content, item.summary or ""]
        if item.stress_source:
            parts.append(item.stress_source)
        if item.source_text:
            parts.append(item.source_text)
        if item.emotion:
            parts.append(item.emotion)
        return " ".join(p for p in parts if p)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """分词：英文按空格/符号分割，中文用 character bigram。"""
        text = str(text).lower()
        tokens: List[str] = []

        # 提取连续的中文块
        chinese_blocks = re.findall(r"[一-鿿]+", text)
        # 提取非中文的字母数字块
        other_blocks = re.findall(r"[a-z0-9]+", text)

        for block in chinese_blocks:
            # 中文用字符 bigram
            for i in range(len(block) - 1):
                tokens.append(block[i:i+2])
            # 也保留单字
            for ch in block:
                tokens.append(ch)

        for block in other_blocks:
            tokens.append(block)

        return [t for t in tokens if len(t) >= 1]

    @staticmethod
    def _document_frequencies(doc_tokens: List[List[str]]) -> Dict[str, int]:
        df: Dict[str, int] = {}
        for tokens in doc_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        return df

    def _idf(self, term: str, n_docs: int, doc_freqs: Dict[str, int]) -> float:
        df = doc_freqs.get(term, 0)
        return math.log((n_docs - df + 0.5) / (max(df, 1) - 0.5) + 1.0)


class MemoryRetriever:
    """多阶段记忆检索：SQL 过滤 + BM25 + 风险召回 + RRF + 重排。"""

    def __init__(self, store=None, bm25=None, rrf_k: int = 60):
        self.store = store or memory_store
        self.bm25 = bm25 or MemoryBM25Retriever()
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        user_id: str,
        query: MemoryQuery,
        top_k: int = 10,
    ) -> List[MemorySearchResult]:
        """执行多阶段检索，返回排序后的记忆列表。"""
        # Stage 1: SQL filter recall — 获取所有 active 候选
        items = await self.store.get_active_items(
            user_id=user_id,
            memory_types=query.memory_types,
            limit=config.MEMORY_MAX_ITEMS_PER_USER,
        )
        if not items:
            return []

        # Stage 2: BM25 lexical recall — 关键词匹配过滤
        extra_terms = []
        if query.stress_source:
            extra_terms.append(query.stress_source)
        if query.emotion:
            extra_terms.append(query.emotion)
        if query.intent:
            extra_terms.append(query.intent)

        bm25_results = self.bm25.retrieve(
            query=query.text,
            items=items,
            limit=top_k * 2,
            extra_terms=extra_terms if extra_terms else None,
        )
        bm25_item_ids = {item.id for item, _ in bm25_results}

        # Stage 3: Score-based recall — 按 importance + recency 补充
        now = datetime.now(timezone.utc)
        scored_items = []
        for item in items:
            if item.id in bm25_item_ids:
                # BM25 匹配的会通过 RRF 融合，跳过这里
                continue
            score = self._score_item(item, query, now)
            if score > 0:
                scored_items.append((item, score))

        # 补充 top items
        scored_items.sort(key=lambda pair: pair[1], reverse=True)

        # RRF fusion: BM25 + scored
        result_sets = [bm25_results, scored_items]
        fused = self._rrf_fusion(result_sets)

        # Multi-factor reranking
        reranked = self._rerank(fused, query, now)

        return reranked[:top_k]

    def _score_item(self, item: MemoryItem, query: MemoryQuery, now: datetime) -> float:
        """基于规则的快速评分（用于非 BM25 匹配的记忆）。"""
        score = 0.0

        # importance
        score += (item.importance or 0.5) * 2.0

        # recency (0-1, higher = newer)
        if item.created_at:
            try:
                age_days = (now - item.created_at).days
                score += max(0, 1.0 - age_days / 180) * 1.0
            except Exception:
                pass

        # emotion match
        if query.emotion and item.emotion and query.emotion == item.emotion:
            score += 0.5

        # stress_source match
        if query.stress_source and item.stress_source:
            if query.stress_source == item.stress_source:
                score += 1.0

        # risk bonus
        if query.need_risk_context and risk_level_index(item.risk_level or "level_0") >= 1:
            score += 1.5

        return score

    # ---------------------------------------------------------------
    # RRF fusion
    # ---------------------------------------------------------------

    def _rrf_fusion(
        self,
        result_sets: List[List[Tuple[MemoryItem, float]]],
    ) -> Dict[str, MemorySearchResult]:
        """Reciprocal Rank Fusion: score = sum(1 / (k + rank))"""
        fused: Dict[str, MemorySearchResult] = {}
        k = self.rrf_k

        for rank_set_idx, results in enumerate(result_sets):
            for rank, (item, _score) in enumerate(results):
                rrf_score = 1.0 / (k + rank + 1)
                if item.id in fused:
                    fused[item.id].score += rrf_score
                else:
                    method = ["bm25", "scored"][rank_set_idx]
                    fused[item.id] = MemorySearchResult(
                        item=item,
                        score=rrf_score,
                        rank=0,
                        retrieval_method=method,
                    )
        return fused

    # ---------------------------------------------------------------
    # Multi-factor reranking
    # ---------------------------------------------------------------

    def _rerank(
        self,
        fused: Dict[str, MemorySearchResult],
        query: MemoryQuery,
        now: datetime,
    ) -> List[MemorySearchResult]:
        """多因子重排。

        final_score =
          0.30 × rrf_score
        + 0.25 × importance
        + 0.15 × recency
        + 0.15 × semantic_similarity (Phase 3b: embedding)
        + 0.10 × emotion_match
        + 0.15 × risk_priority
        """
        results = list(fused.values())
        for r in results:
            item = r.item
            base = r.score

            # importance
            imp = item.importance or 0.5

            # recency
            rec = 0.5
            if item.created_at:
                try:
                    age_days = (now - item.created_at).days
                    rec = max(0, 1.0 - age_days / 180)
                except Exception:
                    pass

            # emotion match
            emo = 0.5 if (query.emotion and item.emotion and query.emotion == item.emotion) else 0.0

            # risk priority
            risk = 1.0 if (query.need_risk_context and risk_level_index(item.risk_level or "level_0") >= 1) else 0.0

            # final
            r.score = (
                0.30 * base
                + 0.25 * imp
                + 0.15 * rec
                + 0.10 * emo
                + 0.20 * risk
            )

        results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results


# 全局单例
memory_retriever = MemoryRetriever()
