import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from models import ContentItem


class BM25Retriever:
    """轻量 BM25 检索器，不依赖外部库。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def retrieve(
        self,
        query: str,
        items: List[ContentItem],
        limit: int = 10,
        extra_terms: List[str] | None = None,
    ) -> List[Tuple[ContentItem, float]]:
        if not items:
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

        scored: List[Tuple[ContentItem, float]] = []
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

    def _build_document(self, item: ContentItem) -> str:
        parts = [
            item.title,
            item.description,
            item.category,
            " ".join(item.tags),
            " ".join(item.emotion_tags),
            item.type,
            item.difficulty or "",
        ]
        return " ".join(parts)

    def _tokenize(self, text: str) -> List[str]:
        chunks = re.findall(r"[\u4e00-\u9fa5]{1,4}|[a-zA-Z0-9_]+", (text or "").lower())
        tokens: List[str] = []
        for chunk in chunks:
            if re.fullmatch(r"[\u4e00-\u9fa5]{2,4}", chunk):
                tokens.append(chunk)
                if len(chunk) >= 3:
                    tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
            else:
                tokens.append(chunk)
        return tokens

    def _document_frequencies(self, doc_tokens: List[List[str]]) -> Dict[str, int]:
        frequencies: Dict[str, int] = {}
        for tokens in doc_tokens:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def _idf(self, term: str, total_docs: int, doc_freqs: Dict[str, int]) -> float:
        doc_freq = doc_freqs.get(term, 0)
        return math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))


bm25_retriever = BM25Retriever()
