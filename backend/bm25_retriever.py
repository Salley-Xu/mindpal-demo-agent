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
        """\u4e2d\u6587 BM25 \u5206\u8bcd\uff1a2-gram \u6ed1\u52a8\u7a97\u53e3 + \u82f1\u6587/\u6570\u5b57\u4fdd\u7559

        \u4e2d\u6587\u90e8\u5206\u751f\u6210\u6240\u6709\u76f8\u90bb\u5b57\u7b26\u7684\u53cc\u5b57\u7ec4\u5408\uff08\u6ed1\u7a97 2-gram\uff09\uff0c
        \u89e3\u51b3\u56fa\u5b9a 4 \u5b57\u5206\u5757\u5bfc\u81f4\u8de8\u5757 bigram \u4e22\u5931\u7684\u95ee\u9898\u3002
        """
        text = (text or "").lower()
        tokens: List[str] = []

        # \u5206\u79bb\u4e2d\u6587\u5e8f\u5217\u548c\u975e\u4e2d\u6587\uff08\u82f1\u6587/\u6570\u5b57\uff09
        chinese_blocks = re.findall(r"[\u4e00-\u9fa5]+", text)
        non_chinese = re.findall(r"[a-zA-Z0-9_]+", text)

        # \u4e2d\u6587\uff1a\u751f\u6210\u5168\u91cf\u6ed1\u7a97 2-gram
        for block in chinese_blocks:
            if len(block) >= 2:
                # \u5168\u90e8\u76f8\u90bb 2-gram
                tokens.extend(block[i:i+2] for i in range(len(block) - 1))
            # \u4fdd\u7559\u5355\u5b57\uff08\u5bf9\u77ed\u5173\u952e\u8bcd\u5982"\u6211" "\u4f60" \u6709\u7528\uff09
            if len(block) == 1:
                tokens.append(block)

        # \u975e\u4e2d\u6587\uff1a\u539f\u6837\u4fdd\u7559
        tokens.extend(non_chinese)

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
