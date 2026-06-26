import math
import re
from collections import Counter
from typing import Dict, List, Tuple

from models import ContentItem


class VectorRetriever:
    """轻量语义近似检索器，使用同义词扩展 + 稀疏向量余弦相似度。"""

    def __init__(self):
        self.synonyms = {
            "焦虑": ["紧张", "担心", "不安", "anxiety"],
            "压力": ["压得喘不过气", "压力大", "stress", "任务过载"],
            "学业": ["考试", "论文", "复习", "成绩", "毕业"],
            "求职": ["找工作", "面试", "实习", "职业规划"],
            "未来": ["方向", "规划", "迷茫", "career"],
            "睡眠": ["失眠", "睡不着", "入睡", "放松"],
            "关系": ["朋友", "室友", "对象", "沟通", "冲突"],
            "放松": ["呼吸", "冥想", "正念", "relaxation"],
        }

    def retrieve(
        self,
        query: str,
        items: List[ContentItem],
        limit: int = 10,
        extra_terms: List[str] | None = None,
    ) -> List[Tuple[ContentItem, float]]:
        if not items:
            return []

        query_tokens = self._expand_tokens(self._tokenize(query))
        for term in extra_terms or []:
            query_tokens.extend(self._expand_tokens(self._tokenize(term)))
        if not query_tokens:
            return []

        query_vector = Counter(query_tokens)
        scored: List[Tuple[ContentItem, float]] = []
        for item in items:
            doc_tokens = self._expand_tokens(self._tokenize(self._build_document(item)))
            if not doc_tokens:
                continue
            score = self._cosine_similarity(query_vector, Counter(doc_tokens))
            if score > 0:
                scored.append((item, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def _build_document(self, item: ContentItem) -> str:
        return " ".join(
            [
                item.title,
                item.description,
                item.category,
                item.type,
                " ".join(item.tags),
                " ".join(item.emotion_tags),
                item.difficulty or "",
            ]
        )

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

    def _expand_tokens(self, tokens: List[str]) -> List[str]:
        expanded = list(tokens)
        for token in tokens:
            for root, synonyms in self.synonyms.items():
                if token == root or token in synonyms:
                    expanded.append(root)
                    expanded.extend(synonyms)
        return expanded

    def _cosine_similarity(self, left: Counter, right: Counter) -> float:
        common_terms = set(left.keys()) & set(right.keys())
        dot = sum(left[term] * right[term] for term in common_terms)
        if dot == 0:
            return 0.0
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / max(left_norm * right_norm, 1e-9)


vector_retriever = VectorRetriever()
