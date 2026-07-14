"""Dense embedding 检索器（bge-small-zh-v1.5）

为生产检索管线接入语义 embedding 检索，与 BM25 + 同义词向量形成 3 路 RRF 融合。
模型文件通过 sentence-transformers 加载，默认从 HuggingFace 在线加载 bge-small-zh-v1.5，
也可通过 DENSE_RETRIEVER_MODEL_PATH 指定本地目录。
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

from models import ContentItem

logger = logging.getLogger(__name__)


class DenseRetriever:
    """基于 sentence-transformer 的稠密向量检索器。"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name or os.getenv(
            "DENSE_RETRIEVER_MODEL_PATH", "BAAI/bge-small-zh-v1.5"
        )
        self.device = device or os.getenv("DENSE_RETRIEVER_DEVICE", "cpu")
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        """延迟加载 embedding 模型。"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                f"DenseRetriever 模型已加载: {self.model_name} (device={self.device})"
            )
        except Exception as e:
            logger.error(f"DenseRetriever 模型加载失败: {e}")
            raise

    def _build_document(self, item: ContentItem) -> str:
        """构建用于编码的文档文本（与 BM25 保持一致）。"""
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

    def retrieve(
        self,
        query: str,
        items: List[ContentItem],
        limit: int = 10,
        extra_terms: Optional[List[str]] = None,
    ) -> List[Tuple[ContentItem, float]]:
        """dense 检索入口。

        将查询和所有内容编码为向量，计算余弦相似度（内积，因 normalize=True），
        返回 Top-K。
        """
        if not items:
            return []
        if not query.strip():
            return []

        self._load_model()

        # 构建查询文本
        query_text = query.strip()
        if extra_terms:
            query_text = query_text + " " + " ".join(extra_terms)

        # 构建文档文本
        doc_texts = [self._build_document(item) for item in items]

        try:
            query_vec = self._model.encode(
                query_text, normalize_embeddings=True, show_progress_bar=False
            )
            doc_vecs = self._model.encode(
                doc_texts,
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Dense 编码失败: {e}")
            return []

        # 余弦 = dot product (因为 normalize_embeddings=True)
        scores = doc_vecs @ query_vec

        # 排序并返回 Top-K
        scored = [
            (item, float(score))
            for item, score in zip(items, scores.tolist())
            if score > 0
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]


# 全局实例（延迟加载，首次 retrieve 时才加载模型）
dense_retriever = DenseRetriever()
