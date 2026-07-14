"""Dense embedding 检索器（bge-small-zh-v1.5）

为生产检索管线接入语义 embedding 检索，与 BM25 + 同义词向量形成 3 路 RRF 融合。
使用 transformers 库直接加载，绕开 sentence-transformers 的 OpenMP 冲突问题。
模型通过均值池化（mean pooling）将 token hidden states 编码为句子向量。
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from models import ContentItem

logger = logging.getLogger(__name__)


# 本地模型目录（优先使用）
_LOCAL_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "third_party", "agent_test_runtime", "models", "bge-small-zh-v1.5"
)
_DEFAULT_MODEL = _LOCAL_MODEL_DIR if os.path.isdir(os.path.join(_LOCAL_MODEL_DIR, "config.json")) else "BAAI/bge-small-zh-v1.5"


class DenseRetriever:
    """基于 transformers 的稠密向量检索器（bge-small-zh-v1.5）。"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name or os.getenv(
            "DENSE_RETRIEVER_MODEL_PATH", _DEFAULT_MODEL
        )
        self.device = device or os.getenv("DENSE_RETRIEVER_DEVICE", "cpu")
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """延迟加载 embedding 模型。"""
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
            logger.info(f"加载 dense 模型: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info(f"DenseRetriever 模型已加载 (device={self.device})")
        except Exception as e:
            logger.error(f"DenseRetriever 模型加载失败: {e}")
            raise

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """均值池化：对非 padding 位置的 token embedding 取平均。"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode(self, texts: List[str]) -> np.ndarray:
        """将文本列表编码为归一化的 embedding 向量矩阵。"""
        with torch.no_grad():
            encoded = self._tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self._model(**encoded)
            embeddings = self._mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy()

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
        if not items:
            return []
        if not query.strip():
            return []

        self._load_model()

        # 构建查询文本
        query_text = query.strip()
        if extra_terms:
            query_text = query_text + " " + " ".join(extra_terms)

        doc_texts = [self._build_document(item) for item in items]

        try:
            query_vec = self._encode([query_text])[0]
            doc_vecs = self._encode(doc_texts)
        except Exception as e:
            logger.error(f"Dense 编码失败: {e}")
            return []

        scores = doc_vecs @ query_vec

        scored = [
            (item, float(score))
            for item, score in zip(items, scores.tolist())
            if score > 0
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]


# 全局实例（延迟加载，首次 retrieve 时才加载模型）
dense_retriever = DenseRetriever()
