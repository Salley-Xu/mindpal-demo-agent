"""Knowledge FAISS Index — 为 knowledge_store 提供稠密向量检索

使用 FAISS + hash embedding（或 bge-small-zh，如 torch 可用）实现，
替代依赖 torch 的 DenseRetriever，提供 O(log n) 的向量检索。

嵌入策略（自动降级）:
  1. bge-small-zh-v1.5  — 如果 torch 可用（最高质量）
  2. hash-zh-v1         — 纯 numpy，零依赖（默认，稳定可用）

依赖:
  - faiss-cpu（必须）
  - numpy
  - transformers / torch（可选，仅 bge 模式需要）
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── 路径 ──

BACKEND_DIR = Path(__file__).resolve().parent
FAISS_STORE_DIR = BACKEND_DIR / "data" / "knowledge_faiss"

# ── 嵌入函数（从 agent_test_data/faiss_utils 改编，避免跨目录依赖） ──

_SYNONYM_MAP = {
    "焦虑": ["紧张", "担心", "不安", "恐慌", "anxiety"],
    "压力": ["压得喘不过气", "压力大", "负荷", "stress", "倦怠"],
    "自杀": ["想死", "轻生", "活不下去", "自伤", "结束生命"],
    "抑郁": ["情绪低落", "沮丧", "消沉", "绝望", "depression"],
    "愤怒": ["生气", "烦躁", "恼火", "暴躁", "anger"],
    "孤独": ["孤单", "寂寞", "被孤立", "loneliness"],
    "考试": ["学业", "复习", "成绩", "毕业", "论文"],
    "睡眠": ["失眠", "睡不着", "入睡", "休息"],
    "关系": ["朋友", "室友", "沟通", "冲突", "家人"],
    "放松": ["呼吸", "冥想", "正念", "relaxation"],
}


def _tokenize(text: str) -> List[str]:
    """中文 2-gram 滑动窗口分词（与 bm25_retriever 保持一致）"""
    import re
    text = text.lower()
    tokens: List[str] = []
    chinese_blocks = re.findall(r"[一-龥]+", text)
    non_chinese = re.findall(r"[a-zA-Z0-9_]+", text)
    for block in chinese_blocks:
        if len(block) >= 2:
            tokens.extend(block[i:i+2] for i in range(len(block) - 1))
        if len(block) == 1:
            tokens.append(block)
    tokens.extend(non_chinese)
    return tokens


def _expand_tokens(tokens: List[str]) -> List[str]:
    expanded = list(tokens)
    for token in tokens:
        for root, synonyms in _SYNONYM_MAP.items():
            if token == root or token in synonyms:
                expanded.append(root)
                expanded.extend(synonyms)
    return expanded


def _hash_embed(texts: List[str], dim: int = 512) -> np.ndarray:
    """Hash embedding：词袋 → hash 桶 → 归一化向量，纯 numpy"""
    import math
    from collections import Counter
    matrix = np.zeros((len(texts), dim), dtype="float32")
    for row_idx, text in enumerate(texts):
        counts = Counter(_expand_tokens(_tokenize(text)))
        for token, count in counts.items():
            bucket = hash(token) % dim
            matrix[row_idx, bucket] += float(count)
        norm = math.sqrt(float(np.dot(matrix[row_idx], matrix[row_idx])))
        if norm > 0:
            matrix[row_idx] /= norm
    return matrix


def _import_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError("需要 faiss-cpu。请执行 `pip install faiss-cpu`")


def _try_import_transformers():
    """尝试导入 transformers/torch，用于 bge 嵌入"""
    try:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import torch
        from transformers import AutoModel, AutoTokenizer
        return torch, AutoModel, AutoTokenizer
    except ImportError:
        return None


_BGE_CACHE: dict = {}  # 全局模型缓存


def _get_bge_model(model_dir: str):
    """缓存 bge 模型（只加载一次）"""
    if model_dir not in _BGE_CACHE:
        imp = _try_import_transformers()
        if imp is None:
            return None
        torch, AutoModel, AutoTokenizer = imp
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        model.eval()
        _BGE_CACHE[model_dir] = (torch, tokenizer, model)
    return _BGE_CACHE[model_dir]


def _bge_embed(texts: List[str], model_dir: str) -> Optional[np.ndarray]:
    """bge-small-zh-v1.5 嵌入（模型缓存，只加载一次）"""
    cached = _get_bge_model(model_dir)
    if cached is None:
        return None
    torch, tokenizer, model = cached
    try:
        rows = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                summed = torch.sum(hidden * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / counts
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            rows.append(emb.cpu().numpy().astype("float32"))
        return np.vstack(rows)
    except Exception as e:
        logger.warning(f"bge 嵌入失败: {e}")
        return None


# ── FAISS 索引管理器 ──

class KnowledgeFaissIndex:
    """knowledge_store 的 FAISS 向量索引

    用法:
        index = KnowledgeFaissIndex()
        index.build(chunks_dict)  # 或 load() 已有索引
        results = index.search(query, top_k=5)
    """

    def __init__(self, store_dir: str = str(FAISS_STORE_DIR)):
        self.store_dir = Path(store_dir)
        self.index: Any = None
        self.meta: List[Dict[str, Any]] = []  # [{chunk_id, doc_id, title}]
        self.embed_dim: int = 512
        self._embed_fn = _hash_embed  # 默认 hash-zh-v1
        self._is_built = False
        self._bge_model_dir: Optional[str] = None
        self._uses_bge: bool = False

        # 寻找本地 bge 模型
        self._try_find_bge_model()

    def _try_find_bge_model(self):
        candidates = [
            BACKEND_DIR / ".." / "third_party" / "agent_test_runtime" / "models" / "bge-small-zh-v1.5",
            BACKEND_DIR / ".." / "agent_test_data" / "hf_models" / "bge-small-zh-v1.5",
        ]
        for path in candidates:
            resolved = path.resolve()
            if (resolved / "config.json").exists():
                self._bge_model_dir = str(resolved)
                break

    def _bge_embed_with_fallback(self, texts: List[str]) -> np.ndarray:
        """bge 嵌入 + hash 降级"""
        emb = _bge_embed(texts, self._bge_model_dir)
        if emb is not None:
            return emb
        logger.warning("bge 嵌入失败，降级到 hash-zh-v1")
        self._uses_bge = False
        self._embed_fn = _hash_embed
        return _hash_embed(texts)

    # ── 构建/持久化 ──

    def build(self, chunks: Dict[str, Any]) -> Dict[str, Any]:
        """从 knowledge_store 的 chunks 构建索引"""
        faiss = _import_faiss()

        texts: List[str] = []
        meta_list: List[Dict[str, Any]] = []

        for chunk_id, chunk in chunks.items():
            title = chunk.title
            tags = " ".join(chunk.metadata.get("tags", []))
            category = chunk.metadata.get("category", "")
            content = chunk.content[:200]

            # 与 knowledge_store._build_search_text 保持一致的权重策略
            search_text = " ".join([title] * 3 + [tags] * 2 + [category] + [content])
            texts.append(search_text)
            meta_list.append({
                "chunk_id": chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
            })

        if not texts:
            raise ValueError("没有文档用于构建 FAISS 索引")

        # 尝试 bge → hash 降级
        embeddings = self._try_embed(texts)
        dim = int(embeddings.shape[1])
        self.embed_dim = dim

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.meta = meta_list
        self._is_built = True

        # 持久化
        self._save()

        return {
            "doc_count": len(texts),
            "dimension": dim,
            "embedding_backend": "bge" if self._embed_fn.__name__ == "_bge_embed_caller" else "hash",
        }

    def _try_embed(self, texts: List[str]) -> np.ndarray:
        """尝试 bge → hash 降级"""
        if self._bge_model_dir:
            emb = _bge_embed(texts, self._bge_model_dir)
            if emb is not None:
                self._uses_bge = True
                self._embed_fn = self._bge_embed_with_fallback
                logger.info(f"FAISS 使用 bge-small-zh 嵌入 (dim={emb.shape[1]})")
                return emb
        logger.info("FAISS 使用 hash-zh-v1 嵌入 (512d, 无需 torch)")
        return _hash_embed(texts)

    def _save(self):
        """持久化 FAISS 索引 + 元数据到磁盘"""
        faiss = _import_faiss()
        self.store_dir.mkdir(parents=True, exist_ok=True)

        idx_path = self.store_dir / "faiss.index"
        meta_path = self.store_dir / "meta.json"
        info_path = self.store_dir / "build_info.json"

        faiss.write_index(self.index, str(idx_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"items": self.meta}, f, ensure_ascii=False)
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({
                "embedding": "bge" if self._uses_bge else "hash",
                "dimension": int(self.index.d),
                "doc_count": len(self.meta),
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"FAISS 索引已持久化: {idx_path} ({len(self.meta)} 篇, {'bge' if self._uses_bge else 'hash'})")

    def load(self) -> bool:
        """从磁盘加载已构建的 FAISS 索引"""
        faiss = _import_faiss()
        idx_path = self.store_dir / "faiss.index"
        meta_path = self.store_dir / "meta.json"

        if not idx_path.exists() or not meta_path.exists():
            logger.info("FAISS 索引不存在，需要重建")
            return False

        try:
            self.index = faiss.read_index(str(idx_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f).get("items", [])
            self._is_built = True

            # 恢复嵌入类型：按 build_info 字段判断
            info_path = self.store_dir / "build_info.json"
            if info_path.exists():
                bi = json.loads(info_path.read_text(encoding="utf-8"))
                emb_type = bi.get("embedding", "")
                if "bge" in emb_type:
                    self._uses_bge = True
                    if self._bge_model_dir:
                        self._embed_fn = self._bge_embed_with_fallback
                    else:
                        # bge model dir not found but index is bge → can't search with bge
                        self._uses_bge = False
                        self._embed_fn = _hash_embed
                        logger.warning("bge model not found, bge index will not be used for search")
                else:
                    self._uses_bge = False
                    self._embed_fn = _hash_embed

            logger.info(f"FAISS 索引已加载 ({len(self.meta)} 篇, dim={self.index.d}, bge={self._uses_bge})")
            return True
        except Exception as e:
            logger.warning(f"FAISS 索引加载失败: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        return self._is_built and self.index is not None

    @property
    def uses_bge(self) -> bool:
        """是否使用 bge-small-zh 真向量（而非 hash）"""
        return self._uses_bge

    # ── 检索 ──

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """向量检索，返回 [(chunk_id, score), ...]"""
        if not self.is_ready:
            logger.warning("FAISS 索引未就绪")
            return []

        t0 = time.time()

        query_vec = self._embed_fn([query])
        distances, indices = self.index.search(query_vec.astype("float32"), top_k)

        results: List[Tuple[str, float]] = []
        seen_docs: set = set()
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            item = self.meta[idx]
            doc_id = item["doc_id"]
            # 去重：同一文档只保留最高分切片
            if doc_id not in seen_docs:
                results.append((item["chunk_id"], float(dist)))
                seen_docs.add(doc_id)

        elapsed = (time.time() - t0) * 1000
        logger.debug(f"FAISS 检索: {elapsed:.0f}ms, top={len(results)}")
        return results

    def rebuild_if_needed(self, chunks: Dict[str, Any]) -> bool:
        """如果 chunks 数量变化，自动重建"""
        if self.is_ready and len(self.meta) == len(chunks):
            return False  # 无需重建
        if self.is_ready:
            logger.info(f"chunks 数量变化 ({len(self.meta)} → {len(chunks)})，重建 FAISS 索引")
        self.build(chunks)
        return True


# ── 全局实例 ──

knowledge_faiss = KnowledgeFaissIndex()
