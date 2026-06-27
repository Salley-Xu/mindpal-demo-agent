import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RUNTIME_DEPS_DIR = PROJECT_ROOT / "third_party" / "agent_test_runtime"
LEGACY_VENDOR_DIR = BASE_DIR / "_vendor"
VENDOR_DIR = RUNTIME_DEPS_DIR / "vendor"
if not VENDOR_DIR.exists() and LEGACY_VENDOR_DIR.exists():
    VENDOR_DIR = LEGACY_VENDOR_DIR
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.append(str(VENDOR_DIR))
YAML_VENDOR_DIR = VENDOR_DIR / "yaml"
if YAML_VENDOR_DIR.exists() and str(YAML_VENDOR_DIR) not in sys.path:
    sys.path.append(str(YAML_VENDOR_DIR))

DEFAULT_EMBEDDING_MODEL = "hash-zh-v1"
DEFAULT_STRONG_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_STRONG_EMBEDDING_LOCAL_DIR = RUNTIME_DEPS_DIR / "models" / "bge-small-zh-v1.5"
DEFAULT_HASH_DIM = 512
SYNONYMS = {
    "焦虑": ["紧张", "担心", "不安", "anxiety"],
    "压力": ["压得喘不过气", "压力大", "stress", "任务过载"],
    "学业": ["考试", "论文", "复习", "成绩", "毕业", "答辩"],
    "求职": ["找工作", "面试", "实习", "职业规划"],
    "未来": ["方向", "规划", "迷茫", "career"],
    "睡眠": ["失眠", "睡不着", "入睡", "放松"],
    "关系": ["朋友", "室友", "对象", "沟通", "冲突", "吵架"],
    "放松": ["呼吸", "冥想", "正念", "relaxation"],
    "安全": ["危险", "危机", "一个人待着", "求助", "支持者"],
}


def _import_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "未安装 faiss-cpu。请先执行 `pip install faiss-cpu` 或安装 requirements.txt 中新增依赖。"
        ) from exc
    return faiss


def _import_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "未安装 sentence-transformers。请先执行 `pip install sentence-transformers` 或安装 requirements.txt 中新增依赖。"
        ) from exc
    return SentenceTransformer


def _import_transformers():
    try:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        os.environ.setdefault("TRANSFORMERS_DISABLE_GENERATION_IMPORT", "1")
        import torch
        from transformers.models.auto.modeling_auto import AutoModel
        from transformers.models.auto.tokenization_auto import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "未安装 transformers 相关依赖。请先将 transformers / tokenizers / huggingface_hub 等依赖安装到 "
            "`third_party/agent_test_runtime/vendor`。"
        ) from exc
    return torch, AutoModel, AutoTokenizer


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_searchable_text(doc: Dict[str, Any]) -> str:
    parts = [
        doc.get("title", ""),
        doc.get("category", ""),
        " ".join(doc.get("emotion_type", []) or []),
        " ".join(doc.get("risk_level", []) or []),
        " ".join(doc.get("tags", []) or []),
        doc.get("recommend_type", ""),
        doc.get("content", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def build_augmented_query(query: str, context: Dict[str, Any]) -> str:
    hints: List[str] = [query]
    emotion_type = context.get("emotion_type")
    risk_level = context.get("risk_level")
    user_preference = context.get("user_preference")
    if emotion_type:
        hints.append(f"情绪:{emotion_type}")
    if risk_level:
        hints.append(f"风险:{risk_level}")
    if user_preference:
        hints.append(f"偏好:{user_preference}")
    return " ".join(hints)


def tokenize_zh_text(text: str) -> List[str]:
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


def expand_tokens(tokens: List[str]) -> List[str]:
    expanded = list(tokens)
    for token in tokens:
        for root, synonyms in SYNONYMS.items():
            if token == root or token in synonyms:
                expanded.append(root)
                expanded.extend(synonyms)
    return expanded


def encode_with_hash_embeddings(
    texts: List[str],
    dim: int = DEFAULT_HASH_DIM,
) -> np.ndarray:
    matrix = np.zeros((len(texts), dim), dtype="float32")
    for row_index, text in enumerate(texts):
        token_counts = Counter(expand_tokens(tokenize_zh_text(text)))
        for token, count in token_counts.items():
            bucket = hash(token) % dim
            matrix[row_index, bucket] += float(count)
        norm = math.sqrt(float(np.dot(matrix[row_index], matrix[row_index])))
        if norm > 0:
            matrix[row_index] /= norm
    return matrix


_TRANSFORMER_EMBEDDER_CACHE: Dict[str, Tuple[Any, Any, Any]] = {}


def resolve_embedding_model_source(model_name: str) -> Tuple[str, bool]:
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path.resolve()), True

    if model_name == DEFAULT_STRONG_EMBEDDING_MODEL:
        local_dir = DEFAULT_STRONG_EMBEDDING_LOCAL_DIR
        required_files = [
            local_dir / "config.json",
            local_dir / "tokenizer_config.json",
            local_dir / "vocab.txt",
        ]
        has_weights = (local_dir / "model.safetensors").exists() or (local_dir / "pytorch_model.bin").exists()
        if local_dir.exists() and has_weights and all(path.exists() for path in required_files):
            return str(local_dir.resolve()), True

    return model_name, False


def _get_transformer_embedder(model_name: str):
    if model_name not in _TRANSFORMER_EMBEDDER_CACHE:
        torch, AutoModel, AutoTokenizer = _import_transformers()
        model_source, is_local_source = resolve_embedding_model_source(model_name)
        if not is_local_source and os.environ.get("ALLOW_REMOTE_HF_MODEL_DOWNLOAD") != "1":
            raise RuntimeError(
                "当前环境默认禁用远程 Hugging Face 模型下载，以避免 Windows 本地运行时崩溃。"
                f" 请先将模型文件放到 `{DEFAULT_STRONG_EMBEDDING_LOCAL_DIR}`，"
                " 或通过 `--embedding-model` 传入本地模型目录；若确认当前网络与依赖稳定，可显式设置"
                " `ALLOW_REMOTE_HF_MODEL_DOWNLOAD=1` 后再尝试在线加载。"
            )
        print(f"[embedding] loading tokenizer/model: {model_source}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=not is_local_source,
            local_files_only=is_local_source,
        )
        model = AutoModel.from_pretrained(
            model_source,
            trust_remote_code=not is_local_source,
            local_files_only=is_local_source,
        )
        model.eval()
        _TRANSFORMER_EMBEDDER_CACHE[model_name] = (torch, tokenizer, model)
    return _TRANSFORMER_EMBEDDER_CACHE[model_name]


def encode_with_transformers_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 16,
) -> np.ndarray:
    torch, tokenizer, model = _get_transformer_embedder(model_name)
    rows: List[np.ndarray] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            summed = torch.sum(hidden * attention_mask, dim=1)
            counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            embeddings = summed / counts
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        rows.append(embeddings.cpu().numpy().astype("float32"))
    return np.vstack(rows)


def encode_texts(
    texts: List[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    if model_name == DEFAULT_EMBEDDING_MODEL:
        return encode_with_hash_embeddings(texts)
    return encode_with_transformers_embeddings(
        texts,
        model_name=model_name,
        batch_size=batch_size,
    )


def build_faiss_store(
    docs: List[Dict[str, Any]],
    store_dir: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 32,
) -> Dict[str, Any]:
    faiss = _import_faiss()
    texts = [build_searchable_text(doc) for doc in docs]
    if not texts:
        raise ValueError("没有可用于构建 FAISS 索引的文档。")

    print(f"[faiss] encoding {len(texts)} documents with model={model_name}")
    embeddings = encode_texts(texts, model_name=model_name, batch_size=batch_size)
    dim = int(embeddings.shape[1])
    print(f"[faiss] building index dim={dim}")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    store_dir.mkdir(parents=True, exist_ok=True)
    index_path = store_dir / "faiss.index"
    meta_path = store_dir / "meta.json"
    build_info_path = store_dir / "build_info.json"
    faiss.write_index(index, str(index_path))
    print(f"[faiss] wrote index to {index_path}")

    meta_payload = {
        "items": [
            {
                "position": idx,
                "doc_id": doc.get("doc_id"),
                "searchable_text": texts[idx],
                "doc": doc,
            }
            for idx, doc in enumerate(docs)
        ]
    }
    write_json(meta_path, meta_payload)

    build_info = {
        "embedding_model": model_name,
        "dimension": dim,
        "doc_count": len(docs),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_type": "IndexFlatIP",
        "normalize_embeddings": True,
        "embedding_backend": "hash" if model_name == DEFAULT_EMBEDDING_MODEL else "transformers",
    }
    write_json(build_info_path, build_info)
    return {
        "index_path": str(index_path),
        "meta_path": str(meta_path),
        "build_info_path": str(build_info_path),
        "doc_count": len(docs),
        "dimension": dim,
        "embedding_model": model_name,
    }


def load_faiss_store(store_dir: Path) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    faiss = _import_faiss()
    index_path = store_dir / "faiss.index"
    meta_path = store_dir / "meta.json"
    build_info_path = store_dir / "build_info.json"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS 索引不存在: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"FAISS 元数据不存在: {meta_path}")
    if not build_info_path.exists():
        raise FileNotFoundError(f"FAISS 构建信息不存在: {build_info_path}")

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
    items = meta.get("items", [])
    return index, items, build_info


class FaissRetrievalBackend:
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.index, self.items, self.build_info = load_faiss_store(store_dir)
        self.embedding_model = self.build_info.get("embedding_model", DEFAULT_EMBEDDING_MODEL)

    def search(self, query: str, context: Dict[str, Any], top_k: int = 3) -> List[str]:
        augmented_query = build_augmented_query(query, context)
        query_embedding = encode_texts([augmented_query], model_name=self.embedding_model)
        distances, indices = self.index.search(query_embedding, top_k)
        result: List[str] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.items):
                continue
            doc_id = self.items[idx].get("doc_id")
            if doc_id:
                result.append(doc_id)
        return result
