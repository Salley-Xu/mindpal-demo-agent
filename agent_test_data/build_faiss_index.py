import argparse
from pathlib import Path

from faiss_utils import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_STRONG_EMBEDDING_LOCAL_DIR,
    build_faiss_store,
    load_jsonl,
)

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent


def resolve_cli_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def main():
    parser = argparse.ArgumentParser(description="Build a local FAISS store from rag_materials.jsonl.")
    parser.add_argument(
        "--input-jsonl",
        default=str(BASE / "rag_materials.jsonl"),
        help="Path to rag materials JSONL.",
    )
    parser.add_argument(
        "--store-dir",
        default=str(BASE / "faiss_store"),
        help="Directory to save faiss.index, meta.json and build_info.json.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Embedding model name or local model directory. `hash-zh-v1` uses local hashed features; "
            "other values use transformers + torch. Remote Hugging Face download is disabled by default, "
            f"so prefer a local directory such as `{DEFAULT_STRONG_EMBEDDING_LOCAL_DIR}`."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used by the embedding encoder.",
    )
    args = parser.parse_args()

    input_path = resolve_cli_path(args.input_jsonl)
    store_dir = resolve_cli_path(args.store_dir)

    docs = load_jsonl(input_path)
    result = build_faiss_store(
        docs=docs,
        store_dir=store_dir,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
    )
    print("[faiss_store]")
    print(f"- input_jsonl: {input_path}")
    print(f"- store_dir: {store_dir}")
    print(f"- embedding_model: {result['embedding_model']}")
    print(f"- doc_count: {result['doc_count']}")
    print(f"- dimension: {result['dimension']}")
    print(f"- index_path: {result['index_path']}")


if __name__ == "__main__":
    main()
