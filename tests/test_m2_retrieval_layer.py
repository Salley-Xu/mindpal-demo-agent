import asyncio
import os
import sys


PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("CHAT_MODEL", "deepseek-chat")
os.environ.setdefault("API_BASE_URL", "https://api.deepseek.com/v1")

from bm25_retriever import bm25_retriever  # noqa: E402
from content_db import content_db  # noqa: E402
from content_recommender import content_recommender  # noqa: E402
from hybrid_retriever import hybrid_retriever  # noqa: E402
from vector_retriever import vector_retriever  # noqa: E402


def test_bm25_retriever_relevance():
    items = content_db.get_all_content()
    results = bm25_retriever.retrieve(
        query="考试焦虑和学业压力太大了",
        items=items,
        limit=3,
        extra_terms=["焦虑", "academic"],
    )
    assert results
    top_ids = [item.id for item, _score in results]
    assert "article_001" in top_ids


def test_vector_and_hybrid_retrieval():
    items = content_db.get_all_content()
    vector_results = vector_retriever.retrieve(
        query="最近面试让我特别紧张，不知道怎么放松",
        items=items,
        limit=5,
        extra_terms=["求职", "焦虑"],
    )
    hybrid_results = hybrid_retriever.retrieve(
        query="最近面试让我特别紧张，不知道怎么放松",
        items=items,
        limit=5,
        extra_terms=["求职", "焦虑"],
    )
    assert vector_results
    assert hybrid_results
    hybrid_top_ids = [item.id for item, _score in hybrid_results[:3]]
    assert any(item_id in hybrid_top_ids for item_id in ["audio_001", "audio_002", "article_003"])


async def test_content_recommender_still_works():
    original_enable_ai = content_recommender.enable_ai_rerank
    content_recommender.enable_ai_rerank = False
    try:
        recs, rationale, scores = await content_recommender.recommend_content(
            user_input="最近考试和未来规划压得我喘不过气，我想找个可以马上做的方法。",
            current_emotion="焦虑",
            conversation_summary={
                "conversation_stage": "resolving",
                "key_concerns": ["academic", "future"],
                "stress_sources": ["学业/求职压力", "未来规划压力"],
            },
            user_profile={
                "risk_level": "low",
                "preferred_support_style": "direct_actionable",
                "preferred_types": ["audio", "article"],
                "preferred_categories": ["academic", "relaxation", "future"],
                "preferred_difficulty": "beginner",
            },
            limit=3,
        )
        assert recs
        assert rationale
        assert scores
        rec_ids = [item.id for item in recs]
        assert any(item_id in rec_ids for item_id in ["article_001", "audio_001", "audio_002", "article_003"])
    finally:
        content_recommender.enable_ai_rerank = original_enable_ai


def main():
    test_bm25_retriever_relevance()
    print("PASS: bm25 retriever relevance")

    test_vector_and_hybrid_retrieval()
    print("PASS: vector and hybrid retrieval")

    asyncio.run(test_content_recommender_still_works())
    print("PASS: content recommender still works")


if __name__ == "__main__":
    main()
