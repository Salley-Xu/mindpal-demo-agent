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

from content_db import content_db  # noqa: E402
from content_recommender import content_recommender  # noqa: E402


def test_content_schema_upgraded():
    item = content_db.get_content_by_id("audio_002")
    assert item is not None
    assert item.risk_levels
    assert item.recommend_type in {"soft", "hard"}
    assert item.priority > 0
    assert item.actionability > 0


def test_query_variants_and_candidate_metadata():
    variants = content_recommender._build_query_variants(
        user_input="最近事情太多，我完全不知道怎么开始。",
        current_emotion="焦虑",
        conversation_summary={
            "key_concerns": ["academic", "future"],
            "stress_sources": ["学业/求职压力"],
            "recent_intents": ["planning"],
        },
        user_profile={"main_stress_sources": ["学业/求职压力"]},
    )
    labels = [variant["label"] for variant in variants]
    assert "raw_input" in labels
    assert "emotion_enhanced" in labels
    assert "stress_source" in labels
    assert "intent_enhanced" in labels
    assert "long_memory" in labels

    candidates = content_recommender._retrieve_candidates(
        user_input="最近事情太多，我完全不知道怎么开始。",
        current_emotion="焦虑",
        conversation_summary={
            "key_concerns": ["academic", "future"],
            "stress_sources": ["学业/求职压力"],
            "recent_intents": ["planning"],
        },
        user_profile={"main_stress_sources": ["学业/求职压力"]},
        limit=6,
    )
    assert candidates
    assert "retrieval_score" in candidates[0]
    assert "matched_queries" in candidates[0]
    assert "retrieval_sources" in candidates[0]


async def test_rerank_pipeline_metadata():
    original_enable_ai = content_recommender.enable_ai_rerank
    content_recommender.enable_ai_rerank = False
    try:
        recs, rationale, scores = await content_recommender.recommend_content(
            user_input="最近考试、面试和未来规划压得我喘不过气，我想要一个具体可执行的方法。",
            current_emotion="焦虑",
            conversation_summary={
                "conversation_stage": "resolving",
                "key_concerns": ["academic", "future"],
                "stress_sources": ["学业/求职压力", "未来规划压力"],
                "recent_intents": ["seeking_help", "planning"],
                "recent_risk_levels": ["low"],
                "recent_recommendation_turns": [],
            },
            user_profile={
                "risk_level": "low",
                "preferred_types": ["audio", "tool"],
                "preferred_categories": ["academic", "mindfulness", "future"],
                "preferred_difficulty": "beginner",
                "main_stress_sources": ["学业/求职压力"],
            },
            limit=3,
        )
        assert recs
        assert rationale
        assert scores
        for item in recs:
            assert item.retrieval_metadata is not None
            assert "retrieval_score" in item.retrieval_metadata
            assert "final_score" in item.retrieval_metadata
            assert "matched_queries" in item.retrieval_metadata
    finally:
        content_recommender.enable_ai_rerank = original_enable_ai


def main():
    test_content_schema_upgraded()
    print("PASS: content schema upgraded")

    test_query_variants_and_candidate_metadata()
    print("PASS: query variants and candidate metadata")

    asyncio.run(test_rerank_pipeline_metadata())
    print("PASS: rerank pipeline metadata")


if __name__ == "__main__":
    main()
