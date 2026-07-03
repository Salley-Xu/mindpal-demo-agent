import asyncio
import os
import sys

import pytest


PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("CHAT_MODEL", "deepseek-chat")
os.environ.setdefault("API_BASE_URL", "https://api.deepseek.com/v1")

from api_endpoints import submit_recommendation_feedback  # noqa: E402
from conversation_manager import ConversationManager  # noqa: E402
from main import app  # noqa: E402
from models import RecommendationFeedbackRequest  # noqa: E402
import api_endpoints  # noqa: E402


@pytest.mark.asyncio
async def test_feedback_endpoint_updates_profile():
    captured = {"patch": None}

    class DummyProfile:
        recommendation_feedback = {"old_item": "neutral"}

    original_get_profile = api_endpoints.UserProfileTool.get_profile
    original_upsert_profile = api_endpoints.UserProfileTool.upsert_profile
    original_record_feedback = api_endpoints.conversation_manager.record_recommendation_feedback

    async def fake_get_profile(user_id):
        assert user_id == "user_fb"
        return DummyProfile()

    async def fake_upsert_profile(user_id, recommendation_feedback=None, **kwargs):
        assert user_id == "user_fb"
        captured["patch"] = recommendation_feedback
        return None

    def fake_record_feedback(user_id, session_id, content_id, feedback):
        assert user_id == "user_fb"
        assert session_id == "session_fb"
        assert content_id == "audio_002"
        assert feedback == "helpful"

    api_endpoints.UserProfileTool.get_profile = fake_get_profile
    api_endpoints.UserProfileTool.upsert_profile = fake_upsert_profile
    api_endpoints.conversation_manager.record_recommendation_feedback = fake_record_feedback
    try:
        result = await submit_recommendation_feedback(
            RecommendationFeedbackRequest(
                user_id="user_fb",
                session_id="session_fb",
                content_id="audio_002",
                feedback="helpful",
            )
        )
        assert result.feedback == "helpful"
        assert captured["patch"]["old_item"] == "neutral"
        assert captured["patch"]["audio_002"] == "helpful"
    finally:
        api_endpoints.UserProfileTool.get_profile = original_get_profile
        api_endpoints.UserProfileTool.upsert_profile = original_upsert_profile
        api_endpoints.conversation_manager.record_recommendation_feedback = original_record_feedback


@pytest.mark.asyncio
async def test_conversation_feedback_updates_summary_state():
    manager = ConversationManager(use_persistence=False)
    manager.add_interaction(
        user_id="user_fb",
        session_id="session_fb",
        user_input="我想试试呼吸练习。",
        emotion="焦虑",
        ai_response="可以试试正念呼吸。",
        emotion_state={"emotion_type": "anxiety", "emotion_intensity": 0.7, "user_intent": "seeking_help"},
        risk_state={"level": "low"},
    )
    manager.record_recommendation_feedback("user_fb", "session_fb", "audio_002", "helpful")
    manager.record_recommendation_feedback("user_fb", "session_fb", "tool_001", "avoid")
    summary = await manager.get_conversation_summary_async("user_fb", "session_fb")
    assert "audio_002" in summary["accepted_recommendations"]
    assert "tool_001" in summary["rejected_recommendations"]


def test_openapi_feedback_schema():
    schema = app.openapi()
    assert "/content/feedback" in schema["paths"]
    request_schema = schema["components"]["schemas"]["RecommendationFeedbackRequest"]["properties"]
    assert "content_id" in request_schema
    assert "feedback" in request_schema


def main():
    asyncio.run(test_feedback_endpoint_updates_profile())
    print("PASS: feedback endpoint updates profile")

    asyncio.run(test_conversation_feedback_updates_summary_state())
    print("PASS: conversation feedback updates summary state")

    test_openapi_feedback_schema()
    print("PASS: feedback openapi schema")


if __name__ == "__main__":
    main()
