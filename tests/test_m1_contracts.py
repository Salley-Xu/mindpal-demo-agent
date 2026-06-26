import asyncio
import os
import sys


PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

# Ensure backend config can initialize in test context.
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("CHAT_MODEL", "deepseek-chat")
os.environ.setdefault("API_BASE_URL", "https://api.deepseek.com/v1")

from models import (  # noqa: E402
    ChatResponse,
    ContentItem,
    ContentRecommendRequest,
    EmotionState,
    RiskState,
    SessionSummary,
)
from main import app  # noqa: E402
import api_endpoints  # noqa: E402


def test_model_instantiation():
    response = ChatResponse(
        response="ok",
        emotion_state=EmotionState(
            current_emotion="焦虑",
            context_emotion="自我怀疑",
            confidence=0.91,
            emotion_trend="consistent",
        ),
        risk_state=RiskState(
            level="warning_high",
            message="需要关注",
            suggestions=["联系朋友"],
            triggers=["撑不住"],
            risk_score=6.5,
        ),
        session_summary=SessionSummary(
            conversation_stage="exploring",
            key_concerns=["academic"],
            turn_count=3,
            emotion_trend="consistent",
            primary_emotion="焦虑",
        ),
        recommendations=[
            ContentItem(
                id="article_001",
                title="测试内容",
                type="article",
                category="academic",
                description="desc",
            )
        ],
        recommendation_rationale="因为匹配",
    )
    assert response.emotion_state is not None
    assert response.risk_state is not None
    assert response.session_summary is not None


def test_openapi_schema():
    schema = app.openapi()

    recommend_post = schema["paths"]["/content/recommend"]["post"]
    assert "requestBody" in recommend_post, "/content/recommend 应声明 requestBody"

    chat_response_schema = schema["components"]["schemas"]["ChatResponse"]
    properties = chat_response_schema["properties"]
    assert "emotion_state" in properties
    assert "risk_state" in properties
    assert "session_summary" in properties


async def test_recommend_endpoint_with_body():
    original_recommend = api_endpoints.content_recommender.recommend_content
    original_get_profile = api_endpoints.UserProfileTool.get_profile

    async def fake_recommend_content(user_input, current_emotion, conversation_summary, user_profile, limit):
        assert user_input == "最近有点焦虑"
        assert current_emotion == "焦虑"
        assert conversation_summary["conversation_stage"] == "exploring"
        assert conversation_summary["key_concerns"] == ["academic", "future"]
        assert limit == 2
        return [], "ok", {"default": 1.0}

    async def fake_get_profile(user_id):
        assert user_id == "user_test"
        return None

    api_endpoints.content_recommender.recommend_content = fake_recommend_content
    api_endpoints.UserProfileTool.get_profile = fake_get_profile
    try:
        result = await api_endpoints.recommend_content(
            ContentRecommendRequest(
                user_input="最近有点焦虑",
                current_emotion="焦虑",
                conversation_stage="exploring",
                key_concerns=["academic", "future"],
                user_id="user_test",
                limit=2,
            )
        )
        assert result.rationale == "ok"
        assert result.match_scores["default"] == 1.0
    finally:
        api_endpoints.content_recommender.recommend_content = original_recommend
        api_endpoints.UserProfileTool.get_profile = original_get_profile


def main():
    test_model_instantiation()
    print("PASS: model instantiation")

    test_openapi_schema()
    print("PASS: openapi schema")

    asyncio.run(test_recommend_endpoint_with_body())
    print("PASS: recommend endpoint body parsing")


if __name__ == "__main__":
    main()
