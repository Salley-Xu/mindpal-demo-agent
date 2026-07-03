import asyncio
import os
import sys
import uuid

from fastapi.testclient import TestClient


PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("CHAT_MODEL", "deepseek-chat")
os.environ.setdefault("API_BASE_URL", "https://api.deepseek.com/v1")

import api_endpoints  # noqa: E402
from conversation_manager import conversation_manager  # noqa: E402
from main import app  # noqa: E402
from models import AgentRunResponse, ChatResponse, ContentItem  # noqa: E402


def test_chat_feedback_summary_regression():
    user_id = f"user_it_{uuid.uuid4().hex[:8]}"
    session_id = f"session_it_{uuid.uuid4().hex[:8]}"
    accepted_content_id = "tool_breath_001"
    rejected_content_id = "video_sleep_002"
    captured_profile_patch = {}
    expected_user_id = user_id

    original_run_agent = api_endpoints.agent_orchestrator.run_agent
    original_get_profile = api_endpoints.UserProfileTool.get_profile
    original_upsert_profile = api_endpoints.UserProfileTool.upsert_profile
    original_use_persistence = conversation_manager.use_persistence

    class DummyProfile:
        recommendation_feedback = {"legacy_item": "neutral"}

    async def fake_run_agent(request):
        emotion_state = {
            "current_emotion": "压力",
            "emotion_type": "stress",
            "context_emotion": "焦虑",
            "emotion_intensity": 0.84,
            "stress_source": "求职",
            "user_intent": "seeking_help",
            "negative_trend": True,
            "confidence": 0.93,
            "emotion_trend": "escalating",
        }
        risk_state = {
            "level": "level_1",
            "message": "需要继续关注压力变化",
            "suggestions": ["联系可信任的人", "尝试短时呼吸练习"],
            "triggers": ["求职压力", "长期疲惫"],
            "risk_score": 0.68,
        }

        conversation_manager.add_interaction(
            user_id=request.user_id,
            session_id=request.session_id,
            user_input=request.text,
            emotion=emotion_state["current_emotion"],
            context_emotion=emotion_state["context_emotion"],
            confidence=emotion_state["confidence"],
            ai_response="先一起把最急的压力点拆开。",
            emotion_state=emotion_state,
            risk_state={"level": risk_state["level"]},
        )
        conversation_manager.mark_recommendation(
            request.user_id,
            request.session_id,
            "soft",
            [accepted_content_id, rejected_content_id],
        )

        summary = await conversation_manager.get_conversation_summary_async(
            request.user_id, request.session_id
        )

        return AgentRunResponse(
            run_id="integration-run",
            chat=ChatResponse(
                response="先一起把最急的压力点拆开。",
                emotion_state=emotion_state,
                risk_state=risk_state,
                session_summary=summary,
                recommendation_decision={
                    "should_recommend": True,
                    "recommend_type": "soft",
                    "score": 0.78,
                    "threshold": 0.7,
                    "reason_codes": ["high_intensity", "seeking_help"],
                    "cooldown_remaining": 0,
                },
                recommendations=[
                    ContentItem(
                        id=accepted_content_id,
                        title="3分钟呼吸放松",
                        type="tool",
                        category="stress",
                        description="适合短时压力升高时使用",
                        recommend_type="soft",
                        source="content_db",
                        retrieval_metadata={
                            "final_score": 0.92,
                            "retrieval_score": 0.88,
                            "best_rank": 1,
                            "matched_queries": ["压力 求职 缓解方法"],
                            "retrieval_sources": ["bm25", "vector"],
                        },
                    ),
                    ContentItem(
                        id=rejected_content_id,
                        title="睡前放松音频",
                        type="audio",
                        category="stress",
                        description="帮助睡前放松",
                        recommend_type="soft",
                        source="content_db",
                        retrieval_metadata={
                            "final_score": 0.74,
                            "retrieval_score": 0.69,
                            "best_rank": 2,
                            "matched_queries": ["长期疲惫 放松"],
                            "retrieval_sources": ["bm25"],
                        },
                    ),
                ],
                recommendation_rationale="当前压力较高，先给你短时可执行的缓解内容。",
            ),
        )

    async def fake_get_profile(user_id):
        assert user_id == expected_user_id
        return DummyProfile()

    async def fake_upsert_profile(user_id, recommendation_feedback=None, **kwargs):
        assert user_id == expected_user_id
        captured_profile_patch.clear()
        captured_profile_patch.update(recommendation_feedback or {})
        return None

    api_endpoints.agent_orchestrator.run_agent = fake_run_agent
    api_endpoints.UserProfileTool.get_profile = fake_get_profile
    api_endpoints.UserProfileTool.upsert_profile = fake_upsert_profile
    conversation_manager.use_persistence = False

    try:
        with TestClient(app) as client:
            chat_resp = client.post(
                "/chat/intelligent",
                json={
                    "text": "最近求职压力很大，晚上也睡不好。",
                    "user_id": user_id,
                    "session_id": session_id,
                },
            )
            assert chat_resp.status_code == 200
            chat_payload = chat_resp.json()
            assert chat_payload["emotion_state"]["current_emotion"] == "压力"
            assert chat_payload["risk_state"]["level"] == "level_1"
            assert chat_payload["recommendation_decision"]["recommend_type"] == "soft"
            assert chat_payload["recommendations"][0]["retrieval_metadata"]["retrieval_sources"] == ["bm25", "vector"]

            summary_resp = client.get(f"/session/{user_id}/{session_id}/summary")
            assert summary_resp.status_code == 200
            summary_payload = summary_resp.json()["summary"]
            assert summary_payload["turn_count"] == 1
            assert summary_payload["stress_sources"] == ["求职"]
            assert summary_payload["recent_risk_levels"] == ["level_1"]
            assert summary_payload["recent_recommendation_turns"] == [1]

            helpful_resp = client.post(
                "/content/feedback",
                json={
                    "user_id": user_id,
                    "session_id": session_id,
                    "content_id": accepted_content_id,
                    "feedback": "helpful",
                },
            )
            assert helpful_resp.status_code == 200

            avoid_resp = client.post(
                "/content/feedback",
                json={
                    "user_id": user_id,
                    "session_id": session_id,
                    "content_id": rejected_content_id,
                    "feedback": "avoid",
                },
            )
            assert avoid_resp.status_code == 200

            summary_after_feedback = client.get(f"/session/{user_id}/{session_id}/summary")
            assert summary_after_feedback.status_code == 200
            updated_summary = summary_after_feedback.json()["summary"]
            assert accepted_content_id in updated_summary["accepted_recommendations"]
            assert rejected_content_id in updated_summary["rejected_recommendations"]

            assert captured_profile_patch["legacy_item"] == "neutral"
            assert captured_profile_patch[rejected_content_id] == "avoid"
    finally:
        api_endpoints.agent_orchestrator.run_agent = original_run_agent
        api_endpoints.UserProfileTool.get_profile = original_get_profile
        api_endpoints.UserProfileTool.upsert_profile = original_upsert_profile
        conversation_manager.use_persistence = original_use_persistence
        asyncio.run(conversation_manager.delete_session_async(user_id, session_id))


def main():
    test_chat_feedback_summary_regression()
    print("PASS: chat -> summary -> feedback regression")


if __name__ == "__main__":
    main()
