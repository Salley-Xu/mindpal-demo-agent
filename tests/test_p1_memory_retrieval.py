import asyncio
import os
import sys
from datetime import datetime, timezone

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

from agent_orchestrator import AgentOrchestrator  # noqa: E402
from agent_tools import MoodEvent, MoodTrackingTool  # noqa: E402


@pytest.mark.asyncio
async def test_build_initial_messages_includes_relevant_long_term_memory():
    orchestrator = AgentOrchestrator()
    original_get_recent_trend = MoodTrackingTool.get_recent_trend

    async def fake_get_recent_trend(user_id: str, limit: int = 20):
        return [
            MoodEvent(
                user_id=user_id,
                session_id="session-a",
                emotion="焦虑",
                emotion_type="anxiety",
                emotion_intensity=0.86,
                stress_source="求职",
                user_intent="seeking_help",
                event_summary="用户围绕求职和面试准备反复感到紧张，担心表达不好。",
                risk_level="medium",
                source="conversation",
                text_snippet="一想到面试就很慌，怕自己讲不清项目。",
                created_at=datetime.now(timezone.utc),
            ),
            MoodEvent(
                user_id=user_id,
                session_id="session-b",
                emotion="难过",
                emotion_type="sadness",
                emotion_intensity=0.72,
                stress_source="关系",
                user_intent="sharing",
                event_summary="用户提到和朋友闹矛盾，心情低落。",
                risk_level="low",
                source="conversation",
                text_snippet="和朋友吵架后觉得很委屈。",
                created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            ),
        ]

    MoodTrackingTool.get_recent_trend = staticmethod(fake_get_recent_trend)
    try:
        messages = await orchestrator._build_initial_messages(
            text="我下周还有面试，现在又开始紧张了。",
            history=[
                {
                    "user_input": "我最近找工作压力很大。",
                    "ai_response": "我们可以先把最担心的部分拆开来看。",
                }
            ],
            summary={
                "conversation_stage": "deepening",
                "primary_emotion": "焦虑",
                "emotion_trend": "escalating",
                "key_concerns": ["future", "self"],
                "current_topic": "学业与求职",
                "recent_intents": ["seeking_help", "planning"],
                "stress_sources": ["求职", "面试"],
                "accepted_recommendations": ["tool_001"],
                "rejected_recommendations": ["audio_003"],
                "compressed_context": "历史对话摘要: 用户持续提到求职和面试压力。",
            },
            urgent_issue={
                "level": "low",
                "message": "",
                "risk_score": 0.18,
            },
            user_id="user_memory_focus",
            emotion_state={
                "current_emotion": "焦虑",
                "emotion_type": "anxiety",
                "context_emotion": "压力",
                "emotion_intensity": 0.81,
                "user_intent": "seeking_help",
                "stress_source": "求职",
                "negative_trend": True,
            },
            user_profile={
                "risk_level": "medium",
                "preferred_support_style": "direct_actionable",
                "avoid_style": ["empty_comfort"],
                "main_stress_sources": ["家庭", "求职", "面试"],
                "recommendation_feedback": {
                    "tool_001": "helpful",
                    "audio_003": "avoid",
                },
            },
            recommendation_decision={
                "should_recommend": False,
                "recommend_type": "none",
                "reason_codes": ["cooldown"],
            },
        )
    finally:
        MoodTrackingTool.get_recent_trend = original_get_recent_trend

    system_prompt = messages[0]["content"]
    assert "【相关长期记忆】" in system_prompt
    assert "记忆检索焦点: 学业与求职, 学业, 求职, 面试" in system_prompt
    assert "相关长期压力源: 求职, 面试" in system_prompt
    assert "稳定支持偏好: 偏好 direct_actionable；避免 empty_comfort" in system_prompt
    assert "历史推荐反馈: 更可能接受 tool_001；明确拒绝 audio_003" in system_prompt
    assert "相似经历1: 情绪=焦虑，主题=求职，风险=level_1" in system_prompt
    assert "用户围绕求职和面试准备反复感到紧张" in system_prompt
    assert "和朋友闹矛盾" not in system_prompt


def main():
    asyncio.run(test_build_initial_messages_includes_relevant_long_term_memory())
    print("PASS: relevant long-term memory is injected into prompt")


if __name__ == "__main__":
    main()
