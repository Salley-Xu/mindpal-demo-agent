import asyncio
import os
import sys
import uuid

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

from agent_orchestrator import agent_orchestrator  # noqa: E402
from agent_tools import UserProfileTool  # noqa: E402
from conversation_manager import conversation_manager  # noqa: E402
from emotion_analyzer import emotion_analyzer  # noqa: E402
from models import AgentRunRequest  # noqa: E402
from urgent_detector import urgent_detector  # noqa: E402


@pytest.mark.asyncio
async def test_high_risk_routes_to_dedicated_safety_response():
    user_id = f"user_safe_{uuid.uuid4().hex[:8]}"
    session_id = f"session_safe_{uuid.uuid4().hex[:8]}"
    llm_called = {"value": False}
    safety_called = {"value": False}

    original_use_persistence = conversation_manager.use_persistence
    original_emotion_method = emotion_analyzer.analyze_with_context_async
    original_safety_method = urgent_detector.generate_crisis_response_async
    original_chat_create = agent_orchestrator.client.chat.completions.create
    original_get_profile = UserProfileTool.get_profile

    async def fake_analyze_with_context_async(text, conversation_summary):
        return "绝望", "无助", 0.96

    async def fake_generate_crisis_response_async(user_input, urgent_issue, conversation_summary):
        safety_called["value"] = True
        assert urgent_issue["level"] == "level_3"
        return "请先联系你现在身边信任的人，并立即拨打心理援助热线。"

    async def fail_if_normal_llm_called(*args, **kwargs):
        llm_called["value"] = True
        raise AssertionError("high risk should not enter normal agent loop")

    async def fake_get_profile(user_id_arg):
        assert user_id_arg == user_id
        return None

    conversation_manager.use_persistence = False
    emotion_analyzer.analyze_with_context_async = fake_analyze_with_context_async
    urgent_detector.generate_crisis_response_async = fake_generate_crisis_response_async
    agent_orchestrator.client.chat.completions.create = fail_if_normal_llm_called
    UserProfileTool.get_profile = fake_get_profile

    try:
        result = await agent_orchestrator.run_agent(
            AgentRunRequest(
                text="我想自杀，真的不想活了。",
                user_id=user_id,
                session_id=session_id,
                return_steps=True,
            )
        )
        chat = result.chat
        assert safety_called["value"] is True
        assert llm_called["value"] is False
        assert chat.response == "请先联系你现在身边信任的人，并立即拨打心理援助热线。"
        assert chat.risk_state is not None
        assert chat.risk_state.level == "level_3"
        assert chat.risk_state.legacy_level == "high"
        assert chat.recommendation_decision is not None
        assert chat.recommendation_decision.should_recommend is False
        assert chat.recommendation_decision.recommend_type == "none"
        assert chat.recommendations is None
        assert any(step.name == "SafetyResponse" for step in (result.steps or []))
        assert all(step.name != "AgentLoop" for step in (result.steps or []))
    finally:
        conversation_manager.use_persistence = original_use_persistence
        emotion_analyzer.analyze_with_context_async = original_emotion_method
        urgent_detector.generate_crisis_response_async = original_safety_method
        agent_orchestrator.client.chat.completions.create = original_chat_create
        UserProfileTool.get_profile = original_get_profile
        await conversation_manager.delete_session_async(user_id, session_id)


@pytest.mark.asyncio
async def test_level_2_routes_to_high_risk_support_mode():
    user_id = f"user_support_{uuid.uuid4().hex[:8]}"
    session_id = f"session_support_{uuid.uuid4().hex[:8]}"
    llm_called = {"value": False}
    support_called = {"value": False}

    original_use_persistence = conversation_manager.use_persistence
    original_emotion_method = emotion_analyzer.analyze_with_context_async
    original_safety_method = urgent_detector.generate_crisis_response_async
    original_chat_create = agent_orchestrator.client.chat.completions.create
    original_get_profile = UserProfileTool.get_profile

    async def fake_analyze_with_context_async(text, conversation_summary):
        return "绝望", "无助", 0.91

    async def fake_generate_crisis_response_async(user_input, urgent_issue, conversation_summary):
        support_called["value"] = True
        assert urgent_issue["level"] == "level_2"
        return "你现在的状态值得认真对待。先不要一个人扛着，联系身边可信任的人，确认自己待在安全环境里。"

    async def fail_if_normal_llm_called(*args, **kwargs):
        llm_called["value"] = True
        raise AssertionError("level_2 should not enter normal agent loop")

    async def fake_get_profile(user_id_arg):
        assert user_id_arg == user_id
        return None

    conversation_manager.use_persistence = False
    emotion_analyzer.analyze_with_context_async = fake_analyze_with_context_async
    urgent_detector.generate_crisis_response_async = fake_generate_crisis_response_async
    agent_orchestrator.client.chat.completions.create = fail_if_normal_llm_called
    UserProfileTool.get_profile = fake_get_profile

    try:
        result = await agent_orchestrator.run_agent(
            AgentRunRequest(
                text="我最近真的快撑不住了，我怕自己会做出什么事。",
                user_id=user_id,
                session_id=session_id,
                return_steps=True,
            )
        )
        chat = result.chat
        assert support_called["value"] is True
        assert llm_called["value"] is False
        assert chat.risk_state is not None
        assert chat.risk_state.level == "level_2"
        assert chat.recommendation_decision is not None
        assert chat.recommendation_decision.should_recommend is False
        assert chat.recommendation_decision.recommend_type == "safety_only"
        assert chat.recommendations is None
        assert any(step.name == "HighRiskSupport" for step in (result.steps or []))
        assert all(step.name != "AgentLoop" for step in (result.steps or []))
    finally:
        conversation_manager.use_persistence = original_use_persistence
        emotion_analyzer.analyze_with_context_async = original_emotion_method
        urgent_detector.generate_crisis_response_async = original_safety_method
        agent_orchestrator.client.chat.completions.create = original_chat_create
        UserProfileTool.get_profile = original_get_profile
        await conversation_manager.delete_session_async(user_id, session_id)


def main():
    asyncio.run(test_high_risk_routes_to_dedicated_safety_response())
    print("PASS: high risk routes to dedicated safety response")

    asyncio.run(test_level_2_routes_to_high_risk_support_mode())
    print("PASS: level 2 routes to high-risk support mode")


if __name__ == "__main__":
    main()
