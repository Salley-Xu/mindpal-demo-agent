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
from output_safety_checker import output_safety_checker  # noqa: E402
from risk_evaluator import risk_evaluator  # noqa: E402
from urgent_detector import urgent_detector  # noqa: E402


def test_output_safety_checker_rewrites_invalidating_response():
    review = output_safety_checker.review(
        response_text="你想太多了，先自己一个人待着冷静一下。",
        risk_state={"level": "level_0"},
        recommendation_decision={"should_recommend": False, "recommend_type": "none"},
    )
    assert review["used_fallback"] is True
    assert "invalidating_language" in review["triggered_rules"]
    assert "encourage_isolation" in review["triggered_rules"]
    assert "你想太多了" not in review["response_text"]
    assert "自己一个人待着" not in review["response_text"]


def test_output_safety_checker_blocks_level_2_recommendations():
    review = output_safety_checker.review(
        response_text="先去看一篇冥想文章，再做个运动计划。",
        risk_state={"level": "level_2"},
        recommendation_decision={"should_recommend": True, "recommend_type": "hard"},
        recommendations=[{"id": "relax-breathing"}],
        recommendation_rationale="先推荐一篇文章帮助放松。",
    )
    assert review["used_fallback"] is True
    assert "ordinary_recommendation_in_high_risk" in review["triggered_rules"]
    assert "high_risk_recommendations_blocked" in review["triggered_rules"]
    assert review["recommendations"] == []
    assert review["recommendation_rationale"] == ""
    assert review["recommendation_decision"]["should_recommend"] is False
    assert review["recommendation_decision"]["recommend_type"] == "safety_only"


@pytest.mark.asyncio
async def test_orchestrator_rewrites_unsafe_level_3_response_before_return():
    user_id = f"user_output_safe_{uuid.uuid4().hex[:8]}"
    session_id = f"session_output_safe_{uuid.uuid4().hex[:8]}"

    original_use_persistence = conversation_manager.use_persistence
    original_emotion_method = emotion_analyzer.analyze_with_context_async
    original_safety_method = urgent_detector.generate_crisis_response_async
    original_chat_create = agent_orchestrator.client.chat.completions.create
    original_get_profile = UserProfileTool.get_profile
    original_evaluate = risk_evaluator.evaluate

    async def fake_analyze_with_context_async(text, conversation_summary):
        return "绝望", "无助", 0.98

    async def fake_generate_crisis_response_async(user_input, urgent_issue, conversation_summary):
        assert urgent_issue["level"] == "level_3"
        return "你一定不会有事，先一个人待着冷静下，我再给你推荐一篇冥想文章。"

    async def fail_if_normal_llm_called(*args, **kwargs):
        raise AssertionError("level_3 should not enter normal agent loop")

    async def fake_get_profile(user_id_arg):
        assert user_id_arg == user_id
        return None

    def fake_evaluate(**kwargs):
        return {
            "level": "level_3",
            "legacy_level": "high",
            "level_index": 3,
            "level_label": "Level 3",
            "message": "",
            "suggestions": [],
            "triggers": [],
            "risk_score": 9.0,
            "raw_score": 9.0,
            "risk_dimensions": {},
            "risk_evidence": {},
            "escalation_reasons": [],
            "recent_risk_levels": [],
            "risk_context": {"subject": "self", "is_third_party_risk": False,
                             "is_help_request": False, "is_discussion_context": False,
                             "is_safe_denial": False},
        }

    conversation_manager.use_persistence = False
    emotion_analyzer.analyze_with_context_async = fake_analyze_with_context_async
    urgent_detector.generate_crisis_response_async = fake_generate_crisis_response_async
    agent_orchestrator.client.chat.completions.create = fail_if_normal_llm_called
    UserProfileTool.get_profile = fake_get_profile
    risk_evaluator.evaluate = fake_evaluate

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
        assert "一定不会有事" not in chat.response
        assert "一个人待着" not in chat.response
        assert "冥想文章" not in chat.response
        assert "请先立刻联系" in chat.response
        assert any(step.name == "OutputSafetyCheck" for step in (result.steps or []))
    finally:
        conversation_manager.use_persistence = original_use_persistence
        emotion_analyzer.analyze_with_context_async = original_emotion_method
        urgent_detector.generate_crisis_response_async = original_safety_method
        agent_orchestrator.client.chat.completions.create = original_chat_create
        UserProfileTool.get_profile = original_get_profile
        risk_evaluator.evaluate = original_evaluate
        await conversation_manager.delete_session_async(user_id, session_id)


def main():
    test_output_safety_checker_rewrites_invalidating_response()
    print("PASS: output safety checker rewrites invalidating response")

    test_output_safety_checker_blocks_level_2_recommendations()
    print("PASS: output safety checker blocks level 2 recommendations")

    asyncio.run(test_orchestrator_rewrites_unsafe_level_3_response_before_return())
    print("PASS: orchestrator rewrites unsafe level 3 response before return")


if __name__ == "__main__":
    main()
