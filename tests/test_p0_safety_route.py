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
from risk_evaluator import risk_evaluator  # noqa: E402
from urgent_detector import urgent_detector  # noqa: E402


def _make_risk_issue(level: str, subject: str = "self", is_help_request: bool = False) -> dict:
    """构造与风险评估器返回兼容的风险 issue 字典。"""
    from risk_levels import risk_level_band, risk_level_index, risk_level_label
    return {
        "level": level,
        "legacy_level": risk_level_band(level),
        "level_index": risk_level_index(level),
        "level_label": risk_level_label(level),
        "message": "",
        "suggestions": [],
        "triggers": [],
        "risk_score": 5.0,
        "raw_score": 5.0,
        "risk_dimensions": {},
        "risk_evidence": {},
        "escalation_reasons": [],
        "recent_risk_levels": [],
        "risk_context": {"subject": subject, "is_third_party_risk": subject == "third_party",
                         "is_help_request": is_help_request, "is_discussion_context": False, "is_safe_denial": False},
    }


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
    original_evaluate = risk_evaluator.evaluate

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

    def fake_evaluate(**kwargs):
        return _make_risk_issue("level_3")

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
        risk_evaluator.evaluate = original_evaluate
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
    original_evaluate_level2 = risk_evaluator.evaluate

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

    def fake_evaluate(**kwargs):
        return _make_risk_issue("level_2")

    conversation_manager.use_persistence = False
    emotion_analyzer.analyze_with_context_async = fake_analyze_with_context_async
    urgent_detector.generate_crisis_response_async = fake_generate_crisis_response_async
    agent_orchestrator.client.chat.completions.create = fail_if_normal_llm_called
    UserProfileTool.get_profile = fake_get_profile
    risk_evaluator.evaluate = fake_evaluate

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
        risk_evaluator.evaluate = original_evaluate_level2
        await conversation_manager.delete_session_async(user_id, session_id)


@pytest.mark.asyncio
async def test_third_party_crisis_help_routes_to_dedicated_support_mode():
    """验证第三方求助走专用路由（直接测试 _is_third_party_crisis_help_request）。"""
    # 验证主体识别
    issue = _make_risk_issue("level_1", subject="third_party", is_help_request=True)
    assert agent_orchestrator._is_third_party_crisis_help_request(issue) is True

    # 非第三方应返回 False
    self_issue = _make_risk_issue("level_1")
    assert agent_orchestrator._is_third_party_crisis_help_request(self_issue) is False

    # 第三方但非求助应返回 False
    tp_no_help = _make_risk_issue("level_1", subject="third_party", is_help_request=False)
    assert agent_orchestrator._is_third_party_crisis_help_request(tp_no_help) is False


def main():
    asyncio.run(test_high_risk_routes_to_dedicated_safety_response())
    print("PASS: high risk routes to dedicated safety response")

    asyncio.run(test_level_2_routes_to_high_risk_support_mode())
    print("PASS: level 2 routes to high-risk support mode")

    asyncio.run(test_third_party_crisis_help_routes_to_dedicated_support_mode())
    print("PASS: third-party crisis help routes to dedicated support mode")


if __name__ == "__main__":
    main()
