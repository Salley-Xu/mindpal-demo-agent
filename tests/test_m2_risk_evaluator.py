import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace


PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("CHAT_MODEL", "deepseek-chat")
os.environ.setdefault("API_BASE_URL", "https://api.deepseek.com/v1")

from risk_evaluator import risk_evaluator  # noqa: E402
from agent_orchestrator import agent_orchestrator  # noqa: E402
from agent_tools import MoodEvent, MoodTrackingTool, UserProfile, UserProfileTool  # noqa: E402
from conversation_manager import conversation_manager  # noqa: E402
from emotion_analyzer import emotion_analyzer  # noqa: E402
from models import AgentRunRequest  # noqa: E402
from urgent_detector import urgent_logger  # noqa: E402


def test_high_risk_detection():
    result = risk_evaluator.evaluate(
        text="我真的不想活了，今晚想结束生命。",
        emotion_state={
            "emotion_type": "sadness",
            "emotion_intensity": 0.95,
            "negative_trend": True,
        },
        conversation_summary={"emotion_trend": "escalating"},
    )
    assert result["level"] == "high"
    assert result["risk_score"] >= 8.0
    assert result["suggestions"]


def test_medium_risk_detection():
    result = risk_evaluator.evaluate(
        text="最近我真的有点崩溃，感觉撑不住了，也觉得自己没用。",
        emotion_state={
            "emotion_type": "stress",
            "emotion_intensity": 0.82,
            "negative_trend": True,
        },
        conversation_summary={"emotion_trend": "consistent"},
    )
    assert result["level"] == "medium"
    assert result["risk_score"] >= 5.0


def test_low_risk_detection():
    result = risk_evaluator.evaluate(
        text="这周有点累，但我想慢慢调整状态。",
        emotion_state={
            "emotion_type": "stress",
            "emotion_intensity": 0.45,
            "negative_trend": False,
        },
        conversation_summary={"emotion_trend": "calming"},
    )
    assert result["level"] == "low"


def test_long_term_risk_context_lifts_score():
    baseline = risk_evaluator.evaluate(
        text="最近有些累，也会怀疑自己是不是没用。",
        emotion_state={
            "emotion_type": "stress",
            "emotion_intensity": 0.58,
            "negative_trend": False,
        },
        conversation_summary={"emotion_trend": "consistent"},
    )
    contextual = risk_evaluator.evaluate(
        text="最近有些累，也会怀疑自己是不是没用。",
        emotion_state={
            "emotion_type": "stress",
            "emotion_intensity": 0.58,
            "negative_trend": False,
        },
        conversation_summary={"emotion_trend": "consistent"},
        long_term_risk_level="high",
        historical_high_risk_count=3,
    )
    assert baseline["level"] == "low"
    assert contextual["risk_score"] > baseline["risk_score"]
    assert contextual["risk_score"] >= baseline["risk_score"] + 3.0


def test_orchestrator_risk_state_builder():
    risk_state = agent_orchestrator._build_risk_state(
        {
            "level": "medium",
            "message": "需要更多支持",
            "suggestions": ["联系朋友"],
            "triggers": ["撑不住"],
            "risk_score": 6.4,
        }
    )
    assert risk_state.level == "medium"
    assert risk_state.risk_score == 6.4


def test_urgent_logger_statistics_compatibility():
    stats = urgent_logger._calculate_statistics(
        [
            {"urgent_level": "high", "risk_score": 9.0},
            {"urgent_level": "medium", "risk_score": 6.0},
            {"urgent_level": "low", "risk_score": 1.0},
            {"urgent_level": "urgent", "risk_score": 10.0},
            {"urgent_level": "warning_high", "risk_score": 7.0},
            {"urgent_level": "warning", "risk_score": 3.0},
        ],
        days=1,
    )
    assert stats["urgent_count"] == 2
    assert stats["warning_high_count"] == 2
    assert stats["warning_count"] == 2
    assert stats["high_count"] == 1
    assert stats["medium_count"] == 1
    assert stats["low_count"] == 1


async def test_run_agent_passes_long_term_risk_context_into_precheck():
    user_id = f"user_risk_ctx_{uuid.uuid4().hex[:8]}"
    session_id = f"session_risk_ctx_{uuid.uuid4().hex[:8]}"
    captured = {}

    original_use_persistence = conversation_manager.use_persistence
    original_emotion_method = emotion_analyzer.analyze_with_context_async
    original_get_profile = UserProfileTool.get_profile
    original_get_recent_trend = MoodTrackingTool.get_recent_trend
    original_evaluate = risk_evaluator.evaluate
    original_chat_create = agent_orchestrator.client.chat.completions.create

    async def fake_analyze_with_context_async(text, conversation_summary):
        return "压力", "疲惫", 0.76

    async def fake_get_profile(user_id_arg):
        assert user_id_arg == user_id
        return UserProfile(
            user_id=user_id_arg,
            risk_level="high",
            preferred_types=[],
            preferred_categories=[],
            preferred_difficulty="beginner",
            preferred_duration_range=None,
            preferred_support_style=None,
            avoid_style=[],
            main_stress_sources=[],
            recommendation_feedback={},
            last_updated=datetime.now(timezone.utc),
        )

    async def fake_get_recent_trend(user_id_arg, limit=20):
        assert user_id_arg == user_id
        return [
            MoodEvent(
                user_id=user_id_arg,
                session_id="s1",
                emotion="焦虑",
                emotion_type="anxiety",
                emotion_intensity=0.9,
                stress_source="求职",
                user_intent="sharing",
                event_summary="历史上出现明显风险升高。",
                risk_level="high",
                source="conversation",
                text_snippet="之前一度觉得撑不住。",
                created_at=datetime.now(timezone.utc),
            ),
            MoodEvent(
                user_id=user_id_arg,
                session_id="s2",
                emotion="难过",
                emotion_type="sadness",
                emotion_intensity=0.8,
                stress_source="学业",
                user_intent="sharing",
                event_summary="又一次高风险波动。",
                risk_level="high",
                source="conversation",
                text_snippet="那几天一直很绝望。",
                created_at=datetime.now(timezone.utc),
            ),
            MoodEvent(
                user_id=user_id_arg,
                session_id="s3",
                emotion="疲惫",
                emotion_type="stress",
                emotion_intensity=0.5,
                stress_source="日常",
                user_intent="sharing",
                event_summary="普通低风险事件。",
                risk_level="low",
                source="conversation",
                text_snippet="最近有点累。",
                created_at=datetime.now(timezone.utc),
            ),
        ]

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {
            "level": "low",
            "message": "",
            "suggestions": [],
            "triggers": [],
            "risk_score": 3.6,
        }

    async def fake_chat_create(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="我们先把今天最难受的部分拆开来说。", tool_calls=None)
                )
            ]
        )

    conversation_manager.use_persistence = False
    emotion_analyzer.analyze_with_context_async = fake_analyze_with_context_async
    UserProfileTool.get_profile = fake_get_profile
    MoodTrackingTool.get_recent_trend = staticmethod(fake_get_recent_trend)
    risk_evaluator.evaluate = fake_evaluate
    agent_orchestrator.client.chat.completions.create = fake_chat_create

    try:
        result = await agent_orchestrator.run_agent(
            AgentRunRequest(
                text="这两天我有点麻木，感觉自己快撑不住了。",
                user_id=user_id,
                session_id=session_id,
                return_steps=True,
            )
        )
        assert result.chat.response == "我们先把今天最难受的部分拆开来说。"
        assert captured["long_term_risk_level"] == "high"
        assert captured["historical_high_risk_count"] == 2
        assert captured["text"] == "这两天我有点麻木，感觉自己快撑不住了。"
        assert captured["conversation_summary"]["primary_emotion"] == "压力"
    finally:
        conversation_manager.use_persistence = original_use_persistence
        emotion_analyzer.analyze_with_context_async = original_emotion_method
        UserProfileTool.get_profile = original_get_profile
        MoodTrackingTool.get_recent_trend = original_get_recent_trend
        risk_evaluator.evaluate = original_evaluate
        agent_orchestrator.client.chat.completions.create = original_chat_create
        await conversation_manager.delete_session_async(user_id, session_id)


def main():
    test_high_risk_detection()
    print("PASS: high risk detection")

    test_medium_risk_detection()
    print("PASS: medium risk detection")

    test_low_risk_detection()
    print("PASS: low risk detection")

    test_long_term_risk_context_lifts_score()
    print("PASS: long-term risk context lifts score")

    test_orchestrator_risk_state_builder()
    print("PASS: orchestrator risk state builder")

    test_urgent_logger_statistics_compatibility()
    print("PASS: urgent logger stats compatibility")

    asyncio.run(test_run_agent_passes_long_term_risk_context_into_precheck())
    print("PASS: agent precheck receives long-term risk context")


if __name__ == "__main__":
    main()
