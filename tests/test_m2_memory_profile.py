import asyncio
import os
import sys
import tempfile

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

from database import DatabaseManager, AsyncDatabaseManager  # noqa: E402
from conversation_manager import ConversationManager  # noqa: E402


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".sqlite3")
    os.close(fd)
    try:
        yield path
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                pass


def test_sync_database_memory_extensions(db_path: str):
    db = DatabaseManager(db_path=db_path, pool_size=1)

    db.upsert_user_profile(
        user_id="user_memory",
        risk_level="warning_high",
        preferences_patch={
            "preferred_support_style": "direct_actionable",
            "avoid_style": ["空泛安慰"],
            "main_stress_sources": ["学业/求职压力"],
            "recommendation_feedback": {"breathing_exercise": "accepted"},
        },
    )
    profile = db.get_user_profile("user_memory")
    assert profile is not None
    assert profile["risk_level"] == "level_2"
    assert profile["preferences"]["preferred_support_style"] == "direct_actionable"
    assert profile["preferences"]["main_stress_sources"] == ["学业/求职压力"]

    created_at = db.add_mood_event(
        user_id="user_memory",
        session_id="session_1",
        emotion="焦虑",
        source="conversation",
        text_snippet="最近面试和考试让我有点焦虑。",
        emotion_type="anxiety",
        emotion_intensity=0.82,
        stress_source="学业/求职压力",
        user_intent="seeking_help",
        event_summary="用户围绕学业和求职表达焦虑，希望获得帮助。",
        risk_level="warning",
    )
    assert created_at

    events = db.get_recent_mood_events("user_memory", limit=5)
    assert len(events) == 1
    assert events[0]["emotion_type"] == "anxiety"
    assert events[0]["emotion_intensity"] == 0.82
    assert events[0]["stress_source"] == "学业/求职压力"
    assert events[0]["risk_level"] == "level_1"


@pytest.mark.asyncio
async def test_async_database_profile_extensions(db_path: str):
    adb = AsyncDatabaseManager(db_path=db_path)
    await adb.upsert_user_profile(
        user_id="user_async",
        risk_level="high",
        preferences_patch={
            "preferred_support_style": "direct_actionable",
            "avoid_style": ["过度说教"],
            "main_stress_sources": ["未来规划压力"],
        },
    )
    profile = await adb.get_user_profile("user_async")
    assert profile is not None
    assert profile["risk_level"] == "level_3"
    assert profile["preferences"]["avoid_style"] == ["过度说教"]


@pytest.mark.asyncio
async def test_conversation_summary_extensions():
    manager = ConversationManager(use_persistence=False)
    manager.add_interaction(
        user_id="user_conv",
        session_id="session_conv",
        user_input="最近考试和找工作压得我喘不过气，我不知道怎么办。",
        emotion="焦虑",
        context_emotion="自我怀疑",
        confidence=0.86,
        ai_response="我们先一起拆分眼前最紧急的压力来源。",
        emotion_state={
            "emotion_type": "anxiety",
            "emotion_intensity": 0.86,
            "stress_source": "学业/求职压力",
            "user_intent": "seeking_help",
            "negative_trend": True,
        },
        risk_state={"level": "medium"},
    )
    manager.add_interaction(
        user_id="user_conv",
        session_id="session_conv",
        user_input="我还是挺迷茫的，但我想做个计划试试看。",
        emotion="压力",
        context_emotion="困惑",
        confidence=0.78,
        ai_response="我们可以先列出今天能完成的最小一步。",
        emotion_state={
            "emotion_type": "stress",
            "emotion_intensity": 0.71,
            "stress_source": "未来规划压力",
            "user_intent": "planning",
            "negative_trend": False,
        },
        risk_state={"level": "low"},
    )

    summary = await manager.get_conversation_summary_async("user_conv", "session_conv")
    session = manager.get_or_create_session("user_conv", "session_conv")

    assert "stress_sources" in summary
    assert "学业/求职压力" in summary["stress_sources"]
    assert "未来规划压力" in summary["stress_sources"]
    assert summary["recent_intents"][-1] == "planning"
    assert summary["recent_risk_levels"][-1] == "level_0"
    assert session["long_term_profile"]["preferred_support_style"] == "direct_actionable"
    assert "main_stress_sources" in session["long_term_profile"]


def main():
    fd, db_path = tempfile.mkstemp(suffix=".sqlite3")
    os.close(fd)
    try:
        test_sync_database_memory_extensions(db_path)
        print("PASS: sync database memory extensions")

        asyncio.run(test_async_database_profile_extensions(db_path))
        print("PASS: async database profile extensions")

        asyncio.run(test_conversation_summary_extensions())
        print("PASS: conversation summary extensions")
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass


if __name__ == "__main__":
    main()
