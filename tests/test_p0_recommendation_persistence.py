import asyncio
import os
import shutil
import sys
import tempfile
import uuid


PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("CHAT_MODEL", "deepseek-chat")
os.environ.setdefault("API_BASE_URL", "https://api.deepseek.com/v1")

import conversation_manager as cm_module  # noqa: E402
from database import AsyncDatabaseManager, DatabaseManager  # noqa: E402
from recommend_gate import recommend_gate  # noqa: E402


def test_recommendation_state_persists_across_manager_reload():
    temp_dir = tempfile.mkdtemp(prefix="mindpal_rec_persist_")
    db_path = os.path.join(temp_dir, "session.db")
    user_id = f"user_rec_{uuid.uuid4().hex[:8]}"
    session_id = f"session_rec_{uuid.uuid4().hex[:8]}"

    original_db_manager = cm_module.db_manager
    original_adb_manager = cm_module.adb_manager

    temp_db_manager = DatabaseManager(db_path=db_path, pool_size=1)
    temp_adb_manager = AsyncDatabaseManager(db_path=db_path)
    cm_module.db_manager = temp_db_manager
    cm_module.adb_manager = temp_adb_manager

    try:
        manager = cm_module.ConversationManager(use_persistence=True)
        manager.add_interaction(
            user_id=user_id,
            session_id=session_id,
            user_input="我最近压力很大，不知道该怎么缓解。",
            emotion="压力",
            context_emotion="焦虑",
            confidence=0.9,
            ai_response="我们先找一个最小的缓解动作。",
            emotion_state={
                "current_emotion": "压力",
                "emotion_type": "stress",
                "emotion_intensity": 0.88,
                "stress_source": "求职",
                "user_intent": "seeking_help",
                "negative_trend": True,
            },
            risk_state={"level": "low"},
        )
        manager.mark_recommendation(
            user_id=user_id,
            session_id=session_id,
            recommend_type="soft",
            item_ids=["audio_001", "tool_001"],
        )
        manager.record_recommendation_feedback(
            user_id=user_id,
            session_id=session_id,
            content_id="audio_001",
            feedback="helpful",
        )
        manager.record_recommendation_feedback(
            user_id=user_id,
            session_id=session_id,
            content_id="tool_001",
            feedback="avoid",
        )

        restored_manager = cm_module.ConversationManager(use_persistence=True)
        restored_session = asyncio.run(
            restored_manager.get_or_create_session_async(user_id, session_id)
        )
        restored_summary = asyncio.run(
            restored_manager.get_conversation_summary_async(user_id, session_id)
        )

        assert len(restored_session["recommendation_events"]) == 1
        assert restored_session["recommendation_events"][0]["turn_number"] == 1
        assert restored_session["recommendation_events"][0]["item_ids"] == ["audio_001", "tool_001"]
        assert restored_session["recommendation_feedback"]["audio_001"] == "helpful"
        assert restored_session["recommendation_feedback"]["tool_001"] == "avoid"
        assert restored_summary["recent_recommendation_turns"] == [1]
        assert restored_summary["accepted_recommendations"] == ["audio_001"]
        assert restored_summary["rejected_recommendations"] == ["tool_001"]

        decision = recommend_gate.decide(
            emotion_state={
                "emotion_intensity": 0.88,
                "user_intent": "seeking_help",
                "negative_trend": True,
                "stress_source": "求职",
            },
            risk_state={"level": "low"},
            conversation_summary=restored_summary,
            user_profile={},
        )
        assert decision["should_recommend"] is False
        assert decision["recommend_type"] == "none"
        assert decision["cooldown_remaining"] == 2
        assert "cooldown_penalty" in decision["reason_codes"]
    finally:
        cm_module.db_manager = original_db_manager
        cm_module.adb_manager = original_adb_manager
        for conn in list(temp_db_manager._connection_pool):
            try:
                conn.close()
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    test_recommendation_state_persists_across_manager_reload()
    print("PASS: recommendation persistence survives manager reload")


if __name__ == "__main__":
    main()
