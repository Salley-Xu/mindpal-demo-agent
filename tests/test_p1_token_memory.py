import os
import sys


PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from conversation_manager import ConversationManager  # noqa: E402


def _add_turn(manager, user_id, session_id, text):
    manager.add_interaction(
        user_id=user_id,
        session_id=session_id,
        user_input=text,
        emotion="压力",
        ai_response="我在这里陪你继续把这件事理清楚。",
        emotion_state={
            "emotion_type": "stress",
            "emotion_intensity": 0.75,
            "stress_source": "求职",
            "user_intent": "seeking_help",
        },
        risk_state={"level": "low"},
    )


def test_token_driven_memory_compresses_before_five_turns():
    manager = ConversationManager(use_persistence=False)
    manager.max_context_tokens = 120
    manager.context_soft_threshold = 0.5
    manager.context_hard_threshold = 0.7

    user_id = "user_token_short"
    session_id = "session_token_short"
    long_text = "我最近在面试准备上特别慌乱，脑子里一直反复想失败会怎样，" * 8

    _add_turn(manager, user_id, session_id, long_text)
    _add_turn(manager, user_id, session_id, long_text)
    _add_turn(manager, user_id, session_id, long_text)

    key = f"{user_id}_{session_id}"
    cache = manager.context_cache[key]
    assert cache["compression_applied"] is True
    assert cache["last_compressed_turn"] == 3
    assert cache["compression_level"] in {"soft", "hard"}
    assert "历史对话摘要" in cache["compressed_context"]


def test_token_driven_memory_does_not_compress_short_turns_only_by_count():
    manager = ConversationManager(use_persistence=False)
    manager.max_context_tokens = 1000
    manager.context_soft_threshold = 0.8
    manager.context_hard_threshold = 0.9

    user_id = "user_token_many"
    session_id = "session_token_many"

    for index in range(6):
        _add_turn(manager, user_id, session_id, f"第{index + 1}轮，我有点烦。")

    key = f"{user_id}_{session_id}"
    cache = manager.context_cache[key]
    assert cache["compression_applied"] is False
    assert cache["compressed_context"] == ""
    assert cache["token_count"] < int(manager.max_context_tokens * manager.context_soft_threshold)


def main():
    test_token_driven_memory_compresses_before_five_turns()
    print("PASS: token-driven memory compresses before five turns")

    test_token_driven_memory_does_not_compress_short_turns_only_by_count()
    print("PASS: token-driven memory avoids count-only compression")


if __name__ == "__main__":
    main()
