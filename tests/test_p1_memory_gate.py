import os
import sys


PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from conversation_manager import ConversationManager  # noqa: E402


def test_memory_gate_requires_repeated_signals_for_implicit_preferences():
    manager = ConversationManager(use_persistence=False)
    user_id = "user_gate"
    session_id = "session_gate"

    manager.add_interaction(
        user_id=user_id,
        session_id=session_id,
        user_input="最近求职压力很大，你能帮帮我吗？",
        emotion="压力",
        ai_response="我们先看看最困扰你的部分。",
        emotion_state={
            "emotion_type": "stress",
            "emotion_intensity": 0.82,
            "stress_source": "求职",
            "user_intent": "seeking_help",
        },
        risk_state={"level": "low"},
    )

    session = manager.get_or_create_session(user_id, session_id)
    profile = session["long_term_profile"]
    assert profile.get("preferred_support_style") is None
    assert profile.get("main_stress_sources", []) == []

    manager.add_interaction(
        user_id=user_id,
        session_id=session_id,
        user_input="还是求职这件事，我现在更想知道下一步怎么做。",
        emotion="焦虑",
        ai_response="我们把下一步拆得更具体一些。",
        emotion_state={
            "emotion_type": "anxiety",
            "emotion_intensity": 0.8,
            "stress_source": "求职",
            "user_intent": "planning",
        },
        risk_state={"level": "low"},
    )

    session = manager.get_or_create_session(user_id, session_id)
    profile = session["long_term_profile"]
    assert profile.get("preferred_support_style") == "direct_actionable"
    assert profile.get("main_stress_sources") == ["求职"]


def test_memory_gate_keeps_explicit_preference_signal():
    manager = ConversationManager(use_persistence=False)
    user_id = "user_gate_explicit"
    session_id = "session_gate_explicit"

    manager.add_interaction(
        user_id=user_id,
        session_id=session_id,
        user_input="不要空泛安慰，直接告诉我怎么做。",
        emotion="焦虑",
        ai_response="好，我会直接给你可执行建议。",
        emotion_state={
            "emotion_type": "anxiety",
            "emotion_intensity": 0.75,
            "stress_source": "面试",
            "user_intent": "seeking_help",
        },
        risk_state={"level": "low"},
    )

    profile = manager.get_or_create_session(user_id, session_id)["long_term_profile"]
    assert profile.get("preferred_support_style") == "direct_actionable"
    assert "empty_comfort" in profile.get("avoid_style", [])
    # 单轮显式偏好应生效，但压力源仍应等待重复出现再写入
    assert profile.get("main_stress_sources", []) == []


def main():
    test_memory_gate_requires_repeated_signals_for_implicit_preferences()
    print("PASS: implicit preference signals require repetition")

    test_memory_gate_keeps_explicit_preference_signal()
    print("PASS: explicit preference signals are preserved")


if __name__ == "__main__":
    main()
