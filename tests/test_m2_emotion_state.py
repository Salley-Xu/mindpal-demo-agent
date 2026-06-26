import os
import sys


PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("CHAT_MODEL", "deepseek-chat")
os.environ.setdefault("API_BASE_URL", "https://api.deepseek.com/v1")

from emotion_analyzer import emotion_analyzer  # noqa: E402
from agent_orchestrator import agent_orchestrator  # noqa: E402
from main import app  # noqa: E402


def test_emotion_payload_builder():
    payload = emotion_analyzer.build_emotion_state_payload(
        text="最近考试和面试压力特别大，我真的有点焦虑，不知道该怎么办。",
        current_emotion="焦虑",
        context_emotion="自我怀疑",
        confidence=0.88,
        conversation_summary={
            "emotion_trend": "escalating",
            "recent_emotions": ["焦虑", "压力", "焦虑"],
            "key_concerns": ["academic", "future"],
        },
    )

    assert payload["emotion_type"] == "anxiety"
    assert payload["emotion_intensity"] >= 0.7
    assert payload["stress_source"] == "学业/求职压力"
    assert payload["user_intent"] == "seeking_help"
    assert payload["negative_trend"] is True


def test_orchestrator_emotion_state():
    emotion_state = agent_orchestrator._build_emotion_state(
        request_text="最近事情太多了，我完全不知道怎么开始，帮帮我。",
        current_emotion="压力",
        context_emotion="困惑",
        confidence=0.8,
        conversation_summary={
            "emotion_trend": "consistent",
            "recent_emotions": ["压力", "焦虑", "压力"],
            "key_concerns": ["future"],
        },
    )

    assert emotion_state.current_emotion == "压力"
    assert emotion_state.emotion_type == "stress"
    assert emotion_state.user_intent == "seeking_help"
    assert isinstance(emotion_state.negative_trend, bool)


def test_openapi_contains_extended_emotion_state():
    schema = app.openapi()
    emotion_state_schema = schema["components"]["schemas"]["EmotionState"]["properties"]

    assert "emotion_type" in emotion_state_schema
    assert "emotion_intensity" in emotion_state_schema
    assert "stress_source" in emotion_state_schema
    assert "user_intent" in emotion_state_schema
    assert "negative_trend" in emotion_state_schema


def main():
    test_emotion_payload_builder()
    print("PASS: emotion payload builder")

    test_orchestrator_emotion_state()
    print("PASS: orchestrator emotion state")

    test_openapi_contains_extended_emotion_state()
    print("PASS: emotion state openapi schema")


if __name__ == "__main__":
    main()
