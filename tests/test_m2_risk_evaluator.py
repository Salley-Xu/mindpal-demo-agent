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

from risk_evaluator import risk_evaluator  # noqa: E402
from agent_orchestrator import agent_orchestrator  # noqa: E402
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


def main():
    test_high_risk_detection()
    print("PASS: high risk detection")

    test_medium_risk_detection()
    print("PASS: medium risk detection")

    test_low_risk_detection()
    print("PASS: low risk detection")

    test_orchestrator_risk_state_builder()
    print("PASS: orchestrator risk state builder")

    test_urgent_logger_statistics_compatibility()
    print("PASS: urgent logger stats compatibility")


if __name__ == "__main__":
    main()
