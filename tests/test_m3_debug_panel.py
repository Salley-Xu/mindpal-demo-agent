import os
import sys


PROJECT_ROOT = os.getcwd()
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if FRONTEND_DIR not in sys.path:
    sys.path.append(FRONTEND_DIR)

from debug_panel import build_debug_snapshot  # noqa: E402


def test_build_debug_snapshot_aggregates_core_sections():
    snapshot = build_debug_snapshot(
        conversation_summary={
            "conversation_stage": "deepening",
            "turn_count": 6,
            "key_concerns": ["academic", "future"],
            "stress_sources": ["论文", "求职"],
            "recent_risk_levels": ["low", "medium"],
            "accepted_recommendations": ["audio_001"],
            "rejected_recommendations": ["video_003"],
            "primary_emotion": "焦虑",
            "emotion_trend": "escalating",
        },
        emotion_state={
            "current_emotion": "压力",
            "emotion_type": "stress",
            "context_emotion": "疲惫",
            "emotion_intensity": 0.82,
            "stress_source": "论文",
            "user_intent": "seeking_help",
            "negative_trend": True,
        },
        risk_state={
            "level": "medium",
            "risk_score": 0.65,
            "message": "需要继续关注压力变化",
            "triggers": ["negative_trend", "stress_escalation"],
        },
        recommendation_decision={
            "should_recommend": True,
            "recommend_type": "soft",
            "score": 0.74,
            "threshold": 0.7,
            "cooldown_remaining": 0,
            "reason_codes": ["high_intensity", "seeking_help"],
        },
        recommendations=[
            {
                "id": "tool_001",
                "title": "3分钟呼吸练习",
                "recommend_type": "soft",
                "source": "content_db",
                "retrieval_metadata": {
                    "final_score": 0.91,
                    "retrieval_score": 0.87,
                    "best_rank": 1,
                    "matched_queries": ["压力 缓解方法"],
                    "retrieval_sources": ["bm25", "vector"],
                },
            }
        ],
        feedback_map={"tool_001": "helpful"},
    )

    assert snapshot["emotion"]["current_emotion"] == "压力"
    assert snapshot["emotion"]["emotion_trend"] == "escalating"
    assert snapshot["risk"]["level"] == "medium"
    assert snapshot["risk"]["score"] == 0.65
    assert snapshot["risk"]["reason"] == "需要继续关注压力变化"
    assert snapshot["risk"]["signals"] == ["negative_trend", "stress_escalation"]
    assert snapshot["risk"]["risk_score"] == 0.65
    assert snapshot["risk"]["message"] == "需要继续关注压力变化"
    assert snapshot["risk"]["triggers"] == ["negative_trend", "stress_escalation"]
    assert snapshot["risk"]["recent_risk_levels"] == ["low", "medium"]
    assert snapshot["decision"]["recommend_type"] == "soft"
    assert snapshot["decision"]["reason_codes"] == ["high_intensity", "seeking_help"]
    assert snapshot["summary"]["stress_sources"] == ["论文", "求职"]
    assert snapshot["summary"]["accepted_recommendations"] == ["audio_001"]
    assert snapshot["recommendations"][0]["feedback"] == "helpful"
    assert snapshot["recommendations"][0]["retrieval"]["retrieval_sources"] == ["bm25", "vector"]


def test_build_debug_snapshot_uses_summary_fallbacks():
    snapshot = build_debug_snapshot(
        conversation_summary={
            "conversation_stage": "initial",
            "turn_count": 1,
            "primary_emotion": "中性",
        },
        recommendations=[],
        feedback_map={},
    )

    assert snapshot["emotion"]["current_emotion"] == "中性"
    assert snapshot["emotion"]["emotion_type"] == "n/a"
    assert snapshot["risk"]["signals"] == []
    assert snapshot["risk"]["triggers"] == []
    assert snapshot["decision"]["should_recommend"] is False
    assert snapshot["summary"]["turn_count"] == 1
    assert snapshot["recommendations"] == []
    assert snapshot["feedback_map"] == {}


def test_build_debug_snapshot_keeps_backward_compatibility_for_legacy_risk_fields():
    snapshot = build_debug_snapshot(
        conversation_summary={"recent_risk_levels": ["low"]},
        risk_state={
            "level": "medium",
            "score": 0.42,
            "reason": "legacy_reason",
            "signals": ["legacy_signal"],
        },
    )

    assert snapshot["risk"]["score"] == 0.42
    assert snapshot["risk"]["reason"] == "legacy_reason"
    assert snapshot["risk"]["signals"] == ["legacy_signal"]
    assert snapshot["risk"]["risk_score"] == 0.42
    assert snapshot["risk"]["message"] == "legacy_reason"
    assert snapshot["risk"]["triggers"] == ["legacy_signal"]


def main():
    test_build_debug_snapshot_aggregates_core_sections()
    print("PASS: debug snapshot aggregates core sections")

    test_build_debug_snapshot_uses_summary_fallbacks()
    print("PASS: debug snapshot uses summary fallbacks")

    test_build_debug_snapshot_keeps_backward_compatibility_for_legacy_risk_fields()
    print("PASS: debug snapshot keeps backward compatibility for legacy risk fields")


if __name__ == "__main__":
    main()
