from typing import Any, Dict, List, Optional


def _clean_value(value: Any, default: str = "n/a") -> Any:
    if value in (None, "", [], {}):
        return default
    return value


def _listify(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def build_debug_snapshot(
    conversation_summary: Optional[Dict[str, Any]] = None,
    emotion_state: Optional[Dict[str, Any]] = None,
    risk_state: Optional[Dict[str, Any]] = None,
    recommendation_decision: Optional[Dict[str, Any]] = None,
    recommendations: Optional[List[Dict[str, Any]]] = None,
    feedback_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    summary = conversation_summary or {}
    emotion = emotion_state or {}
    risk = risk_state or {}
    decision = recommendation_decision or {}
    feedback = feedback_map or {}

    emotion_section = {
        "current_emotion": _clean_value(
            emotion.get("current_emotion") or summary.get("current_emotion") or summary.get("primary_emotion")
        ),
        "emotion_type": _clean_value(emotion.get("emotion_type")),
        "context_emotion": _clean_value(
            emotion.get("context_emotion") or summary.get("context_emotion")
        ),
        "emotion_intensity": _clean_value(emotion.get("emotion_intensity")),
        "stress_source": _clean_value(emotion.get("stress_source")),
        "user_intent": _clean_value(emotion.get("user_intent")),
        "negative_trend": _clean_value(emotion.get("negative_trend")),
        "emotion_trend": _clean_value(
            emotion.get("emotion_trend") or summary.get("emotion_trend")
        ),
    }

    risk_section = {
        "level": _clean_value(risk.get("level")),
        "score": _clean_value(risk.get("risk_score") if risk.get("risk_score") is not None else risk.get("score")),
        "reason": _clean_value(risk.get("message") or risk.get("reason")),
        "signals": _listify(risk.get("triggers") or risk.get("signals")),
        "risk_score": _clean_value(risk.get("risk_score") if risk.get("risk_score") is not None else risk.get("score")),
        "message": _clean_value(risk.get("message") or risk.get("reason")),
        "triggers": _listify(risk.get("triggers") or risk.get("signals")),
        "recent_risk_levels": _listify(summary.get("recent_risk_levels")),
    }

    decision_section = {
        "should_recommend": bool(decision.get("should_recommend", False)),
        "recommend_type": _clean_value(decision.get("recommend_type")),
        "score": _clean_value(decision.get("score")),
        "threshold": _clean_value(decision.get("threshold")),
        "cooldown_remaining": _clean_value(decision.get("cooldown_remaining")),
        "reason_codes": _listify(decision.get("reason_codes")),
    }

    summary_section = {
        "conversation_stage": _clean_value(summary.get("conversation_stage")),
        "turn_count": _clean_value(summary.get("turn_count"), default=0),
        "key_concerns": _listify(summary.get("key_concerns")),
        "stress_sources": _listify(summary.get("stress_sources")),
        "accepted_recommendations": _listify(summary.get("accepted_recommendations")),
        "rejected_recommendations": _listify(summary.get("rejected_recommendations")),
    }

    recommendation_section = []
    for item in recommendations or []:
        retrieval_metadata = item.get("retrieval_metadata") or {}
        content_id = item.get("id", "unknown")
        recommendation_section.append(
            {
                "id": content_id,
                "title": item.get("title", "无标题"),
                "recommend_type": item.get("recommend_type", "unknown"),
                "source": item.get("source", "unknown"),
                "feedback": feedback.get(content_id, "none"),
                "retrieval": {
                    "final_score": _clean_value(retrieval_metadata.get("final_score")),
                    "retrieval_score": _clean_value(retrieval_metadata.get("retrieval_score")),
                    "best_rank": _clean_value(retrieval_metadata.get("best_rank")),
                    "matched_queries": _listify(retrieval_metadata.get("matched_queries")),
                    "retrieval_sources": _listify(retrieval_metadata.get("retrieval_sources")),
                },
            }
        )

    return {
        "emotion": emotion_section,
        "risk": risk_section,
        "decision": decision_section,
        "summary": summary_section,
        "recommendations": recommendation_section,
        "feedback_map": feedback,
    }
