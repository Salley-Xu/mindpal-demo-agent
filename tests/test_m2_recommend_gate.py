import asyncio
import os
import sys
from types import SimpleNamespace

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

from recommend_gate import recommend_gate  # noqa: E402
from content_recommender import content_recommender  # noqa: E402
from conversation_manager import ConversationManager  # noqa: E402
from agent_orchestrator import agent_orchestrator  # noqa: E402
from main import app  # noqa: E402


def test_high_risk_routes_to_safety():
    decision = recommend_gate.decide(
        emotion_state={
            "emotion_intensity": 0.92,
            "user_intent": "seeking_help",
            "negative_trend": True,
            "stress_source": "学业/求职压力",
        },
        risk_state={"level": "level_3"},
        conversation_summary={"turn_count": 3, "recent_recommendation_turns": []},
        user_profile={"preferred_support_style": "direct_actionable"},
    )
    assert decision["should_recommend"] is False
    assert decision["recommend_type"] == "none"
    assert "high_risk_safety_route" in decision["reason_codes"]
    assert "level_3_safety_route" in decision["reason_codes"]


def test_soft_and_hard_recommendation_levels():
    soft_decision = recommend_gate.decide(
        emotion_state={
            "emotion_intensity": 0.62,
            "user_intent": "sharing",
            "negative_trend": False,
            "stress_source": "未来规划压力",
        },
        risk_state={"level": "low"},
        conversation_summary={"turn_count": 2, "recent_recommendation_turns": []},
        user_profile={"preferred_support_style": "direct_actionable"},
    )
    assert soft_decision["should_recommend"] is True
    assert soft_decision["recommend_type"] == "soft"

    hard_decision = recommend_gate.decide(
        emotion_state={
            "emotion_intensity": 0.88,
            "user_intent": "seeking_help",
            "negative_trend": True,
            "stress_source": "学业/求职压力",
        },
        risk_state={"level": "level_1"},
        conversation_summary={"turn_count": 5, "recent_recommendation_turns": []},
        user_profile={
            "preferred_support_style": "direct_actionable",
            "main_stress_sources": ["学业/求职压力"],
        },
    )
    assert hard_decision["should_recommend"] is True
    assert hard_decision["recommend_type"] == "hard"


def test_level_2_disables_ordinary_recommendation():
    decision = recommend_gate.decide(
        emotion_state={
            "emotion_intensity": 0.86,
            "user_intent": "seeking_help",
            "negative_trend": True,
            "stress_source": "学业/求职压力",
        },
        risk_state={"level": "level_2"},
        conversation_summary={"turn_count": 4, "recent_recommendation_turns": []},
        user_profile={"preferred_support_style": "direct_actionable"},
    )
    assert decision["should_recommend"] is False
    assert decision["recommend_type"] == "safety_only"
    assert "level_2_support_route" in decision["reason_codes"]


@pytest.mark.asyncio
async def test_content_recommender_filters_for_level_2():
    recommendations, _, _ = await content_recommender.recommend_content(
        user_input="我最近快撑不住了，想先找点简单的方法缓一缓。",
        current_emotion="压力",
        conversation_summary={"recent_risk_levels": ["level_2"]},
        user_profile={"risk_level": "level_2"},
        limit=5,
    )
    assert recommendations
    assert all(item.recommend_type == "soft" for item in recommendations)
    assert all(item.difficulty in {None, "beginner"} for item in recommendations)
    assert all((item.duration_minutes is None or item.duration_minutes <= 10) for item in recommendations)


def test_cooldown_penalty_and_summary_tracking():
    manager = ConversationManager(use_persistence=False)
    manager.add_interaction(
        user_id="user_gate",
        session_id="session_gate",
        user_input="最近事情有点多。",
        emotion="压力",
        ai_response="我在这里陪你。",
        emotion_state={"emotion_type": "stress", "emotion_intensity": 0.55, "user_intent": "sharing"},
        risk_state={"level": "low"},
    )
    manager.mark_recommendation("user_gate", "session_gate", "hard", ["breathing_001"])
    summary = asyncio.run(manager.get_conversation_summary_async("user_gate", "session_gate"))
    assert summary["recent_recommendation_turns"] == [1]

    decision = recommend_gate.decide(
        emotion_state={"emotion_intensity": 0.7, "user_intent": "seeking_help"},
        risk_state={"level": "low"},
        conversation_summary={"turn_count": 2, "recent_recommendation_turns": [1]},
        user_profile={},
    )
    assert "cooldown_penalty" in decision["reason_codes"]
    assert decision["cooldown_remaining"] >= 1


@pytest.mark.asyncio
async def test_orchestrator_blocks_recommend_tool_when_not_hard():
    tool_call = SimpleNamespace(
        id="tool_1",
        function=SimpleNamespace(name="recommend_content", arguments="{}"),
    )
    result = await agent_orchestrator._execute_single_tool(
        tool_call=tool_call,
        conversation_summary={"primary_emotion": "中性"},
        request_text="我只是想聊聊。",
        user_id="user_gate",
        session_id="session_gate",
        recommendation_decision={"recommend_type": "soft"},
        user_profile={},
    )
    assert "recommend_gate_soft" in result["content"]


def test_openapi_contains_recommendation_decision():
    schema = app.openapi()
    chat_props = schema["components"]["schemas"]["ChatResponse"]["properties"]
    assert "recommendation_decision" in chat_props


# ============================================================
# _calculate_preference_score 纯函数测试
# ============================================================

def test_preference_score_style_match():
    """优先支持风格匹配应加 0.45"""
    score = recommend_gate._calculate_preference_score(
        emotion_state={},
        user_profile={"preferred_support_style": "direct_actionable"},
        conversation_summary={},
    )
    assert score == 0.45


def test_preference_score_stress_match():
    """压力来源匹配应加 0.35"""
    score = recommend_gate._calculate_preference_score(
        emotion_state={"stress_source": "学业/求职压力"},
        user_profile={"main_stress_sources": ["学业/求职压力", "未来规划压力"]},
        conversation_summary={},
    )
    assert score == 0.35


def test_preference_score_rejected_penalty():
    """有被拒绝的推荐时应减 0.25"""
    score = recommend_gate._calculate_preference_score(
        emotion_state={},
        user_profile={},
        conversation_summary={"rejected_recommendations": ["audio_001"]},
    )
    assert score == -0.25


def test_preference_score_combined():
    """多项加成分应累加且不超过 1.0"""
    score = recommend_gate._calculate_preference_score(
        emotion_state={"stress_source": "学业/求职压力"},
        user_profile={
            "preferred_support_style": "direct_actionable",
            "main_stress_sources": ["学业/求职压力"],
        },
        conversation_summary={},
    )
    assert score == 0.80  # 0.45 + 0.35


def test_preference_score_empty_profile():
    """空画像返回 0"""
    score = recommend_gate._calculate_preference_score({}, {}, {})
    assert score == 0.0


def test_preference_score_clamped():
    """分数应被限制在 [-0.3, 1.0] 范围内"""
    # 超出下限
    low = recommend_gate._calculate_preference_score(
        {}, {}, {"rejected_recommendations": ["a", "b", "c", "d", "e"]}
    )
    assert low >= -0.3

    # 超出上限：style(0.45) + stress(0.35) + 更多加分(理论上限)
    profile = {"preferred_support_style": "direct_actionable",
               "main_stress_sources": ["学业/求职压力"]}
    high = recommend_gate._calculate_preference_score(
        {"stress_source": "学业/求职压力"}, profile, {}
    )
    assert high <= 1.0


# ============================================================
# _calculate_cooldown_penalty 纯函数测试
# ============================================================

def test_cooldown_penalty_no_recent():
    """无近期推荐返回 0 冷却"""
    penalty, remaining = recommend_gate._calculate_cooldown_penalty(
        {"turn_count": 5}
    )
    assert penalty == 0.0
    assert remaining == 0


def test_cooldown_penalty_empty_turns():
    """空推荐轮次列表返回 0 冷却"""
    penalty, remaining = recommend_gate._calculate_cooldown_penalty(
        {"turn_count": 5, "recent_recommendation_turns": []}
    )
    assert penalty == 0.0


def test_cooldown_penalty_distance_1():
    """距离 <= 1 轮应返回 0.35 惩罚"""
    penalty, remaining = recommend_gate._calculate_cooldown_penalty(
        {"turn_count": 2, "recent_recommendation_turns": [1]}
    )
    assert penalty == 0.35
    assert remaining >= 1


def test_cooldown_penalty_distance_2():
    """距离 == 2 轮应返回 0.18 惩罚"""
    penalty, remaining = recommend_gate._calculate_cooldown_penalty(
        {"turn_count": 3, "recent_recommendation_turns": [1]}
    )
    assert penalty == 0.18
    assert remaining == 0


def test_cooldown_penalty_distance_3():
    """距离 >= 3 轮无惩罚"""
    penalty, remaining = recommend_gate._calculate_cooldown_penalty(
        {"turn_count": 5, "recent_recommendation_turns": [2]}
    )
    assert penalty == 0.0
    assert remaining == 0


def test_cooldown_penalty_multiple_recent():
    """多个推荐轮次时取最近的一次计算"""
    penalty, remaining = recommend_gate._calculate_cooldown_penalty(
        {"turn_count": 5, "recent_recommendation_turns": [1, 3]}
    )
    assert penalty == 0.18  # 距离 = 5 - 3 = 2


# ============================================================
# decide() 边界条件测试
# ============================================================

def test_decide_none_inputs():
    """所有输入为 None 时不应崩溃"""
    decision = recommend_gate.decide(
        emotion_state=None,
        risk_state=None,
        conversation_summary=None,
        user_profile=None,
    )
    assert isinstance(decision, dict)
    assert "should_recommend" in decision
    assert "recommend_type" in decision


def test_decide_zero_intensity():
    """零情绪强度分数过低，不做推荐"""
    decision = recommend_gate.decide(
        emotion_state={"emotion_intensity": 0.0, "user_intent": "sharing", "negative_trend": False},
        risk_state={"level": "low"},
        conversation_summary={"turn_count": 1, "recent_recommendation_turns": []},
        user_profile={},
    )
    assert decision["should_recommend"] is False
    assert decision["recommend_type"] == "none"


def test_decide_high_risk_high_emotion():
    """高风险+高情绪强度应走安全路由"""
    decision = recommend_gate.decide(
        emotion_state={"emotion_intensity": 0.95, "user_intent": "seeking_help", "negative_trend": True},
        risk_state={"level": "level_3"},
        conversation_summary={"turn_count": 3, "recent_recommendation_turns": []},
        user_profile={},
    )
    assert decision["should_recommend"] is False
    assert decision["recommend_type"] == "none"
    assert "high_risk_safety_route" in decision["reason_codes"]


def main():
    test_high_risk_routes_to_safety()
    print("PASS: high risk routes to safety")

    test_soft_and_hard_recommendation_levels()
    print("PASS: soft and hard recommendation levels")

    test_level_2_disables_ordinary_recommendation()
    print("PASS: level 2 disables ordinary recommendation")

    test_cooldown_penalty_and_summary_tracking()
    print("PASS: cooldown penalty and summary tracking")

    asyncio.run(test_orchestrator_blocks_recommend_tool_when_not_hard())
    print("PASS: orchestrator blocks recommend tool when not hard")

    asyncio.run(test_content_recommender_filters_for_level_2())
    print("PASS: content recommender filters for level 2")

    test_openapi_contains_recommendation_decision()
    print("PASS: recommendation decision openapi schema")

    # New: preference score
    test_preference_score_style_match()
    print("PASS: preference score style match")
    test_preference_score_stress_match()
    print("PASS: preference score stress match")
    test_preference_score_rejected_penalty()
    print("PASS: preference score rejected penalty")
    test_preference_score_combined()
    print("PASS: preference score combined")
    test_preference_score_empty_profile()
    print("PASS: preference score empty profile")
    test_preference_score_clamped()
    print("PASS: preference score clamped")

    # New: cooldown penalty
    test_cooldown_penalty_no_recent()
    print("PASS: cooldown penalty no recent")
    test_cooldown_penalty_empty_turns()
    print("PASS: cooldown penalty empty turns")
    test_cooldown_penalty_distance_1()
    print("PASS: cooldown penalty distance 1")
    test_cooldown_penalty_distance_2()
    print("PASS: cooldown penalty distance 2")
    test_cooldown_penalty_distance_3()
    print("PASS: cooldown penalty distance 3")
    test_cooldown_penalty_multiple_recent()
    print("PASS: cooldown penalty multiple recent")

    # New: edge cases
    test_decide_none_inputs()
    print("PASS: decide none inputs")
    test_decide_zero_intensity()
    print("PASS: decide zero intensity")
    test_decide_high_risk_high_emotion()
    print("PASS: decide high risk high emotion")


if __name__ == "__main__":
    main()
