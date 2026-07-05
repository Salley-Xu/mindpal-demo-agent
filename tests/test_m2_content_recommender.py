"""ContentRecommender 纯函数单元测试"""
import os
import sys

PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from typing import Optional

from content_recommender import content_recommender
from models import ContentItem


# ============================================================
# 辅助函数
# ============================================================

def _make_item(
    id: str = "test_001",
    title: str = "",
    type: str = "article",
    category: str = "stress",
    difficulty: Optional[str] = None,
    duration_minutes: int = 10,
    emotion_tags: list = None,
    tags: list = None,
    popularity: int = 0,
    priority: float = 0.5,
) -> ContentItem:
    return ContentItem(
        id=id,
        title=title or f"Test {id}",
        type=type,
        category=category,
        description="测试内容",
        difficulty=difficulty,
        duration_minutes=duration_minutes,
        emotion_tags=emotion_tags or [],
        tags=tags or [],
        popularity=popularity,
        priority=priority,
    )


# ============================================================
# _build_query_variants
# ============================================================

def test_build_query_variants_full():
    """完整的输入应生成所有 5 种变体"""
    variants = content_recommender._build_query_variants(
        user_input="最近事情太多，我完全不知道怎么开始。",
        current_emotion="焦虑",
        conversation_summary={
            "key_concerns": ["academic", "future"],
            "stress_sources": ["学业/求职压力"],
            "recent_intents": ["planning"],
        },
        user_profile={"main_stress_sources": ["学业/求职压力"]},
    )
    labels = [v["label"] for v in variants]
    assert "raw_input" in labels
    assert "emotion_enhanced" in labels
    assert "stress_source" in labels
    assert "intent_enhanced" in labels
    assert "long_memory" in labels
    # 权重要在合理范围
    for v in variants:
        assert 0 < v["weight"] <= 1.0


def test_build_query_variants_minimal():
    """最小输入应至少返回 raw_input 变体"""
    variants = content_recommender._build_query_variants(
        user_input="你好",
        current_emotion="",
        conversation_summary={},
        user_profile={},
    )
    assert len(variants) == 1
    assert variants[0]["label"] == "raw_input"
    assert variants[0]["weight"] == 1.0


def test_build_query_variants_no_extra_fields():
    """缺少 emotion/stress/intent/profile 时不应崩溃"""
    variants = content_recommender._build_query_variants(
        user_input="测试",
        current_emotion="压力",
        conversation_summary={},
        user_profile={},
    )
    labels = [v["label"] for v in variants]
    assert "raw_input" in labels
    assert "emotion_enhanced" in labels  # 有 emotion 就有此变体


# ============================================================
# _extract_keywords
# ============================================================

def test_extract_keywords_with_chinese():
    """中文文本应提取出有意义的词"""
    keywords = content_recommender._extract_keywords("我感觉最近压力很大，非常焦虑")
    assert "压力" in keywords
    assert "焦虑" in keywords


def test_extract_keywords_empty():
    """空文本返回空列表"""
    keywords = content_recommender._extract_keywords("")
    assert keywords == []


def test_extract_keywords_psych_keywords():
    """应自动补充匹配的心理关键词"""
    keywords = content_recommender._extract_keywords("我最近抑郁了，睡眠不好")
    assert "抑郁" in keywords
    assert "睡眠" in keywords


# ============================================================
# _normalize_user_profile
# ============================================================

def test_normalize_user_profile_empty():
    """空画像应补充所有默认值"""
    profile = content_recommender._normalize_user_profile({})
    assert profile["risk_level"] == "low"
    assert profile["preferred_types"] == []
    assert profile["preferred_categories"] == []
    assert profile["preferred_difficulty"] == "beginner"
    assert profile["preferred_duration_range"] is None


def test_normalize_user_profile_partial():
    """部分画像应补充缺失字段"""
    profile = content_recommender._normalize_user_profile({"risk_level": "level_2"})
    assert profile["risk_level"] == "level_2"
    assert profile["preferred_types"] == []
    assert profile["preferred_difficulty"] == "beginner"


def test_normalize_user_profile_none():
    """None 应被当作空 dict 处理"""
    profile = content_recommender._normalize_user_profile(None)
    assert profile["risk_level"] == "low"


# ============================================================
# _preference_match_score
# ============================================================

def test_preference_match_score_exact_match():
    """精确匹配应得高分"""
    item = _make_item(type="article", category="stress", difficulty="beginner")
    profile = {
        "preferred_types": ["article", "audio"],
        "preferred_categories": ["stress", "anxiety"],
        "preferred_difficulty": "beginner",
    }
    score = content_recommender._preference_match_score(item, profile)
    assert score >= 1.0, f"Expected >= 1.0, got {score}"


def test_preference_match_score_no_match():
    """无匹配应返回 0"""
    item = _make_item(type="video", category="relationship", difficulty="advanced")
    profile = {
        "preferred_types": ["article"],
        "preferred_categories": ["stress"],
        "preferred_difficulty": "beginner",
    }
    score = content_recommender._preference_match_score(item, profile)
    assert score == 0.0


def test_preference_match_score_empty_profile():
    """空画像返回 0"""
    item = _make_item()
    score = content_recommender._preference_match_score(item, {})
    assert score == 0.0


# ============================================================
# _is_item_allowed_for_risk
# ============================================================

def test_is_item_allowed_level_0():
    """Level 0 应允许所有内容"""
    item = _make_item(difficulty="advanced", duration_minutes=60)
    assert content_recommender._is_item_allowed_for_risk(item, "level_0")
    assert content_recommender._is_item_allowed_for_risk(item, "low")


def test_is_item_allowed_level_2_only_soft_beginner():
    """Level 2 只允许 beginner/soft/短内容"""
    soft_beginner = _make_item(difficulty="beginner", duration_minutes=5, type="article")
    hard_advanced = _make_item(difficulty="advanced", duration_minutes=30, type="video")
    long_article = _make_item(difficulty="beginner", duration_minutes=15, type="article")

    assert content_recommender._is_item_allowed_for_risk(soft_beginner, "level_2")
    assert not content_recommender._is_item_allowed_for_risk(hard_advanced, "level_2")
    # duration 超过 10 分钟不应被允许
    assert not content_recommender._is_item_allowed_for_risk(long_article, "level_2")


def test_is_item_allowed_level_3_none():
    """Level 3 不允许任何内容"""
    item = _make_item()
    assert not content_recommender._is_item_allowed_for_risk(item, "level_3")
    assert not content_recommender._is_item_allowed_for_risk(item, "urgent")


# ============================================================
# _calculate_match_scores
# ============================================================

def test_calculate_match_scores_basic():
    """基本匹配度计算"""
    stress_item = _make_item(id="s1", category="stress", emotion_tags=["焦虑"],
                             tags=["academic"], popularity=5)
    items = [stress_item]
    scores = content_recommender._calculate_match_scores(
        recommended_items=items,
        user_input="我压力很大",
        current_emotion="焦虑",
        conversation_summary={"key_concerns": ["academic"], "conversation_stage": "initial"},
        user_profile={},
    )
    assert "s1" in scores
    # emotion match (0.4) + concern match (0.3) + popularity (0.05)
    assert scores["s1"] == 0.75


def test_calculate_match_scores_empty_items():
    """空列表返回空 dict"""
    scores = content_recommender._calculate_match_scores([], "hello", "", {}, {})
    assert scores == {}


# ============================================================
# _adjust_score_with_profile
# ============================================================

def test_adjust_score_with_profile_type_match():
    """类型匹配应加分"""
    item = _make_item(type="audio")
    profile = {"preferred_types": ["audio", "video"], "preferred_categories": [],
               "preferred_difficulty": "beginner"}
    adjusted = content_recommender._adjust_score_with_profile(0.5, item, profile)
    assert adjusted > 0.5


def test_adjust_score_with_profile_no_profile():
    """无画像不改分"""
    item = _make_item()
    adjusted = content_recommender._adjust_score_with_profile(0.5, item, {})
    assert adjusted == 0.5


# ============================================================
# _generate_rationale
# ============================================================

def test_generate_rationale_empty():
    """空推荐列表返回默认提示"""
    rationale = content_recommender._generate_rationale([], "hello", "", {})
    assert "暂时没有找到" in rationale


def test_generate_rationale_different_stages():
    """不同对话阶段使用不同模板"""
    stress_item = _make_item(id="r1", title="放松训练", emotion_tags=["焦虑"])
    concerns = {"key_concerns": ["academic"], "conversation_stage": "exploring"}

    rationale = content_recommender._generate_rationale(
        [stress_item], "我压力很大", "焦虑", concerns
    )
    assert "探索阶段" in rationale
    assert "放松训练" in rationale
    assert "焦虑" in rationale


def test_generate_rationale_initial_stage():
    """initial 阶段应使用初始模板"""
    item = _make_item(id="r2", title="正念冥想")
    rationale = content_recommender._generate_rationale(
        [item], "你好", "", {"conversation_stage": "initial"}
    )
    assert "根据你提到的" in rationale or "根据你的情况" in rationale


# ============================================================
# _rank_candidates (集成测试多个纯函数)
# ============================================================

def test_rank_candidates_empty():
    """空候选列表返回空"""
    result = content_recommender._rank_candidates(
        candidates=[],
        user_input="test",
        current_emotion="",
        conversation_summary={},
        user_profile={},
        limit=5,
    )
    assert result == []


def test_rank_candidates_sorts_by_score():
    """候选应按最终分数降序排列"""
    candidates = [
        {"item": _make_item(id="a", category="stress", emotion_tags=["焦虑"]),
         "retrieval_score": 0.9, "matched_queries": ["raw_input"], "retrieval_sources": ["bm25"]},
        {"item": _make_item(id="b", category="relationship"),
         "retrieval_score": 0.5, "matched_queries": ["raw_input"], "retrieval_sources": ["bm25"]},
    ]
    result = content_recommender._rank_candidates(
        candidates=candidates,
        user_input="我最近压力很大",
        current_emotion="焦虑",
        conversation_summary={"conversation_stage": "initial", "key_concerns": ["academic"]},
        user_profile={},
        limit=5,
    )
    assert len(result) == 2
    # 'a' 有情绪匹配应排在前面
    assert result[0].id == "a"


# ============================================================
# main()
# ============================================================

def main():
    # _build_query_variants
    test_build_query_variants_full()
    print("PASS: build_query_variants full")
    test_build_query_variants_minimal()
    print("PASS: build_query_variants minimal")
    test_build_query_variants_no_extra_fields()
    print("PASS: build_query_variants no extra fields")

    # _extract_keywords
    test_extract_keywords_with_chinese()
    print("PASS: extract_keywords with Chinese")
    test_extract_keywords_empty()
    print("PASS: extract_keywords empty")
    test_extract_keywords_psych_keywords()
    print("PASS: extract_keywords psych keywords")

    # _normalize_user_profile
    test_normalize_user_profile_empty()
    print("PASS: normalize_user_profile empty")
    test_normalize_user_profile_partial()
    print("PASS: normalize_user_profile partial")
    test_normalize_user_profile_none()
    print("PASS: normalize_user_profile None")

    # _preference_match_score
    test_preference_match_score_exact_match()
    print("PASS: preference_match_score exact")
    test_preference_match_score_no_match()
    print("PASS: preference_match_score no match")
    test_preference_match_score_empty_profile()
    print("PASS: preference_match_score empty profile")

    # _is_item_allowed_for_risk
    test_is_item_allowed_level_0()
    print("PASS: is_item_allowed_for_risk level_0")
    test_is_item_allowed_level_2_only_soft_beginner()
    print("PASS: is_item_allowed_for_risk level_2")
    test_is_item_allowed_level_3_none()
    print("PASS: is_item_allowed_for_risk level_3")

    # _calculate_match_scores
    test_calculate_match_scores_basic()
    print("PASS: calculate_match_scores basic")
    test_calculate_match_scores_empty_items()
    print("PASS: calculate_match_scores empty")

    # _adjust_score_with_profile
    test_adjust_score_with_profile_type_match()
    print("PASS: adjust_score_with_profile type match")
    test_adjust_score_with_profile_no_profile()
    print("PASS: adjust_score_with_profile no profile")

    # _generate_rationale
    test_generate_rationale_empty()
    print("PASS: generate_rationale empty")
    test_generate_rationale_different_stages()
    print("PASS: generate_rationale stages")
    test_generate_rationale_initial_stage()
    print("PASS: generate_rationale initial stage")

    # _rank_candidates
    test_rank_candidates_empty()
    print("PASS: rank_candidates empty")
    test_rank_candidates_sorts_by_score()
    print("PASS: rank_candidates sorts by score")


if __name__ == "__main__":
    main()
