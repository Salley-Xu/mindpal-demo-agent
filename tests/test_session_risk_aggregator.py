"""
会话级风险聚合器单元测试。

覆盖 5 类场景：
  1. 当前轮明确高危优先
  2. 风险惯性（上轮高危 → 本轮不允许断崖降级）
  3. 连续升级（窗口内多次轻度信号累积）
  4. 安全确认参与降级
  5. 趋势判定
"""

import os
import sys

PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from session_risk_aggregator import SessionRiskAggregator


def make_agg():
    return SessionRiskAggregator()


# ============================================================
# 1. 当前轮优先原则
# ============================================================

def test_current_turn_level_3_always_wins():
    """当前轮 Level 3 → 会话等级直接设为 3，忽略历史。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_3",
        utterance_binary_prob=0.95,
        utterance_rule_hit=True,
        text="正在割腕",
        conversation_summary={"recent_risk_levels": ["level_0", "level_0"]},
    )
    assert result["session_level"] == "level_3"
    assert "current_turn_level_3" in result["session_aggregation"]["active_rules"]


def test_current_turn_level_2_sets_floor():
    """当前轮 Level 2 → 会话等级至少 2，但可被惯性/升级推高。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_2",
        utterance_binary_prob=0.75,
        utterance_rule_hit=False,
        text="我已经准备好了",
        conversation_summary={},
    )
    assert result["session_level"] == "level_2"
    assert "current_turn_level_2" in result["session_aggregation"]["active_rules"]


# ============================================================
# 2. 风险惯性
# ============================================================

def test_inertia_prev_level_3_keeps_level_2():
    """上轮 Level 3 + 本轮 Level 0 → 会话至少 Level 2。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_0",
        utterance_binary_prob=0.05,
        utterance_rule_hit=False,
        text="算了没事了",
        conversation_summary={"recent_risk_levels": ["level_3"]},
    )
    assert result["session_level"] == "level_2"
    assert "inertia:prev_level_3_floor_2" in result["session_aggregation"]["active_rules"]


def test_inertia_prev_level_2_keeps_level_1():
    """上轮 Level 2 + 本轮 Level 0 → 会话至少 Level 1。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_0",
        utterance_binary_prob=0.1,
        utterance_rule_hit=False,
        text="今天感觉好一点了",
        conversation_summary={"recent_risk_levels": ["level_2"]},
    )
    assert result["session_level"] == "level_1"
    assert "inertia:prev_level_2_floor_1" in result["session_aggregation"]["active_rules"]


def test_inertia_does_not_raise_above_utterance_when_history_is_lower():
    """历史低风险 + 本轮也低风险 → 惯性不触发。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_0",
        utterance_binary_prob=0.05,
        utterance_rule_hit=False,
        text="今天心情不错",
        conversation_summary={"recent_risk_levels": ["level_0", "level_0"]},
    )
    assert result["session_level"] == "level_0"
    assert not any(r.startswith("inertia:") for r in result["session_aggregation"]["active_rules"])


# ============================================================
# 3. 连续升级
# ============================================================

def test_escalation_l2_twice_in_window():
    """窗口内 Level 2 出现 2 次 → 会话至少 Level 2。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_1",
        utterance_binary_prob=0.4,
        utterance_rule_hit=False,
        text="还是很难受",
        conversation_summary={
            "recent_risk_levels": ["level_0", "level_1", "level_2", "level_1", "level_2"],
        },
    )
    assert result["session_level"] == "level_2"
    assert "escalation:l2_twice_in_window" in result["session_aggregation"]["active_rules"]


def test_escalation_l1_thrice_in_window():
    """窗口内 Level 1 出现 3 次 → 会话至少 Level 1。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_0",
        utterance_binary_prob=0.1,
        utterance_rule_hit=False,
        text="希望下周能好点",
        conversation_summary={
            "recent_risk_levels": ["level_0", "level_1", "level_1", "level_1"],
        },
    )
    assert result["session_level"] == "level_1"
    assert "escalation:l1_thrice_in_window" in result["session_aggregation"]["active_rules"]


def test_escalation_not_triggered_for_few_low_signals():
    """窗口内信号不足 → 不触发升级。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_0",
        utterance_binary_prob=0.1,
        utterance_rule_hit=False,
        text="今天还行",
        conversation_summary={
            "recent_risk_levels": ["level_0", "level_1", "level_0"],
        },
    )
    assert result["session_level"] == "level_0"
    assert not any(r.startswith("escalation:") for r in result["session_aggregation"]["active_rules"])


# ============================================================
# 4. 安全确认参与降级
# ============================================================

def test_safety_confirmation_detected():
    """用户明确否认时标记 safety_confirmed。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_1",
        utterance_binary_prob=0.3,
        utterance_rule_hit=False,
        text="没有想自杀，只是最近压力大",
        conversation_summary={
            "recent_risk_levels": ["level_0", "level_1"],
            "recent_intents": ["seeking_help"],
        },
    )
    assert result["session_aggregation"]["safety_confirmed"] is True


def test_no_false_safety_confirmation():
    """普通文本不应标记 safety_confirmed。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_1",
        utterance_binary_prob=0.3,
        utterance_rule_hit=False,
        text="今天真的很焦虑",
        conversation_summary={"recent_risk_levels": ["level_0", "level_1"]},
    )
    assert result["session_aggregation"]["safety_confirmed"] is False


# ============================================================
# 5. 趋势判定
# ============================================================

def test_trend_rising():
    """窗口后半段等级高于前半段 → rising。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_2",
        utterance_binary_prob=0.7,
        utterance_rule_hit=False,
        text="越来越难受了",
        conversation_summary={
            "recent_risk_levels": ["level_0", "level_0", "level_1", "level_1"],
        },
    )
    assert result["risk_trend"] == "rising"


def test_trend_falling():
    """窗口后半段等级低于前半段 → falling。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_1",
        utterance_binary_prob=0.5,
        utterance_rule_hit=False,
        text="比昨天好一点",
        conversation_summary={
            "recent_risk_levels": ["level_2", "level_2", "level_1", "level_0"],
        },
    )
    assert result["risk_trend"] == "falling"


def test_trend_stable():
    """窗口前后半段接近 → stable。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_1",
        utterance_binary_prob=0.3,
        utterance_rule_hit=False,
        text="还是老样子",
        conversation_summary={
            "recent_risk_levels": ["level_1", "level_1", "level_1", "level_1"],
        },
    )
    assert result["risk_trend"] == "stable"


def test_trend_new_session():
    """新会话（无历史）→ stable。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_1",
        utterance_binary_prob=0.3,
        utterance_rule_hit=False,
        text="最近有点焦虑",
        conversation_summary={},
    )
    assert result["risk_trend"] == "stable"


# ============================================================
# 6. 窗口完整性
# ============================================================

def test_window_size_capped():
    """窗口不会超过 WINDOW_SIZE=5。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_1",
        utterance_binary_prob=0.3,
        utterance_rule_hit=False,
        text="测试",
        conversation_summary={
            "recent_risk_levels": ["level_0"] * 20,
        },
    )
    assert len(result["session_aggregation"]["window"]) <= 5


def test_return_structure():
    """返回结构包含所有必需字段。"""
    agg = make_agg()
    result = agg.aggregate(
        utterance_level="level_0",
        utterance_binary_prob=0.05,
        utterance_rule_hit=False,
        text="你好",
        conversation_summary={},
    )
    assert "session_level" in result
    assert "risk_trend" in result
    assert "session_aggregation" in result
    agg_info = result["session_aggregation"]
    assert "utterance_level" in agg_info
    assert "window" in agg_info
    assert "safety_confirmed" in agg_info
    assert "active_rules" in agg_info


def main():
    # 当前轮优先
    test_current_turn_level_3_always_wins()
    print("PASS: current_turn_level_3_always_wins")
    test_current_turn_level_2_sets_floor()
    print("PASS: current_turn_level_2_sets_floor")

    # 风险惯性
    test_inertia_prev_level_3_keeps_level_2()
    print("PASS: inertia_prev_level_3_keeps_level_2")
    test_inertia_prev_level_2_keeps_level_1()
    print("PASS: inertia_prev_level_2_keeps_level_1")
    test_inertia_does_not_raise_above_utterance_when_history_is_lower()
    print("PASS: inertia_not_raise_when_history_lower")

    # 连续升级
    test_escalation_l2_twice_in_window()
    print("PASS: escalation_l2_twice_in_window")
    test_escalation_l1_thrice_in_window()
    print("PASS: escalation_l1_thrice_in_window")
    test_escalation_not_triggered_for_few_low_signals()
    print("PASS: escalation_not_triggered_for_few")

    # 安全确认
    test_safety_confirmation_detected()
    print("PASS: safety_confirmation_detected")
    test_no_false_safety_confirmation()
    print("PASS: no_false_safety_confirmation")

    # 趋势
    test_trend_rising()
    print("PASS: trend_rising")
    test_trend_falling()
    print("PASS: trend_falling")
    test_trend_stable()
    print("PASS: trend_stable")
    test_trend_new_session()
    print("PASS: trend_new_session")

    # 窗口
    test_window_size_capped()
    print("PASS: window_size_capped")
    test_return_structure()
    print("PASS: return_structure")

    print("\n所有测试通过!")


if __name__ == "__main__":
    main()
