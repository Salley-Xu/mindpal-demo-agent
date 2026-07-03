"""
长期风险记忆层（v6.0）单元测试。

覆盖:
  1. risk_event 写入条件
  2. 基线计算（high / medium / low）
  3. 衰减机制
  4. Safety Gate baseline floor
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from risk_memory import RiskMemoryWriter, RiskMemoryReader
from session_risk_aggregator import SessionRiskAggregator
from risk_levels import LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3


# ============================================================
# RiskMemoryWriter 核心逻辑测试
# ============================================================

def test_extract_topics():
    """主题提取应根据关键词正确匹配。"""
    from risk_memory import _extract_topics
    assert "学业" in _extract_topics("考试没考好，很焦虑")
    assert "求职" in _extract_topics("秋招面试挂了")
    assert "睡眠" in _extract_topics("最近一直失眠")
    assert _extract_topics("今天天气不错") == []


def test_recompute_baseline_high_from_level_3():
    """7 天内 Level 3 → baseline high。"""
    now = datetime.now(timezone.utc)
    events = [
        {"event_time": (now - timedelta(days=1)).isoformat(), "max_level": "level_3", "decayed": False},
    ]
    assert RiskMemoryWriter._recompute_baseline(events, now) == "high"


def test_recompute_baseline_high_from_two_level_2():
    """7 天内 2 次 Level 2 → baseline high。"""
    now = datetime.now(timezone.utc)
    events = [
        {"event_time": (now - timedelta(days=1)).isoformat(), "max_level": "level_2", "decayed": False},
        {"event_time": (now - timedelta(days=2)).isoformat(), "max_level": "level_2", "decayed": False},
    ]
    assert RiskMemoryWriter._recompute_baseline(events, now) == "high"


def test_recompute_baseline_medium_from_level_2():
    """14 天内 Level 2 → baseline medium。"""
    now = datetime.now(timezone.utc)
    events = [
        {"event_time": (now - timedelta(days=10)).isoformat(), "max_level": "level_2", "decayed": False},
    ]
    assert RiskMemoryWriter._recompute_baseline(events, now) == "medium"


def test_recompute_baseline_low_no_events():
    """无事件 → baseline low。"""
    now = datetime.now(timezone.utc)
    assert RiskMemoryWriter._recompute_baseline([], now) == "low"


def test_recompute_baseline_decayed_ignored():
    """decayed=True 的事件不参与基线计算。"""
    now = datetime.now(timezone.utc)
    events = [
        {"event_time": (now - timedelta(days=1)).isoformat(), "max_level": "level_3", "decayed": True},
    ]
    assert RiskMemoryWriter._recompute_baseline(events, now) == "low"


def test_apply_decay_marks_old_events():
    """超过 30 天的事件标记为 decayed。"""
    now = datetime.now(timezone.utc)
    memory = {
        "risk_events": [
            {"event_time": (now - timedelta(days=40)).isoformat(), "max_level": "level_3", "decayed": False},
        ],
        "common_triggers": ["学业"],
    }
    result = RiskMemoryWriter._apply_decay(memory, now)
    assert result["risk_events"][0]["decayed"] is True


# ============================================================
# Safety Gate — session_risk_aggregator baseline 参数
# ============================================================

def test_safety_gate_baseline_high_raises_level_0():
    """baseline=high + session_level=0 → 提升到 Level 1。"""
    agg = SessionRiskAggregator()
    result = agg.aggregate(
        utterance_level="level_0",
        utterance_binary_prob=0.05,
        utterance_rule_hit=False,
        text="今天心情不错",
        conversation_summary={},
        baseline="high",
    )
    assert result["session_level"] == LEVEL_1
    assert "baseline:high_floor_1" in result["session_aggregation"]["active_rules"]


def test_safety_gate_baseline_high_does_not_override_high_current():
    """baseline=high 但当前轮已是 Level 2 → 不叠加。"""
    agg = SessionRiskAggregator()
    result = agg.aggregate(
        utterance_level="level_2",
        utterance_binary_prob=0.7,
        utterance_rule_hit=False,
        text="很难受",
        conversation_summary={},
        baseline="high",
    )
    assert result["session_level"] == LEVEL_2
    assert not any(r.startswith("baseline:") for r in result["session_aggregation"]["active_rules"])


def test_safety_gate_baseline_low_no_effect():
    """baseline=low → 不触发任何 Safety Gate。"""
    agg = SessionRiskAggregator()
    result = agg.aggregate(
        utterance_level="level_0",
        utterance_binary_prob=0.05,
        utterance_rule_hit=False,
        text="今天心情不错",
        conversation_summary={},
        baseline="low",
    )
    assert result["session_level"] == LEVEL_0
    assert not any(r.startswith("baseline:") for r in result["session_aggregation"]["active_rules"])


# ============================================================
# RiskMemoryReader
# ============================================================

@pytest.mark.asyncio
async def test_reader_returns_default_for_unknown_user():
    """不存在的用户 → 返回默认 low。"""
    reader = RiskMemoryReader()
    result = await reader.get_baseline("nonexistent_user")
    assert result["baseline"] == "low"
    assert result["common_triggers"] == []


# ============================================================
# RiskMemoryWriter (集成)
# ============================================================

@pytest.mark.asyncio
async def test_writer_skips_low_risk():
    """Level 0-1 不触发写入。"""
    writer = RiskMemoryWriter()
    # 不应抛出异常（内部应直接 return 不操作）
    await writer.update(
        user_id="test_user",
        session_id="sess_001",
        risk_state={"level": "level_0", "risk_context": {"subject": "self"}},
        conversation_summary={},
        emotion_state={},
    )
    # 无断言，只验证不抛异常


@pytest.mark.asyncio
async def test_writer_skips_third_party():
    """第三方语境不写入自身记忆。"""
    writer = RiskMemoryWriter()
    await writer.update(
        user_id="test_user",
        session_id="sess_001",
        risk_state={"level": "level_3", "risk_context": {"subject": "third_party"}},
        conversation_summary={},
        emotion_state={},
    )
    # 无断言，只验证不抛异常


def main():
    # 主题提取
    test_extract_topics()
    print("PASS: extract_topics")

    # 基线计算
    test_recompute_baseline_high_from_level_3()
    print("PASS: baseline high from level_3")
    test_recompute_baseline_high_from_two_level_2()
    print("PASS: baseline high from two level_2")
    test_recompute_baseline_medium_from_level_2()
    print("PASS: baseline medium from level_2")
    test_recompute_baseline_low_no_events()
    print("PASS: baseline low no events")
    test_recompute_baseline_decayed_ignored()
    print("PASS: baseline decayed ignored")

    # 衰减
    test_apply_decay_marks_old_events()
    print("PASS: apply decay")

    # Safety Gate
    test_safety_gate_baseline_high_raises_level_0()
    print("PASS: safety gate baseline high")
    test_safety_gate_baseline_high_does_not_override_high_current()
    print("PASS: safety gate high no override")
    test_safety_gate_baseline_low_no_effect()
    print("PASS: safety gate low no effect")

    # Reader
    asyncio.run(test_reader_returns_default_for_unknown_user())
    print("PASS: reader default")

    # Writer (integration smoke tests)
    asyncio.run(test_writer_skips_low_risk())
    print("PASS: writer skips low risk")
    asyncio.run(test_writer_skips_third_party())
    print("PASS: writer skips third party")

    print("\n所有测试通过!")


if __name__ == "__main__":
    import asyncio
    main()
