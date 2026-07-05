"""
风险评估测试 — 基于 BERT 模型（规则系统已完全移除）。

通过 mock BertRiskPredictor 来控制预测结果，验证：
- 返回字典结构与下游兼容
- risk_context 主体识别
- 风险等级传播到路由决策
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

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

from risk_evaluator import risk_evaluator  # noqa: E402
from agent_orchestrator import agent_orchestrator  # noqa: E402
from agent_tools import MoodEvent, MoodTrackingTool, UserProfile, UserProfileTool  # noqa: E402
from conversation_manager import conversation_manager  # noqa: E402
from emotion_analyzer import emotion_analyzer  # noqa: E402
from models import AgentRunRequest  # noqa: E402
from risk_levels import LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3  # noqa: E402
from urgent_detector import urgent_logger  # noqa: E402


# ============================================================
# Helper: mock BERT 预测器
# ============================================================

def _make_mock_predictor(level: str, fusion_source: str = "model_4class",
                         binary_prob: float = 0.0, rule_matched: bool = False):
    """创建返回指定等级的 mock 预测器。"""
    level_map = {"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}
    idx = level_map.get(level, 0)
    probs = [0.0, 0.0, 0.0, 0.0]
    probs[idx] = 1.0
    mock = MagicMock()
    mock.predict.return_value = {
        "level": level,
        "level_4_prediction": idx,
        "binary_probability": binary_prob,
        "class_probabilities": probs,
        "fusion_source": fusion_source,
        "rule_matched": rule_matched,
    }
    return mock


# ============================================================
# 1. 等级判定（mock BERT 直接返回预期等级）
# ============================================================

def test_high_risk_detection():
    risk_evaluator._predictor = _make_mock_predictor(LEVEL_3, binary_prob=0.95)
    result = risk_evaluator.evaluate(
        text="我真的不想活了，今晚想结束生命。",
    )
    assert result["level"] == LEVEL_3
    assert result["legacy_level"] == "high"
    assert result["risk_score"] >= 8.0
    assert result["suggestions"]
    assert "bert_model_4class" in result["escalation_reasons"]


def test_medium_risk_detection():
    risk_evaluator._predictor = _make_mock_predictor(LEVEL_2, binary_prob=0.78)
    result = risk_evaluator.evaluate(
        text="最近我真的有点崩溃，感觉撑不住了。",
    )
    assert result["level"] == LEVEL_2
    assert result["legacy_level"] == "medium"
    assert result["risk_score"] >= 5.0


def test_low_risk_detection():
    risk_evaluator._predictor = _make_mock_predictor(LEVEL_0, binary_prob=0.05)
    result = risk_evaluator.evaluate(
        text="这周有点累，但我想慢慢调整状态。",
    )
    assert result["level"] == LEVEL_0
    assert result["legacy_level"] == "low"


def test_normal_risk_level_1():
    risk_evaluator._predictor = _make_mock_predictor(LEVEL_1, binary_prob=0.35)
    result = risk_evaluator.evaluate(
        text="最近有点焦虑，睡不好。",
    )
    assert result["level"] == LEVEL_1
    assert result["legacy_level"] == "medium"
    assert result["suggestions"]


# ============================================================
# 2. risk_context 主体识别（纯 regex，不依赖 BERT）
# ============================================================

def test_safe_denial_keeps_level_low():
    risk_evaluator._predictor = _make_mock_predictor(LEVEL_1)
    result = risk_evaluator.evaluate(
        text="我没有想自杀，只是最近压力真的很大。",
    )
    assert result["risk_context"]["is_safe_denial"] is True
    assert result["risk_context"]["subject"] == "self"


def test_third_party_help_request_recognized():
    risk_evaluator._predictor = _make_mock_predictor(LEVEL_1)
    result = risk_evaluator.evaluate(
        text="我朋友说他不想活了，我该怎么办？",
    )
    assert result["risk_context"]["subject"] == "third_party"
    assert result["risk_context"]["is_third_party_risk"] is True
    assert result["risk_context"]["is_help_request"] is True


def test_discussion_context_recognized():
    risk_evaluator._predictor = _make_mock_predictor(LEVEL_0)
    result = risk_evaluator.evaluate(
        text="这部电影里主角最后自杀了，我看完心里很难受。",
    )
    assert result["risk_context"]["subject"] == "discussion"
    assert result["risk_context"]["is_discussion_context"] is True


# ============================================================
# 3. 返回字典结构完整性（下游依赖）
# ============================================================

def test_evaluate_result_structure():
    risk_evaluator._predictor = _make_mock_predictor(LEVEL_2, fusion_source="binary_upgrade")
    result = risk_evaluator.evaluate(
        text="帮帮我，我很难受。",
        conversation_summary={"recent_risk_levels": ["level_1"]},
    )
    # 所有下游依赖的字段必须存在
    required_keys = {
        "level", "legacy_level", "level_index", "level_label",
        "message", "suggestions",
        "risk_score", "raw_score",
        "escalation_reasons", "recent_risk_levels", "risk_context",
    }
    assert required_keys.issubset(result.keys()), f"Missing: {required_keys - result.keys()}"

    # BERT 证据写入 risk_evidence
    assert "bert" in result["risk_evidence"]
    bert_info = result["risk_evidence"]["bert"]
    assert bert_info["fusion_source"] == "binary_upgrade"
    assert bert_info["binary_probability"] == 0.0

    # recent_risk_levels 从 conversation_summary 传入
    assert result["recent_risk_levels"] == ["level_1"]


# ============================================================
# 4. 路由决策传播（agent_orchestrator 集成）
# ============================================================

def test_orchestrator_risk_state_builder():
    """验证 _build_risk_state 接受的 dict 格式兼容新输出。"""
    risk_state = agent_orchestrator._build_risk_state(
        {
            "level": LEVEL_2,
            "message": "检测到高风险倾向",
            "suggestions": ["联系身边可信任的人"],
            "risk_score": 7.5,
            "raw_score": 7.5,
            "risk_evidence": {"bert": {"fusion_source": "model_4class"}},
            "escalation_reasons": ["bert_model_4class"],
        }
    )
    assert risk_state.level == LEVEL_2
    assert risk_state.legacy_level == "medium"
    assert risk_state.risk_score == 7.5
    assert risk_state.escalation_reasons == ["bert_model_4class"]


def test_urgent_logger_statistics_compatibility():
    """紧急日志统计与风险等级无关。"""
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
    assert stats["warning_high_count"] == 1
    assert stats["medium_count"] == 3


# ============================================================
# 5. 完整集成：run_agent 传递风险上下文（mock 整个 evaluate）
# ============================================================

@pytest.mark.asyncio
async def test_run_agent_passes_long_term_risk_context_into_precheck():
    """验证 evaluate 被调用时收到正确的长期风险参数（集成测试）。"""
    user_id = f"user_risk_ctx_{uuid.uuid4().hex[:8]}"
    session_id = f"session_risk_ctx_{uuid.uuid4().hex[:8]}"
    captured = {}

    original_use_persistence = conversation_manager.use_persistence
    original_emotion_method = emotion_analyzer.analyze_with_context_async
    original_get_profile = UserProfileTool.get_profile
    original_get_recent_trend = MoodTrackingTool.get_recent_trend
    original_evaluate = risk_evaluator.evaluate
    original_chat_create = agent_orchestrator.client.chat.completions.create

    async def fake_analyze_with_context_async(text, conversation_summary):
        return "压力", "疲惫", 0.76

    async def fake_get_profile(user_id_arg):
        assert user_id_arg == user_id
        return UserProfile(
            user_id=user_id_arg,
            risk_level="level_3",
            preferred_types=[],
            preferred_categories=[],
            preferred_difficulty="beginner",
            preferred_duration_range=None,
            preferred_support_style=None,
            avoid_style=[],
            main_stress_sources=[],
            recommendation_feedback={},
            last_updated=datetime.now(timezone.utc),
        )

    async def fake_get_recent_trend(user_id_arg, limit=20):
        assert user_id_arg == user_id
        return [
            MoodEvent(
                user_id=user_id_arg,
                session_id="s1",
                emotion="焦虑",
                emotion_type="anxiety",
                emotion_intensity=0.9,
                stress_source="求职",
                user_intent="sharing",
                event_summary="历史上出现明显风险升高。",
                risk_level="level_3",
                source="conversation",
                text_snippet="之前一度觉得撑不住。",
                created_at=datetime.now(timezone.utc),
            ),
            MoodEvent(
                user_id=user_id_arg,
                session_id="s2",
                emotion="难过",
                emotion_type="sadness",
                emotion_intensity=0.8,
                stress_source="学业",
                user_intent="sharing",
                event_summary="又一次高风险波动。",
                risk_level="level_3",
                source="conversation",
                text_snippet="那几天一直很绝望。",
                created_at=datetime.now(timezone.utc),
            ),
            MoodEvent(
                user_id=user_id_arg,
                session_id="s3",
                emotion="疲惫",
                emotion_type="stress",
                emotion_intensity=0.5,
                stress_source="日常",
                user_intent="sharing",
                event_summary="普通低风险事件。",
                risk_level="level_0",
                source="conversation",
                text_snippet="最近有点累。",
                created_at=datetime.now(timezone.utc),
            ),
        ]

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {
            "level": LEVEL_0,
            "legacy_level": "low",
            "level_index": 0,
            "level_label": "Level 0",
            "message": "",
            "suggestions": [],
            "risk_score": 3.6,
            "raw_score": 0.0,
            "risk_evidence": {},
            "escalation_reasons": [],
            "recent_risk_levels": [],
            "risk_context": {"subject": "self"},
        }

    async def fake_chat_create(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="我们先把今天最难受的部分拆开来说。", tool_calls=None)
                )
            ]
        )

    conversation_manager.use_persistence = False
    emotion_analyzer.analyze_with_context_async = fake_analyze_with_context_async
    UserProfileTool.get_profile = fake_get_profile
    MoodTrackingTool.get_recent_trend = staticmethod(fake_get_recent_trend)
    risk_evaluator.evaluate = fake_evaluate
    agent_orchestrator.client.chat.completions.create = fake_chat_create

    try:
        result = await agent_orchestrator.run_agent(
            AgentRunRequest(
                text="这两天我有点麻木，感觉自己快撑不住了。",
                user_id=user_id,
                session_id=session_id,
                return_steps=True,
            )
        )
        assert result.chat.response == "我们先把今天最难受的部分拆开来说。"
        assert captured["long_term_risk_level"] == "level_3"
        assert captured["historical_high_risk_count"] == 2
        assert captured["text"] == "这两天我有点麻木，感觉自己快撑不住了。"
        assert captured["conversation_summary"]["primary_emotion"] == "压力"
    finally:
        conversation_manager.use_persistence = original_use_persistence
        emotion_analyzer.analyze_with_context_async = original_emotion_method
        UserProfileTool.get_profile = original_get_profile
        MoodTrackingTool.get_recent_trend = original_get_recent_trend
        risk_evaluator.evaluate = original_evaluate
        agent_orchestrator.client.chat.completions.create = original_chat_create
        await conversation_manager.delete_session_async(user_id, session_id)


def main():
    test_high_risk_detection()
    print("PASS: high risk detection")
    test_medium_risk_detection()
    print("PASS: medium risk detection")
    test_low_risk_detection()
    print("PASS: low risk detection")
    test_normal_risk_level_1()
    print("PASS: normal risk level 1")
    test_safe_denial_keeps_level_low()
    print("PASS: safe denial context")
    test_third_party_help_request_recognized()
    print("PASS: third party context")
    test_discussion_context_recognized()
    print("PASS: discussion context")
    test_evaluate_result_structure()
    print("PASS: result structure")
    test_orchestrator_risk_state_builder()
    print("PASS: risk state builder")
    test_urgent_logger_statistics_compatibility()
    print("PASS: urgent logger stats")
    asyncio.run(test_run_agent_passes_long_term_risk_context_into_precheck())
    print("PASS: agent precheck integration")


if __name__ == "__main__":
    main()
