# -*- coding: utf-8 -*-
"""
Phase 2 AgentState 单元测试（§31-32）。

覆盖：schema / builder / updater / lifecycle / adapters / shadow consistency。
纯确定性逻辑，不加载 BERT / 不调 LLM。

运行：python evaluation/tests/test_agent_state.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from state.adapters import EmotionAdapter, IntentAdapter, RiskAdapter  # noqa: E402
from state.builder import build_agent_state  # noqa: E402
from state.schema import AgentState, RiskState  # noqa: E402
from state.updater import update_state  # noqa: E402

# ===== Schema =====


def test_schema_defaults():
    s = AgentState()
    assert s.schema_version == "1.0"
    assert s.risk.level == 0
    assert s.intent.labels == []
    assert s.derived.help_seeking is False


def test_schema_no_action_fields():
    """State 禁止包含 Policy 输出字段。"""
    model = AgentState.model_fields
    banned = {"should_recommend", "should_retrieve_memory", "should_safety_intervene",
              "primary_action", "tool_actions", "recommendation_mode", "current_action"}
    for b in banned:
        assert b not in model, f"State 不应包含 {b}"
    # RecommendationState 只存历史
    rec_fields = set(AgentState.model_fields["recommendation"].annotation.model_fields.keys())
    assert "should_recommend" not in rec_fields


def test_schema_validation():
    s = AgentState.model_validate({
        "identity": {"user_id": "u1", "session_id": "s1"},
        "turn": {"turn_index": 1, "current_text": "hi"},
        "risk": {"level": 2},
        "intent": {"labels": ["emotional_expression"], "confidence": 0.8},
    })
    assert s.risk.level == 2
    assert s.intent.labels == ["emotional_expression"]

# ===== Adapters =====


def test_emotion_adapter():
    es = EmotionAdapter.to_state({"current_emotion": "焦虑", "emotion_intensity": 0.8,
                                  "user_intent": "seeking_help", "stress_source": "工作"})
    assert es.current_emotion == "焦虑"
    assert es.intensity == 0.8
    assert es.legacy_user_intent == "seeking_help"


def test_risk_adapter_levels():
    assert RiskAdapter.to_state({"level": "level_3"}).level == 3
    assert RiskAdapter.to_state({"level": "low"}).level == 0
    assert RiskAdapter.to_state({"level": "high"}).level == 3
    assert RiskAdapter.to_state({}).level == 0


def test_risk_adapter_context():
    rs = RiskAdapter.to_state({"level": "level_2", "risk_context": {
        "subject": "third_party", "is_third_party_risk": True, "is_safe_denial": True}})
    assert rs.subject == "third_party"
    assert rs.is_third_party is True
    assert rs.safe_denial is True

# ===== Builder =====


def _build(**kw):
    defaults = dict(
        user_id="u1", session_id="s1", request_text="最近压力好大",
        turn_index=1,
        emotion_state={"current_emotion": "焦虑", "confidence": 0.7},
        urgent_issue={"level": "level_1", "risk_trend": "rising"},
        conversation_summary={"conversation_stage": "exploring", "turn_count": 1},
        user_profile={"preferred_support_style": "direct_actionable"},
    )
    defaults.update(kw)
    return build_agent_state(**defaults)


def test_builder_derived():
    s = _build(intent_result={"labels": ["explicit_help_request"], "confidence": 0.9})
    assert s.derived.help_seeking is True
    assert s.derived.intent_uncertain is False
    assert s.derived.negative_emotion is True  # 焦虑


def test_builder_derived_uncertain():
    s = _build(intent_result={"labels": [], "confidence": 0.2, "is_open_set": True})
    assert s.derived.intent_uncertain is True


def test_builder_memory_signal():
    s = _build(intent_result={"labels": ["memory_reference"], "confidence": 0.9},
               memory_signal={"available_memory_count": 42, "last_retrieved_memory_ids": ["m1"]})
    assert s.memory.explicit_memory_reference is True
    assert s.memory.available_memory_count == 42
    assert s.memory.last_retrieved_memory_ids == ["m1"]


def test_builder_identity_and_meta():
    s = _build()
    assert s.identity.user_id == "u1"
    assert s.identity.session_id == "s1"
    assert s.meta.trace_id != ""

# ===== Updater =====


def test_updater_risk_persistence_accumulate():
    s1 = _build(urgent_issue={"level": "level_2"})
    s2 = update_state(s1, s1.model_copy(deep=True))
    assert s2.risk.persistence == 1
    s3 = update_state(s2, s2.model_copy(deep=True))
    assert s3.risk.persistence == 2


def test_updater_risk_persistence_reset():
    s1 = _build(urgent_issue={"level": "level_2"})
    s1 = update_state(s1, s1.model_copy(deep=True))  # persistence=1
    low = _build(urgent_issue={"level": "level_0"})
    s2 = update_state(s1, low)
    assert s2.risk.persistence == 0


def test_updater_risk_history_window():
    s = None
    for level in ["level_0", "level_1", "level_2", "level_3", "level_0", "level_1", "level_2"]:
        cur = _build(urgent_issue={"level": level})
        s = update_state(s, cur)
    assert len(s.risk.recent_risk_levels) <= 5
    assert s.risk.recent_risk_levels[-1] == 2


def test_updater_turns_since_recommendation():
    s1 = _build()
    s1.recommendation.recent_recommendation_ids = []
    s2 = update_state(s1, s1.model_copy(deep=True))
    assert s2.recommendation.turns_since_last_recommendation == 1
    # 有推荐则 reset
    s3 = s2.model_copy(deep=True)
    s3.recommendation.recent_recommendation_ids = ["c1"]
    s4 = update_state(s2, s3)
    assert s4.recommendation.turns_since_last_recommendation == 0


def test_updater_state_version():
    s1 = _build()
    s2 = update_state(s1, s1.model_copy(deep=True))
    assert s2.meta.state_version == s1.meta.state_version + 1

# ===== Lifecycle / Isolation =====


def test_session_isolation():
    a = _build(user_id="u_a", session_id="s_a", urgent_issue={"level": "level_2"})
    a = update_state(a, a.model_copy(deep=True))  # persistence=1
    b = _build(user_id="u_b", session_id="s_b", urgent_issue={"level": "level_0"})
    b = update_state(None, b)  # 新 session 无 previous
    assert a.risk.persistence == 1
    assert b.risk.persistence == 0
    assert a.identity.session_id == "s_a"
    assert b.identity.session_id == "s_b"

# ===== Shadow Consistency =====


def test_consistency_checker():
    from state.debug import ConsistencyChecker
    s = _build(urgent_issue={"level": "level_1"},
               emotion_state={"current_emotion": "焦虑"},
               conversation_summary={"conversation_stage": "exploring", "turn_count": 1})
    result = ConsistencyChecker.check(s, {
        "risk_level": "level_1", "emotion": "焦虑", "stage": "exploring", "turn_count": 1,
    })
    assert ConsistencyChecker.all_match(result)
    result2 = ConsistencyChecker.check(s, {"risk_level": "level_3", "emotion": "平静"})
    assert result2["risk_match"] is False
    assert result2["emotion_match"] is False


def test_shadow_runner_no_crash():
    from state.shadow import ShadowRunner
    r = ShadowRunner(use_intent=False)
    state = r.run(user_id="u1", session_id="s1", request_text="hi", turn_index=1,
                  emotion_state={"current_emotion": "中性"},
                  urgent_issue={"level": "level_0"},
                  conversation_summary={"conversation_stage": "initial", "turn_count": 1},
                  user_profile={})
    assert state is not None
    assert state.conversation.stage == "initial"


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception:
                failed += 1
                print(f"[FAIL] {name}")
                traceback.print_exc()
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
