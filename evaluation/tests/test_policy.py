# -*- coding: utf-8 -*-
"""
Phase 3 Policy 单元测试（§25）。

覆盖：schema / safety / primary action / tool action / recommendation mode /
      ambiguity / validator / hybrid / shadow。

纯确定性逻辑，不加载 BERT / 不调 LLM。

运行：python evaluation/tests/test_policy.py
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

from policy.ambiguity import ambiguity_detector  # noqa: E402
from policy.deterministic import deterministic_policy  # noqa: E402
from policy.hybrid import hybrid_policy  # noqa: E402
from policy.safety import safety_policy  # noqa: E402
from policy.schema import (  # noqa: E402
    ActionPlan, PolicyResult, PrimaryAction, RecommendationMode, SafetyTarget, ToolAction,
)
from policy.shadow import classify_diff, policy_shadow_runner  # noqa: E402
from policy.validator import policy_validator  # noqa: E402
from state.builder import build_agent_state  # noqa: E402


def _state(text="你好", intent=None, risk_level=0, emotion="中性", risk_ctx=None,
           rejection=False, intensity=0.5, intent_conf=0.9, open_set=False):
    return build_agent_state(
        user_id="u_test", session_id="s_test", request_text=text, turn_index=1,
        intent_result={"labels": intent or [], "confidence": intent_conf,
                       "is_open_set": open_set, "source": "test"},
        emotion_state={"current_emotion": emotion, "emotion_intensity": intensity,
                       "negative_trend": emotion in ("焦虑", "抑郁", "愤怒", "压力")},
        urgent_issue={"level": f"level_{risk_level}", "risk_trend": "new",
                      "risk_context": risk_ctx or {}},
        conversation_summary={"conversation_stage": "initial", "turn_count": 1,
                              "has_rejected_recommendation": rejection},
        user_profile={},
    )


# ===== Schema =====

def test_schema_enum_values():
    assert PrimaryAction.CONTINUE_CHAT.value == "continue_chat"
    assert ToolAction.RECOMMEND_RESOURCE.value == "recommend_resource"
    assert SafetyTarget.THIRD_PARTY.value == "third_party"
    assert RecommendationMode.SAFETY_ONLY.value == "safety_only"


def test_action_plan_defaults():
    p = ActionPlan(primary_action=PrimaryAction.CONTINUE_CHAT)
    assert p.tool_actions == []
    assert p.safety_target == SafetyTarget.NONE
    assert p.recommendation_mode == RecommendationMode.NONE


def test_policy_result_traceable():
    r = PolicyResult(action_plan=ActionPlan(primary_action=PrimaryAction.CONTINUE_CHAT),
                     confidence=0.9, source="rule", matched_rules=["D01"])
    assert r.primary_action == "continue_chat"
    assert r.policy_version == "1.0"
    assert r.fallback_reason is None


# ===== Safety Policy =====

def test_safety_self_l3():
    st = _state("我想自杀", intent=["high_risk_expression"], risk_level=3, emotion="绝望")
    r = safety_policy.decide(st)
    assert r is not None
    assert r.action_plan.primary_action == PrimaryAction.SAFETY_INTERVENTION
    assert r.action_plan.safety_target == SafetyTarget.SELF
    assert r.action_plan.recommendation_mode == RecommendationMode.NONE
    assert r.action_plan.tool_actions == []


def test_safety_self_l2():
    st = _state("我想死", intent=["high_risk_expression"], risk_level=2, emotion="绝望")
    r = safety_policy.decide(st)
    assert r is not None
    assert r.action_plan.safety_target == SafetyTarget.SELF
    # gold：L2 self → rec=none
    assert r.action_plan.recommendation_mode == RecommendationMode.NONE


def test_safety_third_party():
    st = _state("我朋友说想自杀，我该怎么办", intent=["high_risk_expression"], risk_level=2,
                emotion="焦虑",
                risk_ctx={"subject": "third_party", "is_third_party_risk": True, "is_help_request": True})
    r = safety_policy.decide(st)
    assert r is not None
    assert r.action_plan.safety_target == SafetyTarget.THIRD_PARTY
    assert r.action_plan.recommendation_mode == RecommendationMode.SAFETY_ONLY


def test_safety_implicit_high_risk_l2():
    """隐式高风险（intent 非 high_risk_expression，但 risk level 已升级）。"""
    st = _state("我真的撑不下去了", intent=["emotional_expression"], risk_level=2, emotion="绝望")
    r = safety_policy.decide(st)
    assert r is not None
    assert r.action_plan.primary_action == PrimaryAction.SAFETY_INTERVENTION


def test_safety_discussion_not_escalated():
    """讨论语境（电影/新闻）非求助 → 不触发 Safety（INV-11）。"""
    st = _state("我在电影里看到主角想自杀的情节", intent=["casual_chat"], risk_level=0,
                emotion="中性", risk_ctx={"subject": "discussion", "is_discussion_context": True})
    r = safety_policy.decide(st)
    assert r is None or r.action_plan.primary_action != PrimaryAction.SAFETY_INTERVENTION


def test_safety_safe_denial_not_escalated():
    """安全否认 → 不强行升级（INV-11）。"""
    st = _state("我没有想自杀，只是有点难过", intent=["emotional_expression"], risk_level=1,
                emotion="悲伤", risk_ctx={"is_safe_denial": True})
    r = safety_policy.decide(st)
    assert r is None or r.action_plan.primary_action != PrimaryAction.SAFETY_INTERVENTION


def test_safety_risk_plus_resource():
    """风险 + 资源请求 → 仍安全，不推荐普通内容。"""
    st = _state("我想死，有推荐的放松音频吗", intent=["high_risk_expression", "resource_request"],
                risk_level=3, emotion="绝望")
    r = safety_policy.decide(st)
    assert r is not None
    assert r.action_plan.recommendation_mode != RecommendationMode.HARD
    assert ToolAction.RECOMMEND_RESOURCE not in r.action_plan.tool_actions


def test_safety_risk_plus_memory():
    """风险 + 记忆引用 → 仍安全，不做 memory 检索。"""
    st = _state("还记得我之前说过想死吗", intent=["high_risk_expression", "memory_reference"],
                risk_level=3, emotion="绝望")
    r = safety_policy.decide(st)
    assert r is not None
    assert ToolAction.RETRIEVE_MEMORY not in r.action_plan.tool_actions


# ===== S04: high_risk_intent fallback（Contract A，C3.1）=====

def test_safety_intent_high_risk_low():
    """Intent 高 / Risk 低冲突 → S04 兜底触发 self safety（Contract A）。"""
    st = _state("我脑子里全是想死的念头", intent=["high_risk_expression"], risk_level=0,
                emotion="绝望")
    r = safety_policy.decide(st)
    assert r is not None
    assert r.action_plan.primary_action == PrimaryAction.SAFETY_INTERVENTION
    assert r.action_plan.safety_target == SafetyTarget.SELF
    assert "S04" in "".join(r.matched_rules)


def test_safety_intent_high_with_safe_denial():
    """Intent 高但 safe_denial → S04 不触发（INV-11）。"""
    st = _state("我没有想死", intent=["high_risk_expression"], risk_level=0,
                emotion="悲伤", risk_ctx={"is_safe_denial": True})
    r = safety_policy.decide(st)
    assert r is None or r.action_plan.primary_action != PrimaryAction.SAFETY_INTERVENTION


def test_safety_intent_high_with_discussion():
    """Intent 高但 discussion 语境 → S04 不触发（INV-11）。"""
    st = _state("电影里的主角想自杀", intent=["high_risk_expression"], risk_level=0,
                emotion="中性", risk_ctx={"subject": "discussion", "is_discussion_context": True})
    r = safety_policy.decide(st)
    assert r is None or r.action_plan.primary_action != PrimaryAction.SAFETY_INTERVENTION


def test_safety_intent_high_no_fpr_on_normal():
    """正常对话不因 intent 误触发 S04。"""
    st = _state("今天心情不错", intent=["casual_chat"], risk_level=0, emotion="中性")
    r = safety_policy.decide(st)
    assert r is None


# ===== Deterministic: Primary Action =====

def test_primary_information_response():
    st = _state("怎么放松冥想的方法", intent=["information_request"], risk_level=0)
    d = deterministic_policy.decide(st)
    assert d.action_plan.primary_action == PrimaryAction.INFORMATION_RESPONSE
    assert ToolAction.RETRIEVE_KNOWLEDGE in d.action_plan.tool_actions


def test_primary_meta_question():
    st = _state("你是真人吗", intent=["meta_question"], risk_level=0)
    d = deterministic_policy.decide(st)
    assert d.action_plan.primary_action == PrimaryAction.INFORMATION_RESPONSE


def test_primary_vague_utterance():
    st = _state("就是那个……怎么说呢", intent=[], risk_level=0, intent_conf=0.5)
    d = deterministic_policy.decide(st)
    assert d.action_plan.primary_action == PrimaryAction.ASK_CLARIFICATION


def test_primary_continue_chat():
    st = _state("今天天气不错", intent=["casual_chat"], risk_level=0, emotion="中性")
    d = deterministic_policy.decide(st)
    assert d.action_plan.primary_action == PrimaryAction.CONTINUE_CHAT


# ===== Deterministic: Tool Actions =====

def test_tool_resource_recommend():
    st = _state("推荐点冥想音频吧", intent=["resource_request"], risk_level=0)
    d = deterministic_policy.decide(st)
    assert ToolAction.RECOMMEND_RESOURCE in d.action_plan.tool_actions
    assert d.action_plan.recommendation_mode == RecommendationMode.HARD


def test_tool_memory_reference():
    st = _state("我上次说的那个方法，现在可以开始吗", intent=["memory_reference"], risk_level=0)
    d = deterministic_policy.decide(st)
    assert ToolAction.RETRIEVE_MEMORY in d.action_plan.tool_actions
    # INV-07：memory retrieval 不是 primary
    assert d.action_plan.primary_action == PrimaryAction.CONTINUE_CHAT


def test_tool_multi_select():
    st = _state("推荐点放松方法，我上次失眠也问过", intent=["resource_request", "memory_reference"],
                risk_level=0)
    d = deterministic_policy.decide(st)
    assert ToolAction.RECOMMEND_RESOURCE in d.action_plan.tool_actions
    assert ToolAction.RETRIEVE_MEMORY in d.action_plan.tool_actions


def test_tool_rejection_suppresses_rec():
    st = _state("推荐点书", intent=["resource_request"], risk_level=0, rejection=True)
    d = deterministic_policy.decide(st)
    assert d.action_plan.recommendation_mode == RecommendationMode.NONE


# ===== Deterministic: Recommendation Mode =====

def test_rec_resource_hard():
    st = _state("推荐几本缓解焦虑的书吧", intent=["resource_request"], risk_level=0)
    d = deterministic_policy.decide(st)
    assert d.action_plan.recommendation_mode == RecommendationMode.HARD


def test_rec_none_default():
    st = _state("今天有点累", intent=["emotional_expression"], risk_level=0, emotion="疲惫")
    d = deterministic_policy.decide(st)
    assert d.action_plan.recommendation_mode == RecommendationMode.NONE


# ===== Ambiguity =====

def test_ambiguity_clean():
    st = _state("推荐点冥想音频吧", intent=["resource_request"], risk_level=0)
    d = deterministic_policy.decide(st)
    amb = ambiguity_detector.detect(st, d)
    assert not amb.is_ambiguous


def test_ambiguity_conflict():
    st = _state("推荐点方法，也想知道原理", intent=["information_request", "resource_request"],
                risk_level=0)
    d = deterministic_policy.decide(st)
    amb = ambiguity_detector.detect(st, d)
    assert amb.is_ambiguous


def test_ambiguity_intent_uncertain():
    st = _state("就是那个……", intent=[], risk_level=0, intent_conf=0.4, open_set=True)
    d = deterministic_policy.decide(st)
    amb = ambiguity_detector.detect(st, d)
    assert amb.is_ambiguous
    assert "intent_open_set" in amb.reasons


# ===== Validator =====

def test_validator_accepts_valid_plan():
    plan = ActionPlan(primary_action=PrimaryAction.CONTINUE_CHAT,
                      tool_actions=[ToolAction.RETRIEVE_MEMORY],
                      safety_target=SafetyTarget.NONE,
                      recommendation_mode=RecommendationMode.NONE)
    vr = policy_validator.validate_plan(plan)
    assert vr.valid


def test_validator_rejects_safety_only_without_safety():
    plan = ActionPlan(primary_action=PrimaryAction.CONTINUE_CHAT,
                      tool_actions=[],
                      safety_target=SafetyTarget.NONE,
                      recommendation_mode=RecommendationMode.SAFETY_ONLY)
    vr = policy_validator.validate_plan(plan)
    assert not vr.valid
    assert "INV-03" in vr.violated


def test_validator_rejects_third_party_wrong_target():
    plan = ActionPlan(primary_action=PrimaryAction.SAFETY_INTERVENTION,
                      tool_actions=[],
                      safety_target=SafetyTarget.NONE,
                      recommendation_mode=RecommendationMode.NONE)
    st = _state("我朋友想自杀", intent=["high_risk_expression"], risk_level=2,
                risk_ctx={"subject": "third_party", "is_third_party_risk": True})
    vr = policy_validator.validate_result(st, PolicyResult(action_plan=plan))
    assert not vr.valid
    assert "INV-10" in vr.violated


def test_validator_risk_requires_safety():
    plan = ActionPlan(primary_action=PrimaryAction.CONTINUE_CHAT, tool_actions=[],
                      safety_target=SafetyTarget.NONE, recommendation_mode=RecommendationMode.NONE)
    st = _state("我想死", intent=["high_risk_expression"], risk_level=3, emotion="绝望")
    vr = policy_validator.validate_result(st, PolicyResult(action_plan=plan))
    assert not vr.valid
    assert "INV-01" in vr.violated


# ===== Hybrid =====

def test_hybrid_safety_precedence():
    st = _state("我想死", intent=["high_risk_expression"], risk_level=3, emotion="绝望")
    r = hybrid_policy.decide(st, use_llm=False)
    assert r.action_plan.primary_action == PrimaryAction.SAFETY_INTERVENTION
    assert r.source == "rule"


def test_hybrid_deterministic_returns_rule():
    st = _state("推荐点书吧", intent=["resource_request"], risk_level=0)
    r = hybrid_policy.decide(st, use_llm=False)
    assert r.source in ("rule", "deterministic_fallback")
    assert r.action_plan.primary_action == PrimaryAction.CONTINUE_CHAT


def test_hybrid_legacy_mode():
    hybrid_policy.config.mode = "legacy"
    st = _state("推荐点书吧", intent=["resource_request"], risk_level=0)
    r = hybrid_policy.decide(st, use_llm=False)
    assert r.source == "legacy_adapter"
    hybrid_policy.config.mode = "hybrid"


# ===== Shadow =====

def test_shadow_classify_diff():
    a = ActionPlan(primary_action=PrimaryAction.CONTINUE_CHAT, tool_actions=[],
                   safety_target=SafetyTarget.NONE, recommendation_mode=RecommendationMode.NONE)
    b = ActionPlan(primary_action=PrimaryAction.CONTINUE_CHAT, tool_actions=[ToolAction.RETRIEVE_MEMORY],
                   safety_target=SafetyTarget.NONE, recommendation_mode=RecommendationMode.NONE)
    c = ActionPlan(primary_action=PrimaryAction.INFORMATION_RESPONSE, tool_actions=[],
                   safety_target=SafetyTarget.NONE, recommendation_mode=RecommendationMode.NONE)
    assert classify_diff(a, a) == "same"
    assert classify_diff(a, b) == "semantic_equivalent"
    assert classify_diff(a, c) == "different"


def test_shadow_runner_trace():
    st = _state("我想死", intent=["high_risk_expression"], risk_level=3, emotion="绝望")
    trace = policy_shadow_runner.run(st)
    assert trace is not None
    assert trace["diff"] in ("same", "different", "semantic_equivalent")
    assert trace["new_action_plan"]["primary_action"] == "safety_intervention"


def test_shadow_runner_never_raises():
    trace = policy_shadow_runner.run(None)
    assert trace is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))


# ===== Phase 5: DynamicRiskState =====

def test_dynamic_state_escalation():
    from risk.dynamic_state import DynamicRiskTracker
    t = DynamicRiskTracker()
    states = [t.update(level=l, confidence=0.9, text="x") for l in [0, 0, 1, 2, 3]]
    assert [s.level for s in states] == [0, 0, 1, 2, 3]
    assert states[-1].is_high_risk
    assert states[-1].escalation  # 3 > 2


def test_dynamic_state_persistence():
    from risk.dynamic_state import DynamicRiskTracker
    t = DynamicRiskTracker()
    states = [t.update(level=l, confidence=0.9, text="x") for l in [2, 2, 2]]
    assert states[-1].persistence == 3


def test_dynamic_state_safe_denial():
    from risk.dynamic_state import DynamicRiskTracker
    t = DynamicRiskTracker()
    s = t.update(level=1, confidence=0.9, text="我没有想死", context={"is_safe_denial": True})
    assert s.safe_denial is True


def test_dynamic_state_third_party():
    from risk.dynamic_state import DynamicRiskTracker
    t = DynamicRiskTracker()
    s = t.update(level=3, confidence=0.9, text="我朋友想自杀",
                 context={"subject": "third_party", "is_third_party_risk": True})
    assert s.subject == "third_party"
    assert s.is_high_risk
