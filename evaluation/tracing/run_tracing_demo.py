# -*- coding: utf-8 -*-
"""
Phase 7 Task 7.2-7.9：Trace / Replay / Attribution 演示 + 验证。

用法：cd project_root && python evaluation/tracing/run_tracing_demo.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(BACKEND_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from tracing.attribution import error_attributor  # noqa: E402
from tracing.logger import trace_logger  # noqa: E402
from tracing.replay import replay_runner  # noqa: E402


def build_full_turn(trace_id, text, labels, risk_level, emotion, turn_id=1):
    """构造一个完整 traced turn。"""
    t0 = time.time()
    trace = trace_logger.new_trace(trace_id, "s_demo", turn_id)
    trace.input = {"text": text, "text_hash": hash(text) % 10 ** 8}
    trace.perception = {
        "intent_result": {"labels": labels, "confidence": 0.85, "is_open_set": False, "source": "module"},
        "emotion_state": {"current_emotion": emotion, "emotion_intensity": 0.7},
        "urgent_issue": {"level": f"level_{risk_level}", "risk_trend": "new", "risk_context": {}},
        "conversation_summary": {"conversation_stage": "initial", "turn_count": turn_id},
        "user_profile": {},
    }
    # state 摘要（AgentState.to_debug_dict 风格的简化）
    trace.state = {"risk": {"level": risk_level, "trend": "new"},
                   "intent": {"labels": labels},
                   "emotion": {"current_emotion": emotion},
                   "derived": {"high_risk_intent_signal": "high_risk_expression" in labels}}
    trace.latency["perception"] = 80.0
    return trace, t0


def main():
    print("=== PHASE 7 TRACING DEMO ===")

    # 1. 正常 turn（risk 高 → policy 应 safety）
    trace1, t0 = build_full_turn("tr_demo_001", "我想死", ["high_risk_expression"], 3, "绝望")
    from policy.hybrid import hybrid_policy
    from state.builder import build_agent_state
    state1 = build_agent_state(
        user_id="u", session_id="s_demo", request_text="我想死", turn_index=1,
        intent_result={"labels": ["high_risk_expression"], "confidence": 0.85, "is_open_set": False, "source": "module"},
        emotion_state={"current_emotion": "绝望", "emotion_intensity": 0.7},
        urgent_issue={"level": "level_3", "risk_trend": "new", "risk_context": {}},
        conversation_summary={"conversation_stage": "initial", "turn_count": 1}, user_profile={})
    r = hybrid_policy.decide(state1, use_llm=False)
    trace1.policy = {"action_plan": r.action_plan.model_dump(), "source": r.source,
                     "matched_rules": r.matched_rules, "confidence": r.confidence}
    trace1.recommendation = {"mode": r.action_plan.recommendation_mode.value}
    trace1.latency["policy"] = 0.5
    trace1.latency["total"] = round((time.time() - t0) * 1000, 1)
    trace_logger.write(trace1)
    print(f"[1] trace {trace1.trace_id} written (safety decision)")

    # 2. 误差 turn（risk 高但 policy 说 continue → 归因 policy）
    trace2, _ = build_full_turn("tr_demo_002", "我撑不下去了", ["emotional_expression"], 2, "绝望")
    trace2.policy = {"action_plan": {"primary_action": "continue_chat", "tool_actions": [],
                                     "safety_target": "none", "recommendation_mode": "none"},
                     "source": "rule", "matched_rules": ["D04"]}
    trace2.state = {"risk": {"level": 2, "trend": "new"}}
    trace2.recommendation = {"mode": "none"}
    trace2.latency = {"perception": 80.0, "policy": 0.5, "total": 82.0}
    trace_logger.write(trace2)

    attr = error_attributor.attribute(trace2)
    print(f"[2] error attribution: {attr.first_error_layer} ({attr.reason})")

    # 3. 正常 turn（无风险）→ 归因 none
    trace3, _ = build_full_turn("tr_demo_003", "今天天气不错", ["casual_chat"], 0, "中性")
    trace3.policy = {"action_plan": {"primary_action": "continue_chat", "tool_actions": [],
                                     "safety_target": "none", "recommendation_mode": "none"},
                     "source": "rule", "matched_rules": ["D04"]}
    trace3.state = {"risk": {"level": 0, "trend": "new"}}
    trace3.recommendation = {"mode": "none"}
    trace3.latency = {"perception": 80.0, "policy": 0.5, "total": 81.0}
    trace_logger.write(trace3)
    attr3 = error_attributor.attribute(trace3)
    print(f"[3] normal turn attribution: {attr3.first_error_layer}")

    # 4. Replay
    rt = trace_logger.read("tr_demo_001")
    assert rt is not None, "replay: trace 读取失败"
    replay = replay_runner.replay_policy(rt)
    print(f"[4] replay policy: {replay['action_plan']['primary_action']} source={replay['source']}")
    assert replay["action_plan"]["primary_action"] == "safety_intervention"

    # 5. Counterfactual replay（legacy）
    cmp = replay_runner.compare_versions(trace2)
    print(f"[5] counterfactual legacy: {cmp['counterfactual_legacy']['action_plan']['primary_action']}")

    # 6. 目标验证
    traces = trace_logger.list_traces()
    coverage = len(traces) >= 3
    print(f"\n=== TARGETS ===")
    print(f"  Trace coverage: {len(traces)} traces logged → {'PASS' if coverage else 'FAIL'}")
    print(f"  Replay success: {'PASS' if rt is not None else 'FAIL'}")
    print(f"  Version provenance: {'PASS' if 'risk' in rt.versions and rt.versions['risk'] == 'v5_1' else 'FAIL'}")
    print(f"  Error attribution: {'PASS' if attr.first_error_layer == 'policy' else 'FAIL'}")


if __name__ == "__main__":
    main()
