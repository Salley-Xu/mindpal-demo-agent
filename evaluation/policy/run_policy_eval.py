# -*- coding: utf-8 -*-
"""
Policy Evaluation（Phase 3 Preflight P3-0.4 + Task 3.11/3.12）。

流程：Benchmark Case
    → ModulePredictor（predicted 信号）或 Gold（oracle 信号）
    → AgentState
    → Policy（legacy | deterministic | hybrid）
    → ActionPlan
    → 对比 Benchmark Gold。

用法：
  cd backend && PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe ../evaluation/policy/run_policy_eval.py \
      --policy deterministic --state predicted
  # oracle-state（Policy 上限，区分 perception error 与 policy error）
  ... --state oracle --policy hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(BACKEND_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from evaluation.benchmark_schema import BenchmarkCase  # noqa: E402
from evaluation.predictors.module import CANON_TO_CHINESE, ModulePredictor  # noqa: E402
from policy.hybrid import hybrid_policy  # noqa: E402
from policy.legacy_policy_adapter import legacy_policy_adapter  # noqa: E402
from policy.schema import PrimaryAction, RecommendationMode, SafetyTarget, ToolAction  # noqa: E402
from state.builder import build_agent_state  # noqa: E402

PRIMARY_LABELS = ["continue_chat", "ask_clarification", "information_response", "safety_intervention"]
TOOL_LABELS = ["retrieve_memory", "retrieve_knowledge", "recommend_resource"]
SAFETY_LABELS = ["none", "self", "third_party"]
REC_LABELS = ["none", "soft", "hard", "safety_only"]


# ---------------------------------------------------------------------------
# State 构造
# ---------------------------------------------------------------------------

def _subject_enrichment(text: str) -> dict:
    """用生产 risk_evaluator._analyze_context 提取主体信号（subject/third_party/discussion）。"""
    try:
        from risk_evaluator import risk_evaluator
        ctx = risk_evaluator._analyze_context(text)
        return {
            "subject": ctx.get("subject", "self"),
            "is_discussion_context": ctx.get("is_discussion_context", False),
            "is_third_party_risk": ctx.get("is_third_party_risk", False),
            "is_help_request": ctx.get("is_help_request", False),
            "is_safe_denial": ctx.get("is_safe_denial", False),
        }
    except Exception:
        return {}


def build_state_from_prediction(case, pred):
    """用 ModulePredictor 输出 + subject 增强构造 AgentState（predicted 路径）。"""
    risk_map = {"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}
    text = pred.extra.get("_text", "")
    ctx = _subject_enrichment(text)
    risk_ctx = dict(pred.extra.get("risk_context", {}) or {})
    risk_ctx.update(ctx)
    return build_agent_state(
        user_id="u_eval", session_id=case.case_id, request_text=text,
        turn_index=1,
        intent_result={"labels": pred.intent, "confidence": pred.intent_confidence,
                       "is_open_set": False, "source": "module"},
        emotion_state={"current_emotion": CANON_TO_CHINESE.get(pred.emotion, pred.emotion),
                       "emotion_intensity": pred.emotion_intensity,
                       "negative_trend": pred.emotion in
                       {"anxiety", "sadness", "anger", "stress", "fatigue", "panic", "hopelessness"}},
        urgent_issue={"level": f"level_{pred.risk_level}",
                      "risk_trend": pred.risk_trend,
                      "risk_context": risk_ctx},
        conversation_summary={"conversation_stage": "initial", "turn_count": 1},
        user_profile={},
    )


def build_oracle_state(case):
    """用 Benchmark Gold 信号构造 AgentState（oracle 路径，Policy 上限）。"""
    exp = case.expected
    text = [t.content for t in case.conversation if t.role.value == "user"][-1]
    # 从 gold 派生 risk_context
    risk_ctx = {}
    if exp.safety_target.value == "third_party":
        risk_ctx = {"subject": "third_party", "is_third_party_risk": True, "is_help_request": True}
    elif exp.primary_action.value == "safety_intervention":
        risk_ctx = {"subject": "self"}
    return build_agent_state(
        user_id="u_eval", session_id=case.case_id, request_text=text,
        turn_index=1,
        intent_result={"labels": [i.value for i in exp.intent], "confidence": 1.0,
                       "is_open_set": False, "source": "oracle"},
        emotion_state={"current_emotion": CANON_TO_CHINESE.get(exp.emotion, "中性"),
                       "emotion_intensity": exp.emotion_intensity,
                       "negative_trend": exp.emotion in
                       {"anxiety", "sadness", "anger", "stress", "fatigue", "panic", "hopelessness"}},
        urgent_issue={"level": f"level_{exp.risk_level.value}",
                      "risk_trend": exp.risk_trend.value,
                      "risk_context": risk_ctx},
        conversation_summary={"conversation_stage": "initial", "turn_count": 1},
        user_profile={},
    )


def _run_rule_stack(state):
    """完整规则栈：P0 Safety → P1 Deterministic（对齐 §20 'C Deterministic' 定义）。"""
    from policy.safety import safety_policy
    safety_result = safety_policy.decide(state)
    if safety_result is not None:
        return safety_result.action_plan
    from policy.deterministic import deterministic_policy
    return deterministic_policy.decide(state).action_plan


def get_policy(name):
    if name == "legacy":
        return ("legacy", lambda s: legacy_policy_adapter.decide(s).action_plan)
    if name == "deterministic":
        return ("deterministic", _run_rule_stack)
    if name == "hybrid":
        def _h(s):
            hybrid_policy.config.mode = "hybrid"
            return hybrid_policy.decide(s, use_llm=False).action_plan
        return ("hybrid", _h)
    raise ValueError(f"unknown policy: {name}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(args):
    cases = [BenchmarkCase.model_validate(json.loads(l))
             for l in open(PROJECT_ROOT / "evaluation/datasets/agent_benchmark_v1_1.jsonl", encoding="utf-8")
             if l.strip()]
    predictor = ModulePredictor(emotion_channel=args.emotion, risk_channel=args.risk)
    policy_name, policy_fn = get_policy(args.policy)

    rows = []
    t0 = time.time()
    for c in cases:
        state = (build_oracle_state(c) if args.state == "oracle"
                 else build_state_from_prediction(c, predictor.predict(c)))
        ap = policy_fn(state)
        exp = c.expected
        rows.append({
            "case_id": c.case_id,
            "state": args.state,
            "expected": {"primary_action": exp.primary_action.value,
                         "tool_actions": sorted(t.value for t in exp.tool_actions),
                         "safety_target": exp.safety_target.value,
                         "recommendation_mode": exp.recommendation_action.value},
            "predicted": {"primary_action": ap.primary_action.value,
                          "tool_actions": sorted(t.value for t in ap.tool_actions),
                          "safety_target": ap.safety_target.value,
                          "recommendation_mode": ap.recommendation_mode.value},
            "tags": c.tags,
            "text": [t.content for t in c.conversation if t.role.value == "user"][-1][:60],
        })
    elapsed = time.time() - t0

    metrics = compute_metrics(rows)
    metrics["avg_decision_latency_ms"] = round(elapsed / len(rows) * 1000, 2)
    metrics["llm_fallback_rate"] = 0.0  # 本 runner 不使用 LLM

    reports_dir = PROJECT_ROOT / "evaluation" / "policy" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"policy_{policy_name}_{args.state}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "policy": policy_name, "state": args.state,
                            "emotion": args.emotion, "risk": args.risk, "n": len(rows)},
                   "metrics": metrics, "rows": rows}, f, ensure_ascii=False, indent=2)

    print(f"[OK] policy={policy_name} state={args.state} -> {path.name}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return path, metrics, rows


def compute_metrics(rows):
    from evaluation.metrics.agent_metrics import classification_metrics, tool_action_metrics
    pa_t = [r["expected"]["primary_action"] for r in rows]
    pa_p = [r["predicted"]["primary_action"] for r in rows]
    st_t = [r["expected"]["safety_target"] for r in rows]
    st_p = [r["predicted"]["safety_target"] for r in rows]
    rm_t = [r["expected"]["recommendation_mode"] for r in rows]
    rm_p = [r["predicted"]["recommendation_mode"] for r in rows]
    ta_t = [r["expected"]["tool_actions"] for r in rows]
    ta_p = [r["predicted"]["tool_actions"] for r in rows]

    pa = classification_metrics(pa_t, pa_p, PRIMARY_LABELS)
    st = classification_metrics(st_t, st_p, SAFETY_LABELS)
    rm = classification_metrics(rm_t, rm_p, REC_LABELS)
    ta = tool_action_metrics(ta_t, ta_p)
    exact = sum(1 for r in rows
                if r["expected"] == r["predicted"]) / len(rows)

    # Safety Action Recall
    s_tp = sum(1 for r in rows if r["expected"]["primary_action"] == "safety_intervention"
               and r["predicted"]["primary_action"] == "safety_intervention")
    s_total = sum(1 for r in rows if r["expected"]["primary_action"] == "safety_intervention")
    # Safety Target Accuracy（仅 safety case）
    st_acc = sum(1 for r in rows
                 if r["expected"]["safety_target"] == r["predicted"]["safety_target"]) / len(rows)
    return {
        "primary_action_accuracy": pa["accuracy"],
        "primary_action_macro_f1": pa["macro_f1"],
        "safety_action_recall": round(s_tp / s_total, 4) if s_total else 0.0,
        "safety_target_accuracy": round(st_acc, 4),
        "tool_action_micro_f1": ta["micro_f1"],
        "tool_action_macro_f1": ta["macro_f1"],
        "tool_action_exact_match": ta["exact_match"],
        "recommendation_mode_accuracy": rm["accuracy"],
        "policy_exact_match": round(exact, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", default="rule")
    parser.add_argument("--risk", default="rule")
    parser.add_argument("--policy", default="deterministic",
                        choices=["legacy", "deterministic", "hybrid"])
    parser.add_argument("--state", default="predicted", choices=["predicted", "oracle"])
    run(parser.parse_args())
