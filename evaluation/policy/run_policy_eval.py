# -*- coding: utf-8 -*-
"""
Policy Evaluation（Phase 3 Preflight P3-0.4 + Task 3.11）。

流程：Benchmark Case → ModulePredictor（当前系统信号）→ AgentState → Policy → ActionPlan
     → 对比 Benchmark Gold。

用法：
  cd backend && PYTHONIOENCODING=utf-8 /d/anaconda3/python.exe ../evaluation/policy/run_policy_eval.py --risk rule
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
from evaluation.predictors.module import ModulePredictor, CANON_TO_CHINESE  # noqa: E402
from policy.legacy_policy_adapter import legacy_policy_adapter  # noqa: E402
from policy.schema import PrimaryAction, RecommendationMode, SafetyTarget, ToolAction  # noqa: E402
from state.builder import build_agent_state  # noqa: E402

PRIMARY_LABELS = ["continue_chat", "ask_clarification", "information_response", "safety_intervention"]
TOOL_LABELS = ["retrieve_memory", "retrieve_knowledge", "recommend_resource"]
SAFETY_LABELS = ["none", "self", "third_party"]
REC_LABELS = ["none", "soft", "hard", "safety_only"]


def build_state_from_prediction(case, pred):
    """用 ModulePredictor 输出构造 AgentState。"""
    risk_map = {"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}
    return build_agent_state(
        user_id="u_eval", session_id=case.case_id, request_text=pred.extra.get("_text", ""),
        turn_index=1,
        intent_result={"labels": pred.intent, "confidence": pred.intent_confidence,
                       "is_open_set": False, "source": "module"},
        emotion_state={"current_emotion": CANON_TO_CHINESE.get(pred.emotion, pred.emotion),
                       "emotion_intensity": pred.emotion_intensity,
                       "negative_trend": pred.emotion in
                       {"anxiety", "sadness", "anger", "stress", "fatigue", "panic", "hopelessness"}},
        urgent_issue={"level": f"level_{pred.risk_level}",
                      "risk_trend": pred.risk_trend,
                      "risk_context": pred.extra.get("risk_context", {})},
        conversation_summary={"conversation_stage": "initial", "turn_count": 1},
        user_profile={},
    )


def run(args):
    cases = [BenchmarkCase.model_validate(json.loads(l))
             for l in open(PROJECT_ROOT / "evaluation/datasets/agent_benchmark_v1_1.jsonl", encoding="utf-8")
             if l.strip()]
    predictor = ModulePredictor(emotion_channel=args.emotion, risk_channel=args.risk)

    rows = []
    for c in cases:
        pred = predictor.predict(c)
        state = build_state_from_prediction(c, pred)
        result = legacy_policy_adapter.decide(state)
        ap = result.action_plan
        exp = c.expected
        rows.append({
            "case_id": c.case_id,
            "expected": {"primary_action": exp.primary_action.value,
                         "tool_actions": sorted(t.value for t in exp.tool_actions),
                         "safety_target": exp.safety_target.value,
                         "recommendation_mode": exp.recommendation_action.value},
            "predicted": {"primary_action": ap.primary_action.value,
                          "tool_actions": sorted(t.value for t in ap.tool_actions),
                          "safety_target": ap.safety_target.value,
                          "recommendation_mode": ap.recommendation_mode.value},
            "matched_rules": result.matched_rules,
        })

    # 指标
    metrics = compute_metrics(rows)
    reports_dir = PROJECT_ROOT / "evaluation" / "policy" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"current_policy_baseline_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "emotion": args.emotion, "risk": args.risk, "n": len(rows)},
                   "metrics": metrics, "rows": rows}, f, ensure_ascii=False, indent=2)

    print(f"[OK] policy baseline -> {path.name}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


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
    return {
        "primary_action_accuracy": pa["accuracy"],
        "primary_action_macro_f1": pa["macro_f1"],
        "safety_action_recall": round(s_tp / s_total, 4) if s_total else 0.0,
        "safety_target_accuracy": st["accuracy"],
        "tool_action_micro_f1": ta["micro_f1"],
        "tool_action_exact_match": ta["exact_match"],
        "recommendation_mode_accuracy": rm["accuracy"],
        "policy_exact_match": round(exact, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", default="rule")
    parser.add_argument("--risk", default="rule")
    run(parser.parse_args())
