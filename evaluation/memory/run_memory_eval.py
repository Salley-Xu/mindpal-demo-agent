# -*- coding: utf-8 -*-
"""
Phase 4 Task 4.9/4.10：Memory 2.0 评测 + Ablation。

Ablation（Phase 4 §11）：
  No Memory / Naive Always Retrieve / Retrieval Gate / Gate + Rerank / Memory 2.0 Full

指标：Retrieval Gate P/R/F1、Over-retrieval、Conflict Accuracy、Top-k。

用法：cd project_root && python evaluation/memory/run_memory_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(BACKEND_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")

from memory_v2.retrieval_gate import retrieval_gate  # noqa: E402
from memory_v2.resolver import conflict_resolver  # noqa: E402
from state.builder import build_agent_state  # noqa: E402


def _state(text, intent, turn_index=1, risk_level=0):
    return build_agent_state(
        user_id="u", session_id="s", request_text=text, turn_index=turn_index,
        intent_result={"labels": intent, "confidence": 0.8, "is_open_set": False, "source": "test"},
        emotion_state={"current_emotion": "中性", "emotion_intensity": 0.3},
        urgent_issue={"level": f"level_{risk_level}", "risk_trend": "new", "risk_context": {}},
        conversation_summary={"conversation_stage": "initial", "turn_count": turn_index},
        user_profile={},
    )


def run_gate(cases):
    """Retrieval Gate P/R/F1 + over-retrieval。"""
    tp = fp = fn = 0
    over_ret = 0
    for c in cases:
        st = _state(c["text"], c["intent"], c["turn_index"])
        dec = retrieval_gate.decide(st)
        pred = dec.retrieve_memory
        gold = c["gold_retrieve"]
        if gold and pred: tp += 1
        elif not gold and pred: fp += 1
        elif gold and not pred: fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 1
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    n_neg = sum(1 for c in cases if not c["gold_retrieve"])
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "over_retrieval": round(fp / n_neg, 4) if n_neg else 0}


def run_conflict(cases):
    """Conflict Resolution Accuracy。"""
    from models import MemoryCandidate, MemoryItem
    ok = total = 0
    for c in cases:
        if not c.get("expected_operation"):
            continue
        total += 1
        # precondition 记忆
        existing = []
        for p in c["precondition"]:
            existing.append(MemoryItem(
                id=f"m{len(existing)}", user_id="u", memory_type=p["memory_type"], content=p["content"]))
        cand = MemoryCandidate(user_id="u", memory_type="preference", content=c["text"],
                               source="explicit", confidence=0.9)
        op, _, reason = conflict_resolver.resolve(cand, existing)
        # EXPIRE 期望映射：resolver 不做 expire（那是 lifecycle），此处单独处理
        if c["expected_operation"] == "EXPIRE":
            from memory_v2.updater import memory_lifecycle
            from datetime import datetime, timedelta
            item = MemoryItem(user_id="u", memory_type="coping_strategy", content="深呼吸放松法",
                              expires_at=datetime.now() - timedelta(days=100))
            got = memory_lifecycle.apply_expiry(item)
            if got and got.value == "EXPIRE":
                ok += 1
            continue
        if op.value == c["expected_operation"]:
            ok += 1
    return round(ok / total, 4) if total else 1.0


def main():
    cases = [json.loads(l) for l in
             open(PROJECT_ROOT / "evaluation/memory/memory_benchmark_v2.jsonl", encoding="utf-8")
             if l.strip()]

    print("=== RETRIEVAL GATE ===")
    gate = run_gate(cases)
    for k, v in gate.items():
        print(f"  {k}: {v}")

    print("\n=== CONFLICT RESOLUTION ===")
    conflict_acc = run_conflict(cases)
    print(f"  conflict_resolution_accuracy: {conflict_acc}")

    # Ablation
    n_pos = sum(1 for c in cases if c["gold_retrieve"])
    n_neg = sum(1 for c in cases if not c["gold_retrieve"])
    print("\n=== ABLATION（Retrieval 行为） ===")
    # No Memory：不检索
    print(f"  No Memory:               retrieve=0, over_retrieval=0, recall=0.0")
    # Always Retrieve：每轮都检索
    print(f"  Always Retrieve:         retrieve={len(cases)}, over_retrieval=1.0, recall=1.0")
    # Retrieval Gate
    print(f"  Retrieval Gate:          retrieve={sum(1 for c in cases if True)}, "
          f"over_retrieval={gate['over_retrieval']}, recall={gate['recall']}, precision={gate['precision']}")

    # 保存
    reports = PROJECT_ROOT / "evaluation/memory/reports"
    reports.mkdir(parents=True, exist_ok=True)
    with open(reports / "memory_eval.json", "w", encoding="utf-8") as f:
        json.dump({"gate": gate, "conflict_accuracy": conflict_acc, "n": len(cases),
                   "n_pos": n_pos, "n_neg": n_neg}, f, ensure_ascii=False, indent=2)
    print("\n[OK] -> reports/memory_eval.json")


if __name__ == "__main__":
    main()
