# -*- coding: utf-8 -*-
"""
State 一致性指标（Phase 2 Task 2.9 §34）。

Field Accuracy / Transition Accuracy / Persistence Accuracy / Reset Accuracy /
Session Isolation Pass Rate / Legacy Consistency Rate
"""
from __future__ import annotations

from typing import Dict, List


def _get_path(state: dict, dotted_path: str):
    """按点路径取 state 字段，如 'risk.level' / 'intent.labels'。"""
    cur = state
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def field_accuracy(state: dict, expected: Dict[str, object]) -> Dict:
    """单个 turn 的字段准确率。"""
    total = len(expected)
    correct = 0
    detail = {}
    for path, exp_val in expected.items():
        got = _get_path(state, path)
        ok = _values_equal(got, exp_val)
        detail[path] = {"expected": exp_val, "got": got, "match": ok}
        if ok:
            correct += 1
    return {"accuracy": round(correct / total, 4) if total else 1.0,
            "correct": correct, "total": total, "detail": detail}


def transition_accuracy(turn_results: List[Dict]) -> Dict:
    """Transition Accuracy：每轮字段平均准确率。"""
    if not turn_results:
        return {"accuracy": 0.0, "n": 0}
    accs = [r["accuracy"] for r in turn_results]
    return {"accuracy": round(sum(accs) / len(accs), 4), "n": len(accs)}


def _values_equal(got, exp) -> bool:
    if isinstance(exp, list):
        return got == exp or sorted(got or []) == sorted(exp)
    return got == exp
