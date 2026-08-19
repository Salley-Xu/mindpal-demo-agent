# -*- coding: utf-8 -*-
"""
Action 指标（Primary / Tool）单元测试（Phase 0.5 Task 0.5.2 验收）。

运行：python evaluation/tests/test_action_metrics.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics.agent_metrics import (  # noqa: E402
    legacy_intent_metrics,
    memory_behavior_metrics,
    primary_action_metrics,
    tool_action_metrics,
)


def test_primary_action_metrics():
    y_true = ["continue_chat", "safety_intervention", "safety_intervention", "information_response"]
    y_pred = ["continue_chat", "safety_intervention", "continue_chat", "information_response"]
    m = primary_action_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.75
    assert m["safety_tp"] == 1
    assert m["safety_fn"] == 1
    assert m["safety_action_recall"] == 0.5
    assert set(m["per_class"].keys()) == {"continue_chat", "ask_clarification", "information_response", "safety_intervention"}


def test_tool_action_metrics_multilabel():
    y_true = [["retrieve_memory"], ["recommend_resource"], ["retrieve_knowledge", "recommend_resource"]]
    y_pred = [["retrieve_memory"], ["recommend_resource"], ["retrieve_knowledge"]]
    m = tool_action_metrics(y_true, y_pred)
    # retrieve_knowledge: 1 of 3 cases, exact 1 tp / 0 fp / 0 fn (case3 has it, predicted has it)
    assert m["per_tool"]["retrieve_memory"]["tp"] == 1
    assert m["per_tool"]["recommend_resource"]["tp"] == 1
    assert abs(m["exact_match"] - 2 / 3) < 1e-3


def test_tool_action_exact_match():
    y_true = [["retrieve_memory", "recommend_resource"], []]
    y_pred = [["recommend_resource", "retrieve_memory"], []]
    m = tool_action_metrics(y_true, y_pred)
    assert m["exact_match"] == 1.0  # 顺序无关


def test_legacy_intent_coverage():
    labels = ["casual_chat", "emotional_expression", "explicit_help_request", "feedback", "information_request"]
    y_true = [["casual_chat"], ["emotional_expression"], ["explicit_help_request"], ["feedback"], ["information_request"]]
    # 当前系统规则只能预测前 3 类
    y_pred = [["casual_chat"], ["emotional_expression"], ["explicit_help_request"], ["casual_chat"], ["casual_chat"]]
    m = legacy_intent_metrics(y_true, y_pred, labels)
    assert m["legacy_intent_micro_f1"] > 0
    assert m["legacy_intent_macro_f1"] > 0
    assert m["zero_recall_labels"] == ["feedback", "information_request"]
    assert m["covered_labels"] == ["casual_chat", "emotional_expression", "explicit_help_request"]


def test_memory_behavior_metrics():
    # 系统总是检索 → 误报多
    y_true = [True, True, False, False, True]
    y_pred = [True, True, True, True, True]
    m = memory_behavior_metrics(y_true, y_pred)
    assert m["over_retrieval_rate"] == 0.4  # 2/5 检索是不需要的
    assert m["miss_retrieval_rate"] == 0.0  # 需要检索的都检索了


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
