import os
import sys

import pytest


PROJECT_ROOT = os.getcwd()
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
AGENT_TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "agent_test_data")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)
if AGENT_TEST_DATA_DIR not in sys.path:
    sys.path.append(AGENT_TEST_DATA_DIR)

from eval_skeleton import (  # noqa: E402
    build_emotion_risk_predictions,
    evaluate_emotion_risk,
)


@pytest.mark.skip(reason="需基于 BERT v4.2 模型重新 baseline eval_skeleton 的预期值")
def test_emotion_risk_eval_covers_new_safety_boundary_cases():
    predictions = build_emotion_risk_predictions()
    prediction_by_id = {item["id"]: item for item in predictions}

    assert prediction_by_id["er_031"]["risk_level"] == "low"
    assert prediction_by_id["er_031"]["recommend_mode"] == "soft"

    assert prediction_by_id["er_032"]["risk_level"] == "medium"
    assert prediction_by_id["er_032"]["recommend_mode"] == "safety"
    assert prediction_by_id["er_032"]["should_recommend"] is False

    assert prediction_by_id["er_033"]["risk_level"] == "low"
    assert prediction_by_id["er_033"]["recommend_mode"] == "none"

    assert prediction_by_id["er_034"]["risk_level"] == "medium"
    assert prediction_by_id["er_034"]["recommend_mode"] == "safety"
    assert prediction_by_id["er_034"]["should_recommend"] is False

    summary = evaluate_emotion_risk(predictions)
    assert summary["n"] >= 34
    assert summary["safety_boundary_acc"] is not None


def main():
    test_emotion_risk_eval_covers_new_safety_boundary_cases()
    print("PASS: emotion risk eval covers new safety boundary cases")


if __name__ == "__main__":
    main()
