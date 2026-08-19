# -*- coding: utf-8 -*-
"""
Phase 1 Intent 单元测试（§42）。

覆盖：schema / threshold / calibration / open_set / metrics / legacy rule / hybrid 门控逻辑。
只测确定性逻辑，不加载 BERT 模型、不调用 LLM API。

运行：python evaluation/tests/test_intent.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from data.intent.intent_schema import INTENT_LABELS, IntentData  # noqa: E402
from evaluation.intent.calibration import (  # noqa: E402
    apply_temperature, brier_score, ece_score, fit_temperature,
    optimize_per_label_thresholds,
)
from evaluation.intent.metrics import intent_metrics  # noqa: E402
from evaluation.intent.open_set import MaxScoreOpenSet, ood_metrics  # noqa: E402
from evaluation.intent.predictors import (  # noqa: E402
    LegacyRulePredictor, RuleClassifierPredictor,
)


def test_intent_schema():
    d = IntentData(id="t1", text="我很焦虑，怎么办？",
                   labels=["emotional_expression", "explicit_help_request"])
    assert len(d.labels) == 2
    assert d.is_ood is False


def test_legacy_rule_predictor():
    p = LegacyRulePredictor()
    assert p.predict("我该怎么办？")["labels"] == ["explicit_help_request"]
    assert p.predict("今天天气不错")["labels"] == ["casual_chat"]
    assert p.predict("最近很焦虑")["labels"] == ["emotional_expression"]


def test_rule_classifier():
    p = RuleClassifierPredictor()
    r = p.predict("推荐几个放松的练习")
    assert "resource_request" in r["labels"]
    # 高危优先
    r2 = p.predict("我不想活了，有什么办法？")
    assert "high_risk_expression" in r2["labels"]
    assert "explicit_help_request" in r2["labels"]
    # 有任务意图时不标 casual
    r3 = p.predict("推荐点放松的音频")
    assert "casual_chat" not in r3["labels"]


def test_intent_metrics():
    y_true = [["a"], ["a", "b"], ["b"]]
    y_pred = [["a"], ["a"], ["b"]]
    m = intent_metrics(y_true, y_pred, ["a", "b"])
    assert m["macro_f1"] > 0 and m["micro_f1"] > 0
    assert 0 <= m["exact_match"] <= 1
    assert 0 <= m["hamming_loss"] <= 1


def test_per_label_threshold():
    probs = np.array([[0.9, 0.1], [0.3, 0.8], [0.2, 0.2]])
    labels = np.array([[1, 0], [0, 1], [0, 0]])
    thr = optimize_per_label_thresholds(probs, labels)
    assert len(thr) == 2
    assert all(0 <= t <= 1 for t in thr)


def test_temperature_scaling_improves_calibration():
    # 构造 over-confident 但部分错误的分值
    logits = np.array([[3.0, -3.0], [3.0, -3.0], [-3.0, 3.0]])
    labels = np.array([[1, 0], [0, 1], [0, 1]])
    T = fit_temperature(logits, labels)
    assert 0.1 < T < 10.0
    raw = 1 / (1 + np.exp(-logits))
    cal = apply_temperature(logits, T)
    assert ece_score(cal.tolist(), labels.tolist()) <= ece_score(raw.tolist(), labels.tolist()) + 1e-6
    assert brier_score(cal, labels) <= brier_score(raw, labels) + 1e-6


def test_open_set_decision():
    os_ = MaxScoreOpenSet(threshold=0.35)
    assert os_.is_open_set({"a": 0.1}) is True       # 低分 → open
    assert os_.is_open_set({"a": 0.9}) is False      # 高分 → in-domain
    assert os_.is_open_set({}) is True               # 空 → open


def test_ood_metrics_direction():
    # 域内分高、OOD 分低 → AUROC 高
    in_s = [0.8, 0.75, 0.7, 0.85]
    ood_s = [0.2, 0.15, 0.3, 0.25]
    m = ood_metrics(in_s, ood_s, threshold=0.5)
    assert m["auroc"] > 0.9
    assert m["ood_recall"] == 1.0
    assert m["in_domain_false_reject"] == 0.0


def test_hybrid_gate_logic():
    """Hybrid 门控：低置信/空分 → 触发 fallback；高置信 → 直接 classifier。"""
    from evaluation.intent.hybrid import HybridIntentPredictor

    class FakeClassifier:
        def predict(self, text, context=None):
            if text == "high":
                return {"labels": ["casual_chat"], "confidence": 0.9,
                        "is_open_set": False, "source": "classifier",
                        "label_scores": {"casual_chat": 0.9}}
            return {"labels": [], "confidence": 0.1, "is_open_set": True,
                    "source": "classifier", "label_scores": {}}

    class FakeLLM:
        def predict(self, text, context=None):
            return {"labels": ["information_request"], "confidence": 0.9,
                    "is_open_set": False, "source": "llm", "label_scores": {}}

    h = HybridIntentPredictor(classifier=FakeClassifier(), open_set=MaxScoreOpenSet(0.35),
                              enable_llm_fallback=True, llm=FakeLLM())
    assert h.predict("high")["source"] == "classifier"      # 高置信不走 LLM
    assert h.predict("low")["source"] == "llm"              # 空分触发 fallback


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
