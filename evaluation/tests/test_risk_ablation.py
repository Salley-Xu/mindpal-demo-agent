# -*- coding: utf-8 -*-
"""
Risk Ablation 单元测试（Phase 0.5 Task 0.5.3 验收）。
只测确定性逻辑（规则 / 上下文），不加载 BERT 模型。

运行：python evaluation/tests/test_risk_ablation.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.predictors.risk_ablation import (  # noqa: E402
    HIGH_RISK_KEYWORDS_L2,
    HIGH_RISK_KEYWORDS_L3,
    RiskAblationPredictor,
)


def _predictor():
    # 不加载模型（懒加载），只测确定性逻辑
    return RiskAblationPredictor(model_path="dummy")


def test_rule_only_levels():
    p = _predictor()
    assert p.rule_only_level("我真的想死了") == "level_3"
    assert p.rule_only_level("我撑不住了，好累") == "level_2"
    assert p.rule_only_level("最近很焦虑，睡不好") == "level_1"
    assert p.rule_only_level("今天天气不错") == "level_0"


def test_rule_match_high_risk():
    p = _predictor()
    assert p._match_rule("我正在割腕") is True
    assert p._match_rule("今天天气不错") is False


def test_rule_override_level():
    p = _predictor()
    raw_hit = {"rule_matched": True, "raw_label": 0}
    raw_clean = {"rule_matched": False, "raw_label": 3}
    assert p.rule_override_level(raw_hit) == "level_3"
    assert p.rule_override_level(raw_clean) == "level_3"  # raw label 3


def test_context_discussion_floor_0():
    p = _predictor()
    # 讨论语境 + 非规则命中 → level_0
    raw = {"rule_matched": False, "raw_label": 2}
    assert p.context_level("这部电影里主角最后自杀了吗？", raw, "level_2") == "level_0"


def test_context_third_party_floor_0():
    p = _predictor()
    raw = {"rule_matched": False, "raw_label": 3}
    # 第三方非求助 → level_0
    assert p.context_level("我朋友说他看过一部讲自杀的电影。", raw, "level_3") == "level_0"


def test_context_safe_denial_degrade():
    p = _predictor()
    raw = {"rule_matched": False, "raw_label": 1}
    # 安全否认：当前 level_2 且不低于 utterance(1) → 降一级到 level_1
    assert p.context_level("我没有想自杀的想法，只是最近压力大。", raw, "level_2") == "level_1"


def test_variants_structure():
    """一条假 case 的 variants 结构应包含全部 5 个变体。"""
    from evaluation.benchmark_schema import BenchmarkCase
    case = BenchmarkCase(case_id="ra_test", conversation=[{"role": "user", "content": "最近好累"}],
                         expected={"risk_level": 0})
    p = _predictor()
    # 只验证结构构造（mock 掉 raw_bert/full_pipeline 的模型调用）
    p.raw_bert = lambda text: {"raw_label": 0, "raw_confidence": 0.5, "probs_4": [0.9, 0.05, 0.03, 0.02],
                               "binary_prob": 0.1, "rule_matched": False}
    p.full_pipeline_level = lambda text, summary: {"level": "level_0", "risk_trend": "new",
                                                   "risk_context": {}, "session_aggregation": {}}
    rec = p.analyze(case)
    assert set(rec["variants"].keys()) == {"raw_bert", "rule_override", "context_rules", "full_pipeline", "rule_only"}
    assert rec["expected"] == 0
    assert "raw_bert" in rec and "context" in rec and "aggregator" in rec


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
