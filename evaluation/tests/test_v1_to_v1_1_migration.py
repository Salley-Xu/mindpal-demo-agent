# -*- coding: utf-8 -*-
"""
v1 → v1.1 迁移单元测试（Phase 0.5 Task 0.5.1 验收）。

运行：python evaluation/tests/test_v1_to_v1_1_migration.py
"""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark_schema import (  # noqa: E402
    BenchmarkCase,
    migrate_v1_case_to_v1_1,
    migrate_v1_cases_to_v1_1,
)

BENCH_DIR = PROJECT_ROOT / "evaluation" / "datasets"


def _v1_case(case_id, agent_action, rec="none", retrieve_knowledge=False, memory_needed=False):
    return {
        "case_id": case_id,
        "conversation": [{"role": "user", "content": "测试输入"}],
        "expected": {
            "intent": ["casual_chat"],
            "emotion": "neutral",
            "risk_level": 0,
            "memory_needed": memory_needed,
            "recommendation_action": rec,
            "agent_action": agent_action,
            "should_retrieve_knowledge": retrieve_knowledge,
        },
        "tags": ["test"],
        "source": "template",
    }


def test_continue_chat():
    m = migrate_v1_case_to_v1_1(_v1_case("m_001", "continue_chat"))
    e = m["expected"]
    assert e["primary_action"] == "continue_chat"
    assert e["tool_actions"] == []
    assert e["safety_target"] == "none"
    assert "agent_action" not in e
    assert "should_retrieve_knowledge" not in e


def test_retrieve_memory_becomes_tool():
    m = migrate_v1_case_to_v1_1(_v1_case("m_002", "retrieve_memory", memory_needed=True))
    e = m["expected"]
    assert e["primary_action"] == "continue_chat"
    assert e["tool_actions"] == ["retrieve_memory"]


def test_recommend_resource_becomes_tool():
    m = migrate_v1_case_to_v1_1(_v1_case("m_003", "recommend_resource", rec="hard"))
    e = m["expected"]
    assert e["primary_action"] == "continue_chat"
    assert e["tool_actions"] == ["recommend_resource"]


def test_safety_intervention_self_target():
    m = migrate_v1_case_to_v1_1(_v1_case("m_004", "safety_intervention"))
    e = m["expected"]
    assert e["primary_action"] == "safety_intervention"
    assert e["safety_target"] == "self"


def test_third_party_support_migration():
    m = migrate_v1_case_to_v1_1(_v1_case("m_005", "safety_intervention", rec="third_party_support"))
    e = m["expected"]
    assert e["primary_action"] == "safety_intervention"
    assert e["recommendation_action"] == "safety_only"
    assert e["safety_target"] == "third_party"


def test_retrieve_knowledge_merged_into_tools():
    m = migrate_v1_case_to_v1_1(
        _v1_case("m_006", "information_response", retrieve_knowledge=True))
    e = m["expected"]
    assert e["primary_action"] == "information_response"
    assert e["tool_actions"] == ["retrieve_knowledge"]


def test_input_not_mutated():
    original = _v1_case("m_007", "continue_chat")
    migrated = migrate_v1_case_to_v1_1(original)
    assert "agent_action" in original["expected"]  # 原 dict 未变
    assert "agent_action" not in migrated["expected"]


def test_migrate_full_v1_dataset_matches_v1_1_seed():
    v1 = [json.loads(l) for l in open(BENCH_DIR / "agent_benchmark_v1.jsonl", encoding="utf-8") if l.strip()]
    v11 = [json.loads(l) for l in open(BENCH_DIR / "agent_benchmark_v1_1.jsonl", encoding="utf-8") if l.strip()]
    mig = migrate_v1_cases_to_v1_1(v1)
    mig_by_id = {c["case_id"]: c["expected"] for c in mig}
    seed_by_id = {c["case_id"]: c["expected"] for c in v11}
    assert len(mig) == len(v1) == 100
    for c in mig:
        BenchmarkCase.model_validate(c)  # 迁移后的完整 case 必须通过 v1.1 校验
    # 迁移出的 100 条必须与 v1.1 文件中的种子 base 完全一致（v1.1 在其上扩充）
    for cid, mig_exp in mig_by_id.items():
        assert cid in seed_by_id, f"{cid} 不在 v1.1 中"
        assert mig_exp == seed_by_id[cid], f"{cid} 迁移不一致"


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
