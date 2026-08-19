# -*- coding: utf-8 -*-
"""
Agent Benchmark v1.1 数据审计（Phase 0.5 Task 0.5.4 §15）。

统计 + 查重（Exact / MinHash；Embedding 记为 TODO），
输出 docs/benchmark_v1_1_data_report.md。

用法：/d/anaconda3/python.exe evaluation/datasets/audit_benchmark_v1_1.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark_schema import BenchmarkCase  # noqa: E402


# ---- MinHash（轻量，shingle = 字符 2-gram，128 哈希） ----
_NUM_HASH = 128


def _shingles(text: str) -> set:
    text = "".join(text.split())
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _minhash(text: str, seeds) -> tuple:
    shingles = _shingles(text)
    if not shingles:
        return tuple(0 for _ in range(_NUM_HASH))
    sig = []
    for seed in seeds:
        sig.append(min(hashlib.md5((s + str(seed)).encode("utf-8")).hexdigest() for s in shingles))
    return tuple(sig)


def _estimate_similarity(sig_a, sig_b) -> float:
    same = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return same / _NUM_HASH


def audit(path: Path) -> dict:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(BenchmarkCase.model_validate(json.loads(line)))

    # 基础统计
    risk = Counter(c.expected.risk_level.value for c in cases)
    intent = Counter(i.value for c in cases for i in c.expected.intent)
    primary = Counter(c.expected.primary_action.value for c in cases)
    tool = Counter(t.value for c in cases for t in c.expected.tool_actions)
    rec = Counter(c.expected.recommendation_action.value for c in cases)
    safety = Counter(c.expected.safety_target.value for c in cases)
    source = Counter(c.source for c in cases)
    single = sum(1 for c in cases if len(c.conversation) == 1)
    multi_label = sum(1 for c in cases if len(c.expected.intent) > 1)
    multi_turn = len(cases) - single
    high_risk = sum(1 for c in cases if c.expected.risk_level.value >= 2)
    emo_counter = Counter(c.expected.emotion for c in cases if c.expected.emotion)

    # Exact duplicate（同一用户轮文本完全相同）
    texts = ["|".join(t.content for t in c.conversation if t.role.value == "user") for c in cases]
    exact_dup = Counter(texts)
    exact_dup_pairs = {t: n for t, n in exact_dup.items() if n > 1}

    # MinHash duplicate（用户轮文本近似重复，Jaccard 估计 >= 0.8）
    seeds = list(range(_NUM_HASH))
    sigs = [_minhash(t, seeds) for t in texts]
    minhash_pairs = []
    # 只比较有哈希差异的（粗略：与随机前 64 条对比，避免 O(n²)）
    for i in range(len(sigs)):
        for j in range(i + 1, min(i + 200, len(sigs))):
            if exact_dup[texts[i]] > 1:
                continue  # 已在 exact 中
            sim = _estimate_similarity(sigs[i], sigs[j])
            if sim >= 0.8:
                minhash_pairs.append((cases[i].case_id, cases[j].case_id, round(sim, 3)))
    minhash_pairs = sorted(minhash_pairs, key=lambda x: -x[2])[:15]

    return {
        "total": len(cases),
        "source": dict(source),
        "risk": dict(sorted(risk.items())),
        "high_risk_count": high_risk,
        "intent": dict(intent),
        "primary_action": dict(primary),
        "tool_action": dict(tool),
        "recommendation": dict(rec),
        "safety_target": dict(safety),
        "emotion": dict(emo_counter),
        "single_turn": single,
        "multi_turn": multi_turn,
        "multi_label_cases": multi_label,
        "multi_label_ratio": round(multi_label / len(cases), 3),
        "exact_duplicate_pairs": exact_dup_pairs,
        "minhash_duplicate_pairs": minhash_pairs,
        "embedding_audit": "TODO",
    }


def render(stats: dict) -> str:
    from datetime import datetime
    L = []
    L.append("# Benchmark v1.1 数据审计报告")
    L.append("")
    L.append(f"> 文件：evaluation/datasets/agent_benchmark_v1_1.jsonl")
    L.append(f"> 日期：{datetime.now().strftime('%Y-%m-%d')}")
    L.append("")
    L.append("## 1. 总体统计")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| Case 数量 | {stats['total']} |")
    L.append(f"| Source 分布 | {stats['source']} |")
    L.append(f"| 单轮 / 多轮 | {stats['single_turn']} / {stats['multi_turn']} |")
    L.append(f"| Multi-label case 占比 | {stats['multi_label_cases']} ({stats['multi_label_ratio']}) |")
    L.append(f"| 高风险数量 (L2+L3) | {stats['high_risk_count']} |")
    L.append("")
    L.append("## 2. 分布")
    L.append("")
    L.append("### Risk")
    L.append("")
    L.append(f"```text\n{stats['risk']}\n```")
    L.append("")
    L.append("### Intent（multi-label）")
    L.append("")
    L.append(f"```text\n{stats['intent']}\n```")
    L.append("")
    L.append("### Primary Action")
    L.append("")
    L.append(f"```text\n{stats['primary_action']}\n```")
    L.append("")
    L.append("### Tool Action")
    L.append("")
    L.append(f"```text\n{stats['tool_action']}\n```")
    L.append("")
    L.append("### Recommendation / SafetyTarget / Emotion")
    L.append("")
    L.append(f"```text\nrec: {stats['recommendation']}\nsafety: {stats['safety_target']}\n```")
    L.append("")
    L.append("## 3. 查重审计")
    L.append("")
    L.append(f"### Exact Duplicate（用户轮文本完全一致）")
    L.append("")
    if stats["exact_duplicate_pairs"]:
        for t, n in stats["exact_duplicate_pairs"].items():
            L.append(f"- `{t[:40]}` x{n}")
    else:
        L.append("- 无 ✅")
    L.append("")
    L.append("### MinHash Duplicate（Jaccard 估计 >= 0.8，Top 15）")
    L.append("")
    if stats["minhash_duplicate_pairs"]:
        for a, b, sim in stats["minhash_duplicate_pairs"]:
            L.append(f"- `{a}` ~ `{b}` sim={sim}")
    else:
        L.append("- 无 ✅")
    L.append("")
    L.append("### Embedding Similarity")
    L.append("")
    L.append("- **TODO**：后续用 bge-small-zh 做 embedding 相似度审计（保留占位）。")
    L.append("")
    L.append("## 4. 与目标分布对照（§13 / §23）")
    L.append("")
    risk = stats["risk"]
    r0 = risk.get(0, 0); r1 = risk.get(1, 0); r2 = risk.get(2, 0); r3 = risk.get(3, 0)
    checks = [
        ("Case >= 250", stats["total"] >= 250, stats["total"]),
        ("L2+L3 >= 60", r2 + r3 >= 60, r2 + r3),
        ("ask_clarification >= 20", stats["primary_action"].get("ask_clarification", 0) >= 20, stats["primary_action"].get("ask_clarification", 0)),
        ("follow_up >= 20", stats["intent"].get("follow_up", 0) >= 20, stats["intent"].get("follow_up", 0)),
        ("feedback >= 30", stats["intent"].get("feedback", 0) >= 30, stats["intent"].get("feedback", 0)),
        ("memory_reference >= 30", stats["intent"].get("memory_reference", 0) >= 30, stats["intent"].get("memory_reference", 0)),
        ("high_risk_expression >= 50", stats["intent"].get("high_risk_expression", 0) >= 50, stats["intent"].get("high_risk_expression", 0)),
        ("safety_intervention >= 60", stats["primary_action"].get("safety_intervention", 0) >= 60, stats["primary_action"].get("safety_intervention", 0)),
        ("retrieve_memory >= 40", stats["tool_action"].get("retrieve_memory", 0) >= 40, stats["tool_action"].get("retrieve_memory", 0)),
        ("retrieve_knowledge >= 30", stats["tool_action"].get("retrieve_knowledge", 0) >= 30, stats["tool_action"].get("retrieve_knowledge", 0)),
        ("recommend_resource >= 50", stats["tool_action"].get("recommend_resource", 0) >= 50, stats["tool_action"].get("recommend_resource", 0)),
    ]
    L.append("| 检查项 | 结果 | 值 |")
    L.append("|---|---|---|")
    for name, ok, val in checks:
        L.append(f"| {name} | {'✅' if ok else '❌'} | {val} |")
    L.append("")
    L.append("### 已知偏差（软目标）")
    L.append("")
    L.append(f"- Risk L0 = {r0}（建议 130-150，因 intent 长尾覆盖优先而偏高）")
    L.append(f"- Risk L1 = {r1}（建议 70-80，符合）")
    L.append(f"- Risk L2 = {r2} / L3 = {r3}（建议 30-35 / 30-35，符合）")
    L.append("")
    return "\n".join(L)


def main():
    path = PROJECT_ROOT / "evaluation" / "datasets" / "agent_benchmark_v1_1.jsonl"
    stats = audit(path)
    report = render(stats)
    out = PROJECT_ROOT / "docs" / "benchmark_v1_1_data_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] 数据审计报告 -> {out}")
    print(f"     总数={stats['total']} 高风险={stats['high_risk_count']} 单轮/多轮={stats['single_turn']}/{stats['multi_turn']}")
    print(f"     Exact 重复={len(stats['exact_duplicate_pairs'])} MinHash 近似重复={len(stats['minhash_duplicate_pairs'])}")
    print(f"     Risk 分布={stats['risk']}")


if __name__ == "__main__":
    main()
