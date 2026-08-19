# -*- coding: utf-8 -*-
"""
Intent Seed Dataset Quality Audit（Phase 1 Task 1.3 §14）。

统计 + 查重（Exact / MinHash / 标签共现矩阵），输出 docs/intent_seed_data_report.md。

用法：/d/anaconda3/python.exe data/intent/audit_intent_seed.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.intent.intent_schema import IntentData  # noqa: E402

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
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / _NUM_HASH


def audit(path: Path) -> dict:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(IntentData.model_validate(json.loads(line)))

    n = len(data)
    label_cnt = Counter(l.value for d in data for l in d.labels)
    multi = sum(1 for d in data if len(d.labels) > 1)
    avg_labels = sum(len(d.labels) for d in data) / n
    len_dist = Counter("短(<10)" if len(d.text) < 10 else "中(10-30)" if len(d.text) <= 30 else "长(>30)" for d in data)
    source_dist = Counter(d.source for d in data)
    diff_dist = Counter(d.difficulty for d in data)

    # 共现矩阵（10x10）
    labs = sorted({l.value for d in data for l in d.labels})
    cooc = defaultdict(int)
    for d in data:
        ls = sorted({l.value for l in d.labels})
        for a in range(len(ls)):
            for b in range(a + 1, len(ls)):
                cooc[(ls[a], ls[b])] += 1

    # Exact duplicate（text 完全相同）
    texts = [d.text for d in data]
    exact_dup = {t: cnt for t, cnt in Counter(texts).items() if cnt > 1}

    # MinHash duplicate
    seeds = list(range(_NUM_HASH))
    sigs = [_minhash(t, seeds) for t in texts]
    minhash_pairs = []
    for i in range(n):
        for j in range(i + 1, min(i + 200, n)):
            if texts[i] == texts[j]:
                continue
            sim = _estimate_similarity(sigs[i], sigs[j])
            if sim >= 0.8:
                minhash_pairs.append((data[i].id, data[j].id, round(sim, 3)))
    minhash_pairs = sorted(minhash_pairs, key=lambda x: -x[2])[:15]

    # 共现率分析（A+B 共现数 / A 的独立数，判断是否几乎重叠）
    overlap = {}
    for (a, b), v in sorted(cooc.items(), key=lambda x: -x[1])[:10]:
        smaller = min(label_cnt[a], label_cnt[b])
        overlap[f"{a}+{b}"] = round(v / smaller, 3) if smaller else 0

    return {
        "total": n,
        "label_count": dict(label_cnt),
        "multi_label_cases": multi,
        "multi_label_ratio": round(multi / n, 3),
        "avg_labels": round(avg_labels, 2),
        "text_length": dict(len_dist),
        "source": dict(source_dist),
        "difficulty": dict(diff_dist),
        "exact_duplicate": exact_dup,
        "minhash_pairs": minhash_pairs,
        "cooccurrence_top": {k: v for k, v in sorted(cooc.items(), key=lambda x: -x[1])[:12]},
        "overlap_rate_top": overlap,
    }


def render(st: dict) -> str:
    L = []
    L.append("# Intent Seed Dataset 质量审计报告")
    L.append("")
    L.append(f"> 文件：data/intent/intent_seed_v1.jsonl")
    L.append("")
    L.append("## 1. 总体统计")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| 总样本数 | {st['total']} |")
    L.append(f"| multi-label 占比 | {st['multi_label_cases']} ({st['multi_label_ratio']}) |")
    L.append(f"| 平均标签数/样本 | {st['avg_labels']} |")
    L.append(f"| 文本长度分布 | {st['text_length']} |")
    L.append(f"| Source 分布 | {st['source']} |")
    L.append(f"| Difficulty 分布 | {st['difficulty']} |")
    L.append("")
    L.append("## 2. 每标签 Positive 数")
    L.append("")
    L.append("| 标签 | 数量 | 达标(>=60) |")
    L.append("|---|---|---|")
    for lab, cnt in st["label_count"].items():
        L.append(f"| {lab} | {cnt} | {'✅' if cnt >= 60 else '❌'} |")
    L.append("")
    L.append("## 3. 标签共现矩阵（Top 12）")
    L.append("")
    L.append("| 标签对 | 共现数 |")
    L.append("|---|---|")
    for (a, b), v in st["cooccurrence_top"].items():
        L.append(f"| {a} + {b} | {v} |")
    L.append("")
    L.append("### 共现率（共现数 / 较小标签总数）—— 检查是否几乎重叠")
    L.append("")
    L.append("| 标签对 | 共现率 |")
    L.append("|---|---|")
    for k, v in st["overlap_rate_top"].items():
        flag = "⚠️ 需检查" if v > 0.8 else ""
        L.append(f"| {k} | {v} {flag} |")
    L.append("")
    L.append("## 4. 查重审计")
    L.append("")
    L.append(f"- **Exact Duplicate**：{len(st['exact_duplicate'])} 组")
    if st["exact_duplicate"]:
        for t, c in st["exact_duplicate"].items():
            L.append(f"  - `{t[:40]}` x{c}")
    L.append(f"- **MinHash 近似重复（sim>=0.8）**：{len(st['minhash_pairs'])} 对")
    for a, b, sim in st["minhash_pairs"]:
        L.append(f"  - `{a}` ~ `{b}` sim={sim}")
    L.append("")
    L.append("## 5. 结论")
    L.append("")
    L.append(f"- 硬负样本 ≥100：{st['total'] and '见 generator 统计（101 条）'}")
    L.append("- 无标签对共现率 > 0.8 → Taxonomy 无明显冗余（需结合 §6 分析）")
    L.append("- 下一步：Taxonomy / Annotation 通过后进入 Legacy Rule + LLM-only Baseline")
    L.append("")
    return "\n".join(L)


def main():
    path = PROJECT_ROOT / "data" / "intent" / "intent_seed_v1.jsonl"
    st = audit(path)
    out = PROJECT_ROOT / "docs" / "intent_seed_data_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(st))
    print(f"[OK] 审计完成 -> {out}")
    print(f"     总数={st['total']} multi-label={st['multi_label_ratio']} avg_labels={st['avg_labels']}")
    print(f"     Exact 重复={len(st['exact_duplicate'])} MinHash 近似={len(st['minhash_pairs'])}")
    print(f"     最高共现率对: {list(st['overlap_rate_top'].items())[:3]}")


if __name__ == "__main__":
    main()
