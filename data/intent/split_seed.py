# -*- coding: utf-8 -*-
"""
Intent Seed 分组切分（Phase 1 §19）：train 70% / dev 15% / test 15%。

按模板桶（case_id 前缀，如 iseed_c / iseed_e ...）做 Group Split，
避免同一模板改写同时进入 train/test。

用法：/d/anaconda3/python.exe data/intent/split_seed.py
输出：data/intent/intent_train_v1.jsonl / intent_dev_v1.jsonl / intent_test_v1.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.intent.intent_schema import IntentData  # noqa: E402


def group_key(d: IntentData) -> str:
    # 变体与其源模板同组（防泄漏）
    if d.source == "template_variant" and d.notes and "variant_of=" in d.notes:
        src = d.notes.split("variant_of=")[1].strip()
        parts = src.split("_")
        return "_".join(parts[:2]) if len(parts) >= 3 else src
    # iseed_xxx_NNN -> iseed_xxx
    parts = d.id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 3 else d.id


def split_group(data: list, rng: random.Random):
    """按模板桶 Group Split（用于 Task 1.7 的 LLM 扩充数据，防同模板泄漏）。"""
    groups = defaultdict(list)
    for d in data:
        groups[group_key(d)].append(d)
    group_labels = {g: set(l.value for d in items for l in d.labels) for g, items in groups.items()}
    group_names = sorted(groups.keys())
    rng.shuffle(group_names)
    n = len(group_names)
    n_train = round(n * 0.7)
    n_dev = round(n * 0.15)
    train_groups = group_names[:n_train]
    dev_groups = group_names[n_train:n_train + n_dev]
    test_groups = group_names[n_train + n_dev:]

    all_labels = set(l.value for d in data for l in d.labels)
    test_labels = set(l.value for g in test_groups for d in groups[g] for l in d.labels)
    for lab in (all_labels - test_labels):
        for g in list(train_groups):
            if lab in group_labels[g]:
                test_groups.append(g)
                train_groups.remove(g)
                break

    train = [d for g in train_groups for d in groups[g]]
    dev = [d for g in dev_groups for d in groups[g]]
    test = [d for g in test_groups for d in groups[g]]
    return train, dev, test


def split_stratified(data: list, rng: random.Random):
    """分层随机切分（seed 模板是独立手写文本，泄漏风险低；保证 test 标签平衡）。"""
    data = list(data)
    rng.shuffle(data)
    n = len(data)
    n_train = int(n * 0.7)
    n_dev = int(n * 0.15)
    train, dev, test = data[:n_train], data[n_train:n_train + n_dev], data[n_train + n_dev:]
    # 若 test 缺标签，从 dev/train 抽补
    all_labels = set(l.value for d in data for l in d.labels)
    test_labels = set(l.value for d in test for l in d.labels)
    for lab in (all_labels - test_labels):
        for pool in (dev, train):
            for d in pool:
                if any(l.value == lab for l in d.labels):
                    test.append(d)
                    pool.remove(d)
                    break
            if lab in test_labels:
                break
    return train, dev, test


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["group", "stratified"], default="stratified",
                        help="group 用于 LLM 扩充数据防泄漏；stratified 用于 seed 手写模板")
    parser.add_argument("--data", default="intent_seed_v1.jsonl",
                        help="输入文件名（data/intent/ 下）")
    parser.add_argument("--out-suffix", default="v1", help="输出文件后缀，如 v1 / v1_exp")
    args = parser.parse_args()

    src = PROJECT_ROOT / "data" / "intent" / args.data
    data = [IntentData.model_validate(json.loads(l)) for l in open(src, encoding="utf-8") if l.strip()]

    rng = random.Random(2026)
    if args.mode == "group":
        train, dev, test = split_group(data, rng)
    else:
        train, dev, test = split_stratified(data, rng)

    def write(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for d in items:
                f.write(json.dumps(d.model_dump(), ensure_ascii=False) + "\n")

    suffix = args.out_suffix
    train_path = PROJECT_ROOT / "data" / "intent" / f"intent_train_{suffix}.jsonl"
    dev_path = PROJECT_ROOT / "data" / "intent" / f"intent_dev_{suffix}.jsonl"
    test_path = PROJECT_ROOT / "data" / "intent" / f"intent_test_{suffix}.jsonl"
    write(train_path, train)
    write(dev_path, dev)
    write(test_path, test)

    print(f"[OK] {args.mode} 切分完成: train={len(train)} dev={len(dev)} test={len(test)}")
    for name, items in [("train", train), ("dev", dev), ("test", test)]:
        cnt = Counter(l.value for d in items for l in d.labels)
        print(f"  {name:5s} 标签覆盖: {len(cnt)} 类 {dict(cnt)}")


if __name__ == "__main__":
    main()
