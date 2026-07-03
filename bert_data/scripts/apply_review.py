# 人工复核结果写入脚本
# 5 条 silver + 1 条 gold, 2 条丢弃

import json
from pathlib import Path

PROCESSED = Path("d:/project/project/bert_data/processed")

# 加载全量数据
with open(PROCESSED / "mapped_v3_all.jsonl", "r", encoding="utf-8") as f:
    all_records = [json.loads(line) for line in f]

# 构造 ID → 记录 的映射
record_map = {r["sample_id"]: r for r in all_records}

# ===== 确认的复核修正 =====

corrections = {
    # 样本 #1: level_2 → level_1, 修正 tags
    "sos-hl-1k-200397": {
        "cssrs_lite_level": 1,
        "evidence_tags": ["passive_death_wish", "protective_factor", "severe_distress"],
        "label_quality": "silver",
        "review_reason": "",
        "mapping_method": "human",
    },
    # 样本 #2: 保持 level_0, 升 silver
    "sos-hl-1k-200991": {
        "label_quality": "silver",
        "review_reason": "",
        "mapping_method": "human",
    },
    # 样本 #3: 保持 level_2, 加 protective_factor tag, 升 silver
    "sos-hl-1k-201200": {
        "evidence_tags": ["active_ideation", "method_signal", "protective_factor"],
        "label_quality": "silver",
        "review_reason": "",
        "mapping_method": "human",
    },
    # 样本 #5: hypothetical, 加 discussion_context + protective_factor
    "sos-hl-1k-201246": {
        "subject_context": "hypothetical",
        "evidence_tags": ["discussion_context", "protective_factor"],
        "label_quality": "silver",
        "review_reason": "",
        "mapping_method": "human",
    },
    # 样本 #6: level_2 → level_3, gold 标签
    "mentalglm-300020": {
        "cssrs_lite_level": 3,
        "evidence_tags": ["active_ideation", "intent_signal", "method_signal", "plan_signal", "severe_distress"],
        "label_quality": "gold",
        "review_reason": "",
        "mapping_method": "human",
    },
}

discard_ids = {
    "sos-hl-1k-201244",   # 无语义 ":)晚安"
    "mentalglm-300024",   # 与 #2 重复
}

# 应用修正
for sid, fixes in corrections.items():
    if sid in record_map:
        r = record_map[sid]
        for k, v in fixes.items():
            r[k] = v
        print(f"  OK 更新 {sid}: level={r['cssrs_lite_level']}, quality={r['label_quality']}")

# 丢弃无用样本
filtered = [r for r in all_records if r["sample_id"] not in discard_ids]
print(f"\n  丢弃 {len(all_records) - len(filtered)} 条 (无语义/重复)")
print(f"  保留 {len(filtered)} 条")

# 按 quality 分流
gold = [r for r in filtered if r["label_quality"] == "gold"]
silver = [r for r in filtered if r["label_quality"] == "silver"]
review = [r for r in filtered if r["label_quality"] == "review"]

print(f"\n  gold:   {len(gold)}")
print(f"  silver: {len(silver)}")
print(f"  review: {len(review)}")

# 导出更新后的全量数据
with open(PROCESSED / "mapped_v3_all.jsonl", "w", encoding="utf-8") as f:
    for r in filtered:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 导出 gold
with open(PROCESSED / "gold_v3.jsonl", "w", encoding="utf-8") as f:
    for r in gold:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 导出 silver
with open(PROCESSED / "silver_v3.jsonl", "w", encoding="utf-8") as f:
    for r in silver:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 导出 review (剩余)
with open(PROCESSED / "review_queue.jsonl", "w", encoding="utf-8") as f:
    for r in review:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\n  已更新: mapped_v3_all.jsonl, gold_v3.jsonl, silver_v3.jsonl, review_queue.jsonl")

# ===== 重新切分 =====

import random
from collections import defaultdict, Counter
random.seed(42)

# 从 silver 切 train/dev/test (gold 放 test)
groups = defaultdict(list)
for r in silver:
    groups[r["group_id"]].append(r)

group_ids = list(groups.keys())
random.shuffle(group_ids)

level_groups = defaultdict(list)
for gid in group_ids:
    level = groups[gid][0]["cssrs_lite_level"]
    level_groups[level].append(gid)

train_gids, dev_gids, test_gids = set(), set(), set()
for level, gids in level_groups.items():
    random.shuffle(gids)
    n_total = len(gids)
    n_test = max(1, round(n_total * 0.1))
    n_dev = max(1, round(n_total * 0.15))
    test_gids.update(gids[:n_test])
    dev_gids.update(gids[n_test:n_test + n_dev])
    train_gids.update(gids[n_test + n_dev:])

train = [r for r in silver if r["group_id"] in train_gids]
dev = [r for r in silver if r["group_id"] in dev_gids]
test = [r for r in silver if r["group_id"] in test_gids]

# gold 全部放入 test
test.extend(gold)

# dev 中加少量 review
random.shuffle(review)
if review:
    dev.extend(review[:min(10, len(review))])

# 统计
for name, data in [("train", train), ("dev", dev), ("test", test)]:
    lc = Counter(r["cssrs_lite_level"] for r in data)
    print(f"  {name}: {len(data)} 条, level分布: {dict(sorted(lc.items()))}")

# 导出
for name, data in [("train", train), ("dev", dev), ("test", test)]:
    with open(PROCESSED / f"{name}_v3.jsonl", "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\nDone. 数据集已重建")
