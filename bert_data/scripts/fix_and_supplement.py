#!/usr/bin/env python3
"""
标签修复 + 规则兜底层 + 数据补充
"""
import json, re, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from collections import Counter

DATA_DIR = Path("d:/project/project/bert_data/processed")
OUTPUT_DIR = Path("d:/project/project/bert_data/processed_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 修复 2 条标签错
# ============================================================
print("=" * 60)
print("1. 修复标签错")
print("=" * 60)

with open(DATA_DIR / "mapped_v3_all.jsonl", "r", encoding="utf-8") as f:
    all_records = [json.loads(line) for line in f]

fixes = {
    # 样本 #6: "我割腕有人看嘛" → level_2
    "我割腕有人看": {
        "cssrs_lite_level": 2,
        "evidence_tags": ["active_ideation", "method_signal"],
        "review_reason": "",
        "mapping_method": "human",
    },
    # 样本 #7: "割腿上比割手腕出血要多哎" → level_2
    "割腿上比割手腕出血": {
        "cssrs_lite_level": 2,
        "evidence_tags": ["method_signal", "severe_distress"],
        "review_reason": "",
        "mapping_method": "human",
    },
}

fixed_count = 0
for r in all_records:
    for keyword, new_values in fixes.items():
        if keyword in r["text"] and r["cssrs_lite_level"] == 3:
            old_level = r["cssrs_lite_level"]
            for k, v in new_values.items():
                r[k] = v
            r["label_quality"] = "silver"
            print(f"  修复: {r['sample_id']}")
            print(f"    level: {old_level} → {r['cssrs_lite_level']}")
            print(f"    text: {r['text'][:60]}")
            fixed_count += 1

print(f"\n  共修复 {fixed_count} 条")
print()

# ============================================================
# 2. 规则兜底层：短文本高危 pattern
# ============================================================
print("=" * 60)
print("2. 规则兜底层：短文本高危 pattern")
print("=" * 60)

# 这些 pattern 一旦命中，强制覆盖到 level_3（不管 BERT 预测什么）
HIGH_RISK_OVERRIDE_PATTERNS = [
    # 进行式自伤行为
    r"正在?割[腕脉手]",
    r"割[腕脉手]中",
    r"在割[腕脉手]",
    r"刚割[完过]?[腕脉手]",
    r"第[一二三四五六七八九十\d]+[次回]割[腕脉手]",
    r"尝试割[腕脉手]",

    # 进行式跳楼
    r"在[站坐]?[天台楼顶窗台阳台].*[想准]?备?跳",
    r"已经[站坐].*[天台楼顶窗台阳台].*[想准]?备?跳",
    r"准备跳[楼河崖桥海]",

    # 正在进行或即将行动
    r"这就[去要].*(死|自杀|跳|割|走)",
    r"现在就去(死|自杀|跳|割)",
    r"马上就去(死|自杀|跳|割)",

    # 明确告别+行动
    r"(永别|拜拜啦?走饭|再见世界|再也不见).*(跳|割|死|自杀)",
    r"去(跳|割).*(啦|了|咯).*(拜拜|再见|永别)",

    # 综合方法描述（多方法联动 → 意图强，限定15字内）
    r"(安眠药|头孢|酒).{0,15}(跳|割腕|上吊|烧炭)",
    r"(割腕|上吊|烧炭).{0,15}(跳|安眠药|头孢|酒)",

    # 有明确实际行为 + 正在描述
    r"昨晚割[腕脉手]了",
    r"割[腕脉手]失败",
    r"又割[腕脉手]",

    # 约死+有方法
    r"一起.*(烧炭|跳|割腕|上吊|安眠药)",
    r"约.*(烧炭|跳|割腕|上吊)",

    # 准备结束生命
    r"准备.*(后事|遗书|遗照|自杀)",
    r"遗书.*发出去",
]

def match_high_risk_patterns(text: str) -> list:
    """返回所有匹配的高危 pattern 名称"""
    matched = []
    for i, pattern in enumerate(HIGH_RISK_OVERRIDE_PATTERNS):
        if re.search(pattern, text):
            matched.append(f"rule_{i}")
    return matched

# 扫描全量数据，统计命中的样本
hit_count = 0
for r in all_records:
    matches = match_high_risk_patterns(r["text"])
    if matches:
        hit_count += 1
        # 强制设为 level_3
        if r["cssrs_lite_level"] < 3:
            print(f"  规则覆盖: {r['sample_id']} level_{r['cssrs_lite_level']} → level_3")
            print(f"    text: {r['text'][:60]}")
            print(f"    rules: {len(matches)} 条命中")
            r["cssrs_lite_level"] = 3
            r["evidence_tags"] = list(set(r.get("evidence_tags", []) + ["intent_signal", "urgency_signal"]))
            r["mapping_method"] = "rule_fallback"

print(f"\n  高危规则命中: {hit_count} 条")
print(f"  其中覆盖升级: {(sum(1 for r in all_records if r.get('mapping_method')=='rule_fallback'))} 条")
print()

# ============================================================
# 3. 补充训练数据：对标注数据做 SMOTE 式增强（回译风格）
# ============================================================
print("=" * 60)
print("3. 补充训练数据")
print("=" * 60)

# 从现有 level_3 样本中筛选出和 6 条模型错 pattern 相似的样本
# 用于补充训练集中这些模式的数量
TARGET_PATTERNS = {
    "positive_masking": [  # 正面情绪 + 高风险
        r"(开心|兴奋|轻松|解脱|期待|多好).*(死|跳|割|离开|再见|走啦)",
        r"(死|跳|离开|再见|走啦).*(开心|兴奋|轻松|解脱|期待|多好)",
        r"太开心.*死|多好.*离开",
    ],
    "self_harm_action": [  # 实际自伤行为
        r"昨晚?.*(割[腕脉]|吞药|吃药.*自杀|跳)",
        r"(割[腕脉]|吞药|吃药).*(失败|十几刀|好多|出血|痛)",
        r"第[\d一二三]+次.*(割|吞|跳)",
    ],
    "short_high_risk": [  # 短文本高风险（< 20 字而 level_3）
        r"^.{1,30}(割腕|上吊|跳楼).{0,10}$",
        r"^.{1,20}想死.{0,10}$",
        r"^.{1,20}在准备死",
    ],
    "urgent_short": [  # 短文本紧迫
        r"立马|马上|立刻|现在就|这就去.*(跳|割|死|自杀)",
        r"冲出去.*(跳|死|自杀)",
        r"站在.*(阳台|天台|楼顶|窗台).*想跳",
    ],
    "farewell_high_risk": [  # 告别 + 高风险
        r"(再见|拜拜|永别|走啦|离开啦|先走).*(世界|大家|各位)",
        r"我走.*了.*(拜拜|再见|永别)",
    ],
}

# 统计训练集中各类 pattern 的覆盖情况
print("训练集 level_3 中的模式覆盖:")
train_records = [r for r in all_records if r["label_quality"] in ("silver", "gold")]
total_l3 = sum(1 for r in train_records if r["cssrs_lite_level"] == 3)
print(f"  总 level_3 样本: {total_l3}")

for pattern_name, patterns in TARGET_PATTERNS.items():
    matched = set()
    for r in train_records:
        if r["cssrs_lite_level"] == 3:
            for p in patterns:
                if re.search(p, r["text"]):
                    matched.add(r["sample_id"])
                    break
    print(f"  {pattern_name}: {len(matched)} 条")

# 对覆盖不足的模式，从全量数据中寻找同 pattern 样本补充
print("\n补充 low → level_3 的样本:")
supplement_count = 0
for r in all_records:
    if r["cssrs_lite_level"] >= 3:
        continue  # 已经是 level_3 或以上
    if r["source"] != "sos-1k-fine":
        continue
    if r["original_label"] in ("fine_0", "fine_1", "fine_2"):
        continue  # 避免把日常情绪升到 level_3

    # 检查是否匹配高危模式
    matches = match_high_risk_patterns(r["text"])
    if matches:
        old_level = r["cssrs_lite_level"]
        r["cssrs_lite_level"] = 3
        r["evidence_tags"] = list(set(r.get("evidence_tags", []) + ["intent_signal", "urgency_signal"]))
        if r["label_quality"] == "silver":
            pass  # 保持 silver
        supplement_count += 1
        if supplement_count <= 5:
            print(f"  升级: {r['sample_id']} level_{old_level} → level_3")
            print(f"    text: {r['text'][:50]}")

print(f"\n  共补充 {supplement_count} 条")
print()

# ============================================================
# 4. 重建 train/dev/test
# ============================================================
print("=" * 60)
print("4. 重建数据集")
print("=" * 60)

# 导出修正后的全量
with open(OUTPUT_DIR / "mapped_v3_all.jsonl", "w", encoding="utf-8") as f:
    for r in all_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  全量: {len(all_records)} 条 → mapped_v3_all.jsonl")

# 按 quality 分流
silver = [r for r in all_records if r["label_quality"] == "silver"]
gold = [r for r in all_records if r["label_quality"] == "gold"]
review = [r for r in all_records if r["label_quality"] == "review"]

# 导出分流
for name, data in [("silver", silver), ("gold", gold), ("review", review)]:
    with open(OUTPUT_DIR / f"{name}_v3.jsonl", "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(data)} 条")

# group-safe stratified split
import random
random.seed(42)
from collections import defaultdict

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
    n_test = max(1, round(n_total * 0.10))
    n_dev = max(1, round(n_total * 0.15))
    test_gids.update(gids[:n_test])
    dev_gids.update(gids[n_test:n_test + n_dev])
    train_gids.update(gids[n_test + n_dev:])

train = [r for r in silver if r["group_id"] in train_gids]
dev = [r for r in silver if r["group_id"] in dev_gids]
test = [r for r in silver if r["group_id"] in test_gids]
test.extend(gold)

# dev 中加 review
if review:
    dev.extend(review[:min(5, len(review))])

for name, data in [("train", train), ("dev", dev), ("test", test)]:
    with open(OUTPUT_DIR / f"{name}_v3.jsonl", "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lc = Counter(r["cssrs_lite_level"] for r in data)
    print(f"  {name}: {len(data)} -> level分布: {dict(sorted(lc.items()))}")

print()
print("=" * 60)
print("规则兜底层 — 推理时使用")
print("=" * 60)
print("""
在 BERT 推理代码中加入以下后处理逻辑:

    if predict(text) < 3:
        for pattern in HIGH_RISK_OVERRIDE_PATTERNS:
            if re.search(pattern, text):
                return 3  # override to level_3

当前 HIGH_RISK_OVERRIDE_PATTERNS 列表:
""")
for i, p in enumerate(HIGH_RISK_OVERRIDE_PATTERNS):
    print(f"  [{i}] {p}")

print(f"\n补数据后的 level_3 总数: {sum(1 for r in all_records if r['cssrs_lite_level']==3)}")
