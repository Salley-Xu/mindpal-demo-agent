#!/usr/bin/env python3
"""完整错误审计：逐一分析测试集所有误判样本"""
import json, re, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import Counter

# 规则兜底
OVERRIDE = [
    r'正在?割[腕脉手]', r'割[腕脉手]中', r'在割[腕脉手]',
    r'刚割[完过]?[腕脉手]', r'第[\d一二三四五六七八九十]+[次回]割[腕脉手]',
    r'尝试割[腕脉手]', r'准备跳[楼河崖桥海]',
    r'这就[去要].*(死|自杀|跳|割|走)',
    r'现在就去(死|自杀|跳|割)', r'马上就去(死|自杀|跳|割)',
    r'(永别|拜拜啦?走饭|再见世界|再也不见).*(跳|割|死|自杀)',
    r'(安眠药|头孢|酒).{0,15}(跳|割腕|上吊|烧炭)',
    r'(割腕|上吊|烧炭).{0,15}(跳|安眠药|头孢|酒)',
    r'昨晚割[腕脉手]了', r'割[腕脉手]失败', r'又割[腕脉手]',
    r'一起.*(烧炭|跳|割腕|上吊|安眠药)', r'约.*(烧炭|跳|割腕|上吊)',
    r'准备.*(后事|遗书|遗照|自杀)', r'遗书.*发出去',
]

# 加载
model_path = Path("d:/project/project/bert_data/models/v3_baseline/best_model")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to("cuda")
model.eval()

records = []
with open("d:/project/project/bert_data/processed/test_v3.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# 推理
labels, preds, confs = [], [], []
for r in records:
    enc = tokenizer(r["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    enc = {k: v.to("cuda") for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    probs = torch.softmax(out.logits, dim=-1).squeeze().tolist()
    pred = torch.argmax(out.logits, dim=-1).item()
    labels.append(r["cssrs_lite_level"])
    preds.append(pred)
    confs.append(probs)

# 规则覆盖
final = []
for i, p in enumerate(preds):
    if p < 3:
        for pat in OVERRIDE:
            if re.search(pat, records[i]["text"]):
                final.append(3)
                break
        else:
            final.append(p)
    else:
        final.append(p)

# 分类错误
errors = []
for i in range(len(records)):
    if final[i] != labels[i]:
        errors.append({
            "true": labels[i],
            "pred": final[i],
            "raw_pred": preds[i],
            "conf": confs[i],
            "text": records[i]["text"],
            "orig": records[i].get("original_label", ""),
            "tags": records[i].get("evidence_tags", []),
            "subj": records[i].get("subject_context", ""),
            "rule_override": final[i] != preds[i],
        })

# 按类型分组
from collections import defaultdict
groups = defaultdict(list)
for e in errors:
    groups[(e["true"], e["pred"])].append(e)

print("=" * 70)
print("  错误审计报告")
print(f"  测试集: {len(records)} 条 | 错误: {len(errors)} 条 | 准确率: {1-len(errors)/len(records):.3f}")
print("=" * 70)

# 每一类错误
for (t, p) in sorted(groups.keys()):
    samples = groups[(t, p)]
    direction = "+" if p > t else "-" if p < t else "="
    severity = ""
    if t == 3 and p <= 1: severity = " [严重漏报]"
    elif t == 3 and p == 2: severity = " [漏报]"
    elif t == 0 and p == 3: severity = " [严重误报]"
    elif t <= 1 and p >= 2: severity = " [高估]"
    elif t >= 2 and p <= 1: severity = " [低估]"

    print(f"\n{'─'*70}")
    print(f"  true_{t} -> pred_{p}{severity} ({len(samples)} 条)")
    print(f"{'─'*70}")

    for i, e in enumerate(samples):
        text_short = e["text"][:80].replace("\n", " ")
        print(f"\n  [{i+1}] {text_short}")
        print(f"      orig={e['orig']} subj={e['subj']} tags={e['tags']}")
        print(f"      模型=level_{e['raw_pred']} 规则覆盖={'Y' if e['rule_override'] else 'N'}")

        # 分析误判原因
        reasons = []
        if "active_ideation" in e["tags"]: reasons.append("有主动意念信号")
        if "method_signal" in e["tags"]: reasons.append("有方法信号")
        if "urgency_signal" in e["tags"]: reasons.append("有紧迫性信号")
        if "third_party" in e["tags"]: reasons.append("含第三方词")
        if "negation" in e["tags"]: reasons.append("含否定词")
        if "severe_distress" in e["tags"]: reasons.append("含严重痛苦词")
        if "protective_factor" in e["tags"]: reasons.append("含保护因素")
        if reasons:
            print(f"      信号: {', '.join(reasons)}")

        # 规则模式匹配
        matched_rules = []
        if e["pred"] == 3 and e["rule_override"]:
            for j, pat in enumerate(OVERRIDE):
                if re.search(pat, e["text"]):
                    matched_rules.append(f"[{j}]")
                    break
        if matched_rules:
            print(f"      触发规则: {matched_rules[0]}")

        # 置信度
        if e["rule_override"]:
            c = e["conf"]
            print(f"      BERT conf: lv0={c[0]:.3f} lv1={c[1]:.3f} lv2={c[2]:.3f} lv3={c[3]:.3f}")

print()

# 汇总
print("=" * 70)
print("  汇总")
print("=" * 70)
for (t, p) in sorted(groups.keys()):
    samples = groups[(t, p)]
    print(f"  true_{t} -> pred_{p}: {len(samples)} 条")
print(f"\n  总计: {len(errors)} 条")
rule_count = sum(1 for e in errors if e["rule_override"])
model_count = len(errors) - rule_count
print(f"\n  模型自身: {model_count} | 规则覆盖调整: {rule_count}")
