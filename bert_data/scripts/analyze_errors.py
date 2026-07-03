#!/usr/bin/env python3
"""
导出特定错误样本进行人工分析
重点分析 7 类错误：
  1. true_3 -> pred_0  (最危险: 高风险被漏成无风险)
  2. true_3 -> pred_1  (危险: 高风险被低估)
  3. true_0 -> pred_2  (误报: 无风险被高估)
  4. true_0 -> pred_3  (严重误报: 无风险被标为最高)
  5. true_2 -> pred_3  (过度高估)
  6. true_3 -> pred_2  (低估高风险)
  7. true_1 -> pred_2  (边界高估)
"""

import json
import torch
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加载模型
model_path = Path("models/v3_baseline/best_model")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda")
model.to(device)
model.eval()

# 加载 test 集
records = []
with open("processed/test_v3.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# 推理
all_preds = []
for r in records:
    text = r["text"]
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    pred = torch.argmax(out.logits, dim=-1).item()
    all_preds.append(pred)

# 收集指定错误类型: (true, pred) pairs
target_errors = [
    (3, 0, "true_3->pred_0  HIGHEST RISK MISSED"),
    (3, 1, "true_3->pred_1  HIGH RISK UNDERESTIMATED"),
    (0, 2, "true_0->pred_2  FALSE POSITIVE (to level_2)"),
    (0, 3, "true_0->pred_3  FALSE POSITIVE (to level_3)"),
    (2, 3, "true_2->pred_3  OVER-CLASSIFY"),
    (3, 2, "true_3->pred_2  UNDER-CLASSIFY (high risk)"),
    (1, 2, "true_1->pred_2  BOUNDARY OVER-ESTIMATE"),
]

errors = {key: [] for _, _, key in target_errors}
# create a (t,p) -> key lookup
error_key_map = {(t, p): key for (t, p, key) in target_errors}
for idx, (r, pred) in enumerate(zip(records, all_preds)):
    true_label = r["cssrs_lite_level"]
    for (t, p, key) in target_errors:
        if true_label == t and pred == p:
            errors[key].append((idx, r, pred))

# 使用规则检测工具分析文本
def analyze_text(text):
    """提取文本中最显著的风险信号"""
    signals = []
    import re
    if re.search(r"(割腕|割脉|切腕|割手腕|割血管)", text): signals.append("有自伤行为")
    if re.search(r"(跳楼|跳河|跳海|跳桥|跳崖)", text): signals.append("提及跳楼")
    if re.search(r"(上吊|吊死|自缢)", text): signals.append("提及上吊")
    if re.search(r"(烧炭|一氧化碳|煤气)", text): signals.append("提及烧炭")
    if re.search(r"(安眠药|吞药|吃药.*死|药物过量)", text): signals.append("提及服药")
    if re.search(r"(今晚|今天|现在|马上|立刻).*(死|自杀|跳|割|离开)", text): signals.append("紧迫性")
    if re.search(r"(遗书|最后|告别|永别)", text): signals.append("有告别/遗书")
    if re.search(r"控制不住|忍不住|害怕.*(失控|控制不住)", text): signals.append("失控感")
    if re.search(r"想.*(死|自杀|离开|消失|不存在|结束)", text): signals.append("有死亡愿望")
    if re.search(r"(不想活|活不下去|撑不下去|熬不下去)", text): signals.append("绝望/放弃")
    if re.search(r"(想.*跳|想.*割|想.*上吊)", text): signals.append("有具体方法")
    if re.search(r"(约.*一起|一起.*死|有没有.*一起)", text): signals.append("约死行为")
    if re.search(r"(太累了|好累|累.*死|活.*没意思)", text): signals.append("严重疲惫/无意义")
    if re.search(r"(害怕|恐惧|绝望|痛苦|煎熬|生不如死)", text): signals.append("严重痛苦")
    if re.search(r"(第三|他|她|别人|新闻|听说|有一个|有位)", text): signals.append("可能非本人")
    if re.search(r"(不是|没有|不会|不想|别|开玩笑)", text): signals.append("否定表达")
    return signals if signals else ["无明确风险信号"]

# 写出分析报告
output_path = "processed/error_analysis_report.md"
with open(output_path, "w", encoding="utf-8") as out:
    out.write("# 错误样本分析报告\n\n")
    out.write(f"分析日期: 2026-07-02\n")
    out.write(f"模型: hfl/chinese-macbert-base (best_model)\n")
    out.write(f"测试集: test_v3.jsonl ({len(records)} 条)\n\n")
    out.write("---\n\n")

    for (t, p, key) in target_errors:
        samples = errors.get(key, [])
        out.write(f"## {key}\n\n")
        out.write(f"**数量: {len(samples)} 条**\n\n")
        out.write("| # | 文本 | 当前标签 | 模型预测 | 规则信号 | 人工判定 | 是否为标签错 |\n")
        out.write("|---|------|---------|---------|---------|---------|------------|\n")

        for i, (idx, r, pred) in enumerate(samples):
            text = r["text"]
            text_clean = text.replace("\n", " ").replace("|", "/")[:100]
            signals = "; ".join(analyze_text(text))

            out.write(f"| {i+1} | {text_clean} | level_{t} | level_{pred} | {signals} | (待填) | (待填) |\n")

        out.write("\n---\n\n")

    # 每个错误类型选 2 个典型样本详细展开
    out.write("## 典型样本详细分析\n\n")

    for (t, p, key) in target_errors:
        samples = errors.get(key, [])
        if not samples:
            continue
        out.write(f"### {key}\n\n")
        for i, (idx, r, pred) in enumerate(samples[:2]):
            text = r["text"]
            text_display = text[:300].replace("\n", "\n  ")

            out.write(f"#### 样本 #{i+1}\n\n")
            out.write(f"**文本:**\n```\n  {text_display}\n```\n\n")
            out.write(f"**当前标签:** level_{t}  |  **模型预测:** level_{pred}\n\n")
            out.write(f"**原始标签:** {r.get('original_label', 'N/A')}\n\n")
            out.write(f"**证据标签:** {r.get('evidence_tags', [])}\n\n")
            out.write(f"**判定依据分析:**\n\n")
            out.write("| 信号 | 存在 | 说明 |\n")
            out.write("|------|------|------|\n")
            sigs = analyze_text(text)
            for s in sigs:
                out.write(f"| {s} | 是 | (人工分析) |\n")
            out.write(f"\n**人工判定:** \n\n")
            out.write(f"**结论:** \n\n")
        out.write("---\n\n")

    # 汇总统计
    out.write("## 汇总\n\n")
    out.write("| 错误类型 | 数量 | 标签错占比 | 模型错占比 |\n")
    out.write("|---------|------|-----------|-----------|\n")
    for (t, p, key) in target_errors:
        samples = errors.get(key, [])
        out.write(f"| {key} | {len(samples)} | (待填) | (待填) |\n")

print(f"报告已生成: {output_path}")
print()

# 同时直接在终端输出关键信息
for (t, p, key) in target_errors:
    samples = errors.get(key, [])
    print(f"\n{'='*60}")
    print(f"  {key} ({len(samples)} 条)")
    print(f"{'='*60}")
    for i, (idx, r, pred) in enumerate(samples[:3]):
        text = r["text"][:80].replace("\n", " ")
        sigs = "|".join(analyze_text(r["text"]))
        print(f"  [{i+1}] true=level_{t}, pred=level_{pred}, orig={r.get('original_label','')}")
        print(f"      tags={r.get('evidence_tags',[])}")
    if len(samples) > 3:
        print(f"      ... 还有 {len(samples)-3} 条")
