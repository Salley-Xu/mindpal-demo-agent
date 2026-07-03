#!/usr/bin/env python3
"""导出 test 集中 level_3 漏报样本（true_3 但模型预测 < 3）"""
import json, torch, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = Path("d:/project/project/bert_data/models/v3_baseline/best_model")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda")
model.to(device)
model.eval()

records = []
with open("d:/project/project/bert_data/processed/test_v3.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

errors = []
for r in records:
    true_lv = r["cssrs_lite_level"]
    enc = tokenizer(r["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred = torch.argmax(logits, dim=-1).item()
    if true_lv == 3 and pred < 3:
        errors.append((r, pred, probs))

errors.sort(key=lambda x: x[1])  # pred_0 优先

with open("d:/project/project/bert_data/processed/level3_missed.txt", "w", encoding="utf-8") as out:
    out.write("test 集中 level_3 漏报样本（true_3 但模型预测 < 3）\n")
    out.write(f"总计: {len(errors)} / 16 条\n\n")

    for i, (r, pred, probs) in enumerate(errors):
        out.write(f"{'='*70}\n")
        out.write(f"样本 #{i+1}\n\n")
        out.write(f"文本:\n  {r['text']}\n\n")
        out.write(f"当前标签:  level_3\n")
        out.write(f"模型预测:  level_{pred}\n")
        out.write(f"置信度分布:  level_0={probs[0]:.3f}, level_1={probs[1]:.3f}, level_2={probs[2]:.3f}, level_3={probs[3]:.3f}\n\n")
        out.write(f"原始标签:  {r.get('original_label','')}\n")
        out.write(f"证据标签:  {r.get('evidence_tags',[])}\n")
        out.write(f"主体语境:  {r.get('subject_context','')}\n")
        out.write(f"来源:      {r.get('source','')}\n\n")
        out.write(f"人工判定:\n")
        out.write(f"  正确 level?  (待填)\n")
        out.write(f"  问题来源:   (标签错 / 模型错 / 边界模糊)\n")
        out.write(f"  备注:\n\n")

print(f"已导出 {len(errors)} 条到 processed/level3_missed.txt")
