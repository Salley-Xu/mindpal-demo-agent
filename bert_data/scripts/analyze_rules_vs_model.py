#!/usr/bin/env python3
"""
分析测试集中误报/漏报的来源：模型 vs 规则兜底层

重点分析 4 类错误：
  true_0 → pred_3  (最严重误报：模型/规则把无风险判为最高级)
  true_0 → pred_2  (严重误报)
  true_1 → pred_3  (高估)
  true_3 → pred_1  (漏报：高风险被低估)
"""
import json, re, os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ====== 规则兜底 pattern（和 fix_and_supplement.py 保持一致） ======
HIGH_RISK_OVERRIDE_PATTERNS = [
    r"正在?割[腕脉手]",
    r"割[腕脉手]中",
    r"在割[腕脉手]",
    r"刚割[完过]?[腕脉手]",
    r"第[一二三四五六七八九十\d]+[次回]割[腕脉手]",
    r"尝试割[腕脉手]",
    r"在[站坐]?[天台楼顶窗台阳台].*[想准]?备?跳",
    r"已经[站坐].*[天台楼顶窗台阳台].*[想准]?备?跳",
    r"准备跳[楼河崖桥海]",
    r"这就[去要].*(死|自杀|跳|割|走)",
    r"现在就去(死|自杀|跳|割)",
    r"马上就去(死|自杀|跳|割)",
    r"(永别|拜拜啦?走饭|再见世界|再也不见).*(跳|割|死|自杀)",
    r"去(跳|割).*(啦|了|咯).*(拜拜|再见|永别)",
    r"(安眠药|头孢|酒).{0,15}(跳|割腕|上吊|烧炭)",
    r"(割腕|上吊|烧炭).{0,15}(跳|安眠药|头孢|酒)",
    r"昨晚割[腕脉手]了",
    r"割[腕脉手]失败",
    r"又割[腕脉手]",
    r"一起.*(烧炭|跳|割腕|上吊|安眠药)",
    r"约.*(烧炭|跳|割腕|上吊)",
    r"准备.*(后事|遗书|遗照|自杀)",
    r"遗书.*发出去",
]

def match_rule(text):
    for i, pat in enumerate(HIGH_RISK_OVERRIDE_PATTERNS):
        if re.search(pat, text):
            return i
    return None

# ====== 加载模型 ======
model_path = Path("d:/project/project/bert_data/models/v3_baseline/best_model")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda")
model.to(device)
model.eval()

# ====== 加载测试集 ======
records = []
with open("d:/project/project/bert_data/processed/test_v3.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

# ====== 推理: 模型预测 vs 规则覆盖 ======
results = []
for r in records:
    true_lv = r["cssrs_lite_level"]
    text = r["text"]

    enc = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    model_pred = torch.argmax(logits, dim=-1).item()

    # 规则覆盖
    rule_idx = match_rule(text)
    final_pred = 3 if (rule_idx is not None and model_pred < 3) else model_pred

    results.append({
        "true": true_lv,
        "model_pred": model_pred,
        "final_pred": final_pred,
        "rule_hit": rule_idx is not None,
        "rule_idx": rule_idx,
        "probs": probs,
        "text": text,
        "orig": r.get("original_label",""),
        "tags": r.get("evidence_tags",[]),
    })

# ====== 分析 4 类错误 ======
targets = [
    (0, 3, "true_0 -> pred_3 最严重误报"),
    (0, 2, "true_0 -> pred_2 严重误报"),
    (1, 3, "true_1 -> pred_3 高估"),
    (3, 1, "true_3 -> pred_1 漏报"),
]

output_path = "d:/project/project/bert_data/processed/rule_vs_model_analysis.txt"
with open(output_path, "w", encoding="utf-8") as out:
    out.write("规则 vs 模型 误报分析\n")
    out.write("=" * 70 + "\n\n")

    for true_lv, pred_lv, title in targets:
        matched = [r for r in results if r["true"] == true_lv and r["final_pred"] == pred_lv]

        out.write(f"{'='*70}\n")
        out.write(f"{title}  ({len(matched)} 条)\n")
        out.write(f"{'='*70}\n\n")

        # 按来源分类
        from_model = [r for r in matched if not r["rule_hit"]]
        from_rule = [r for r in matched if r["rule_hit"]]

        out.write(f"来源: 模型={len(from_model)}, 规则覆盖={len(from_rule)}\n\n")

        # 如果规则导致的，列出具体哪个 pattern
        if from_rule:
            out.write("--- 规则覆盖导致的 ---\n")
            for r in from_rule:
                ri = r["rule_idx"]
                out.write(f"  pattern[{ri}]: {HIGH_RISK_OVERRIDE_PATTERNS[ri]}\n")
                out.write(f"  text: {r['text'][:80]}\n")
                out.write(f"  model_pred={r['model_pred']}, final={r['final_pred']}\n")
                out.write(f"  conf: [{r['probs'][0]:.3f}, {r['probs'][1]:.3f}, {r['probs'][2]:.3f}, {r['probs'][3]:.3f}]\n")
                out.write(f"  orig={r['orig']}, tags={r['tags']}\n\n")

        if from_model:
            out.write("--- 模型自身预测错误 ---\n")
            for r in from_model:
                out.write(f"  text: {r['text'][:80]}\n")
                out.write(f"  model_pred={r['model_pred']}, conf: [{r['probs'][0]:.3f}, {r['probs'][1]:.3f}, {r['probs'][2]:.3f}, {r['probs'][3]:.3f}]\n")
                out.write(f"  orig={r['orig']}, tags={r['tags']}\n\n")

        out.write("\n")

print(f"分析完成 → {output_path}")
