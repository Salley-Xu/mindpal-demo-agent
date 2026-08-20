# -*- coding: utf-8 -*-
"""
Phase 5 Task 5.2：Risk Checkpoint Shootout。

独立跨源 Risk Test = agent_benchmark_v1_1（362）+ policy_independent_test_v1（430）= 792 条对话型风险标注。

方法：
  A rule-only（ModulePredictor._rule_risk 规则分类器）
  B v4_2_domain_only_v2（当前生产 MultiTaskBERT）
  C v4_3_coral（CORAL 序数回归）
  D v4_2_ft_conversational（对话微调版本）

指标：Macro F1 / Per-level F1 / High-risk(L2+L3) Recall-Precision / FPR / FNR / ECE / Brier。

用法：cd project_root && python evaluation/risk/run_risk_shootout.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(BACKEND_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from evaluation.benchmark_schema import BenchmarkCase  # noqa: E402


def build_test():
    """从 benchmark + policy independent 构建跨源风险测试。"""
    pairs = []  # (text, gold_level)
    for path in ["evaluation/datasets/agent_benchmark_v1_1.jsonl",
                 "evaluation/policy/policy_independent_test_v1.jsonl"]:
        for l in open(PROJECT_ROOT / path, encoding="utf-8"):
            if not l.strip():
                continue
            c = BenchmarkCase.model_validate(json.loads(l))
            text = [t.content for t in c.conversation if t.role.value == "user"][-1]
            pairs.append((text, c.expected.risk_level.value))
    return pairs


def _rule_predict(text):
    from evaluation.predictors.module import ModulePredictor, HIGH_RISK_KEYWORDS_L3, HIGH_RISK_KEYWORDS_L2
    import re
    if any(k in text for k in HIGH_RISK_KEYWORDS_L3):
        return 3, [0.0, 0.0, 0.0, 1.0]
    if any(k in text for k in HIGH_RISK_KEYWORDS_L2):
        return 2, [0.0, 0.0, 1.0, 0.0]
    # 负面情绪 → L1
    neg = ["焦虑", "抑郁", "愤怒", "压力", "绝望", "害怕", "孤独", "疲惫", "恐慌"]
    if any(e in text for e in neg):
        return 1, [0.0, 1.0, 0.0, 0.0]
    return 0, [1.0, 0.0, 0.0, 0.0]


def load_bert(path):
    from bert_risk_predictor import BertRiskPredictor
    return BertRiskPredictor(model_path=str(PROJECT_ROOT / path), device="cpu")


def evaluate(pairs, predict_fn, name):
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    y_true = [g for _, g in pairs]
    preds = []
    confs = []
    for text, _ in pairs:
        level, probs = predict_fn(text)
        preds.append(level)
        confs.append(probs)
    y_pred = preds

    n = len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3])
    acc = accuracy_score(y_true, y_pred)
    per_level = {}
    for lvl in range(4):
        per_level[f"L{lvl}_f1"] = round(f1_score([1 if x == lvl else 0 for x in y_true],
                                                 [1 if x == lvl else 0 for x in y_pred]), 4)

    # High-risk = L2/L3
    hr_true = [1 if l >= 2 else 0 for l in y_true]
    hr_pred = [1 if l >= 2 else 0 for l in y_pred]
    hr_recall = recall_score(hr_true, hr_pred, zero_division=0)
    hr_precision = precision_score(hr_true, hr_pred, zero_division=0)
    tp = sum(1 for a, b in zip(hr_true, hr_pred) if a and b)
    fp = sum(1 for a, b in zip(hr_true, hr_pred) if not a and b)
    fn = sum(1 for a, b in zip(hr_true, hr_pred) if a and not b)
    tn = sum(1 for a, b in zip(hr_true, hr_pred) if not a and not b)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    # ECE / Brier（用 high-risk 概率，即 P(L2)+P(L3)）
    conf_hr = [max(0.0, min(1.0, probs[2] + probs[3])) for probs in confs]
    brier = sum((c - a) ** 2 for c, a in zip(conf_hr, hr_true)) / n
    # ECE（10 bins）
    bins = 10
    bin_edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        idx = [j for j, c in enumerate(conf_hr) if lo <= c < hi]
        if not idx:
            continue
        avg_conf = sum(conf_hr[j] for j in idx) / len(idx)
        acc_bin = sum(hr_true[j] for j in idx) / len(idx)
        ece += len(idx) / n * abs(avg_conf - acc_bin)

    return {
        "method": name, "n": n,
        "macro_f1": round(macro_f1, 4), "accuracy": round(acc, 4),
        **per_level,
        "high_risk_recall": round(hr_recall, 4),
        "high_risk_precision": round(hr_precision, 4),
        "fpr": round(fpr, 4), "fnr": round(fnr, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "ece": round(ece, 4), "brier": round(brier, 4),
        "latency_ms": 0.0,
    }


def main():
    pairs = build_test()
    print(f"independent risk test: {len(pairs)} cases")
    from collections import Counter
    print(f"  gold level dist: {dict(sorted(Counter(g for _, g in pairs).items()))}")

    methods = []
    t0 = time.time()
    methods.append(evaluate(pairs, _rule_predict, "rule_only"))
    print("[OK] rule_only  done")

    for name, path in [
        ("v4_2_domain_only_v2", "bert_data/models/v4_2_domain_only_v2/best_model"),
        ("v4_3_coral", "bert_data/models/v4_3_coral/best_model"),
        ("v4_2_ft_conversational", "bert_data/models/v4_2_ft_conversational/best_model"),
    ]:
        if not (PROJECT_ROOT / path / "model_state.pt").exists():
            print(f"[SKIP] {name}: model not found")
            continue
        model = load_bert(path)
        t_start = time.time()
        def _bert(text, _m=model):
            r = _m.predict(text)
            lvl = int(r["level"].split("_")[1])
            return lvl, r["class_probabilities"]
        m = evaluate(pairs, _bert, name)
        m["latency_ms"] = round((time.time() - t_start) / len(pairs) * 1000, 2)
        methods.append(m)
        print(f"[OK] {name} done ({m['latency_ms']}ms/case)")

    print(f"\ntotal time: {time.time()-t0:.1f}s")
    print("\n=== SHOOTOUT TABLE ===")
    hdr = ["method", "macro_f1", "accuracy", "HR_recall", "HR_prec", "FPR", "FNR",
           "L0_f1", "L1_f1", "L2_f1", "L3_f1", "ECE", "Brier", "latency_ms"]
    key_map = {"HR_recall": "high_risk_recall", "HR_prec": "high_risk_precision",
               "FPR": "fpr", "FNR": "fnr"}
    print(" | ".join(f"{h:>12}" for h in hdr))
    for m in methods:
        vals = []
        for h in hdr:
            k = key_map.get(h, h)
            v = m.get(k, 0)
            vals.append(f"{v if isinstance(v, str) else round(v, 4):>12}")
        print(" | ".join(vals))

    reports = PROJECT_ROOT / "evaluation" / "risk" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports / f"risk_shootout_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "n": len(pairs),
                   "methods": methods}, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] -> {path.name}")


if __name__ == "__main__":
    main()
