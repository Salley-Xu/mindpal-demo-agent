# -*- coding: utf-8 -*-
"""
Intent Small Model 训练（Phase 1 Task 1.6）。

用法：
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 \
  /d/anaconda3/python.exe evaluation/intent/train_small_model.py --epochs 5

输出：
  models/intent/best_model/   （model_state.pt + tokenizer）
  evaluation/intent/reports/classifier_baseline.<json|md>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from data.intent.intent_schema import INTENT_LABELS, IntentData  # noqa: E402
from evaluation.intent.metrics import intent_metrics  # noqa: E402
from evaluation.intent.small_model import IntentClassifier, SmallModelPredictor  # noqa: E402

LABEL2ID = {lab: i for i, lab in enumerate(INTENT_LABELS)}


class IntentDataset(Dataset):
    def __init__(self, items: list, tokenizer, max_length: int = 128):
        self.examples = []
        for d in items:
            enc = tokenizer(d.text, truncation=True, padding="max_length",
                            max_length=max_length, return_tensors="pt")
            target = torch.zeros(len(INTENT_LABELS))
            for lab in d.labels:
                target[LABEL2ID[lab.value]] = 1.0
            self.examples.append((enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_items(path: Path):
    return [IntentData.model_validate(json.loads(l)) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--model-name", default="hfl/chinese-macbert-base")
    parser.add_argument("--device", default="cpu", help="cpu（默认，规避本机 CUDA 内核不兼容）")
    parser.add_argument("--data-suffix", default="v1", help="数据文件后缀（train/dev/test_<suffix>）")
    parser.add_argument("--out-model", default="best_model", help="模型输出子目录名")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "intent"
    sfx = args.data_suffix
    train_items = load_items(data_dir / f"intent_train_{sfx}.jsonl")
    dev_items = load_items(data_dir / f"intent_dev_{sfx}.jsonl")
    test_items = load_items(data_dir / f"intent_test_{sfx}.jsonl")
    print(f"[INFO] train={len(train_items)} dev={len(dev_items)} test={len(test_items)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = IntentDataset(train_items, tokenizer, args.max_length)
    dev_ds = IntentDataset(dev_items, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device)
    model = IntentClassifier(model_name=args.model_name, num_labels=len(INTENT_LABELS))
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_dev_f1 = -1
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attn, target = (x.to(device) for x in batch)
            optimizer.zero_grad()
            logits = model(input_ids, attn)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        # dev 评估
        dev_metrics = evaluate(model, dev_items, device, tokenizer, args.max_length)
        print(f"  epoch {epoch+1}/{args.epochs} loss={total_loss/len(train_loader):.4f} "
              f"dev_macroF1={dev_metrics['macro_f1']} dev_exact={dev_metrics['exact_match']}")
        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            save_dir = PROJECT_ROOT / "models" / "intent" / args.out_model
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model_state.pt")
            tokenizer.save_pretrained(str(save_dir))
    elapsed = time.time() - t0
    print(f"[OK] 训练完成，耗时 {elapsed:.0f}s，best_dev_macroF1={best_dev_f1:.4f}")

    # 用训练保存的模型在 test 上评测（不是默认 best_model）
    predictor = SmallModelPredictor(model_dir=str(PROJECT_ROOT / "models" / "intent" / args.out_model))
    test_preds = [predictor.predict(d.text) for d in test_items]
    test_true = [[l.value for l in d.labels] for d in test_items]
    test_metrics = intent_metrics(test_true, [p["labels"] for p in test_preds], INTENT_LABELS)

    # 输出报告
    reports_dir = PROJECT_ROOT / "evaluation" / "intent" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"classifier_baseline_{ts}"
    payload = {
        "meta": {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "model": args.model_name, "epochs": args.epochs, "lr": args.lr,
                 "device": str(device), "train": len(train_items), "dev": len(dev_items),
                 "test": len(test_items), "elapsed_s": round(elapsed, 1),
                 "best_dev_macro_f1": round(best_dev_f1, 4)},
        "dev_metrics": {"macro_f1": round(best_dev_f1, 4)},
        "test_metrics": test_metrics,
    }
    with open(reports_dir / f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(reports_dir / f"{base}.md", "w", encoding="utf-8") as f:
        f.write(build_md(payload))

    print(f"[OK] 报告 -> evaluation/intent/reports/{base}.md")
    print(f"  test Macro F1={test_metrics['macro_f1']} Micro F1={test_metrics['micro_f1']} Exact={test_metrics['exact_match']}")
    print(f"  test 每标签: { {k: v['f1'] for k, v in test_metrics['per_class'].items()} }")


def evaluate(model, items, device, tokenizer, max_length):
    model.eval()
    ds = IntentDataset(items, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=32)
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, attn, target = (x.to(device) for x in batch)
            logits = model(input_ids, attn).cpu()
            probs = torch.sigmoid(logits)
            for i in range(probs.size(0)):
                pred = [INTENT_LABELS[j] for j, p in enumerate(probs[i]) if p >= 0.5]
                gold = [INTENT_LABELS[j] for j, p in enumerate(target[i]) if p == 1.0]
                y_true.append(gold)
                y_pred.append(pred)
    return intent_metrics(y_true, y_pred, INTENT_LABELS)


def build_md(payload) -> str:
    m = payload["test_metrics"]
    L = ["# Intent Small Model Baseline", "", f"- 日期: {payload['meta']['date']}", f"- 模型: {payload['meta']['model']} epochs={payload['meta']['epochs']} lr={payload['meta']['lr']}",
         "", "## Test 指标", "", "| 指标 | 值 |", "|---|---|",
         f"| Macro F1 | {m['macro_f1']} |", f"| Micro F1 | {m['micro_f1']} |",
         f"| Exact Match | {m['exact_match']} |", f"| Hamming Loss | {m['hamming_loss']} |",
         "", "## Per-label F1", "", "| label | P | R | F1 |", "|---|---|---|---|"]
    for lab, d in m["per_class"].items():
        L.append(f"| {lab} | {d['precision']} | {d['recall']} | {d['f1']} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
