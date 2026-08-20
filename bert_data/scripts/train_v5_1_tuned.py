#!/usr/bin/env python3
"""
v5.0 Risk 训练 — 在 v4.2 基础上用 Targeted Expansion（hard negatives + 隐式风险 + 安全负例）微调。

目标：修复对话域高 FPR（benign→L3）+ 隐式风险漏检。

数据：bert_data/processed_v4/train_v5_1_expansion.jsonl（1027 条）
起点：v4_2_multitask_calibrated（与 v4_2_domain_only_v2 相同基座）
低 LR + 少 epoch，防止灾难性遗忘。
"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "backend")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import json, random
import numpy as np
from pathlib import Path
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from bert_risk_predictor import MultiTaskBERT


class Config:
    train_path = "bert_data/processed_v4/train_v5_1_expansion.jsonl"
    output_dir = "bert_data/models/v5_1_tuned"
    base_checkpoint = "bert_data/models/v4_2_multitask_calibrated/best_model/model_state.pt"

    model_name = "hfl/chinese-macbert-base"
    max_length = 128
    batch_size = 8
    learning_rate = 3e-6  # 更低 LR，防遗忘
    weight_decay = 0.01
    num_epochs = 15
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    device = "cpu"
    seed = 42
    loss_weight_aux = 0.3


cfg = Config()
cfg.output_dir = Path(cfg.output_dir)
cfg.output_dir.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT_DIR = Path(PROJECT_ROOT)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path):
    records = []
    with open(PROJECT_ROOT_DIR / path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class RiskDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text = r.get("text", "")
        label_4 = r.get("cssrs_lite_level", 0)
        label_2 = 1 if label_4 >= 2 else 0
        enc = self.tokenizer(text, truncation=True, padding="max_length",
                              max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label_4": torch.tensor(label_4, dtype=torch.long),
            "label_2": torch.tensor(label_2, dtype=torch.long),
        }


def main():
    print("=" * 60)
    print("  v5.0 Targeted Expansion 微调")
    print(f"  从 {cfg.base_checkpoint} 出发")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=True)
    model = MultiTaskBERT(cfg.model_name)
    state = torch.load(cfg.base_checkpoint, map_location=cfg.device, weights_only=True)
    model.load_state_dict(state)
    model.to(cfg.device)
    print("  加载 checkpoint ✓")

    train_records = load_data(cfg.train_path)
    random.shuffle(train_records)
    split = int(len(train_records) * 0.85)
    train_data = train_records[:split]
    dev_data = train_records[split:]

    train_ds = RiskDataset(train_data, tokenizer, cfg.max_length)
    dev_ds = RiskDataset(dev_data, tokenizer, cfg.max_length)

    train_levels = Counter(r["cssrs_lite_level"] for r in train_data)
    dev_levels = Counter(r["cssrs_lite_level"] for r in dev_data)
    print(f"  训练: {len(train_data)}, levels: {dict(sorted(train_levels.items()))}")
    print(f"  验证: {len(dev_data)}, levels: {dict(sorted(dev_levels.items()))}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    no_decay = ["bias", "LayerNorm.weight"]
    opt_grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(opt_grouped, lr=cfg.learning_rate)
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps)

    _tl = Counter(r["cssrs_lite_level"] for r in train_data)
    _total = sum(_tl.values())
    _w4 = torch.tensor([_total / (4 * _tl.get(i, 1)) for i in range(4)], dtype=torch.float).to(cfg.device)
    loss_fn_4 = nn.CrossEntropyLoss(weight=_w4)
    loss_fn_2 = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            iids = batch["input_ids"].to(cfg.device)
            am = batch["attention_mask"].to(cfg.device)
            l4 = batch["label_4"].to(cfg.device)
            l2 = batch["label_2"].to(cfg.device)
            logits_4, logits_2 = model(iids, am)
            loss = loss_fn_4(logits_4, l4) + cfg.loss_weight_aux * loss_fn_2(logits_2, l2)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in DataLoader(dev_ds, batch_size=cfg.batch_size):
                iids = batch["input_ids"].to(cfg.device)
                am = batch["attention_mask"].to(cfg.device)
                logits_4, _ = model(iids, am)
                preds = torch.argmax(logits_4, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["label_4"].tolist())

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
        f1_m = np.mean(precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=[0, 1, 2, 3], zero_division=0)[2])
        acc = accuracy_score(all_labels, all_preds)

        print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | MF1: {f1_m:.4f} | t2→p0: {cm[2][0]}")

        if f1_m > best_f1:
            best_f1 = f1_m
            save_path = cfg.output_dir / "best_model"
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / "model_state.pt")
            tokenizer.save_pretrained(save_path)
            print(f"    → New best MF1={f1_m:.4f}, saved")

    print(f"\n  Done. Best dev MF1: {best_f1:.4f}")


if __name__ == "__main__":
    set_seed(cfg.seed)
    main()
