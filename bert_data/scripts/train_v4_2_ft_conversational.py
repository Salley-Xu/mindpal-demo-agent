#!/usr/bin/env python3
"""
v4.2 对话域 Fine-tune — 从原始 v4.2 checkpoint 出发，仅用合成对话数据微调。

目标：保留 SOS-1K 知识的同时，将模型适应到对话域。
"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "backend")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import json, re, random
import numpy as np
from pathlib import Path
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class Config:
    # 仅用合成对话数据 fine-tune
    train_path = "bert_data/processed_v4/risk_synth_conversational.jsonl"
    dev_path = "bert_data/processed/dev_v3.jsonl"  # 仍用 SOS-1K dev 做验证
    output_dir = "bert_data/models/v4_2_ft_conversational"
    base_checkpoint = "bert_data/models/v4_2_multitask_calibrated/best_model/model_state.pt"

    model_name = "hfl/chinese-macbert-base"
    max_length = 128
    batch_size = 8  # 小 batch（数据少）
    learning_rate = 1e-5  # 更低学习率（fine-tune）
    weight_decay = 0.01
    num_epochs = 16  # 更多 epoch（数据少）
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    logging_steps = 10
    device = "cpu"
    seed = 42
    loss_weight_main = 1.0
    loss_weight_aux = 0.3


cfg = Config()
cfg.output_dir = Path(cfg.output_dir)
cfg.output_dir.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT_DIR = Path(PROJECT_ROOT)


from bert_risk_predictor import MultiTaskBERT  # 复用原始架构
from risk_levels import LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3


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
    print("  v4.2 对话域 Fine-tune")
    print(f"  从 {cfg.base_checkpoint} 出发")
    print(f"  训练集: {cfg.train_path}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=True)
    model = MultiTaskBERT(cfg.model_name)
    model.to(cfg.device)

    # 加载原始 checkpoint
    state = torch.load(cfg.base_checkpoint, map_location=cfg.device, weights_only=True)
    model.load_state_dict(state)
    print(f"\n  加载 checkpoint: {cfg.base_checkpoint}")

    train_records = load_data(cfg.train_path)
    dev_records = load_data(cfg.dev_path)

    train_ds = RiskDataset(train_records, tokenizer, cfg.max_length)
    dev_ds = RiskDataset(dev_records, tokenizer, cfg.max_length)

    train_levels = Counter(r["cssrs_lite_level"] for r in train_records)
    print(f"  训练: {len(train_records)} 条, levels: {dict(sorted(train_levels.items()))}")
    print(f"  验证: {len(dev_records)} 条")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    no_decay = ["bias", "LayerNorm.weight"]
    opt_grouped = [
        {"params": [p for n,p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n,p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(opt_grouped, lr=cfg.learning_rate)
    total_steps = len(train_loader) * cfg.num_epochs
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps*cfg.warmup_ratio),
        num_training_steps=total_steps)

    loss_fn_4 = nn.CrossEntropyLoss()
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

        # Dev 评估
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

        cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
        f1_m = np.mean(precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=[0,1,2,3], zero_division=0)[2])
        acc = accuracy_score(all_labels, all_preds)

        print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Dev Acc: {acc:.4f} | Macro F1: {f1_m:.4f} | t2→p0: {cm[2][0]}")

        if f1_m > best_f1:
            best_f1 = f1_m
            save_path = cfg.output_dir / "best_model"
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / "model_state.pt")
            tokenizer.save_pretrained(save_path)
            print(f"    → New best, saved to {save_path}")

    print(f"\n  Fine-tune done. Best Macro F1: {best_f1:.4f}")
    print(f"  Model saved to: {cfg.output_dir / 'best_model'}")


if __name__ == "__main__":
    set_seed(cfg.seed)
    main()
