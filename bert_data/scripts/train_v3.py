#!/usr/bin/env python3
"""
风险语义识别 — v3 Baseline 训练脚本

使用 hfl/chinese-macbert-base 对 cssrs_lite_level (0-3) 四分类。

Usage:
    cd D:/project/project/bert_data
    set KMP_DUPLICATE_LIB_OK=TRUE
    python scripts/train_v3.py
"""

import os
import json
import sys
import time
import random
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

# 抑制 OpenMP 警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============= 配置 =============
class Config:
    # 路径
    train_path = "processed/train_v3.jsonl"
    dev_path = "processed/dev_v3.jsonl"
    test_path = "processed/test_v3.jsonl"
    output_dir = "models/v3_baseline"

    # 模型
    model_name = "hfl/chinese-macbert-base"
    num_labels = 4
    max_length = 128

    # 训练
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 5
    warmup_ratio = 0.1
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0

    # 日志
    logging_steps = 20
    eval_steps = 100
    save_steps = 500

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 随机种子
    seed = 42

    # 类别权重 (基于数据分布调整)
    use_class_weights = True

    # 标签映射
    id2label = {0: "level_0", 1: "level_1", 2: "level_2", 3: "level_3"}
    label2id = {"level_0": 0, "level_1": 1, "level_2": 2, "level_3": 3}


cfg = Config()
cfg.output_dir = Path(cfg.output_dir)
cfg.output_dir.mkdir(parents=True, exist_ok=True)


# ============= 工具 =============
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_metrics(labels, preds):
    acc = accuracy_score(labels, preds)

    # 宏平均
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    # 加权平均
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    # 每类指标
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2, 3], zero_division=0
    )

    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "weighted_f1": f1_weighted,
        "per_class_f1": {f"level_{i}": f1_per[i] for i in range(len(f1_per))},
        "per_class_precision": {f"level_{i}": p_per[i] for i in range(len(p_per))},
        "per_class_recall": {f"level_{i}": r_per[i] for i in range(len(r_per))},
        "confusion_matrix": cm.tolist(),
    }


# ============= 数据集 =============
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
        label = r.get("cssrs_lite_level", 0)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "sample_id": r.get("sample_id", ""),
        }


# ============= 训练 =============
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        print(f"  设备: {cfg.device}")

        # 加载模型和 tokenizer
        print(f"  加载模型: {cfg.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=cfg.num_labels,
            id2label=cfg.id2label,
            label2id=cfg.label2id,
        ).to(self.device)

        # 加载数据
        print(f"  加载数据...")
        train_records = load_jsonl(cfg.train_path)
        dev_records = load_jsonl(cfg.dev_path)
        test_records = load_jsonl(cfg.test_path)
        print(f"    train: {len(train_records)}")
        print(f"    dev:   {len(dev_records)}")
        print(f"    test:  {len(test_records)}")

        # 打印训练集分布
        train_levels = Counter(r["cssrs_lite_level"] for r in train_records)
        print(f"    train levels: {dict(sorted(train_levels.items()))}")

        self.train_dataset = RiskDataset(train_records, self.tokenizer, cfg.max_length)
        self.dev_dataset = RiskDataset(dev_records, self.tokenizer, cfg.max_length)
        self.test_dataset = RiskDataset(test_records, self.tokenizer, cfg.max_length)

        # 类别权重
        if cfg.use_class_weights:
            total = sum(train_levels.values())
            weights = [total / max(1, train_levels.get(i, 0)) for i in range(cfg.num_labels)]
            weights = [w / sum(weights) * cfg.num_labels for w in weights]  # 归一化
            self.class_weights = torch.tensor(weights, dtype=torch.float).to(self.device)
            print(f"    类别权重: {dict(zip([f'level_{i}' for i in range(cfg.num_labels)], [f'{w:.2f}' for w in weights]))}")
        else:
            self.class_weights = None

    def train(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  开始训练 v3 Baseline")
        print(f"  模型: {cfg.model_name}")
        print(f"  批次大小: {cfg.batch_size}")
        print(f"  学习率: {cfg.learning_rate}")
        print(f"  Epochs: {cfg.num_epochs}")
        print(f"{'='*60}\n")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        dev_loader = DataLoader(
            self.dev_dataset,
            batch_size=cfg.batch_size,
        )

        # 优化器
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)

        # 学习率调度
        total_steps = len(train_loader) * cfg.num_epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # 损失函数
        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        global_step = 0
        best_f1 = 0.0
        best_epoch = 0
        train_losses = []
        eval_results = []

        for epoch in range(1, cfg.num_epochs + 1):
            self.model.train()
            total_loss = 0
            epoch_start = time.time()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                loss = loss_fn(logits, labels)

                # 权重正则化
                if self.class_weights is not None:
                    loss = loss * self.class_weights.mean()

                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % cfg.logging_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"    Epoch {epoch}/{cfg.num_epochs} | Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr_now:.2e}")

            # Epoch 结束时评估
            avg_epoch_loss = total_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)

            dev_metrics = self.evaluate(dev_loader, "dev")
            eval_results.append(dev_metrics)

            epoch_time = time.time() - epoch_start
            print(f"\n  Epoch {epoch} 完成 ({epoch_time:.1f}s)")
            print(f"    Train Loss: {avg_epoch_loss:.4f}")
            print(f"    Dev Accuracy: {dev_metrics['accuracy']:.4f}")
            print(f"    Dev Macro F1: {dev_metrics['macro_f1']:.4f}")
            print(f"    Per-class F1: {dev_metrics['per_class_f1']}")
            print()

            # 保存最优模型
            if dev_metrics["macro_f1"] > best_f1:
                best_f1 = dev_metrics["macro_f1"]
                best_epoch = epoch
                self.save_model(cfg.output_dir / "best_model")
                print(f"    -> 新最优模型 (F1={best_f1:.4f}), 已保存\n")

        # 加载最优模型做最终测试
        print(f"  加载最优模型 (Epoch {best_epoch}, F1={best_f1:.4f}) 测试...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.output_dir / "best_model",
        ).to(self.device)

        test_loader = DataLoader(self.test_dataset, batch_size=cfg.batch_size)
        test_metrics = self.evaluate(test_loader, "test")

        # 输出最终报告
        self.print_final_report(test_metrics, train_losses, eval_results, best_epoch)

        # 保存训练配置和结果
        self.save_results(test_metrics, best_epoch, train_losses, eval_results)

    def evaluate(self, loader, name="dev"):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return compute_metrics(all_labels, all_preds)

    def save_model(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # 保存配置
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        config["id2label"] = {int(k): v for k, v in cfg.id2label.items()}
        config["label2id"] = cfg.label2id
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def print_final_report(self, test_metrics, train_losses, eval_results, best_epoch):
        print(f"\n{'='*60}")
        print(f"  v3 Baseline 训练完成")
        print(f"{'='*60}")
        print(f"\n  [Train Loss]")
        for i, loss in enumerate(train_losses):
            print(f"    Epoch {i+1}: {loss:.4f}")

        print(f"\n  [Dev 最优] Epoch {best_epoch}")
        best = eval_results[best_epoch - 1]
        print(f"    Accuracy:  {best['accuracy']:.4f}")
        print(f"    Macro F1:  {best['macro_f1']:.4f}")
        print(f"    Per-class F1:")
        for lv in range(4):
            print(f"      level_{lv}: {best['per_class_f1'][f'level_{lv}']:.4f}")

        print(f"\n  [Test 最终结果]")
        print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"    Macro F1:  {test_metrics['macro_f1']:.4f}")
        print(f"    Weighted F1: {test_metrics['weighted_f1']:.4f}")
        print(f"    Per-class:")
        print(f"    {'Level':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print(f"    {'-'*46}")
        for lv in range(4):
            p = test_metrics['per_class_precision'][f'level_{lv}']
            r = test_metrics['per_class_recall'][f'level_{lv}']
            f = test_metrics['per_class_f1'][f'level_{lv}']
            print(f"    {'level_'+str(lv):<10} {p:<12.4f} {r:<12.4f} {f:<12.4f}")

        print(f"\n  [Test Confusion Matrix]")
        cm = test_metrics['confusion_matrix']
        print(f"    {'':<10} {'pred_0':<8} {'pred_1':<8} {'pred_2':<8} {'pred_3':<8}")
        for i, row in enumerate(cm):
            print(f"    {'true_'+str(i):<10} {row[0]:<8} {row[1]:<8} {row[2]:<8} {row[3]:<8}")

        # 总体评估
        print(f"\n  [综合评估]")
        weighted_f1 = test_metrics['weighted_f1']
        macro_f1 = test_metrics['macro_f1']
        if weighted_f1 >= 0.75:
            print(f"    Weighted F1={weighted_f1:.3f} >= 0.75: ✅ 达标")
        else:
            print(f"    Weighted F1={weighted_f1:.3f} < 0.75: ⚠️ 低于目标线")
        if macro_f1 >= 0.65:
            print(f"    Macro F1={macro_f1:.3f} >= 0.65: ✅ 各等级均衡性达标")
        else:
            print(f"    Macro F1={macro_f1:.3f} < 0.65: ⚠️ level_0 或 level_3 可能偏低")
        # 混淆矩阵分析
        cm = np.array(test_metrics['confusion_matrix'])
        total = cm.sum()
        off_diag = total - np.trace(cm)
        if off_diag / total < 0.2:
            print(f"    混淆率 {off_diag/total*100:.1f}% < 20%: ✅ 跨等级混淆可控")
        else:
            print(f"    混淆率 {off_diag/total*100:.1f}% >= 20%: ⚠️ 跨等级混淆偏多")
        print()

    def save_results(self, test_metrics, best_epoch, train_losses, eval_results):
        results = {
            "config": {
                "model_name": cfg.model_name,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "num_epochs": cfg.num_epochs,
                "max_length": cfg.max_length,
                "weight_decay": cfg.weight_decay,
                "use_class_weights": cfg.use_class_weights,
            },
            "train_losses": train_losses,
            "best_epoch": best_epoch,
            "dev_best": eval_results[best_epoch - 1],
            "test": test_metrics,
        }
        with open(cfg.output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  结果已保存到 {cfg.output_dir / 'results.json'}")

        # 保存分类报告文本
        with open(cfg.output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write("v3 Baseline Classification Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Model: {cfg.model_name}\n")
            f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"Macro F1: {test_metrics['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {test_metrics['weighted_f1']:.4f}\n\n")
            f.write("Per-class:\n")
            f.write(f"{'Level':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
            f.write("-" * 46 + "\n")
            for lv in range(4):
                p = test_metrics['per_class_precision'][f'level_{lv}']
                r = test_metrics['per_class_recall'][f'level_{lv}']
                f1 = test_metrics['per_class_f1'][f'level_{lv}']
                f.write(f"{'level_'+str(lv):<10} {p:<12.4f} {r:<12.4f} {f1:<12.4f}\n")
            f.write("\nConfusion Matrix:\n")
            for row in test_metrics['confusion_matrix']:
                f.write(f"  {row}\n")


# ============= 主函数 =============
def main():
    print("=" * 60)
    print("  风险语义识别 - v3 Baseline 训练")
    print("=" * 60)

    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
