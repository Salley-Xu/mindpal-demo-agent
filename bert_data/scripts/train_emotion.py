#!/usr/bin/env python3
"""
train_emotion.py — 训练 BERT 情绪分类模型

架构：
  BERT backbone (hfl/chinese-macbert-base)
    └── classifier → 8 类情绪 (neutral, positive, anxiety, stress, sadness, anger, confusion, helplessness)
"""

import os, json, random
import numpy as np
from pathlib import Path
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# ====== 标签配置（5 类：confusion/stress/helplessness 合并到 anxiety） ======
NUM_EMOTIONS = 5
IDX_TO_EMOTION = {0: "neutral", 1: "positive", 2: "anxiety", 3: "sadness", 4: "anger"}
EMOTION_TO_IDX = {v: k for k, v in IDX_TO_EMOTION.items()}


# ====== 配置 ======
class Config:
    data_dir = "bert_data/processed_v4"
    output_dir = "bert_data/models/emotion_v1"

    model_name = "hfl/chinese-macbert-base"
    max_length = 128
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 10          # 增加轮数，配合早停
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    logging_steps = 20
    early_stop_patience = 3  # dev loss 连续 3 轮不降则停止
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42


cfg = Config()
cfg.output_dir = Path(cfg.output_dir)
cfg.output_dir.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ====== 模型 ======
class EmotionBERT(nn.Module):
    def __init__(self, model_name="hfl/chinese-macbert-base", num_emotions=NUM_EMOTIONS):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_emotions)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits


# ====== 数据集 ======
class EmotionDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=128):
        self.texts = []
        self.labels = []
        self.weights = []

        for r in records:
            label_str = r.get("emotion", "")
            if label_str not in EMOTION_TO_IDX:
                continue
            self.texts.append(r["text"])
            self.labels.append(EMOTION_TO_IDX[label_str])
            # source 权重：人工标注 > LLM 伪标签
            source_weight = {
                "manual": 1.5,
                "silver": 1.0,
                "llm_pseudo": 0.8,
            }.get(r.get("source", "llm_pseudo"), 1.0)
            # confidence 权重：高置信度样本权重略高
            conf_weight = min(1.0, (r.get("confidence", 0.8) or 0.8) * 1.2)
            self.weights.append(source_weight * conf_weight)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "weight": torch.tensor(self.weights[idx], dtype=torch.float),
        }


# ====== Focal Loss ======
class FocalLoss(nn.Module):
    """Focal Loss — 聚焦难分类样本，缓解类别不平衡"""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, labels):
        ce_loss = nn.functional.cross_entropy(logits, labels, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测概率
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# ====== 训练 ======
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        set_seed(cfg.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        # 加载数据
        train_records = load_jsonl(Path(cfg.data_dir) / "emotion_train.jsonl")
        dev_records = load_jsonl(Path(cfg.data_dir) / "emotion_dev.jsonl")
        self.test_records = load_jsonl(Path(cfg.data_dir) / "emotion_test.jsonl")

        self.train_dataset = EmotionDataset(train_records, self.tokenizer, cfg.max_length)
        self.dev_dataset = EmotionDataset(dev_records, self.tokenizer, cfg.max_length)
        self.test_dataset = EmotionDataset(self.test_records, self.tokenizer, cfg.max_length)

        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg.batch_size, shuffle=True)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=cfg.batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=cfg.batch_size)

        # 类别权重（基于训练集频率）
        label_counts = Counter(self.train_dataset.labels)
        total = len(self.train_dataset.labels)
        class_weights = [total / max(1, label_counts[i]) for i in range(NUM_EMOTIONS)]
        mean_w = np.mean(class_weights)
        class_weights = [w / mean_w for w in class_weights]
        self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print(f"类别权重: { {IDX_TO_EMOTION[i]: round(w, 2) for i, w in enumerate(class_weights)} }")

        # 模型
        self.model = EmotionBERT(cfg.model_name).to(self.device)
        self.criterion = FocalLoss(weight=self.class_weights, gamma=2.0)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        total_steps = len(self.train_loader) * cfg.num_epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        self.best_dev_loss = float("inf")
        self.patience_counter = 0
        self.best_epoch = -1

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            if (step + 1) % self.cfg.logging_steps == 0:
                print(f"  Step {step+1}/{len(self.train_loader)} Loss: {loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def evaluate(self, loader, records=None):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_EMOTIONS))

        # 逐类指标
        per_class = {}
        for i in range(NUM_EMOTIONS):
            p, r, f, _ = precision_recall_fscore_support(
                [1 if x == i else 0 for x in all_labels],
                [1 if x == i else 0 for x in all_preds],
                average="binary", zero_division=0
            )
            per_class[IDX_TO_EMOTION[i]] = {
                "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            }

        return {
            "loss": round(avg_loss, 4),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }

    def train(self):
        print(f"训练集: {len(self.train_dataset)} 条")
        print(f"开发集: {len(self.dev_dataset)} 条")
        print(f"测试集: {len(self.test_dataset)} 条")
        print(f"设备: {self.device}")
        print()

        for epoch in range(1, self.cfg.num_epochs + 1):
            print(f"Epoch {epoch}/{self.cfg.num_epochs}")
            train_loss = self.train_epoch()
            dev_metrics = self.evaluate(self.dev_loader)
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Dev Loss: {dev_metrics['loss']} | Acc: {dev_metrics['accuracy']:.4f} | Macro F1: {dev_metrics['macro_f1']:.4f}")
            print(f"  Per-class F1: { {k: v['f1'] for k, v in dev_metrics['per_class'].items()} }")

            # 早停
            if dev_metrics["loss"] < self.best_dev_loss:
                self.best_dev_loss = dev_metrics["loss"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_model()
                print(f"  → 保存最佳模型 (epoch {epoch})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.early_stop_patience:
                    print(f"  早停触发 (连续 {self.cfg.early_stop_patience} 轮未改善)")
                    break
            print()

        print(f"最佳模型: epoch {self.best_epoch}")

    def save_model(self):
        self.model.eval()
        save_dir = self.cfg.output_dir / "best_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model_state.pt")
        self.tokenizer.save_pretrained(save_dir)
        print(f"模型已保存到: {save_dir}")

    def test(self):
        """在测试集上评估最佳模型"""
        # 重新加载最佳模型
        save_dir = self.cfg.output_dir / "best_model"
        model = EmotionBERT(self.cfg.model_name).to(self.device)
        state_dict = torch.load(save_dir / "model_state.pt", map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        self.model = model

        test_metrics = self.evaluate(self.test_loader, self.test_records)
        print("=" * 50)
        print("测试集结果:")
        print(f"  Loss: {test_metrics['loss']} | Acc: {test_metrics['accuracy']:.4f} | Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"  混淆矩阵:\n{np.array(test_metrics['confusion_matrix'])}")
        print("\n  逐类指标:")
        for label, metrics in sorted(test_metrics["per_class"].items()):
            print(f"    {label:15s} P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")

        # 保存结果
        results = {
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.cfg).items() if not k.startswith("_")},
            "best_epoch": self.best_epoch,
            "test_metrics": test_metrics,
        }
        with open(self.cfg.output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {self.cfg.output_dir / 'results.json'}")
        return test_metrics


# ====== 入口 ======
if __name__ == "__main__":
    import sys
    # 支持从项目根目录或 bert_data/scripts 目录运行
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # bert_data/scripts/../../ → agent_version/
    os.chdir(project_root)
    print(f"工作目录: {os.getcwd()}")

    trainer = Trainer(cfg)
    trainer.train()
    trainer.test()
