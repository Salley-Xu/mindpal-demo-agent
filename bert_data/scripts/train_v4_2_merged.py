#!/usr/bin/env python3
"""
v4.2 合并训练版 — 在 SOS-1K + 对话域合成数据上训练。

与 train_v4_2.py 仅数据路径不同，模型架构完全一致。
"""
import os, sys
# 将项目根目录加入路径，使得 backend/ 模块可导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json, re, time, random
import numpy as np
from pathlib import Path
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# ====== 配置 ======
class Config:
    # 使用合并训练集（SOS-1K 936 + 合成对话 600）
    train_path = "bert_data/processed_v4/train_merged.jsonl"
    dev_path = "bert_data/processed/dev_v3.jsonl"
    test_path = "bert_data/processed/test_v3.jsonl"
    output_dir = "bert_data/models/v4_2_merged_v2"

    model_name = "hfl/chinese-macbert-base"
    max_length = 128
    batch_size = 16
    learning_rate = 1e-5
    weight_decay = 0.01
    num_epochs = 10
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    logging_steps = 30
    device = "cpu"
    seed = 42

    # 多任务权重
    loss_weight_main = 1.0
    loss_weight_aux = 0.3
    binary_threshold = 0.50


cfg = Config()
cfg.output_dir = Path(cfg.output_dir)
cfg.output_dir.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT_DIR = Path(PROJECT_ROOT)


# ====== 规则兜底 ======
HIGH_RISK_OVERRIDE_PATTERNS = [
    re.compile(r"正在?割[腕脉手]"), re.compile(r"割[腕脉手]中"),
    re.compile(r"在割[腕脉手]"), re.compile(r"刚割[完过]?[腕脉手]"),
    re.compile(r"第[一二三四五六七八九十\d]+[次回]割[腕脉手]"),
    re.compile(r"尝试割[腕脉手]"), re.compile(r"准备跳[楼河崖桥海]"),
    re.compile(r"这就[去要].*(死|自杀|跳|割|走)"),
    re.compile(r"现在就去(死|自杀|跳|割)"), re.compile(r"马上就去(死|自杀|跳|割)"),
    re.compile(r"(永别|拜拜啦?走饭|再见世界|再也不见).*(跳|割|死|自杀)"),
    re.compile(r"(安眠药|头孢|酒).{0,15}(跳|割腕|上吊|烧炭)"),
    re.compile(r"(割腕|上吊|烧炭).{0,15}(跳|安眠药|头孢|酒)"),
    re.compile(r"昨晚割[腕脉手]了"), re.compile(r"割[腕脉手]失败"), re.compile(r"又割[腕脉手]"),
    re.compile(r"一起.*(烧炭|跳|割腕|上吊|安眠药)"), re.compile(r"约.*(烧炭|跳|割腕|上吊)"),
    re.compile(r"准备.*(后事|遗书|遗照|自杀)"), re.compile(r"遗书.*发出去"),
]


def rule_match(text):
    return any(p.search(text) for p in HIGH_RISK_OVERRIDE_PATTERNS)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_data(path):
    full_path = PROJECT_ROOT_DIR / path
    records = []
    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ====== 模型 ======
class MultiTaskBERT(nn.Module):
    def __init__(self, model_name="hfl/chinese-macbert-base"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = nn.Dropout(0.1)
        hidden = self.bert.config.hidden_size
        self.classifier_4 = nn.Linear(hidden, 4)
        self.classifier_2 = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier_4(pooled), self.classifier_2(pooled)


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


# ====== 训练器 ======
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        print(f"  Device: {cfg.device}")
        print(f"  Model: {cfg.model_name} (MultiTask: 4class + binary)")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=True)
        self.model = MultiTaskBERT(cfg.model_name).to(self.device)

        train_records = load_data(cfg.train_path)
        dev_records = load_data(cfg.dev_path)
        test_records = load_data(cfg.test_path)

        self.train_ds = RiskDataset(train_records, self.tokenizer, cfg.max_length)
        self.dev_ds = RiskDataset(dev_records, self.tokenizer, cfg.max_length)
        self.test_ds = RiskDataset(test_records, self.tokenizer, cfg.max_length)

        train_levels = Counter(r["cssrs_lite_level"] for r in train_records)
        print(f"  Train: {len(train_records)}, levels: {dict(sorted(train_levels.items()))}")
        print(f"  Dev: {len(dev_records)}, Test: {len(test_records)}")

    def train(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  v4.2 Merged Training")
        print(f"  loss = CE(4class) + {cfg.loss_weight_aux} * CE(binary)")
        print(f"{'='*60}")

        train_loader = DataLoader(self.train_ds, batch_size=cfg.batch_size, shuffle=True)

        no_decay = ["bias", "LayerNorm.weight"]
        opt_grouped = [
            {"params": [p for n,p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
            {"params": [p for n,p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(opt_grouped, lr=cfg.learning_rate)
        total_steps = len(train_loader) * cfg.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps*cfg.warmup_ratio),
            num_training_steps=total_steps)

        # 类别权重（Level 0 偏少需要提权）
        _tl = Counter(r["cssrs_lite_level"] for r in self.train_ds.records)
        _total = sum(_tl.values())
        _w4 = torch.tensor([_total / (4 * _tl.get(i, 1)) for i in range(4)], dtype=torch.float).to(self.device)
        print(f"  4类权重: {_w4.cpu().tolist()}")
        loss_fn_4 = nn.CrossEntropyLoss(weight=_w4)
        loss_fn_2 = nn.CrossEntropyLoss()

        best_f1 = 0.0
        best_epoch = 0
        global_step = 0

        for epoch in range(1, cfg.num_epochs + 1):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(train_loader):
                iids = batch["input_ids"].to(self.device)
                am = batch["attention_mask"].to(self.device)
                l4 = batch["label_4"].to(self.device)
                l2 = batch["label_2"].to(self.device)

                logits_4, logits_2 = self.model(iids, am)
                loss = loss_fn_4(logits_4, l4) + cfg.loss_weight_aux * loss_fn_2(logits_2, l2)

                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.logging_steps == 0:
                    print(f"    Epoch {epoch}/{cfg.num_epochs} | Step {global_step} | Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            dev_metrics = self.evaluate(self.dev_ds)
            print(f"\n  Epoch {epoch} | Train Loss: {avg_loss:.4f}")
            print(f"    Dev Acc: {dev_metrics['acc']:.4f} | Macro F1: {dev_metrics['macro_f1']:.4f}")
            print(f"    Dev t2->p0: {dev_metrics['t2_to_p0']}")

            combined_score = dev_metrics["macro_f1"] - 0.05 * dev_metrics["t2_to_p0"]
            if combined_score > best_f1:
                best_f1 = combined_score
                best_epoch = epoch
                self.save_model(cfg.output_dir / "best_model")
                print(f"    -> New best (score={combined_score:.4f})")
            print()

        # ---- 阈值校准 ----
        print(f"\n{'='*60}")
        print(f"  Binary threshold calibration on Dev")
        print(f"{'='*60}")
        self.load_model(cfg.output_dir / "best_model")
        best_threshold, _ = self.calibrate_threshold()
        cfg.binary_threshold = best_threshold

        # ---- 测试 ----
        print(f"\n{'='*60}")
        print(f"  Test with threshold={cfg.binary_threshold:.2f} + rules")
        print(f"{'='*60}")
        test_results = self.evaluate_test()

        # ---- 保存 ----
        self.save_results(test_results, best_epoch, best_threshold)
        print(f"\n  Results saved to {cfg.output_dir / 'results.json'}")

    def predict_batch(self, dataset):
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size)
        all_preds_4, all_probs_2 = [], []
        for batch in loader:
            iids = batch["input_ids"].to(self.device)
            am = batch["attention_mask"].to(self.device)
            with torch.no_grad():
                logits_4, logits_2 = self.model(iids, am)
            preds_4 = torch.argmax(logits_4, dim=-1).cpu().tolist()
            probs_2 = torch.softmax(logits_2, dim=-1)[:, 1].cpu().tolist()
            all_preds_4.extend(preds_4)
            all_probs_2.extend(probs_2)
        return all_preds_4, all_probs_2

    def evaluate(self, dataset):
        preds_4, _ = self.predict_batch(dataset)
        labels = [r["cssrs_lite_level"] for r in dataset.records]
        cm = confusion_matrix(labels, preds_4, labels=[0, 1, 2, 3])
        f1_m = np.mean(precision_recall_fscore_support(
            labels, preds_4, average=None, labels=[0, 1, 2, 3], zero_division=0)[2])
        return {
            "acc": accuracy_score(labels, preds_4),
            "macro_f1": f1_m,
            "t2_to_p0": int(cm[2][0]) if cm.shape[0] > 2 else 0,
        }

    def calibrate_threshold(self):
        preds_4, probs_2 = self.predict_batch(self.dev_ds)
        labels = [r["cssrs_lite_level"] for r in self.dev_ds.records]
        bl_labels = [1 if l >= 2 else 0 for l in labels]

        best_f1 = 0.0
        best_threshold = 0.50
        for th in np.arange(0.30, 0.81, 0.02):
            bl_preds = [1 if p >= th else 0 for p in probs_2]
            f1 = precision_recall_fscore_support(bl_labels, bl_preds, average="binary", zero_division=0)[2]
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = th

        print(f"  Best binary threshold: {best_threshold:.2f} (binary F1={best_f1:.4f})")
        return best_threshold, {}

    def evaluate_test(self):
        th = self.cfg.binary_threshold
        preds_4, probs_2 = self.predict_batch(self.test_ds)
        labels = [r["cssrs_lite_level"] for r in self.test_ds.records]
        texts = [r["text"] for r in self.test_ds.records]

        # 三级融合
        all_preds = []
        for i in range(len(texts)):
            if rule_match(texts[i]):
                all_preds.append(3)
            elif probs_2[i] > th and preds_4[i] < 2:
                all_preds.append(2)
            else:
                all_preds.append(preds_4[i])

        cm = confusion_matrix(labels, all_preds, labels=[0, 1, 2, 3])
        p, r, f1, _ = precision_recall_fscore_support(
            labels, all_preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
        f1_m = np.mean(f1)
        acc = accuracy_score(labels, all_preds)

        print(f"\n  Test Results (threshold={th:.2f} + rules):")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Macro F1:  {f1_m:.4f}")
        print(f"    Per-class:")
        print(f"    {'Level':<10} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        for i in range(4):
            print(f"    {'level_'+str(i):<10} {p[i]:<8.4f} {r[i]:<8.4f} {f1[i]:<8.4f}")
        print(f"    CM:")
        print(f"    {'':<10} {'p0':<5} {'p1':<5} {'p2':<5} {'p3':<5}")
        for i in range(4):
            print(f"    {'t'+str(i):<10} {cm[i][0]:<5} {cm[i][1]:<5} {cm[i][2]:<5} {cm[i][3]:<5}")
        print(f"    t2->p0: {cm[2][0] if cm.shape[0] > 2 else 'N/A'}")
        rule_caught = sum(1 for t in texts if rule_match(t))
        print(f"    Rule triggers: {rule_caught}")

        return {
            "accuracy": acc, "macro_f1": f1_m,
            "per_class_f1": {f"level_{i}": float(f1[i]) for i in range(4)},
            "per_class_precision": {f"level_{i}": float(p[i]) for i in range(4)},
            "per_class_recall": {f"level_{i}": float(r[i]) for i in range(4)},
            "confusion_matrix": cm.tolist(),
            "t2_to_p0": int(cm[2][0]) if cm.shape[0] > 2 else 0,
            "binary_threshold": th,
        }

    def save_model(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model_state.pt")
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        self.model.load_state_dict(
            torch.load(Path(path) / "model_state.pt", map_location=self.device))

    @staticmethod
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: Trainer._to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [Trainer._to_native(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return Trainer._to_native(obj.tolist())
        return obj

    def save_results(self, test_results, best_epoch, best_threshold):
        saved = Trainer._to_native({
            "config": {
                "model": "MultiTaskBERT",
                "train_data": "train_merged.jsonl (SOS-1K + synth_conversational)",
                "binary_threshold": best_threshold,
                "num_rules": len(HIGH_RISK_OVERRIDE_PATTERNS),
            },
            "best_epoch": best_epoch,
            "test": test_results,
        })
        with open(cfg.output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)


# ====== 主入口 ======
if __name__ == "__main__":
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    trainer.train()
