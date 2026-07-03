#!/usr/bin/env python3
"""
v4.1: 多任务模型 — 4分类主任务 + high_risk_binary 辅助头

架构：
  BERT backbone (shared)
    ├── classifier_4 → 4分类 (cssrs_lite_level 0-3)
    └── classifier_2 → 二分类 (high_risk: 0=level_0/1, 1=level_2/3)

推理融合：
  if pred_4 < 2 and pred_2 == 1:
      final_pred = 2  # 至少提升到 level_2

目标：修复 t2→p0 漏报（level_2 被压到 level_0）
"""

import os, json, re, sys, time, random
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
    train_path = "processed/train_v3.jsonl"
    dev_path = "processed/dev_v3.jsonl"
    test_path = "processed/test_v3.jsonl"
    output_dir = "models/v4_1_multitask"

    model_name = "hfl/chinese-macbert-base"
    max_length = 128
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 6
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    logging_steps = 20
    eval_steps = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # 多任务权重
    loss_weight_main = 1.0      # 4分类 loss 权重
    loss_weight_aux = 0.3       # 二分类 loss 权重

    # 推理融合：二分类判风险但4分类判低时，最低升到 level_2
    fusion_min_level = 2


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


# ====== 多任务模型 ======
class MultiTaskBERT(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size  # 768
        self.classifier_4 = nn.Linear(hidden_size, 4)   # 主任务
        self.classifier_2 = nn.Linear(hidden_size, 2)   # 辅助任务

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits_4 = self.classifier_4(pooled)
        logits_2 = self.classifier_2(pooled)
        return logits_4, logits_2


# ====== 数据集 ======
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
        label_2 = 1 if label_4 >= 2 else 0  # high_risk_binary

        encoding = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels_4": torch.tensor(label_4, dtype=torch.long),
            "labels_2": torch.tensor(label_2, dtype=torch.long),
        }


# ====== 训练器 ======
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        print(f"  设备: {cfg.device}")
        print(f"  模型: {cfg.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = MultiTaskBERT(cfg.model_name).to(self.device)

        train_records = load_jsonl(cfg.train_path)
        dev_records = load_jsonl(cfg.dev_path)
        test_records = load_jsonl(cfg.test_path)

        self.train_ds = RiskDataset(train_records, self.tokenizer, cfg.max_length)
        self.dev_ds = RiskDataset(dev_records, self.tokenizer, cfg.max_length)
        self.test_ds = RiskDataset(test_records, self.tokenizer, cfg.max_length)
        self.test_records = test_records

        train_levels = Counter(r["cssrs_lite_level"] for r in train_records)
        print(f"    train: {len(train_records)}, levels: {dict(sorted(train_levels.items()))}")

        # 类别权重（4分类）
        total = sum(train_levels.values())
        weights_4 = [total / max(1, train_levels.get(i, 0)) for i in range(4)]
        weights_4 = [w / sum(weights_4) * 4 for w in weights_4]
        self.weights_4 = torch.tensor(weights_4, dtype=torch.float).to(self.device)
        print(f"    4分类权重: {[f'{w:.2f}' for w in weights_4]}")

        # 二分类权重
        high = sum(1 for r in train_records if r["cssrs_lite_level"] >= 2)
        low = len(train_records) - high
        w0 = len(train_records) / max(1, low) * 2
        w1 = len(train_records) / max(1, high) * 2
        w_sum = w0 + w1
        self.weights_2 = torch.tensor([w0/w_sum*2, w1/w_sum*2], dtype=torch.float).to(self.device)
        print(f"    2分类权重: low={w0/w_sum*2:.2f}, high={w1/w_sum*2:.2f}")

    def train(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  v4.1 多任务训练")
        print(f"  loss = {cfg.loss_weight_main} * CE_4 + {cfg.loss_weight_aux} * CE_2")
        print(f"{'='*60}\n")

        train_loader = DataLoader(self.train_ds, batch_size=cfg.batch_size, shuffle=True)
        dev_loader = DataLoader(self.dev_ds, batch_size=cfg.batch_size)

        # 优化器
        no_decay = ["bias", "LayerNorm.weight"]
        opt_grouped = [
            {"params": [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(opt_grouped, lr=cfg.learning_rate)

        total_steps = len(train_loader) * cfg.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * cfg.warmup_ratio),
            num_training_steps=total_steps,
        )

        loss_fn_4 = nn.CrossEntropyLoss(weight=self.weights_4)
        loss_fn_2 = nn.CrossEntropyLoss(weight=self.weights_2)

        best_f1 = 0.0
        best_epoch = 0
        global_step = 0

        for epoch in range(1, cfg.num_epochs + 1):
            self.model.train()
            total_loss = 0

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_4 = batch["labels_4"].to(self.device)
                labels_2 = batch["labels_2"].to(self.device)

                logits_4, logits_2 = self.model(input_ids, attention_mask)
                loss = (cfg.loss_weight_main * loss_fn_4(logits_4, labels_4)
                      + cfg.loss_weight_aux * loss_fn_2(logits_2, labels_2))

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

            # 评估
            dev_metrics = self.evaluate(dev_loader)
            print(f"\n  Epoch {epoch} | Train Loss: {avg_loss:.4f}")
            print(f"    Dev 4class Acc: {dev_metrics['accuracy_4']:.4f} | Macro F1: {dev_metrics['macro_f1_4']:.4f}")
            print(f"    Dev binary Acc: {dev_metrics['accuracy_2']:.4f} | Binary F1: {dev_metrics['binary_f1']:.4f}")

            if dev_metrics["macro_f1_4"] > best_f1:
                best_f1 = dev_metrics["macro_f1_4"]
                best_epoch = epoch
                self.save_model(cfg.output_dir / "best_model")
                print(f"    -> 新最优模型 (F1={best_f1:.4f})")

            print()

        # 最终测试
        print(f"  加载最优模型 (Epoch {best_epoch}) 测试...")
        self.load_model(cfg.output_dir / "best_model")
        test_loader = DataLoader(self.test_ds, batch_size=cfg.batch_size)
        test_metrics = self.evaluate(test_loader, is_test=True)
        self.print_report(test_metrics)

        # 带规则兜底的最终指标
        self.final_eval_with_rules()

        # 保存结果
        self.save_results(test_metrics, best_epoch)

    def evaluate(self, loader, is_test=False):
        self.model.eval()
        all_preds_4, all_labels_4 = [], []
        all_preds_2, all_labels_2 = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_4 = batch["labels_4"].to(self.device)
                labels_2 = batch["labels_2"].to(self.device)

                logits_4, logits_2 = self.model(input_ids, attention_mask)

                preds_4 = torch.argmax(logits_4, dim=-1)
                preds_2 = torch.argmax(logits_2, dim=-1)

                all_preds_4.extend(preds_4.cpu().numpy())
                all_labels_4.extend(labels_4.cpu().numpy())
                all_preds_2.extend(preds_2.cpu().numpy())
                all_labels_2.extend(labels_2.cpu().numpy())

        # 4分类指标
        acc_4 = accuracy_score(all_labels_4, all_preds_4)
        p4, r4, f1_4, _ = precision_recall_fscore_support(
            all_labels_4, all_preds_4, average=None, labels=[0,1,2,3], zero_division=0)
        f1_macro_4 = np.mean(f1_4)
        cm_4 = confusion_matrix(all_labels_4, all_preds_4, labels=[0,1,2,3])

        # 二分类指标
        acc_2 = accuracy_score(all_labels_2, all_preds_2)
        _, _, f1_2, _ = precision_recall_fscore_support(
            all_labels_2, all_preds_2, average="binary", pos_label=1, zero_division=0)

        results_4 = {
            "accuracy_4": acc_4, "macro_f1_4": f1_macro_4,
            "per_class_f1": {f"level_{i}": float(f1_4[i]) for i in range(4)},
            "per_class_precision": {f"level_{i}": float(p4[i]) for i in range(4)},
            "per_class_recall": {f"level_{i}": float(r4[i]) for i in range(4)},
            "confusion_matrix_4": cm_4.tolist(),
        }

        # 如果是测试集，做融合推理
        if is_test:
            fused_preds = []
            for p4, p2 in zip(all_preds_4, all_preds_2):
                if p4 < 2 and p2 == 1:
                    fused_preds.append(cfg.fusion_min_level)
                else:
                    fused_preds.append(p4)

            acc_fused = accuracy_score(all_labels_4, fused_preds)
            pf, rf, f1_f, _ = precision_recall_fscore_support(
                all_labels_4, fused_preds, average=None, labels=[0,1,2,3], zero_division=0)
            f1_macro_fused = np.mean(f1_f)
            cm_fused = confusion_matrix(all_labels_4, fused_preds, labels=[0,1,2,3])

            results_4["fused_accuracy"] = acc_fused
            results_4["fused_macro_f1"] = f1_macro_fused
            results_4["fused_per_class_f1"] = {f"level_{i}": float(f1_f[i]) for i in range(4)}
            results_4["fused_confusion_matrix"] = cm_fused.tolist()
            # 显示融合效果
            upgrades = sum(1 for p4, p2 in zip(all_preds_4, all_preds_2) if p4 < 2 and p2 == 1)
            print(f"    融合: {upgrades} 条被升级, 4分类 Macro F1={f1_macro_4:.4f} → 融合后={f1_macro_fused:.4f}")

        results_4["accuracy_2"] = acc_2
        results_4["binary_f1"] = f1_2
        return results_4

    def save_model(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model_state.pt")
        self.tokenizer.save_pretrained(path)
        # 保存配置
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_name": cfg.model_name,
                "architecture": "MultiTaskBERT",
                "tasks": ["cssrs_lite_level_4class", "high_risk_binary"],
                "id2label": {"0": "level_0", "1": "level_1", "2": "level_2", "3": "level_3"},
                "fusion_min_level": cfg.fusion_min_level,
            }, f, ensure_ascii=False, indent=2)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(Path(path) / "model_state.pt", map_location=self.device))

    def final_eval_with_rules(self):
        """带规则兜底的最终评测"""
        # 对测试记录做推理 + 规则兜底
        records = self.test_records
        all_true, all_pred = [], []

        for i in range(0, len(records), cfg.batch_size):
            batch = records[i:i+cfg.batch_size]
            texts = [r["text"] for r in batch]
            enc = self.tokenizer(texts, truncation=True, padding=True, max_length=cfg.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits_4, logits_2 = self.model(enc["input_ids"], enc["attention_mask"])
            preds_4 = torch.argmax(logits_4, dim=-1).cpu().numpy()
            preds_2 = torch.argmax(logits_2, dim=-1).cpu().numpy()

            for j, r in enumerate(batch):
                true_lv = r["cssrs_lite_level"]
                p4 = preds_4[j]
                p2 = preds_2[j]

                # 融合
                if p4 < 2 and p2 == 1:
                    final_p = cfg.fusion_min_level
                else:
                    final_p = p4

                # 规则兜底
                if final_p < 3:
                    from fix_and_supplement import match_high_risk_patterns
                    # inline rules
                    _OVERRIDE = [
                        r'正在?割[腕脉手]', r'割[腕脉手]中', r'在割[腕脉手]',
                        r'刚割[完过]?[腕脉手]', r'第[\d一二三四五六七八九十]+[次回]割[腕脉手]',
                        r'尝试割[腕脉手]', r'准备跳[楼河崖桥海]',
                        r'这就[去要].*(死|自杀|跳|割|走)',
                        r'现在就去(死|自杀|跳|割)', r'马上就去(死|自杀|跳|割)',
                        r'(永别|拜拜啦?走饭|再见世界|再也不见).*(跳|割|死|自杀)',
                        r'(安眠药|头孢|酒).{0,15}(跳|割腕|上吊|烧炭)',
                        r'(割腕|上吊|烧炭).{0,15}(跳|安眠药|头孢|酒)',
                        r'昨晚割[腕脉手]了', r'割[腕脉手]失败', r'又割[腕脉手]',
                        r'一起.*(烧炭|跳|割腕|上吊|安眠药)', r'约.*(烧炭|跳|割腕|上吊)',
                        r'准备.*(后事|遗书|遗照|自杀)', r'遗书.*发出去',
                    ]
                    for pat in _OVERRIDE:
                        if re.search(pat, r["text"]):
                            final_p = 3
                            break

                all_true.append(true_lv)
                all_pred.append(final_p)

        cm = confusion_matrix(all_true, all_pred, labels=[0,1,2,3])
        p, r, f1, _ = precision_recall_fscore_support(all_true, all_pred, average=None, labels=[0,1,2,3], zero_division=0)
        f1_m = np.mean(f1)
        f1_w = precision_recall_fscore_support(all_true, all_pred, average="weighted", zero_division=0)[2]
        acc = accuracy_score(all_true, all_pred)

        print(f"\n  [最终: 多任务融合 + 规则兜底]")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Macro F1:  {f1_m:.4f}")
        print(f"    Weighted:  {f1_w:.4f}")
        print(f"    Per-class:")
        print(f"    {'Level':<10} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        for i in range(4):
            print(f"    {'level_'+str(i):<10} {p[i]:<8.4f} {r[i]:<8.4f} {f1[i]:<8.4f}")
        print(f"    Confusion Matrix:")
        print(f"    {'':<10} {'p0':<5} {'p1':<5} {'p2':<5} {'p3':<5}")
        for i in range(4):
            print(f"    {'t'+str(i):<10} {cm[i][0]:<5} {cm[i][1]:<5} {cm[i][2]:<5} {cm[i][3]:<5}")

        # 保存最终结果
        self.final_results = {
            "accuracy": acc, "macro_f1": f1_m, "weighted_f1": f1_w,
            "per_class_f1": {f"level_{i}": float(f1[i]) for i in range(4)},
            "per_class_precision": {f"level_{i}": float(p[i]) for i in range(4)},
            "per_class_recall": {f"level_{i}": float(r[i]) for i in range(4)},
            "confusion_matrix": cm.tolist(),
        }

    def print_report(self, metrics):
        print(f"\n{'='*60}")
        print(f"  v4.1 多任务测试结果")
        print(f"{'='*60}")
        print(f"\n  4分类:")
        print(f"    Accuracy:  {metrics['accuracy_4']:.4f}")
        print(f"    Macro F1:  {metrics['macro_f1_4']:.4f}")
        print(f"    Per-class F1:")
        for lv in range(4):
            print(f"      level_{lv}: {metrics['per_class_f1'][f'level_{lv}']:.4f}")
        print(f"\n  二分类:")
        print(f"    Accuracy:  {metrics['accuracy_2']:.4f}")
        print(f"    Binary F1: {metrics['binary_f1']:.4f}")

        if "fused_accuracy" in metrics:
            print(f"\n  融合推理 (多头融合):")
            print(f"    Accuracy:  {metrics['fused_accuracy']:.4f}")
            print(f"    Macro F1:  {metrics['fused_macro_f1']:.4f}")
            print(f"    Per-class F1:")
            for lv in range(4):
                print(f"      level_{lv}: {metrics['fused_per_class_f1'][f'level_{lv}']:.4f}")
            print(f"    4分类 CM:")
            self._print_cm(metrics['confusion_matrix_4'])
            print(f"    融合 CM:")
            self._print_cm(metrics.get('fused_confusion_matrix', [[0]]))

    def _print_cm(self, cm):
        print(f"    {'':<10} {'p0':<5} {'p1':<5} {'p2':<5} {'p3':<5}")
        for i in range(len(cm)):
            print(f"    {'t'+str(i):<10} {cm[i][0]:<5} {cm[i][1]:<5} {cm[i][2]:<5} {cm[i][3]:<5}")

    def save_results(self, metrics, best_epoch):
        all_results = {
            "config": {
                "model_name": cfg.model_name,
                "architecture": "MultiTaskBERT",
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "num_epochs": cfg.num_epochs,
                "loss_weight_main": cfg.loss_weight_main,
                "loss_weight_aux": cfg.loss_weight_aux,
                "fusion_min_level": cfg.fusion_min_level,
            },
            "best_epoch": best_epoch,
            "test_4class": {
                "accuracy_4": metrics["accuracy_4"],
                "macro_f1_4": metrics["macro_f1_4"],
                "per_class": {lv: metrics['per_class_f1'][f'level_{lv}'] for lv in range(4)},
            },
            "test_binary": {
                "accuracy_2": metrics["accuracy_2"],
                "binary_f1": metrics["binary_f1"],
            },
        }
        if "fused_accuracy" in metrics:
            all_results["test_fused"] = {
                "accuracy": metrics["fused_accuracy"],
                "macro_f1": metrics["fused_macro_f1"],
            }
        if hasattr(self, "final_results"):
            all_results["test_final_with_rules"] = self.final_results

        with open(cfg.output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存到 {cfg.output_dir / 'results.json'}")


# ====== 主函数 ======
def main():
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
