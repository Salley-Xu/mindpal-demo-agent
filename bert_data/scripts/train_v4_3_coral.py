#!/usr/bin/env python3
"""
v4.3: CORAL (Consistent Rank Logits) 有序回归

替换 v4.2 的 CrossEntropy 多分类 + binary 辅助头方案。

改进:
  1. CORAL loss 强制模型学习等级间的顺序关系 (0 < 1 < 2 < 3)
  2. 去掉 binary 辅助头（AUROC<0.5，已在前端禁用）
  3. 训练目标从 4 分类 softmax 改为 3 个二分类子任务：
     - 是否 ≥ Level 1
     - 是否 ≥ Level 2
     - 是否 ≥ Level 3
  4. 输出层从 4 神经元改为 3 神经元

用法:
    cd D:/project/project/bert_data
    set KMP_DUPLICATE_LIB_OK=TRUE
    python scripts/train_v4_3_coral.py
"""

import os, json, re, time, random
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
    output_dir = "models/v4_3_coral"

    model_name = "hfl/chinese-macbert-base"
    max_length = 128
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 8
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    logging_steps = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # CORAL 预测阈值（在 dev 上搜索后覆盖）
    coral_threshold = 0.50


cfg = Config()
cfg.output_dir = Path(cfg.output_dir)
cfg.output_dir.mkdir(parents=True, exist_ok=True)


# ====== 规则兜底（保持与 v4.2 一致） ======
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


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ====== CORAL 工具函数 ======

def level_to_coral(level_4: int) -> list:
    """
    将 4 级标签转为 CORAL 有序目标。

    level_0 → [0, 0, 0]  (不低于 Level 1)
    level_1 → [1, 0, 0]  (≥ Level 1，但 < Level 2)
    level_2 → [1, 1, 0]  (≥ Level 1 且 ≥ Level 2，但 < Level 3)
    level_3 → [1, 1, 1]  (≥ Level 1 且 ≥ Level 2 且 ≥ Level 3)
    """
    return [1.0 if level_4 > k else 0.0 for k in range(3)]


def coral_to_level(coral_probs: list, threshold: float = 0.50) -> int:
    """将 CORAL 概率转为 0-3 等级。"""
    level = 0
    for i in range(3):
        if coral_probs[i] >= threshold:
            level = i + 1
        else:
            break
    return level


# ====== 模型 ======
class CoralBERT(nn.Module):
    """
    CORAL 有序回归 BERT 模型。

    输出 3 个 logits，对应 3 个二分类子任务：
      logit_0: 是否 ≥ Level 1
      logit_1: 是否 ≥ Level 2
      logit_2: 是否 ≥ Level 3
    """
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        hidden = self.bert.config.hidden_size
        self.coral_output = nn.Linear(hidden, 3)  # 3 CORAL logits

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.coral_output(pooled)


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
        coral_target = level_to_coral(label_4)
        enc = self.tokenizer(text, truncation=True, padding="max_length",
                              max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "coral_target": torch.tensor(coral_target, dtype=torch.float),
            "label_4": torch.tensor(label_4, dtype=torch.long),
        }


# ====== CORAL 权重计算 ======
def compute_coral_weights(train_records):
    """
    对 3 个 COROL 子任务分别计算正负样本权重。
    每个子任务: 是否 ≥ level_k
    """
    n = len(train_records)
    weights = []
    for k in range(3):
        pos = sum(1 for r in train_records if r["cssrs_lite_level"] > k)
        neg = n - pos
        if pos == 0 or neg == 0:
            weights.append(1.0)
        else:
            w = n / (2.0 * pos), n / (2.0 * neg)  # pos_weight, neg_weight
            weights.append(w[1] / w[0])  # BCEWithLogitsLoss pos_weight
    return torch.tensor(weights, dtype=torch.float)


# ====== 训练器 ======
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        print(f"  Device: {cfg.device}")
        print(f"  Model: {cfg.model_name} (CORAL ordinal regression)")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=True)
        self.model = CoralBERT(cfg.model_name).to(self.device)

        train_records = load_jsonl(cfg.train_path)
        self.dev_records = load_jsonl(cfg.dev_path)
        self.test_records = load_jsonl(cfg.test_path)

        self.train_ds = RiskDataset(train_records, self.tokenizer, cfg.max_length)
        self.dev_ds = RiskDataset(self.dev_records, self.tokenizer, cfg.max_length)
        self.test_ds = RiskDataset(self.test_records, self.tokenizer, cfg.max_length)

        train_levels = Counter(r["cssrs_lite_level"] for r in train_records)
        print(f"  Train: {len(train_records)}, levels: {dict(sorted(train_levels.items()))}")
        print(f"  Dev: {len(self.dev_records)}, Test: {len(self.test_records)}")

        # CORAL 子任务权重
        self.coral_weights = compute_coral_weights(train_records).to(self.device)
        print(f"  CORAL pos_weights: {self.coral_weights.cpu().tolist()}")

    def train(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  v4.3 CORAL Training")
        print(f"  loss = BCEWithLogitsLoss(coral_logits, coral_target)")
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

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.coral_weights)

        best_f1 = 0.0
        best_epoch = 0
        global_step = 0

        for epoch in range(1, cfg.num_epochs + 1):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(train_loader):
                iids = batch["input_ids"].to(self.device)
                am = batch["attention_mask"].to(self.device)
                targets = batch["coral_target"].to(self.device)

                logits = self.model(iids, am)
                loss = loss_fn(logits, targets)

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
            dev_metrics = self.evaluate_dev(threshold=0.50)
            print(f"\n  Epoch {epoch} | Train Loss: {avg_loss:.4f}")
            print(f"    Dev Acc: {dev_metrics['acc']:.4f} | Macro F1: {dev_metrics['macro_f1']:.4f}")
            print(f"    Dev t2->p0: {dev_metrics['t2_to_p0']}")

            if dev_metrics["macro_f1"] > best_f1:
                best_f1 = dev_metrics["macro_f1"]
                best_epoch = epoch
                self.save_model(cfg.output_dir / "best_model")
                print(f"    -> New best (F1={best_f1:.4f})")
            print()

        # ---- 阈校准 ----
        print(f"\n{'='*60}")
        print(f"  CORAL threshold calibration on Dev")
        print(f"{'='*60}")
        self.load_model(cfg.output_dir / "best_model")
        best_threshold, calib_report = self.calibrate_threshold()
        cfg.coral_threshold = best_threshold

        # ---- 测试 ----
        print(f"\n{'='*60}")
        print(f"  Test with threshold={cfg.coral_threshold:.2f} + rules")
        print(f"{'='*60}")
        test_results = self.evaluate_test()

        # ---- 保存 ----
        self.save_results(test_results, best_epoch, calib_report)
        print(f"\n  Results saved to {cfg.output_dir / 'results.json'}")

    def predict_coral(self, texts):
        """批量预测，返回 CORAL 概率 + 4 级预测。"""
        self.model.eval()
        all_probs, all_preds = [], []
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = texts[i:i+self.cfg.batch_size]
            enc = self.tokenizer(batch, truncation=True, padding=True,
                                  max_length=self.cfg.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(enc["input_ids"], enc["attention_mask"])
            probs = torch.sigmoid(logits).cpu().numpy()
            for row in probs:
                all_probs.append(row.tolist())
                all_preds.append(coral_to_level(row))
        return all_probs, all_preds

    def evaluate_dev(self, threshold=0.50):
        """Dev 评估（纯 CORAL，无规则融合）。"""
        texts = [r["text"] for r in self.dev_records]
        labels = [r["cssrs_lite_level"] for r in self.dev_records]
        _, preds = self.predict_coral(texts)

        cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
        f1_m = np.mean(precision_recall_fscore_support(
            labels, preds, average=None, labels=[0, 1, 2, 3], zero_division=0)[2])

        return {
            "acc": accuracy_score(labels, preds),
            "macro_f1": f1_m,
            "t2_to_p0": int(cm[2][0]) if cm.shape[0] > 2 else 0,
        }

    def calibrate_threshold(self):
        """在 Dev 上搜索最优 CORAL 阈值。"""
        texts = [r["text"] for r in self.dev_records]
        labels = [r["cssrs_lite_level"] for r in self.dev_records]
        probs, _ = self.predict_coral(texts)

        best_f1 = 0.0
        best_threshold = 0.50
        results = []

        for th in np.arange(0.20, 0.81, 0.02):
            th = round(th, 2)
            preds = [coral_to_level(p, th) for p in probs]
            _, _, f1s, _ = precision_recall_fscore_support(
                labels, preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
            macro_f1 = np.mean(f1s)
            cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
            t2_to_p0 = cm[2][0] if cm.shape[0] > 2 else 0
            t0_to_p3 = cm[0][3] if cm.shape[0] > 0 else 0
            # 组合评分：macro F1 优先，t2→p0 和 t0→p3 惩罚
            score = macro_f1 - 0.1 * t2_to_p0 - 0.1 * t0_to_p3
            results.append((th, macro_f1, t2_to_p0, t0_to_p3, score))
            if score > best_f1:
                best_f1 = score
                best_threshold = th

        print(f"\n  Threshold scan:")
        print(f"  Best threshold: {best_threshold:.2f} (score={best_f1:.2f})")

        # 输出最优阈值下的详细指标
        final_preds = [coral_to_level(p, best_threshold) for p in probs]
        cm = confusion_matrix(labels, final_preds, labels=[0, 1, 2, 3])
        _, _, f1s, _ = precision_recall_fscore_support(
            labels, final_preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
        print(f"  Dev CORAL results (th={best_threshold:.2f}):")
        print(f"    Per-class F1: {[f'{f1s[i]:.4f}' for i in range(4)]}")
        print(f"    t2->p0: {cm[2][0]}")

        return best_threshold, {
            "best_threshold": best_threshold,
            "dev_results": results,
            "dev_fused_cm": cm.tolist(),
        }

    def evaluate_test(self):
        """Test 评估（CORAL + 规则兜底）。"""
        th = self.cfg.coral_threshold
        texts = [r["text"] for r in self.test_records]
        labels = [r["cssrs_lite_level"] for r in self.test_records]
        probs, preds_4 = self.predict_coral(texts)

        all_preds = []
        for i, text in enumerate(texts):
            # 规则兜底（优先级最高）
            if rule_match(text):
                all_preds.append(3)
            else:
                # CORAL 预测
                all_preds.append(coral_to_level(probs[i], th))

        cm = confusion_matrix(labels, all_preds, labels=[0, 1, 2, 3])
        p, r, f1, _ = precision_recall_fscore_support(
            labels, all_preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
        f1_m = np.mean(f1)
        f1_w = precision_recall_fscore_support(
            labels, all_preds, average="weighted", zero_division=0)[2]
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

        rule_caught = sum(1 for r in self.test_records if rule_match(r["text"]))
        print(f"    Rule triggers: {rule_caught}")

        # Level 2 预测统计
        pred_levels = Counter(all_preds)
        print(f"    Predicted distribution: {dict(sorted(pred_levels.items()))}")

        return {
            "accuracy": acc, "macro_f1": f1_m, "weighted_f1": f1_w,
            "per_class_f1": {f"level_{i}": float(f1[i]) for i in range(4)},
            "per_class_precision": {f"level_{i}": float(p[i]) for i in range(4)},
            "per_class_recall": {f"level_{i}": float(r[i]) for i in range(4)},
            "confusion_matrix": cm.tolist(),
            "t2_to_p0": int(cm[2][0]) if cm.shape[0] > 2 else 0,
            "t3_fn": int(cm[3][0] + cm[3][1]) if cm.shape[0] > 3 else 0,
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
        """递归将 numpy 类型转为 Python 原生类型。"""
        if isinstance(obj, dict):
            return {k: Trainer._to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Trainer._to_native(v) for v in obj]
        if isinstance(obj, tuple):
            return [Trainer._to_native(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return Trainer._to_native(obj.tolist())
        return obj

    def save_results(self, test_results, best_epoch, calib_report):
        saved = Trainer._to_native({
            "config": {
                "model": "CoralBERT",
                "coral_threshold": self.cfg.coral_threshold,
                "num_rules": len(HIGH_RISK_OVERRIDE_PATTERNS),
            },
            "best_epoch": best_epoch,
            "calibration": calib_report,
            "test": {
                "accuracy": test_results["accuracy"],
                "macro_f1": test_results["macro_f1"],
                "weighted_f1": test_results["weighted_f1"],
                "per_class_f1": test_results["per_class_f1"],
                "per_class_precision": test_results["per_class_precision"],
                "per_class_recall": test_results["per_class_recall"],
                "confusion_matrix": test_results["confusion_matrix"],
                "t2_to_p0": test_results["t2_to_p0"],
                "t3_fn": test_results["t3_fn"],
                "rule_triggers": int(sum(
                    1 for r in self.test_records if rule_match(r["text"])
                )),
            },
        })
        with open(self.cfg.output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)


# ====== 主入口 ======
if __name__ == "__main__":
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    trainer.train()
