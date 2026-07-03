#!/usr/bin/env python3
"""
v4.2: 多任务 + binary 阈值校准 + 规则兜底

架构：
  BERT backbone
    ├── classifier_4 → 4分类
    └── classifier_2 → high_risk_binary

推理融合（三级）:
  1. 强规则命中 → level_3
  2. binary_prob > threshold AND pred_4 < 2 → level_2
  3. 否则 → pred_4
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
    output_dir = "models/v4_2_multitask_calibrated"

    model_name = "hfl/chinese-macbert-base"
    max_length = 128
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 6
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    logging_steps = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # 多任务权重
    loss_weight_main = 1.0
    loss_weight_aux = 0.3

    # 阈值校准（在 dev 上搜索后写入）
    binary_threshold = 0.50  # 初始值，校准后覆盖


cfg = Config()
cfg.output_dir = Path(cfg.output_dir)
cfg.output_dir.mkdir(parents=True, exist_ok=True)

# ====== 规则兜底 ======
HIGH_RISK_OVERRIDE_PATTERNS = [
    r'正在?割[腕脉手]', r'割[腕脉手]中', r'在割[腕脉手]',
    r'刚割[完过]?[腕脉手]', r'第[一二三四五六七八九十\d]+[次回]割[腕脉手]',
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


def rule_match(text):
    for pat in HIGH_RISK_OVERRIDE_PATTERNS:
        if re.search(pat, text):
            return True
    return False


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
class MultiTaskBERT(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
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
            "labels_4": torch.tensor(label_4, dtype=torch.long),
            "labels_2": torch.tensor(label_2, dtype=torch.long),
        }


# ====== 训练器 ======
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        print(f"  Device: {cfg.device}")
        print(f"  Model: {cfg.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = MultiTaskBERT(cfg.model_name).to(self.device)

        train_records = load_jsonl(cfg.train_path)
        self.dev_records = load_jsonl(cfg.dev_path)
        self.test_records = load_jsonl(cfg.test_path)

        self.train_ds = RiskDataset(train_records, self.tokenizer, cfg.max_length)
        self.dev_ds = RiskDataset(self.dev_records, self.tokenizer, cfg.max_length)
        self.test_ds = RiskDataset(self.test_records, self.tokenizer, cfg.max_length)

        train_levels = Counter(r["cssrs_lite_level"] for r in train_records)
        print(f"  Train: {len(train_records)}, levels: {dict(sorted(train_levels.items()))}")
        print(f"  Dev: {len(self.dev_records)}, Test: {len(self.test_records)}")

        # 4分类权重
        total = sum(train_levels.values())
        w4 = [total / max(1, train_levels.get(i, 0)) for i in range(4)]
        w4 = [w / sum(w4) * 4 for w in w4]
        self.weights_4 = torch.tensor(w4, dtype=torch.float).to(self.device)

        # 2分类权重
        hi = sum(1 for r in train_records if r["cssrs_lite_level"] >= 2)
        lo = len(train_records) - hi
        w2 = [len(train_records)/max(1,lo)*2, len(train_records)/max(1,hi)*2]
        ws = sum(w2)
        self.weights_2 = torch.tensor([w2[0]/ws*2, w2[1]/ws*2], dtype=torch.float).to(self.device)

    def train(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  v4.2 Training (multi-task + threshold calibration)")
        print(f"  loss = {cfg.loss_weight_main} * CE_4 + {cfg.loss_weight_aux} * CE_2")
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

        loss_fn_4 = nn.CrossEntropyLoss(weight=self.weights_4)
        loss_fn_2 = nn.CrossEntropyLoss(weight=self.weights_2)

        best_f1 = 0.0
        best_epoch = 0
        global_step = 0

        for epoch in range(1, cfg.num_epochs + 1):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(train_loader):
                iids = batch["input_ids"].to(self.device)
                am = batch["attention_mask"].to(self.device)
                l4 = batch["labels_4"].to(self.device)
                l2 = batch["labels_2"].to(self.device)

                logits_4, logits_2 = self.model(iids, am)
                loss = cfg.loss_weight_main * loss_fn_4(logits_4, l4) \
                     + cfg.loss_weight_aux * loss_fn_2(logits_2, l2)

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
            dev_metrics = self.evaluate_dev()
            print(f"\n  Epoch {epoch} | Train Loss: {avg_loss:.4f}")
            print(f"    Dev 4class Acc: {dev_metrics['acc_4']:.4f} | Macro F1: {dev_metrics['macro_f1_4']:.4f}")
            print(f"    Dev binary Acc: {dev_metrics['acc_2']:.4f} | Binary F1: {dev_metrics['binary_f1']:.4f}")
            print(f"    Dev t2->p0: {dev_metrics['t2_to_p0']}")

            if dev_metrics["macro_f1_4"] > best_f1:
                best_f1 = dev_metrics["macro_f1_4"]
                best_epoch = epoch
                self.save_model(cfg.output_dir / "best_model")
                print(f"    -> New best (F1={best_f1:.4f})")
            print()

        # ---- 阈值校准 ----
        print(f"\n{'='*60}")
        print(f"  Binary threshold calibration on Dev")
        print(f"{'='*60}")
        self.load_model(cfg.output_dir / "best_model")
        best_threshold, calib_report = self.calibrate_threshold()
        cfg.binary_threshold = best_threshold

        # ---- 测试 ----
        print(f"\n{'='*60}")
        print(f"  Test with threshold={cfg.binary_threshold:.2f} + rules")
        print(f"{'='*60}")
        test_results = self.evaluate_test()

        # ---- 保存 ----
        self.save_results(test_results, best_epoch, calib_report)
        print(f"\n  Results saved to {cfg.output_dir / 'results.json'}")

    def evaluate_dev(self):
        """Dev 评估（纯模型，无融合无规则）"""
        self.model.eval()
        loader = DataLoader(self.dev_ds, batch_size=self.cfg.batch_size)
        all_p4, all_l4 = [], []
        all_p2, all_l2 = [], []

        with torch.no_grad():
            for batch in loader:
                iids = batch["input_ids"].to(self.device)
                am = batch["attention_mask"].to(self.device)
                l4 = batch["labels_4"].to(self.device)
                l2 = batch["labels_2"].to(self.device)

                logits_4, logits_2 = self.model(iids, am)
                all_p4.extend(torch.argmax(logits_4, dim=-1).cpu().numpy())
                all_l4.extend(l4.cpu().numpy())
                all_p2.extend(torch.argmax(logits_2, dim=-1).cpu().numpy())
                all_l2.extend(l2.cpu().numpy())

        cm = confusion_matrix(all_l4, all_p4, labels=[0,1,2,3])
        f1_macro = np.mean(precision_recall_fscore_support(
            all_l4, all_p4, average=None, labels=[0,1,2,3], zero_division=0)[2])
        _, _, f1_2, _ = precision_recall_fscore_support(
            all_l2, all_p2, average="binary", pos_label=1, zero_division=0)

        return {
            "acc_4": accuracy_score(all_l4, all_p4),
            "macro_f1_4": f1_macro,
            "acc_2": accuracy_score(all_l2, all_p2),
            "binary_f1": f1_2,
            "t2_to_p0": int(cm[2][0]) if cm.shape[0] > 2 else 0,
            "cm_4": cm.tolist(),
        }

    def calibrate_threshold(self):
        """
        在 Dev 上搜索最优 binary threshold。
        目标：最小化 t2→p0，同时控制 t0→p3 和 t3 漏报。
        """
        # 收集所有 dev 样本的模型输出
        self.model.eval()
        loader = DataLoader(self.dev_ds, batch_size=self.cfg.batch_size)
        all_p4, all_bin_probs = [], []
        all_texts, all_true = [], []

        with torch.no_grad():
            for batch in loader:
                iids = batch["input_ids"].to(self.device)
                am = batch["attention_mask"].to(self.device)
                # items from dataset
                # need texts and true labels
                break  # We'll use a different approach

        # 直接用 dev_records 配合模型推理
        all_p4, all_bin_probs, all_true = [], [], []
        for i in range(0, len(self.dev_records), self.cfg.batch_size):
            batch = self.dev_records[i:i+self.cfg.batch_size]
            texts = [r["text"] for r in batch]
            enc = self.tokenizer(texts, truncation=True, padding=True,
                                 max_length=self.cfg.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits_4, logits_2 = self.model(enc["input_ids"], enc["attention_mask"])
            probs_2 = torch.softmax(logits_2, dim=-1)
            all_p4.extend(torch.argmax(logits_4, dim=-1).cpu().numpy())
            all_bin_probs.extend(probs_2[:, 1].cpu().numpy())
            all_true.extend([r["cssrs_lite_level"] for r in batch])

        # 搜索最优阈值
        best_score = -1
        best_threshold = 0.5
        results = []

        for th in np.arange(0.20, 0.81, 0.02):
            th = round(th, 2)
            # 融合推理
            fused = []
            for p4, bp in zip(all_p4, all_bin_probs):
                if p4 < 2 and bp > th:
                    fused.append(2)
                else:
                    fused.append(p4)

            cm = confusion_matrix(all_true, fused, labels=[0,1,2,3])
            t2_to_p0 = cm[2][0] if cm.shape[0] > 2 else 0
            t0_to_p3 = cm[0][3] if cm.shape[0] > 0 else 0
            t3_fn = sum([cm[3][0] + cm[3][1]]) if cm.shape[0] > 3 else 0

            # 目标分数 = -t2_to_p0 - 0.5*t0_to_p3 - 0.5*t3_fn
            # t2→p0 权重最高，t0→p3 和 t3 fn 权重减半
            score = -t2_to_p0 - 0.5 * t0_to_p3 - 0.5 * t3_fn

            results.append((th, t2_to_p0, t0_to_p3, t3_fn, score))

            if score > best_score:
                best_score = score
                best_threshold = th

        # 打印校准报告
        print(f"\n  Threshold scan results:")
        print(f"  {'th':<6} {'t2->0':<8} {'t0->3':<8} {'t3_fn':<8} {'score':<8}")
        for th, t20, t03, t3fn, sc in results:
            marker = " <--" if th == best_threshold else ""
            if th in [round(best_threshold-0.06,2), best_threshold, round(best_threshold+0.06,2)]:
                print(f"  {th:<6.2f} {t20:<8} {t03:<8} {t3fn:<8} {sc:<8.2f}{marker}")

        print(f"\n  Best threshold: {best_threshold:.2f} (score={best_score:.2f})")

        # 用最优阈值重新计算 dev 融合指标
        fused = []
        upgrades = 0
        for p4, bp in zip(all_p4, all_bin_probs):
            if p4 < 2 and bp > best_threshold:
                fused.append(2)
                upgrades += 1
            else:
                fused.append(p4)

        cm = confusion_matrix(all_true, fused, labels=[0,1,2,3])
        _, _, f1s, _ = precision_recall_fscore_support(
            all_true, fused, average=None, labels=[0,1,2,3], zero_division=0)
        print(f"  Dev fusion results (th={best_threshold:.2f}):")
        print(f"    Upgrades: {upgrades}")
        print(f"    Per-class F1: {[f'{f1s[i]:.4f}' for i in range(4)]}")
        print(f"    t2->p0: {cm[2][0] if cm.shape[0] > 2 else 'N/A'}")

        return best_threshold, {
            "best_threshold": best_threshold,
            "dev_results": results,
            "upgrades": upgrades,
            "dev_fused_cm": cm.tolist(),
        }

    def evaluate_test(self):
        """Test 评估（含融合 + 规则兜底）"""
        th = self.cfg.binary_threshold
        all_true, all_pred = [], []

        for i in range(0, len(self.test_records), self.cfg.batch_size):
            batch = self.test_records[i:i+self.cfg.batch_size]
            texts = [r["text"] for r in batch]
            enc = self.tokenizer(texts, truncation=True, padding=True,
                                 max_length=self.cfg.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits_4, logits_2 = self.model(enc["input_ids"], enc["attention_mask"])
            preds_4 = torch.argmax(logits_4, dim=-1).cpu().numpy()
            bin_probs = torch.softmax(logits_2, dim=-1)[:, 1].cpu().numpy()

            for j, r in enumerate(batch):
                true_lv = r["cssrs_lite_level"]
                p4 = preds_4[j]
                bp = bin_probs[j]

                # 三级融合
                # 1) 强规则 → level_3
                if rule_match(r["text"]):
                    final_p = 3
                # 2) binary 判风险 + 4分类 < 2 → level_2
                elif bp > th and p4 < 2:
                    final_p = 2
                # 3) 保持原预测
                else:
                    final_p = p4

                all_true.append(true_lv)
                all_pred.append(final_p)

        cm = confusion_matrix(all_true, all_pred, labels=[0,1,2,3])
        p, r, f1, _ = precision_recall_fscore_support(
            all_true, all_pred, average=None, labels=[0,1,2,3], zero_division=0)
        f1_m = np.mean(f1)
        f1_w = precision_recall_fscore_support(
            all_true, all_pred, average="weighted", zero_division=0)[2]
        acc = accuracy_score(all_true, all_pred)

        print(f"\n  Test Results (th={th:.2f} + rules):")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Macro F1:  {f1_m:.4f}")
        print(f"    Weighted:  {f1_w:.4f}")
        print(f"    Per-class:")
        print(f"    {'Level':<10} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        for i in range(4):
            print(f"    {'level_'+str(i):<10} {p[i]:<8.4f} {r[i]:<8.4f} {f1[i]:<8.4f}")
        print(f"    CM:")
        print(f"    {'':<10} {'p0':<5} {'p1':<5} {'p2':<5} {'p3':<5}")
        for i in range(4):
            print(f"    {'t'+str(i):<10} {cm[i][0]:<5} {cm[i][1]:<5} {cm[i][2]:<5} {cm[i][3]:<5}")
        print(f"    t2->p0: {cm[2][0] if cm.shape[0] > 2 else 'N/A'}")

        # 单独统计各级贡献
        rule_caught = sum(1 for r in self.test_records if rule_match(r["text"]))
        print(f"    Rule triggers: {rule_caught}")

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

    def save_results(self, test_results, best_epoch, calib_report):
        all_results = {
            "config": {
                "model_name": cfg.model_name,
                "architecture": "MultiTaskBERT",
                "loss_weight_main": cfg.loss_weight_main,
                "loss_weight_aux": cfg.loss_weight_aux,
                "binary_threshold": cfg.binary_threshold,
                "num_rules": len(HIGH_RISK_OVERRIDE_PATTERNS),
                "fusion": "rules_first -> binary_fusion -> 4class_fallback",
            },
            "best_epoch": best_epoch,
            "calibration": calib_report,
            "test": test_results,
        }
        with open(cfg.output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)


def main():
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
