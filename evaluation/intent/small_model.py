# -*- coding: utf-8 -*-
"""
Intent Small Model（Phase 1 Task 1.6）。

架构：Encoder(macbert-base) → [CLS] → Dropout → Linear(10) → Sigmoid（推理时）
Loss：BCEWithLogitsLoss（multi-label，不用 softmax）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.intent.intent_schema import INTENT_LABELS  # noqa: E402
from evaluation.intent.predictors import IntentPredictor  # noqa: E402


class IntentClassifier(nn.Module):
    """multi-label intent 分类器。"""

    def __init__(self, model_name: str = "hfl/chinese-macbert-base", num_labels: int = 10):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class SmallModelPredictor(IntentPredictor):
    """加载已训练模型进行推理。"""

    name = "classifier"

    def __init__(self, model_dir: Optional[str] = None, device: str = "cpu",
                 thresholds: Optional[List[float]] = None):
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        if model_dir is None:
            model_dir = str(PROJECT_ROOT / "models" / "intent" / "best_model")
        self.model_dir = Path(model_dir)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = IntentClassifier(num_labels=len(INTENT_LABELS))
        state = torch.load(str(self.model_dir / "model_state.pt"), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        # 阈值（未校准默认 0.5）
        self.thresholds = thresholds or [0.5] * len(INTENT_LABELS)

    @torch.no_grad()
    def get_logits(self, text: str) -> List[float]:
        """返回原始 logits（供校准 / 阈值优化）。"""
        enc = self.tokenizer(text, truncation=True, padding="max_length",
                             max_length=128, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        logits = self.model(input_ids, attn).cpu().squeeze().tolist()
        return logits

    @torch.no_grad()
    def predict(self, text: str, context: Optional[List[dict]] = None) -> dict:
        enc = self.tokenizer(text, truncation=True, padding="max_length",
                             max_length=128, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        logits = self.model(input_ids, attn).cpu()
        probs = torch.sigmoid(logits).squeeze().tolist()
        labels = [INTENT_LABELS[i] for i, p in enumerate(probs)
                  if p >= self.thresholds[i]]
        conf = max(probs) if probs else 0.0
        return {
            "labels": labels,
            "confidence": round(conf, 4),
            "is_open_set": max(probs) < 0.1 if probs else True,  # 后续由 open_set 模块接管
            "source": "classifier",
            "label_scores": {INTENT_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
        }
