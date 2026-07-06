"""
bert_emotion_predictor.py — BERT 情绪分类推理封装

模型架构：EmotionBERT（hfl/chinese-macbert-base + 8 类分类头）
标签体系：neutral, positive, anxiety, stress, sadness, anger, confusion, helplessness
"""

import json
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# 5 类规范化情绪标签（confusion/stress/helplessness 合并到 anxiety）
NUM_EMOTIONS = 5
IDX_TO_EMOTION = {
    0: "neutral",
    1: "positive",
    2: "anxiety",
    3: "sadness",
    4: "anger",
}
EMOTION_TO_CHINESE = {
    "neutral": "中性",
    "positive": "快乐",
    "anxiety": "焦虑",
    "sadness": "抑郁",
    "anger": "愤怒",
    # 兼容旧名
    "中性": "中性", "快乐": "快乐", "焦虑": "焦虑",
    "抑郁": "抑郁", "愤怒": "愤怒",
    # 合并标签保留映射
    "stress": "焦虑", "confusion": "焦虑", "helplessness": "焦虑",
    "压力": "焦虑", "困惑": "焦虑", "无助": "焦虑",
}


class EmotionBERT(nn.Module):
    """单任务 BERT 情绪分类模型（5 类：neutral/positive/anxiety/sadness/anger）"""

    def __init__(self, model_name: str = "hfl/chinese-macbert-base", num_emotions: int = NUM_EMOTIONS):
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


class BertEmotionPredictor:
    """BERT 情绪分类推理封装"""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.6,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 加载模型
        self.model = EmotionBERT(model_name="hfl/chinese-macbert-base")
        state_path = os.path.join(model_path, "model_state.pt")
        if os.path.exists(state_path):
            state_dict = torch.load(state_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            logger.info(f"加载情绪模型权重: {state_path}")
        else:
            logger.warning(f"未找到模型权重: {state_path}，使用未初始化模型")

        self.model.to(self.device)
        self.model.eval()

        # 加载 Temperature Scaling 参数（可选）
        self.temperature = 1.0
        temp_path = os.path.join(model_path, "temperature.json")
        if os.path.exists(temp_path):
            with open(temp_path, "r") as f:
                temp_data = json.load(f)
            self.temperature = temp_data.get("temperature", 1.0)
            logger.info(f"Temperature Scaling: T={self.temperature}")

    @torch.no_grad()
    def predict(self, text: str) -> tuple[str, float]:
        """
        预测情绪标签

        Args:
            text: 输入文本

        Returns:
            (emotion_label, confidence)
            emotion_label 为英文规范化标签（如 "stress", "anxiety"）
            confidence 为 softmax 概率（0~1）
        """
        if not text or not text.strip():
            return "neutral", 0.0

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Temperature Scaling
        scaled_logits = logits / self.temperature
        probabilities = torch.softmax(scaled_logits, dim=-1)
        confidence, pred_idx = torch.max(probabilities, dim=-1)

        label = IDX_TO_EMOTION[int(pred_idx)]
        return label, round(float(confidence), 4)

    @torch.no_grad()
    def predict_with_probs(self, text: str) -> dict:
        """
        预测情绪标签并返回完整概率分布

        Returns:
            {
                "label": "stress",
                "confidence": 0.85,
                "probabilities": {"neutral": 0.02, "stress": 0.85, ...},
                "probabilities_array": [0.02, 0.0, 0.02, 0.85, ...]
            }
        """
        if not text or not text.strip():
            probs = {k: 0.0 for k in IDX_TO_EMOTION.values()}
            probs["neutral"] = 1.0
            return {"label": "neutral", "confidence": 0.0, "probabilities": probs}

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(logits, dim=-1)[0]

        confidence, pred_idx = torch.max(probabilities, dim=-1)
        label = IDX_TO_EMOTION[int(pred_idx)]

        return {
            "label": label,
            "confidence": round(float(confidence), 4),
            "probabilities": {
                IDX_TO_EMOTION[i]: round(float(probabilities[i]), 4)
                for i in range(NUM_EMOTIONS)
            },
        }

    def predict_chinese(self, text: str) -> tuple[str, float]:
        """返回中文情绪标签（供前端展示和 prompts 使用）"""
        label, confidence = self.predict(text)
        return EMOTION_TO_CHINESE.get(label, "中性"), confidence
