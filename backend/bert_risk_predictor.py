"""
BERT 风险预测模块

封装 v4.2 MultiTaskBERT 模型（4分类 + 二分类 + 规则兜底 + 阈值融合），
提供与 risk_evaluator 兼容的风险等级预测接口。

Usage:
    predictor = BertRiskPredictor("bert_data/models/v4_2_multitask_calibrated/best_model")
    result = predictor.predict("正在割腕")
    # => {"level": "level_3", "fusion_source": "rule_override", ...}
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from risk_levels import LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3, normalize_risk_level

logger = logging.getLogger(__name__)

# ============================================================
# 规则兜底：20 条高危 pattern（与 train_v4_2.py 一致）
# ============================================================
HIGH_RISK_OVERRIDE_PATTERNS = [
    # 进行式自伤行为
    re.compile(r"正在?割[腕脉手]"),
    re.compile(r"割[腕脉手]中"),
    re.compile(r"在割[腕脉手]"),
    re.compile(r"刚割[完过]?[腕脉手]"),
    re.compile(r"第[一二三四五六七八九十\d]+[次回]割[腕脉手]"),
    re.compile(r"尝试割[腕脉手]"),
    # 进行式跳楼
    re.compile(r"准备跳[楼河崖桥海]"),
    # 正在进行或即将行动
    re.compile(r"这就[去要].*(死|自杀|跳|割|走)"),
    re.compile(r"现在就去(死|自杀|跳|割)"),
    re.compile(r"马上就去(死|自杀|跳|割)"),
    # 明确告别+行动
    re.compile(r"(永别|拜拜啦?走饭|再见世界|再也不见).*(跳|割|死|自杀)"),
    # 综合方法描述
    re.compile(r"(安眠药|头孢|酒).{0,15}(跳|割腕|上吊|烧炭)"),
    re.compile(r"(割腕|上吊|烧炭).{0,15}(跳|安眠药|头孢|酒)"),
    # 有明确实际行为
    re.compile(r"昨晚割[腕脉手]了"),
    re.compile(r"割[腕脉手]失败"),
    re.compile(r"又割[腕脉手]"),
    # 约死
    re.compile(r"一起.*(烧炭|跳|割腕|上吊|安眠药)"),
    re.compile(r"约.*(烧炭|跳|割腕|上吊)"),
    # 准备结束生命
    re.compile(r"准备.*(后事|遗书|遗照|自杀)"),
    re.compile(r"遗书.*发出去"),
]


def rule_match(text: str) -> bool:
    """检查文本是否命中高危规则兜底。"""
    for pattern in HIGH_RISK_OVERRIDE_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ============================================================
# 模型架构（与 train_v4_2.py 一致）
# ============================================================
class MultiTaskBERT(nn.Module):
    """多任务 BERT 模型：共享 backbone + 4分类头 + 二分类头"""

    def __init__(self, model_name: str = "hfl/chinese-macbert-base"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        hidden = self.bert.config.hidden_size  # 768
        self.classifier_4 = nn.Linear(hidden, 4)
        self.classifier_2 = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier_4(pooled), self.classifier_2(pooled)


# ============================================================
# 主预测器
# ============================================================
class BertRiskPredictor:
    """
    BERT 风险预测器。

    封装 v4.2 MultiTaskBERT 模型，实现三级融合推理：
      1. 规则兜底（高危 pattern）→ level_3
      2. binary_prob > threshold 且 4分类 < 2 → level_2
      3. 否则 → 4分类预测
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        binary_threshold: float = 0.50,
        max_length: int = 128,
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.binary_threshold = binary_threshold
        self.max_length = max_length

        logger.info(
            "BERT 风险预测器初始化: model=%s device=%s threshold=%.2f",
            model_path, self.device, binary_threshold,
        )

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        logger.info("  Tokenizer 加载完成: %s", self.tokenizer.__class__.__name__)

        # 加载模型
        self.model = MultiTaskBERT("hfl/chinese-macbert-base")
        state_dict_path = self.model_path / "model_state.pt"
        if not state_dict_path.exists():
            raise FileNotFoundError(f"模型权重不存在: {state_dict_path}")
        self.model.load_state_dict(
            torch.load(str(state_dict_path), map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("  模型加载完成: %s (%.1f MB)", state_dict_path.name, state_dict_path.stat().st_size / 1e6)

    def predict(self, text: str) -> Dict:
        """
        对单条文本进行风险预测，返回融合后的结果。

        返回:
            level: str                  归一化风险等级 "level_0"~"level_3"
            level_4_prediction: int     4分类原始预测 0-3
            binary_probability: float   二分类高风险概率 0-1
            class_probabilities: List[float]  4类置信度 [p0,p1,p2,p3]
            fusion_source: str          融合来源
            rule_matched: bool          是否命中规则兜底
        """
        # 规则兜底（优先级最高）
        if rule_match(text):
            return {
                "level": LEVEL_3,
                "level_4_prediction": 3,
                "binary_probability": 1.0,
                "class_probabilities": [0.0, 0.0, 0.0, 1.0],
                "fusion_source": "rule_override",
                "rule_matched": True,
            }

        # BERT 推理
        try:
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits_4, logits_2 = self.model(enc["input_ids"], enc["attention_mask"])

            pred_4 = int(torch.argmax(logits_4, dim=-1).cpu().item())
            probs_4 = torch.softmax(logits_4, dim=-1).cpu().squeeze().tolist()
            binary_prob = float(torch.softmax(logits_2, dim=-1)[0, 1].cpu().item())
        except Exception as e:
            logger.error("BERT 推理失败，回退到 LEVEL_0: %s", e)
            return {
                "level": LEVEL_0,
                "level_4_prediction": 0,
                "binary_probability": 0.0,
                "class_probabilities": [0.0, 0.0, 0.0, 0.0],
                "fusion_source": "inference_error",
                "rule_matched": False,
            }

        # 二级融合：binary 升级
        if binary_prob > self.binary_threshold and pred_4 < 2:
            return {
                "level": LEVEL_2,
                "level_4_prediction": pred_4,
                "binary_probability": round(binary_prob, 4),
                "class_probabilities": [round(p, 4) for p in probs_4],
                "fusion_source": "binary_upgrade",
                "rule_matched": False,
            }

        # 默认：4 分类结果
        level_map = {0: LEVEL_0, 1: LEVEL_1, 2: LEVEL_2, 3: LEVEL_3}
        return {
            "level": level_map.get(pred_4, LEVEL_0),
            "level_4_prediction": pred_4,
            "binary_probability": round(binary_prob, 4),
            "class_probabilities": [round(p, 4) for p in probs_4],
            "fusion_source": "model_4class",
            "rule_matched": False,
        }

    def predict_level(self, text: str) -> str:
        """快捷方法：只返回归一化风险等级字符串。"""
        return self.predict(text)["level"]

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """批量预测多条文本。"""
        return [self.predict(t) for t in texts]
