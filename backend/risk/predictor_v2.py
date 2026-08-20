# -*- coding: utf-8 -*-
"""
Risk Predictor v2（Phase 5 Task 5.5 / 5.9 接口冻结）。

封装 Risk v5.1 模型 + Calibrated Fusion（P23>0.20）→ RiskResult v2。

- 单轮：v5_1 BERT → class_probs → RiskFusion（level + high_risk_prob）
- 多轮：交给 DynamicRiskTracker（dynamic_state.py）
- 版本溯源：risk_version = "v5_1"
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from risk.calibration import risk_calibration
from risk.fusion import risk_fusion

logger = logging.getLogger(__name__)

# v5.1 模型路径（可被 env 覆盖）
DEFAULT_RISK_MODEL = "bert_data/models/v5_1_tuned/best_model"


class RiskPredictorV2:
    """Risk 2.0 单轮预测器。"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model_path = model_path or DEFAULT_RISK_MODEL
        self.device = device
        self._bert = None
        self.risk_version = "v5_1"

    def _get_bert(self):
        if self._bert is None:
            from bert_risk_predictor import BertRiskPredictor
            self._bert = BertRiskPredictor(model_path=self.model_path, device=self.device)
        return self._bert

    def predict(self, text: str) -> Dict:
        """单轮预测 → RiskResult v2。"""
        raw = self._get_bert().predict(text)
        probs = raw["class_probabilities"]
        cal = risk_calibration.calibrate(probs)
        fused = risk_fusion.fuse(probs, binary_prob=raw.get("binary_probability"),
                                 rule_hit=raw.get("rule_matched", False))
        return {
            "level": fused["level"],
            "high_risk_prob": fused["high_risk_prob"],
            "class_probabilities": cal["calibrated_probs"],
            "fusion_source": fused["source"],
            "rule_matched": raw.get("rule_matched", False),
            "risk_version": self.risk_version,
            "threshold": risk_calibration.HIGH_RISK_THRESHOLD,
        }

    def predict_level(self, text: str) -> int:
        return self.predict(text)["level"]


risk_predictor_v2 = RiskPredictorV2()
