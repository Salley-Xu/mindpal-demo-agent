# -*- coding: utf-8 -*-
"""
Risk Calibration（Phase 5 Task 5.4）。

从 Phase 5.4 阈值扫描冻结的校准参数：
  - high-risk 阈值 P(L2)+P(L3) > 0.20
  - 决策等级由 P2/P3 相对大小定 L2/L3
  - 低风险 argmax(L0/L1)

记录各层决策阈值（可 Trace）。
"""
from __future__ import annotations

from typing import List


class RiskCalibration:
    """校准配置 + 决策阈值。"""

    HIGH_RISK_THRESHOLD = 0.20        # 冻结（Phase 5 扫描最佳点）
    L3_VS_L2_DECISION = "p3_ge_p2"    # L3 当 P3>=P2
    LOW_RISK_DECISION = "argmax_l0_l1"

    def calibrate(self, probs: List[float]) -> dict:
        """返回校准后的概率 + 决策。"""
        p = [max(0.0, min(1.0, x)) for x in probs]
        total = sum(p) or 1.0
        p = [x / total for x in p]  # 归一化
        return {
            "calibrated_probs": [round(x, 4) for x in p],
            "high_risk_prob": round(p[2] + p[3], 4),
            "decision": "high_risk" if p[2] + p[3] > self.HIGH_RISK_THRESHOLD else "low_risk",
            "threshold": self.HIGH_RISK_THRESHOLD,
        }


risk_calibration = RiskCalibration()
