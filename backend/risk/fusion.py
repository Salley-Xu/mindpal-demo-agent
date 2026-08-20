# -*- coding: utf-8 -*-
"""
Risk Fusion（Phase 5 Task 5.3）。

在 Phase 5.2/5.4 阈值扫描基础上冻结的最佳工作点：
  P(L2)+P(L3) > 0.20 → high-risk（HR recall 0.936 / FPR 0.022，792 跨源独立）

规则：
  - 4-class 概率融合：high_risk = P2 + P3 > threshold
  - rule 命中（L3 关键词）优先 → level_3
"""
from __future__ import annotations

from typing import List, Optional

# 冻结的融合阈值（Phase 5 扫描最佳点）
HIGH_RISK_PROB_THRESHOLD = 0.20


class RiskFusion:
    """4-class 概率 → 风险等级融合。"""

    def fuse(self, probs: List[float], binary_prob: Optional[float] = None,
             rule_hit: bool = False) -> dict:
        """返回 { level, high_risk_prob, source }。"""
        if rule_hit:
            return {"level": 3, "high_risk_prob": 1.0, "source": "rule_override"}

        p2 = probs[2] if len(probs) > 2 else 0.0
        p3 = probs[3] if len(probs) > 3 else 0.0
        high_risk_prob = p2 + p3

        if high_risk_prob > HIGH_RISK_PROB_THRESHOLD:
            # 高风险：取 P3 与 P2 中的高者定级
            level = 3 if p3 >= p2 else 2
        else:
            # 低风险：argmax of L0/L1
            level = 0 if probs[0] >= probs[1] else 1

        return {"level": level, "high_risk_prob": round(high_risk_prob, 4),
                "source": "calibrated_fusion"}


risk_fusion = RiskFusion()
