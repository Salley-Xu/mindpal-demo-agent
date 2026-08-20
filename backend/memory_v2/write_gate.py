# -*- coding: utf-8 -*-
"""
Write Gate v2（Phase 4 Task 4.3）。

特征（Phase 4 §6）：importance / stability / future usefulness / confidence / privacy / duplication。
禁止"所有事实都写记忆"。

输出：是否写入 + 重要性 + 原因。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from memory_v2.schema import MEMORY_TYPE_CONFIGS, MemoryType
from models import MemoryCandidate
from state.schema import AgentState

# 隐私敏感类型（不写细节）
_SENSITIVE_TYPES = {MemoryType.PROFILE.value}
_IMPORTANCE_THRESHOLD = 0.45


@dataclass
class GateDecision:
    should_write: bool
    importance: float
    reasons: List[str]


class WriteGateV2:
    """记忆写入门控。"""

    def decide(self, candidate: MemoryCandidate, state: Optional[AgentState] = None) -> GateDecision:
        reasons: List[str] = []
        try:
            mtype = MemoryType(candidate.memory_type)
            cfg = MEMORY_TYPE_CONFIGS.get(mtype)
        except ValueError:
            cfg = None

        # 1. importance：基础（type stability/priority）+ 显式信号
        importance = 0.3
        if cfg:
            importance += cfg.stability * 0.3
        if candidate.source == "explicit":
            importance += 0.2
            reasons.append("explicit_source")
        if candidate.risk_level and candidate.risk_level in ("level_2", "level_3"):
            importance += 0.25
            reasons.append("risk_relevant")
        if candidate.confidence and candidate.confidence > 0.8:
            importance += 0.1
        importance = min(1.0, importance)

        # 2. stability：临时情绪/单轮噪声不写
        if candidate.memory_type == MemoryType.EVENT.value and not candidate.source_text:
            importance -= 0.1
            reasons.append("low_signal_event")

        # 3. privacy：敏感内容不写细节
        if candidate.memory_type in _SENSITIVE_TYPES and candidate.confidence < 0.7:
            reasons.append("privacy_gated")
            return GateDecision(False, importance, reasons)

        should_write = importance >= _IMPORTANCE_THRESHOLD
        reasons.append(f"importance:{importance:.2f}")
        return GateDecision(should_write, round(importance, 3), reasons)


write_gate_v2 = WriteGateV2()
