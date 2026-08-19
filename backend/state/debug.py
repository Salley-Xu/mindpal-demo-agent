# -*- coding: utf-8 -*-
"""
State Debug / Shadow Trace（Phase 2 Task 2.7 §30）。

最小 State Trace：不记录长期记忆原文、Profile 全量、完整 history。
只记录 ID、类型、结构化摘要。Phase 7 再做完整 Observability。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from state.schema import AgentState

logger = logging.getLogger(__name__)

_SHADOW_LOG = Path(__file__).resolve().parents[1] / "logs" / "agent_state_shadow.jsonl"


class ShadowTracer:
    """Shadow-mode 一致性追踪（§29）。"""

    def __init__(self, log_path: Path = _SHADOW_LOG):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_turn(self, state: AgentState, consistency: Optional[Dict] = None) -> None:
        """记录一轮 state + 一致性检查结果。"""
        record = {
            "trace_id": state.meta.trace_id,
            "session_id": state.identity.session_id,
            "turn_index": state.turn.turn_index,
            "state_version": state.meta.state_version,
            "state": state.to_debug_dict(),
            "consistency": consistency or {},
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning("shadow trace 写入失败: %s", e)

    def read_last(self, n: int = 20) -> list:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(l) for l in lines[-n:]]


class ConsistencyChecker:
    """Shadow-mode 一致性检查（§28）。"""

    @staticmethod
    def check(state: AgentState, legacy: Dict) -> Dict:
        """对比 AgentState 与 legacy 变量。"""
        return {
            "risk_match": _match_risk(state.risk.level, legacy.get("risk_level")),
            "emotion_match": _match_emotion(state.emotion.current_emotion, legacy.get("emotion")),
            "stage_match": _match_stage(state.conversation.stage, legacy.get("stage")),
            "turn_count_match": _match_turn(state.conversation.turn_count, legacy.get("turn_count")),
        }

    @staticmethod
    def all_match(result: Dict) -> bool:
        return all(v is True for k, v in result.items() if k.endswith("_match"))


def _match_risk(state_level: int, legacy_level) -> bool:
    from state.adapters import _risk_int
    return _risk_int(legacy_level) == state_level


def _match_emotion(state_emotion: str, legacy_emotion) -> bool:
    return str(legacy_emotion or "") == str(state_emotion)


def _match_stage(state_stage: str, legacy_stage) -> bool:
    return str(legacy_stage or "") == str(state_stage)


def _match_turn(state_turn: int, legacy_turn) -> bool:
    return int(legacy_turn or 0) == int(state_turn)


shadow_tracer = ShadowTracer()
