# -*- coding: utf-8 -*-
"""
Trace Logger（Phase 7 Task 7.2-7.5）。

写/读 JSONL。同一 turn 一个 trace_id。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config import config
from tracing.schema import TurnTrace

logger = logging.getLogger(__name__)

_TRACE_DIR = os.path.join(getattr(config, "LOG_DIR", "logs"), "traces")
_TRACE_ENABLED = True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceLogger:
    """统一 Trace 写入/读取。"""

    def new_trace(self, trace_id: str, session_id: str, turn_id: int) -> TurnTrace:
        return TurnTrace(trace_id=trace_id, session_id=session_id, turn_id=turn_id,
                         versions={
                             "intent": "phase1_5_final",
                             "risk": "v5_1",
                             "state": "agentstate_v1",
                             "policy": "agentpolicy_v1",
                             "memory": "memory_v2",
                             "recommendation": "recommendation_v2",
                         })

    def write(self, trace: TurnTrace) -> None:
        if not _TRACE_ENABLED:
            return
        try:
            os.makedirs(_TRACE_DIR, exist_ok=True)
            path = Path(_TRACE_DIR) / f"{trace.trace_id}.json"
            data = trace.model_dump()
            data["logged_at"] = _now()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:  # noqa: BLE001
            logger.warning("trace 写入失败: %s", e)

    def read(self, trace_id: str) -> Optional[TurnTrace]:
        path = Path(_TRACE_DIR) / f"{trace_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return TurnTrace(**json.load(f))
        except Exception as e:  # noqa: BLE001
            logger.warning("trace 读取失败 %s: %s", trace_id, e)
            return None

    def list_traces(self, limit: int = 50) -> List[str]:
        if not os.path.isdir(_TRACE_DIR):
            return []
        files = sorted(Path(_TRACE_DIR).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in files[:limit]]


trace_logger = TraceLogger()
