# -*- coding: utf-8 -*-
"""
Trace Context（Phase 7 Task 7.1）。

同一 turn 一个 trace_id，模块共享。基于 threading.local 传播。
"""
from __future__ import annotations

import threading
import uuid
from typing import Dict, Optional


class _TraceContext(threading.local):
    def __init__(self):
        self.trace_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.turn_id: int = 0
        self.extra: Dict[str, str] = {}


_context = _TraceContext()


def new_trace(session_id: str = "", turn_id: int = 0) -> str:
    """开启新 trace（同 turn 一个 trace_id）。"""
    trace_id = uuid.uuid4().hex[:16]
    _context.trace_id = trace_id
    _context.session_id = session_id
    _context.turn_id = turn_id
    _context.extra = {}
    return trace_id


def get_trace_id() -> Optional[str]:
    return _context.trace_id


def get_context() -> dict:
    return {"trace_id": _context.trace_id, "session_id": _context.session_id,
            "turn_id": _context.turn_id, **{f"extra.{k}": v for k, v in _context.extra.items()}}


def set_extra(key: str, value: str) -> None:
    _context.extra[key] = value


def clear() -> None:
    _context.trace_id = None
    _context.session_id = None
    _context.turn_id = 0
    _context.extra = {}
