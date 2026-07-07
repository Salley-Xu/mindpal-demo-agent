"""推荐模块全链路追踪

为每次推荐请求记录结构化 trace 到 JSONL 日志文件，支持离线分析和评估。
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config import config

logger = logging.getLogger(__name__)

# 日志文件路径
_TRACE_LOG_DIR = getattr(config, "LOG_DIR", "logs")
_TRACE_LOG_PATH = os.path.join(_TRACE_LOG_DIR, "recommendation_trace.jsonl")
_TRACE_ENABLED = getattr(config, "RECOMMEND_TRACE_ENABLED", True)


class TraceEvent(BaseModel):
    """单次推荐请求的全链路追踪记录。"""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: str = ""
    user_id: str = ""
    session_id: str = ""
    turn_id: int = 0

    # ── 阶段 1：门控输入 ──
    gate_inputs: Dict[str, float] = Field(default_factory=dict)
    emotion_state: Dict[str, Any] = Field(default_factory=dict)
    risk_state: Dict[str, Any] = Field(default_factory=dict)
    conversation_summary: Dict[str, Any] = Field(default_factory=dict)
    user_profile: Dict[str, Any] = Field(default_factory=dict)

    # ── 阶段 2：门控输出 ──
    gate_output: Dict[str, Any] = Field(default_factory=dict)

    # ── 阶段 3：推荐结果 ──
    query_variants: List[str] = Field(default_factory=list)
    candidate_count: int = 0
    rerank_used: bool = False
    rerank_success: bool = False
    recommendation_ids: List[str] = Field(default_factory=list)
    recommendation_scores: List[float] = Field(default_factory=list)

    # ── 阶段 4：安全与持久化 ──
    safety_overridden: bool = False
    safety_before: Dict[str, Any] = Field(default_factory=dict)
    safety_after: Dict[str, Any] = Field(default_factory=dict)
    persisted: bool = False

    # ── 性能 ──
    latency_ms: int = 0


def write_trace(event: TraceEvent):
    """将 trace 事件写入 JSONL 日志文件。"""
    if not _TRACE_ENABLED:
        return
    try:
        os.makedirs(_TRACE_LOG_DIR, exist_ok=True)
        line = json.dumps(event.model_dump(), ensure_ascii=False, default=str)
        with open(_TRACE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.warning(f"写入推荐 trace 失败: {e}")


def read_traces(limit: int = 0) -> List[TraceEvent]:
    """从日志文件读取 trace 事件（最近优先）。"""
    if not os.path.exists(_TRACE_LOG_PATH):
        return []
    events = []
    try:
        with open(_TRACE_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(TraceEvent(**json.loads(line)))
                    except Exception:
                        pass
    except Exception as e:
        logger.warning(f"读取推荐 trace 失败: {e}")
    # 按时间逆序排列（最新靠前）
    events.sort(key=lambda e: e.timestamp, reverse=True)
    if limit > 0:
        events = events[:limit]
    return events
