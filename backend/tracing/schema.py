# -*- coding: utf-8 -*-
"""
Unified Trace Schema（Phase 7 Task 7.1）。

同一 turn 一个 trace_id，所有模块共享。
避免写入：全量敏感 profile / 全量长期记忆原文 / 不必要完整 prompt。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TurnTrace(BaseModel):
    """单轮完整决策链路 Trace。"""
    trace_id: str = ""
    session_id: str = ""
    turn_id: int = 0

    input: Dict[str, Any] = Field(default_factory=dict)            # 必要输入 + hash
    perception: Dict[str, Any] = Field(default_factory=dict)       # intent/emotion/risk
    state: Dict[str, Any] = Field(default_factory=dict)            # AgentState 摘要
    policy: Dict[str, Any] = Field(default_factory=dict)           # PolicyResult
    tools: Dict[str, Any] = Field(default_factory=dict)            # tool 选择
    memory: Dict[str, Any] = Field(default_factory=dict)           # 记忆命中/丢弃
    recommendation: Dict[str, Any] = Field(default_factory=dict)   # 推荐决策
    response: Dict[str, Any] = Field(default_factory=dict)         # 响应摘要
    feedback: Dict[str, Any] = Field(default_factory=dict)         # 反馈
    latency: Dict[str, float] = Field(default_factory=dict)        # 分阶段耗时
    versions: Dict[str, str] = Field(default_factory=dict)         # 各模块版本
    llm_calls: List[Dict[str, Any]] = Field(default_factory=list)  # LLM 调用记录
    error: Optional[Dict[str, Any]] = None                         # 错误 + 归因层


class AttributionResult(BaseModel):
    """Error Attribution（Phase 7 Task 7.7）。"""
    trace_id: str = ""
    first_error_layer: str = ""    # perception|state|policy|memory|recommendation|execution|llm|safety
    reason: str = ""
