# -*- coding: utf-8 -*-
"""
AgentState Persistence（Phase 2 Task 2.5 §24-25）。

策略：每轮保存 latest state（session 维度），不新增第二套长期数据库。
Debug 模式可写 full history JSONL；Production 默认只保留 latest。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from state.schema import AgentState

logger = logging.getLogger(__name__)

# latest state 内存缓存：{user_id:session_id -> AgentState}
_latest_state: Dict[str, AgentState] = {}

# Debug history JSONL（可选）
_DEBUG_HISTORY_DIR = Path(__file__).resolve().parents[1] / "data" / "agent_state_history"


def _key(user_id: str, session_id: str) -> str:
    return f"{user_id}:{session_id}"


class AgentStateStore:
    """AgentState 快照存储（latest + 可选 history）。"""

    def __init__(self, persist_history: bool = False):
        self.persist_history = persist_history
        if persist_history:
            _DEBUG_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    def save_latest(self, state: AgentState) -> None:
        _latest_state[_key(state.identity.user_id, state.identity.session_id)] = state
        if self.persist_history:
            self._append_history(state)

    def load_latest(self, user_id: str, session_id: str) -> Optional[AgentState]:
        return _latest_state.get(_key(user_id, session_id))

    def clear(self, user_id: str, session_id: str) -> None:
        _latest_state.pop(_key(user_id, session_id), None)

    def _append_history(self, state: AgentState) -> None:
        try:
            path = _DEBUG_HISTORY_DIR / f"{state.identity.session_id}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(state.to_debug_dict(), ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning("state history 写入失败: %s", e)


# 全局单例（默认不写 history，仅 latest）
agent_state_store = AgentStateStore(persist_history=os.getenv("AGENT_STATE_HISTORY", "false").lower() == "true")
