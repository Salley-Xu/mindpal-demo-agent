# -*- coding: utf-8 -*-
"""
Context-aware Intent 输入构建（Phase 1.5 Task 1.5.1 §5.2）。

输入格式比较：
  Variant A: current only        -> [current_user]
  Variant B: +1 turn             -> [prev_assistant, current_user]
  Variant C: +2 turns            -> [prev_user, prev_assistant, current_user]
"""
from __future__ import annotations

from typing import Dict, List


def build_context_input(conversation: List[Dict], context_turns: int = 0) -> str:
    """
    从 conversation（含当前 user 轮）构造上下文输入。

    conversation: [{"role": "user"/"assistant", "content": str}, ...]，最后一条为当前 user 轮
    context_turns: 0 / 1 / 2（包含的历史轮数，不含当前轮）

    返回：拼接文本，或退化为当前文本（无历史时）。
    """
    if not conversation:
        return ""
    # 最后一条是当前 user 轮（兼容 dict 和 Pydantic 对象）
    def _get(turn, key):
        if isinstance(turn, dict):
            return turn.get(key, "")
        return getattr(turn, key, "") or ""

    current = conversation[-1]
    current_text = _get(current, "content")
    prev = conversation[:-1]  # 历史轮

    if context_turns <= 0 or not prev:
        return current_text

    # 取最近 N 个历史轮
    hist = prev[-context_turns:]
    parts = []
    for turn in hist:
        role = _get(turn, "role") or "user"
        content = _get(turn, "content")
        if role == "assistant":
            parts.append(f"assistant：{content}")
        else:
            parts.append(f"user：{content}")
    parts.append(f"user：{current_text}")
    return "\n".join(parts)


def context_turns_used(conversation: List[Dict], context_turns: int) -> int:
    """实际使用的历史轮数（无历史则 0）。"""
    if not conversation:
        return 0
    return min(context_turns, len(conversation) - 1) if context_turns > 0 else 0
