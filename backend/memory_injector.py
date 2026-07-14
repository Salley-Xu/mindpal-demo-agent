"""
MemoryInjector — Token-budget 控制的长时记忆注入。

优先级策略（从高到低）：
  1. 风险基线 / 触发词（若有）
  2. 显式偏好 / 回避（稳定注入）
  3. 当前主题相关压力源
  4. 相似情绪事件（top 1-2）
  5. 推荐反馈摘要
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from config import config
from models import InjectedMemoryContext, MemorySearchResult

logger = logging.getLogger(__name__)


class MemoryInjector:
    """在 token budget 内选择和格式化注入内容。"""

    def __init__(self):
        pass

    def build_context(
        self,
        memories: List[MemorySearchResult],
        risk_context: Optional[Dict[str, Any]],
        token_budget: int,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> InjectedMemoryContext:
        """
        在 token budget 内组装记忆注入文本。

        参数:
            memories: 经 MemoryRetriever 排序后的结果
            risk_context: RiskMemoryStore.get_recent_risk_context() 的结果
            token_budget: 最大可用 tokens
            token_counter: token 计数函数，默认 len//2 估算

        返回:
            InjectedMemoryContext: 含注入文本、用量、跳过原因
        """
        if token_counter is None:
            token_counter = lambda t: max(1, len(str(t)) // 2)

        lines: List[str] = []
        used = 0
        included: List[str] = []
        dropped: Dict[str, str] = {}
        now_token: Callable[[str], int] = token_counter

        # ---------------------------------------------------------------
        # Step 1: 风险上下文（若有则固定注入）
        # ---------------------------------------------------------------
        if risk_context and risk_context.get("baseline", "low") != "low":
            risk_line = self._format_risk_context(risk_context)
            risk_tokens = now_token(risk_line)
            if used + risk_tokens <= token_budget:
                lines.append(risk_line)
                used += risk_tokens

        # ---------------------------------------------------------------
        # Step 2: 按优先级分类注入
        # ---------------------------------------------------------------
        preference_lines = []
        stress_lines = []
        event_lines = []
        other_lines = []

        for mr in memories:
            item = mr.item
            if item.memory_type in ("support_preference", "avoid_preference"):
                preference_lines.append((mr, self._format_memory_for_inject(mr)))
            elif item.memory_type == "stress_source":
                stress_lines.append((mr, self._format_memory_for_inject(mr)))
            elif item.memory_type == "mood_event":
                event_lines.append((mr, self._format_memory_for_inject(mr)))
            else:
                other_lines.append((mr, self._format_memory_for_inject(mr)))

        # 2a: 偏好（2-3 条，稳定注入）
        for mr, text in preference_lines[:2]:
            tokens = now_token(text)
            if used + tokens <= token_budget:
                lines.append(text)
                used += tokens
                included.append(mr.item.id)
            else:
                dropped[mr.item.id] = "budget_exceeded"

        # 2b: 压力源（1-2 条）
        for mr, text in stress_lines[:2]:
            tokens = now_token(text)
            if used + tokens <= token_budget:
                lines.append(text)
                used += tokens
                included.append(mr.item.id)
            else:
                dropped[mr.item.id] = "budget_exceeded"

        # 2c: 情绪事件（1-2 条，仅在有余量时）
        for mr, text in event_lines[:2]:
            tokens = now_token(text)
            if used + tokens <= token_budget:
                lines.append(text)
                used += tokens
                included.append(mr.item.id)
            elif len(event_lines) > 1:
                # 跳过低分事件
                dropped[mr.item.id] = "budget_exceeded"

        # 2d: 其他（有多余预算时才注入）
        for mr, text in other_lines:
            tokens = now_token(text)
            if used + tokens <= token_budget:
                lines.append(text)
                used += tokens
                included.append(mr.item.id)
            else:
                break

        # 记录所有跳过的
        all_ids = {mr.item.id for mr in memories}
        for mid in all_ids:
            if mid not in included and mid not in dropped:
                dropped[mid] = "low_priority"

        result_text = "\n".join(lines) if lines else ""

        return InjectedMemoryContext(
            text=result_text,
            used_tokens=used,
            included_memory_ids=included,
            dropped_memory_ids=list(set(dropped.keys())),
            drop_reasons=dropped,
        )

    # ---------------------------------------------------------------
    # 格式化
    # ---------------------------------------------------------------

    @staticmethod
    def _format_risk_context(risk_context: Dict[str, Any]) -> str:
        """格式化风险上下文为注入用文本。"""
        parts = []
        baseline = risk_context.get("baseline", "low")
        if baseline != "low":
            label = {"low": "低", "medium": "中", "high": "高"}
            parts.append(f"长期风险基线：{label.get(baseline, baseline)}")

        triggers = risk_context.get("common_triggers", [])
        if triggers:
            parts.append(f"常见触发因素：{'、'.join(triggers[:3])}")

        return "【风险安全】" + "；".join(parts)

    @staticmethod
    def _format_memory_for_inject(mr: MemorySearchResult) -> str:
        """格式化单条记忆为注入用文本。"""
        item = mr.item
        mtype = item.memory_type

        if mtype == "support_preference":
            content = item.content[:80]
            return f"- 支持偏好：{content}"
        elif mtype == "avoid_preference":
            content = item.content[:80]
            return f"- 回避偏好：{content}"
        elif mtype == "stress_source":
            content = item.summary or item.content
            return f"- 相关压力源：{content[:80]}"
        elif mtype == "mood_event":
            content = item.summary or item.content
            return f"- 相似经历：{content[:80]}"
        elif mtype == "coping_strategy":
            content = item.content[:80]
            return f"- 有效策略：{content}"
        elif mtype == "recommendation_feedback":
            content = item.content[:80]
            return f"- 推荐反馈：{content}"
        else:
            content = (item.summary or item.content)[:80]
            return f"- 相关信息：{content}"


# 全局单例
memory_injector = MemoryInjector()
