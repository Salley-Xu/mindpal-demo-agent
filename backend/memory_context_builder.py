"""
MemoryContextBuilder — AgentOrchestrator 到记忆系统的桥接。

将当前上下文（用户输入、情绪、风险）转换为记忆检索查询，
调用 MemoryRetriever + MemoryInjector，返回结构化注入结果。
"""

import logging
from typing import Any, Dict, Optional

from config import config
from memory_retriever import memory_retriever
from memory_injector import memory_injector
from risk_memory_store import risk_memory_store
from models import InjectedMemoryContext, MemoryQuery
from risk_levels import risk_level_index

logger = logging.getLogger(__name__)


class MemoryContextBuilder:
    """构建 prompt 注入用的记忆上下文。"""

    def __init__(self, retriever=None, injector=None, risk_store=None):
        self.retriever = retriever or memory_retriever
        self.injector = injector or memory_injector
        self.risk_store = risk_store or risk_memory_store

    async def build(
        self,
        user_id: str,
        user_input: str,
        emotion_state: Optional[Dict[str, Any]] = None,
        risk_state: Optional[Dict[str, Any]] = None,
        conversation_summary: Optional[Dict[str, Any]] = None,
    ) -> InjectedMemoryContext:
        """构建记忆上下文。

        返回:
            InjectedMemoryContext: 含注入文本、token 用量、跳过原因。
        """
        emotion_state = emotion_state or {}
        risk_state = risk_state or {}
        summary = conversation_summary or {}

        # 1. 构建检索查询
        risk_level = risk_state.get("level", "level_0")
        need_risk = risk_level_index(risk_level) >= 1

        query = MemoryQuery(
            text=user_input,
            emotion=emotion_state.get("current_emotion") or emotion_state.get("emotion_type"),
            risk_level=risk_level,
            stress_source=emotion_state.get("stress_source"),
            intent=emotion_state.get("user_intent"),
            need_risk_context=need_risk,
            need_preference=True,
            memory_types=[
                "support_preference",
                "avoid_preference",
                "stress_source",
                "mood_event",
                "coping_strategy",
                "recommendation_feedback",
            ],
            top_k=15,
        )

        # 2. 检索
        memories = await self.retriever.retrieve(
            user_id=user_id,
            query=query,
            top_k=15,
        )

        # 3. 风险上下文
        risk_context = None
        if need_risk:
            try:
                risk_context = await self.risk_store.get_recent_risk_context(user_id)
            except Exception:
                pass

        # 4. 计算 token budget
        token_budget = int(config.MAX_CONTEXT_TOKENS * config.MEMORY_INJECTION_BUDGET_RATIO)
        if token_budget < 64:
            token_budget = 64

        # 5. 注入
        result = self.injector.build_context(
            memories=memories,
            risk_context=risk_context,
            token_budget=token_budget,
            token_counter=None,
        )

        return result


# 全局单例
memory_context_builder = MemoryContextBuilder()
