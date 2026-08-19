# -*- coding: utf-8 -*-
"""
LLM Fallback（Phase 3 Task 3.8，P4 最低优先级层）。

仅用于 P3 Ambiguity Gate 判定为 ambiguous 的 State。

输入：
    AgentState compact serialization
    + Allowed Action Schema
    + Policy Invariants

输出严格 JSON（见 Phase 3 §14），必须经过：
    - Schema validation
    - Policy invariant validation
    - Safety override validation

LLM 永远不能覆盖 Safety：若 LLM 输出违反 Safety 不变量，Validator 会拒绝；
若 state.risk 已判定需安全干预，本层不允许被调用（由 hybrid.py 保证）。

生产环境调用 DeepSeek Chat（OpenAI 兼容）；评测环境可用 --llm none 关闭
（此时 fallback 返回 deterministic 决策并标记 source=llm_off）。
"""
from __future__ import annotations

import json
import re
from typing import Optional

from policy.engine import DeterministicDecision, build_policy_result
from policy.schema import ActionPlan, PolicyResult, PrimaryAction, RecommendationMode, SafetyTarget, ToolAction
from policy.validator import policy_validator
from state.schema import AgentState

try:
    from config import config
    _OPENAI = True
except Exception:  # pragma: no cover
    _OPENAI = False

# 允许的动作集合（LLM 可选）
_ALLOWED_SCHEMA = {
    "primary_action": ["continue_chat", "ask_clarification", "information_response", "safety_intervention"],
    "tool_actions": ["retrieve_memory", "retrieve_knowledge", "recommend_resource"],
    "safety_target": ["none", "self", "third_party"],
    "recommendation_mode": ["none", "soft", "hard", "safety_only"],
}

_INVARIANTS_PROMPT = (
    "Safety 优先级最高："
    "risk>=2 时必须 safety_intervention；"
    "safety_only 不允许普通推荐；"
    "safety_target=third_party 必须 primary=safety_intervention；"
    "安全否认/讨论语境不强行升级。"
)

_ENUM_MAP = {
    "primary_action": PrimaryAction,
    "safety_target": SafetyTarget,
    "recommendation_mode": RecommendationMode,
}


class LLMFallbackPolicy:
    """P4 LLM Fallback（可关闭）。"""

    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self._client = None
        self.model = None

    # ---- 客户端（懒加载，对齐 agent_orchestrator 模式） ----

    def _get_client(self):
        if self._client is None:
            if not _OPENAI:
                raise RuntimeError("config 不可用，无法初始化 LLM client")
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.API_BASE_URL)
            self.model = config.CHAT_MODEL
        return self._client

    # ---- 状态序列化 ----

    @staticmethod
    def compact_state(state: AgentState) -> dict:
        """AgentState 压缩序列化（不泄露敏感原文，含 Policy 决策所需信号）。"""
        return {
            "intent": {"labels": state.intent.labels, "confidence": round(state.intent.confidence, 3),
                       "is_open_set": state.intent.is_open_set},
            "emotion": {"current_emotion": state.emotion.current_emotion,
                        "intensity": state.emotion.intensity,
                        "negative_trend": state.derived.negative_emotion},
            "risk": {"level": state.risk.level, "trend": state.risk.trend,
                     "is_third_party": state.risk.is_third_party,
                     "is_discussion": state.risk.is_discussion,
                     "safe_denial": state.risk.safe_denial,
                     "subject": state.risk.subject},
            "derived": state.derived.model_dump(),
            "conversation": {"stage": state.conversation.stage, "turn_count": state.conversation.turn_count},
            "recommendation": {"rejection_detected": state.recommendation.rejection_detected,
                               "turns_since_last_recommendation": state.recommendation.turns_since_last_recommendation},
            "memory": {"explicit_reference": state.memory.explicit_memory_reference,
                       "available_count": state.memory.available_memory_count},
        }

    def _build_prompt(self, state: AgentState, ambiguity) -> str:
        state_json = json.dumps(self.compact_state(state), ensure_ascii=False)
        allowed_json = json.dumps(_ALLOWED_SCHEMA, ensure_ascii=False)
        return (
            "你是 MindPal 的决策引擎，根据对话 AgentState 输出 ActionPlan。\n\n"
            f"AgentState:\n{state_json}\n\n"
            f"允许的动作（仅从这些值选择）:\n{allowed_json}\n\n"
            f"不变量:\n{_INVARIANTS_PROMPT}\n\n"
            "输出严格 JSON（不要输出任何其它文字）:\n"
            '{"primary_action": "...", "tool_actions": ["..."], "safety_target": "...", "recommendation_mode": "..."}'
        )

    # ---- 主入口 ----

    async def adecide(self, state: AgentState, ambiguity=None) -> Optional[PolicyResult]:
        """异步 LLM Fallback（生产路径）。"""
        if not self.enabled:
            return None
        client = self._get_client()
        prompt = self._build_prompt(state, ambiguity)
        try:
            resp = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:  # pragma: no cover
            return self._fallback_result(state, f"llm_error:{type(e).__name__}")
        return self._parse_and_validate(state, raw)

    def decide(self, state: AgentState, ambiguity=None) -> Optional[PolicyResult]:
        """同步入口（评测路径）。要求当前线程无运行中的事件循环。"""
        if not self.enabled:
            return None
        import asyncio
        return asyncio.run(self.adecide(state, ambiguity))

    # ---- 解析 + 校验 ----

    def _parse_and_validate(self, state: AgentState, raw: str) -> Optional[PolicyResult]:
        data = self._extract_json(raw)
        if data is None:
            return self._fallback_result(state, "llm_invalid_json")
        try:
            plan = ActionPlan(
                primary_action=_ENUM_MAP["primary_action"](data.get("primary_action", "continue_chat")),
                tool_actions=[ToolAction(t) for t in data.get("tool_actions", []) if t in _ALLOWED_SCHEMA["tool_actions"]],
                safety_target=_ENUM_MAP["safety_target"](data.get("safety_target", "none")),
                recommendation_mode=_ENUM_MAP["recommendation_mode"](data.get("recommendation_mode", "none")),
            )
        except (ValueError, TypeError):
            return self._fallback_result(state, "llm_invalid_enum")

        # Safety override validation：LLM 不得覆盖 Safety
        vr = policy_validator.validate_result(state, build_policy_result(plan, source="llm"))
        if not vr.valid:
            return self._fallback_result(state, f"llm_invariant_violated:{','.join(vr.violated)}")
        return build_policy_result(
            plan,
            confidence=0.6,
            source="llm",
            matched_rules=["LLM_FALLBACK"],
            fallback_reason=None,
        )

    @staticmethod
    def _extract_json(raw: str) -> Optional[dict]:
        raw = raw.strip()
        # 剥离可能的 markdown 围栏
        fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
        if fence:
            raw = fence.group(1)
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                return None
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _fallback_result(state: AgentState, reason: str) -> PolicyResult:
        """LLM 失败时回退到 Deterministic 决策（Safety 由 hybrid 层保证）。"""
        from policy.deterministic import deterministic_policy
        det = deterministic_policy.decide(state)
        return det.to_policy_result(source="deterministic_fallback", fallback_reason=reason)


llm_fallback = LLMFallbackPolicy()
