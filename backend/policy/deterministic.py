# -*- coding: utf-8 -*-
"""
Deterministic Policy（Phase 3 Task 3.3 / 3.4 / 3.5，P1 确定性规则层）。

输入：AgentState v1
输出：DeterministicDecision（ActionPlan + confidence + matched_rules + conflicts）

规则依据（Frozen Benchmark v1.1 gold + docs/legacy_policy_audit.md）：

Primary Action（互斥）：
    D01  information_request 意图       → information_response   （gold 36/39）
    D02  meta_question 意图             → information_response   （gold 18/21）
    D03  含糊表达（省略号/短句/犹豫）    → ask_clarification      （gold 21 case 的主信号）
    D04  否则                            → continue_chat

Tool Actions（可多选，非安全场景）：
    D05  resource_request 意图          → recommend_resource     （gold 45/48）
    D06  memory_reference 意图          → retrieve_memory        （gold 40/46）
    D07  information_request 意图       → retrieve_knowledge     （gold 30/39）

Recommendation Mode（gold：rec 以 none 为主，soft 是少数派）：
    D08  resource_request 意图          → hard                   （gold 37/48）
    D09  检测到 rejection / cooldown     → none（抑制）
    D10  否则                            → none

INV 约束：
    - Safety 由 P0 层先行处理，本层不触碰 safety 场景
    - recommendation 独立于 primary_action（INV-06）
    - retrieve_memory 是 tool 不是 primary（INV-07）
    - retrieve_knowledge 可与 information_response 共存（INV-08）
"""
from __future__ import annotations

from typing import List, Optional

from policy.engine import DeterministicDecision
from policy.schema import ActionPlan, PrimaryAction, RecommendationMode, SafetyTarget, ToolAction
from state.schema import AgentState

# 意图标签（对齐 evaluation/benchmark_schema.py IntentLabel）
_INTENT_INFORMATION = {"information_request", "meta_question"}
_INTENT_MEMORY = {"memory_reference"}
_INTENT_RESOURCE = {"resource_request"}

# 含糊表达标记（gold ask_clarification 的主信号：省略号 / 犹豫 / 开放式询问）
_VAGUE_MARKERS = [
    "……", "...", "。。", "怎么说", "该怎么说", "不太确定",
    "你猜", "你觉得呢", "有点不对劲", "遇到点麻烦", "压在心里",
    "算了吧", "不说了", "帮帮忙", "什么办法都行",
]


class DeterministicPolicy:
    """P1 确定性规则层（Task 3.3 + 3.4 + 3.5 合并在一个层内，保证可解释）。"""

    def decide(self, state: AgentState) -> DeterministicDecision:
        labels = set(state.intent.labels)
        rules: List[str] = []
        conflicts: List[str] = []

        # ---------------- Primary Action（Task 3.3） ----------------
        primary = PrimaryAction.CONTINUE_CHAT
        primary_conf = 0.9

        if labels & _INTENT_INFORMATION:
            primary = PrimaryAction.INFORMATION_RESPONSE
            primary_conf = 0.92
            rules.append("D01_information_request")
        elif self._is_vague_utterance(state):
            # 含糊表达（省略号/犹豫/开放式）→ 澄清（gold：21 个 ask_clarification case 的主信号）
            primary = PrimaryAction.ASK_CLARIFICATION
            primary_conf = 0.78
            rules.append("D03_vague_clarification")
        elif state.derived.intent_uncertain and not labels:
            # 无意图 + 低置信 → 澄清
            primary = PrimaryAction.ASK_CLARIFICATION
            primary_conf = 0.72
            rules.append("D03_no_intent_uncertain")
        elif state.derived.intent_uncertain:
            # 有意图但低置信 → 保守 continue（避免过度澄清打断情绪表达）
            primary = PrimaryAction.CONTINUE_CHAT
            primary_conf = 0.78
            rules.append("D04_continue_low_conf")
        else:
            rules.append("D04_continue_chat")

        # ---------------- Tool Actions（Task 3.4） ----------------
        tools: List[ToolAction] = []
        tool_conf = 0.85

        if labels & _INTENT_RESOURCE:
            tools.append(ToolAction.RECOMMEND_RESOURCE)
            rules.append("D05_resource_recommend")
        if labels & _INTENT_MEMORY:
            tools.append(ToolAction.RETRIEVE_MEMORY)
            rules.append("D06_memory_reference")
        if labels & _INTENT_INFORMATION:
            tools.append(ToolAction.RETRIEVE_KNOWLEDGE)
            rules.append("D07_information_knowledge")

        # ---------------- Recommendation Mode（Task 3.5） ----------------
        # gold：rec 以 none 为主，resource_request → hard，其余 → none
        rec = RecommendationMode.NONE
        if state.recommendation.rejection_detected:
            rules.append("D09_rejection_suppress")
        elif labels & _INTENT_RESOURCE:
            rec = RecommendationMode.HARD
            rules.append("D08_resource_hard")

        # 冲突检测：同轮既明确 info 又带 hard 推荐 → 低置信（留给高层）
        if rec == RecommendationMode.HARD and primary == PrimaryAction.INFORMATION_RESPONSE:
            conflicts.append("info_primary_with_hard_rec")

        plan = ActionPlan(
            primary_action=primary,
            tool_actions=tools,
            safety_target=SafetyTarget.NONE,
            recommendation_mode=rec,
        )
        confidence = min(primary_conf, tool_conf)
        if conflicts:
            confidence = min(confidence, 0.6)
        return DeterministicDecision(
            action_plan=plan,
            confidence=confidence,
            matched_rules=rules,
            conflicts=conflicts,
        )

    @staticmethod
    def _is_vague_utterance(state: AgentState) -> bool:
        """含糊表达检测（gold ask_clarification 的信号：省略号 / 犹豫 / 开放式询问）。"""
        text = (state.turn.current_text or "").strip()
        if not text:
            return False
        for marker in _VAGUE_MARKERS:
            if marker in text:
                return True
        # 极短文本（< 6 字符）且无明确意图 → 倾向澄清（避免误伤资源/信息请求）
        if len(text) <= 5 and not state.intent.labels:
            return True
        return False


deterministic_policy = DeterministicPolicy()
