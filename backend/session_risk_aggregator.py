"""
会话级风险聚合器（v5.0）

在 BERT 单轮预测之上做多层状态机聚合：
  1. 当前轮优先原则
  2. 风险惯性（不因一轮好转而断崖降级）
  3. 连续升级检测（多轮轻度信号累积触发更高等级）
  4. 安全确认参与降级
  5. 趋势判定（rising / stable / falling）

用法:
    aggregator = SessionRiskAggregator()
    result = aggregator.aggregate(
        utterance_level="level_1",
        utterance_binary_prob=0.35,
        utterance_rule_hit=False,
        text="最近压力很大",
        conversation_summary={"recent_risk_levels": ["level_0", "level_2"]},
    )
    # => {"session_level": "level_2", "risk_trend": "rising", ...}
"""

import logging
import re
from typing import Any, Dict, List, Optional

from risk_levels import (
    LEVEL_0,
    LEVEL_1,
    LEVEL_2,
    LEVEL_3,
    normalize_risk_level,
    risk_level_index,
)

logger = logging.getLogger(__name__)


class SessionRiskAggregator:
    """会话级风险聚合器。"""

    WINDOW_SIZE = 5

    # 安全确认关键词（同 risk_evaluator._analyze_context 保持一致）
    SAFE_DENIAL_PATTERNS = [
        "没有想自杀",
        "没有想伤害自己",
        "没有想伤害别人",
        "不会自杀",
        "不会伤害自己",
        "我没有想死",
    ]

    def aggregate(
        self,
        utterance_level: str,
        utterance_binary_prob: float,
        utterance_rule_hit: bool,
        text: str,
        conversation_summary: Optional[Dict[str, Any]] = None,
        baseline: str = "low",
    ) -> Dict[str, Any]:
        """
        对当前轮 BERT 单轮预测进行会话级聚合。

        参数:
            utterance_level: BERT 输出的单轮等级 (level_0~level_3)
            utterance_binary_prob: BERT 二分类高风险概率 (0~1)
            utterance_rule_hit: 是否命中 BERT 内部规则兜底
            text: 当前轮用户输入（用于安全确认检测）
            conversation_summary: 会话摘要（含 recent_risk_levels 等）
            baseline: 长期风险基线 (low/medium/high)，来自风险记忆层 v6.0

        返回:
            session_level: str          聚合后的会话级风险等级
            risk_trend: str             趋势
            session_aggregation: dict   聚合详情（调试/审计用）
        """
        summary = conversation_summary or {}
        recent = [
            normalize_risk_level(lv)
            for lv in (summary.get("recent_risk_levels", []) or [])
        ]
        window = (recent + [utterance_level])[-self.WINDOW_SIZE:]

        active_rules: List[str] = []
        session_level = utterance_level
        utterance_idx = risk_level_index(utterance_level)

        # ============================================================
        # 规则 1：当前轮优先 — 当前轮明确高危时直接采用
        # ============================================================
        if utterance_idx >= 3:
            session_level = LEVEL_3
            active_rules.append("current_turn_level_3")
            return self._build_result(session_level, window, active_rules, text, summary, baseline)

        if utterance_idx >= 2:
            # 当前轮 Level 2 是下限，惯性/升级只可能更高
            session_level = LEVEL_2
            active_rules.append("current_turn_level_2")

        # ============================================================
        # 规则 2：风险惯性 — 不让风险等级断崖下降
        # ============================================================
        if recent:
            prev = recent[-1]
            prev_idx = risk_level_index(prev)

            if prev_idx >= 3 and risk_level_index(session_level) < 2:
                session_level = LEVEL_2
                active_rules.append("inertia:prev_level_3_floor_2")
            elif prev_idx >= 2 and risk_level_index(session_level) < 1:
                session_level = LEVEL_1
                active_rules.append("inertia:prev_level_2_floor_1")

        # ============================================================
        # 规则 3：连续升级 — 窗口内多次轻度信号累加
        # ============================================================
        l2_count = sum(1 for lv in window if risk_level_index(lv) >= 2)
        l1_count = sum(1 for lv in window if risk_level_index(lv) >= 1)

        if l2_count >= 2 and risk_level_index(session_level) < 2:
            session_level = LEVEL_2
            active_rules.append("escalation:l2_twice_in_window")
        elif l1_count >= 3 and risk_level_index(session_level) < 1:
            session_level = LEVEL_1
            active_rules.append("escalation:l1_thrice_in_window")

        # ============================================================
        # 规则 4：安全确认参与降级
        # ============================================================
        is_safe_denial = any(pattern in text for pattern in self.SAFE_DENIAL_PATTERNS)
        has_level_2_in_window = any(risk_level_index(lv) >= 2 for lv in window[:-1])  # 排除当前轮
        # 检查用户意图是否包含求助（从 conversation_summary 读取）
        recent_intents = summary.get("recent_intents", []) or []
        has_help_seeking = any(
            intent in {"seeking_help", "seeking_relief", "planning"}
            for intent in recent_intents
        )
        # 检查是否当前轮有保护因素（从 emotion_state 读取 — 但此处没有，用 has_help_seeking 近似）

        if is_safe_denial and not has_level_2_in_window and has_help_seeking:
            # 降一级，但不低于当前轮 BERT 等级
            current_idx = risk_level_index(session_level)
            if current_idx > utterance_idx and current_idx > 0:
                lowered = self._level_at_index(current_idx - 1)
                if risk_level_index(lowered) >= utterance_idx:
                    session_level = lowered
                    active_rules.append("safety_confirmation_deescalate")

        # ============================================================
        # 规则 5：Safety Gate — 长期基线轻量修正（v6.0）
        # 只在 utterance_level 和 session_level 都 < 2 时生效
        # ============================================================
        if baseline == "high" and risk_level_index(session_level) < 1:
            session_level = LEVEL_1
            active_rules.append("baseline:high_floor_1")
        elif baseline == "medium" and risk_level_index(session_level) < 1:
            # medium 基线只在当前轮或上轮非 Level 0 时提升
            utterance_is_0 = risk_level_index(utterance_level) == 0
            prev_is_0 = bool(recent) and risk_level_index(recent[-1]) == 0
            if not (utterance_is_0 and prev_is_0):
                session_level = LEVEL_1
                active_rules.append("baseline:medium_floor_1")

        # ============================================================
        # 趋势判定
        # ============================================================
        risk_trend = self._detect_trend(window)

        return {
            "session_level": session_level,
            "risk_trend": risk_trend,
            "session_aggregation": {
                "utterance_level": utterance_level,
                "utterance_binary_prob": round(utterance_binary_prob, 4),
                "utterance_rule_hit": utterance_rule_hit,
                "window": window,
                "safety_confirmed": is_safe_denial,
                "active_rules": active_rules,
                "baseline_applied": baseline,  # v2.0: 审计字段，记录使用的基线
            },
        }

    # ---------------------------------------------------------------
    # 内部方法
    # ---------------------------------------------------------------

    def _build_result(
        self,
        session_level: str,
        window: List[str],
        active_rules: List[str],
        text: str,
        summary: Dict[str, Any],
        baseline: str = "low",
    ) -> Dict[str, Any]:
        """快捷构建返回（当前轮明确高危时直接调用）。"""
        is_safe_denial = any(pattern in text for pattern in self.SAFE_DENIAL_PATTERNS)
        return {
            "session_level": session_level,
            "risk_trend": self._detect_trend(window),
            "session_aggregation": {
                "utterance_level": session_level,
                "utterance_binary_prob": 0.0,
                "utterance_rule_hit": False,
                "window": window,
                "safety_confirmed": is_safe_denial,
                "active_rules": active_rules,
                "baseline_applied": baseline,  # v2.0: 审计字段
            },
        }

    def _detect_trend(self, window: List[str]) -> str:
        """
        比较窗口前后半段的平均风险等级。

        返回 "rising" | "stable" | "falling"。
        """
        if len(window) < 2:
            return "stable"

        mid = len(window) // 2
        first_half = window[:mid]
        second_half = window[mid:]

        if not first_half or not second_half:
            return "stable"

        avg_first = sum(risk_level_index(lv) for lv in first_half) / len(first_half)
        avg_second = sum(risk_level_index(lv) for lv in second_half) / len(second_half)

        diff = avg_second - avg_first
        if diff > 0.3:
            return "rising"
        elif diff < -0.3:
            return "falling"
        return "stable"

    @staticmethod
    def _level_at_index(idx: int) -> str:
        """将 0-3 索引转为等级字符串。"""
        return [LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3][idx]
