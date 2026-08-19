# -*- coding: utf-8 -*-
"""
Shadow-mode 集成（Phase 2 Task 2.6 §26-29）。

旧流程继续驱动生产逻辑；AgentState 只旁路构建、更新、记录与一致性检查。
本模块绝不抛出异常（行为保持重构）。
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from state.builder import build_agent_state
from state.debug import ConsistencyChecker, shadow_tracer
from state.persistence import agent_state_store
from state.schema import AgentState
from state.updater import update_state

logger = logging.getLogger(__name__)


class ShadowRunner:
    """每轮旁路构建 AgentState + 一致性检查 + 追踪。"""

    def __init__(self, use_intent: bool = False):
        self.use_intent = use_intent
        self._intent_predictor = None

    def _get_intent(self, text: str, conversation=None):
        """惰性加载冻结的 Intent Service（可选，避免拖慢生产）。"""
        if not self.use_intent:
            return None
        try:
            if self._intent_predictor is None:
                import sys
                from pathlib import Path
                root = Path(__file__).resolve().parents[2]
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                from evaluation.intent.small_model import SmallModelPredictor
                self._intent_predictor = SmallModelPredictor(
                    model_dir=str(root / "models/intent/phase1_5_final_model"))
            conv = conversation or [{"role": "user", "content": text}]
            result = self._intent_predictor.predict(text, (conv, 1))
            return result
        except Exception as e:  # noqa: BLE001
            logger.debug("shadow intent 调用失败（不影响生产）: %s", e)
            return None

    def run(
        self,
        *,
        user_id: str,
        session_id: str,
        request_text: str,
        turn_index: int,
        emotion_state: Optional[Dict],
        urgent_issue: Optional[Dict],
        conversation_summary: Optional[Dict],
        user_profile: Optional[Dict],
        risk_baseline: Optional[str] = None,
        historical_high_risk_count: int = 0,
        previous_state: Optional[AgentState] = None,
        trace_id: Optional[str] = None,
        conversation: Optional[list] = None,
    ) -> Optional[AgentState]:
        """旁路构建并追踪 AgentState。失败静默返回 None。"""
        try:
            intent_result = self._get_intent(request_text, conversation)
            summary = conversation_summary or {}
            current = build_agent_state(
                user_id=user_id,
                session_id=session_id,
                request_text=request_text,
                turn_index=turn_index,
                intent_result=intent_result,
                emotion_state=emotion_state,
                urgent_issue=urgent_issue,
                conversation_summary=summary,
                user_profile=user_profile,
                risk_baseline=risk_baseline,
                historical_high_risk_count=historical_high_risk_count,
                trace_id=trace_id,
                source_versions={"intent": "phase1_5_final" if intent_result else "legacy",
                                 "risk": "v4_2_domain_only_v2"},
            )
            state = update_state(previous_state, current)

            # 一致性检查（legacy 对比）
            consistency = ConsistencyChecker.check(state, {
                "risk_level": (urgent_issue or {}).get("level"),
                "emotion": (emotion_state or {}).get("current_emotion"),
                "stage": summary.get("conversation_stage"),
                "turn_count": summary.get("turn_count"),
            })

            # 保存 latest + 追踪
            agent_state_store.save_latest(state)
            shadow_tracer.log_turn(state, consistency)
            return state
        except Exception as e:  # noqa: BLE001
            logger.warning("shadow state 构建失败（不影响生产）: %s", e)
            return None


# 全局单例（默认不开 Intent，避免拖慢生产；可配置开启）
shadow_runner = ShadowRunner(use_intent=False)
