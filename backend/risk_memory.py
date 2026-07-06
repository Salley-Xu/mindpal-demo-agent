"""
长期风险记忆层（v2.0 / Phase 1）

跨会话维护用户近期风险基线、反复触发因素、保护因素和支持偏好。
**主存储已从 user_profile.preferences JSON blob 迁移到独立表。**

用法:
    # 写入（每轮对话结束后）
    await RiskMemoryWriter().update(user_id, session_id, risk_state, summary, emotion_state)

    # 读取（每轮对话开始时）
    memory = await RiskMemoryReader().get_baseline(user_id)
    # => {"baseline": "low", "common_triggers": [...], "protective_factors": [...]}
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from risk_levels import LEVEL_2, LEVEL_3, normalize_risk_level, risk_level_index
from risk_memory_store import risk_memory_store
from models import RiskEvent as RiskEventModel

logger = logging.getLogger(__name__)


# ============================================================
# Topic 提取（保持不变）
# ============================================================

_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "学业": ["考试", "学习", "论文", "毕业", "成绩", "复习", "实验", "考研", "绩点"],
    "求职": ["求职", "面试", "offer", "秋招", "春招", "简历", "工作", "实习", "算法岗"],
    "人际关系": ["朋友", "室友", "同学", "对象", "伴侣", "家人", "父母", "导师", "吵架"],
    "自我评价": ["没用", "不行", "差劲", "失败", "自卑", "不自信", "否定", "能力"],
    "睡眠": ["失眠", "睡不着", "早醒", "熬夜", "睡眠", "入睡"],
    "未来": ["未来", "迷茫", "方向", "前途", "规划", "不知道怎么办"],
    "经济": ["钱", "经济", "贫困", "学费", "欠", "债"],
}


def _extract_topics(text: str) -> List[str]:
    """从文本中提取匹配的风险主题。"""
    if not text:
        return []
    found = []
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            found.append(topic)
    return found


# ============================================================
# 写入器
# ============================================================

class RiskMemoryWriter:
    """长期风险记忆写入器，每轮对话结束后调用。

    v2.0: 主写入目标为 RiskMemoryStore（独立表），
    JSON 兼容写入降级为轻量摘要（仅用于回滚安全）。
    """

    async def update(
        self,
        user_id: str,
        session_id: str,
        risk_state: Dict[str, Any],
        conversation_summary: Dict[str, Any],
        emotion_state: Dict[str, Any],
    ) -> None:
        """更新用户长期风险记忆。"""
        from database import adb_manager
        from agent_tools import UserProfileTool

        level = risk_state.get("level", "level_0")
        level_idx = risk_level_index(level)
        context = risk_state.get("risk_context", {})

        # 第三方/讨论语境不写入用户自身记忆
        if context.get("subject") in ("third_party", "discussion"):
            return

        # 只关注 Level 2+ 的事件
        if level_idx < 2:
            return

        # 构造事件数据
        now = datetime.now()
        topic = self._infer_topic(conversation_summary, emotion_state)
        summary = self._build_event_summary(conversation_summary, emotion_state)
        safety_confirmed = context.get("is_safe_denial", False)

        # ============================================================
        # 主写入：RiskMemoryStore（独立表，事务安全）
        # ============================================================
        try:
            _, baseline = await risk_memory_store.add_risk_event_with_baseline_update(
                RiskEventModel(
                    user_id=user_id,
                    session_id=session_id,
                    risk_level=level,
                    subject=context.get("subject", "self"),
                    topic=topic,
                    summary=summary[:120] if summary else "",
                    evidence_snippet=emotion_state.get("source_text"),
                    safety_confirmed=safety_confirmed,
                )
            )
            if topic:
                await risk_memory_store.add_risk_trigger(user_id, topic)
        except Exception as e:
            logger.error("RiskMemoryStore 写入失败（影响主链路）: %s", e)
            raise

        # ============================================================
        # 兼容写入：JSON 轻量摘要（Phase 2 可移除）
        # ============================================================
        try:
            profile = await UserProfileTool.get_profile(user_id)
            if profile:
                preferences = dict(profile.preferences or {})
                memory: Dict[str, Any] = preferences.get("risk_memory", {})
                if not memory:
                    memory = self._empty_memory()

                memory["baseline"] = baseline.baseline
                memory["baseline_updated_at"] = now.isoformat()
                memory["last_high_risk_time"] = now.isoformat()
                memory["decay_status"] = "active"
                if safety_confirmed:
                    memory["last_safety_confirmation"] = now.isoformat()
                if topic and topic not in memory["common_triggers"]:
                    memory["common_triggers"].append(topic)
                    memory["common_triggers"] = memory["common_triggers"][-5:]

                preferences["risk_memory"] = memory
                await adb_manager.upsert_user_profile(
                    user_id=user_id,
                    risk_level=level,
                    preferences_patch=preferences,
                )
        except Exception as e:
            logger.warning("JSON 兼容写入失败（非致命）: %s", e)

    # ---------------------------------------------------------------
    # 内部
    # ---------------------------------------------------------------

    @staticmethod
    def _empty_memory() -> Dict[str, Any]:
        return {
            "baseline": "low",
            "baseline_updated_at": None,
            "risk_events": [],
            "common_triggers": [],
            "protective_factors": [],
            "last_high_risk_time": None,
            "last_safety_confirmation": None,
            "decay_status": "active",
        }

    @staticmethod
    def _infer_topic(
        conversation_summary: Dict[str, Any],
        emotion_state: Dict[str, Any],
    ) -> str:
        """从对话摘要和情绪状态推断风险主题。"""
        stress_source = emotion_state.get("stress_source")
        if stress_source:
            topics = _extract_topics(str(stress_source))
            if topics:
                return topics[0]

        stress_sources = conversation_summary.get("stress_sources", []) or []
        for src in stress_sources:
            topics = _extract_topics(str(src))
            if topics:
                return topics[0]

        current_topic = conversation_summary.get("current_topic", "")
        if current_topic:
            topics = _extract_topics(current_topic)
            if topics:
                return topics[0]

        return "未分类"

    @staticmethod
    def _build_event_summary(
        conversation_summary: Dict[str, Any],
        emotion_state: Dict[str, Any],
    ) -> str:
        """生成风险事件摘要。"""
        emotion = emotion_state.get("current_emotion") or emotion_state.get("emotion_type") or "负面"
        stress_source = emotion_state.get("stress_source") or ""
        if stress_source:
            return f"因{stress_source}产生{emotion}情绪，表达明显痛苦"
        return f"表达{emotion}情绪，存在较高风险"

    # ---------------------------------------------------------------
    # 以下方法保留供已有测试兼容调用（新代码走 RiskMemoryStore）
    # ---------------------------------------------------------------

    @staticmethod
    def _apply_decay(memory: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        """（兼容）旧 JSON 衰减逻辑，测试用。"""
        m = dict(memory)
        events = m.get("risk_events", [])
        for ev in events:
            try:
                et = datetime.fromisoformat(ev["event_time"])
                if now - et > timedelta(days=30):
                    ev["decayed"] = True
            except (ValueError, TypeError):
                ev["decayed"] = True

        active_triggers = set()
        for ev in events:
            if ev.get("decayed"):
                continue
            topic = ev.get("topic", "")
            if topic:
                active_triggers.add(topic)

        existing = m.get("common_triggers", [])
        kept = [t for t in existing if t in active_triggers]
        for t in existing:
            if t not in active_triggers and len(kept) < 3 and t not in kept:
                kept.append(t)
        m["common_triggers"] = kept
        return m

    @staticmethod
    def _recompute_baseline(risk_events: List[Dict[str, Any]], now: datetime) -> str:
        """（兼容）旧 JSON 基线计算，测试用。"""
        events_7d = []
        events_14d = []

        for ev in risk_events:
            if ev.get("decayed"):
                continue
            try:
                et = datetime.fromisoformat(ev["event_time"])
            except (ValueError, TypeError):
                continue
            days = (now - et).days
            if days < 7:
                events_7d.append(ev)
            if days < 14:
                events_14d.append(ev)

        if any(risk_level_index(ev.get("max_level", "level_0")) >= 3 for ev in events_7d):
            return "high"
        l2_in_7d = sum(1 for ev in events_7d if risk_level_index(ev.get("max_level", "level_0")) >= 2)
        if l2_in_7d >= 2:
            return "high"
        if any(risk_level_index(ev.get("max_level", "level_0")) >= 2 for ev in events_14d):
            return "medium"
        l1_in_14d = sum(1 for ev in events_14d if risk_level_index(ev.get("max_level", "level_0")) >= 1)
        if l1_in_14d >= 3:
            return "medium"
        return "low"


# ============================================================
# 读取器
# ============================================================

class RiskMemoryReader:
    """长期风险记忆读取器，每轮对话开始时调用。

    v2.0: 主读取源为 risk_baselines / risk_triggers 表，
    旧 JSON 作为 fallback（回滚兼容）。
    """

    async def get_baseline(self, user_id: str) -> Dict[str, Any]:
        """
        读取用户长期风险基线。

        返回:
            baseline: str          "low" / "medium" / "high"
            common_triggers: list  反复触发主题
            protective_factors: list 保护因素
        """
        # ============================================================
        # 主读取：RiskMemoryStore（独立表）
        # ============================================================
        try:
            baseline = await risk_memory_store.get_baseline(user_id)
            triggers = await risk_memory_store.get_active_triggers(user_id)
            pfs = await risk_memory_store.get_protective_factors(user_id)

            if baseline:
                return {
                    "baseline": baseline.baseline,
                    "common_triggers": [t.trigger for t in triggers],
                    "protective_factors": [pf.factor for pf in pfs],
                    "last_high_risk_time": (
                        baseline.last_high_risk_time.isoformat()
                        if baseline.last_high_risk_time else None
                    ),
                    "decay_status": baseline.decay_status,
                }
        except Exception as e:
            logger.warning("RiskMemoryStore 读取失败，尝试 JSON fallback: %s", e)

        # ============================================================
        # Fallback：旧 JSON（Phase 2 可移除）
        # ============================================================
        try:
            from agent_tools import UserProfileTool

            profile = await UserProfileTool.get_profile(user_id)
            if not profile:
                return self._default()

            preferences = profile.preferences or {}
            memory = preferences.get("risk_memory", {})
            if not memory:
                return self._default()

            now = datetime.now()
            memory = RiskMemoryWriter._apply_decay(memory, now)
            baseline = memory.get("baseline", "low")

            updated_at = memory.get("baseline_updated_at")
            if updated_at:
                try:
                    updated_dt = datetime.fromisoformat(updated_at)
                    if now - updated_dt > timedelta(hours=1):
                        new_baseline = RiskMemoryWriter._recompute_baseline(
                            memory.get("risk_events", []), now
                        )
                        if new_baseline != baseline:
                            baseline = new_baseline
                except (ValueError, TypeError):
                    pass

            return {
                "baseline": baseline,
                "common_triggers": memory.get("common_triggers", []),
                "protective_factors": memory.get("protective_factors", []),
                "last_high_risk_time": memory.get("last_high_risk_time"),
                "decay_status": memory.get("decay_status", "active"),
            }
        except Exception:
            return self._default()

    @staticmethod
    def _default() -> Dict[str, Any]:
        return {
            "baseline": "low",
            "common_triggers": [],
            "protective_factors": [],
            "last_high_risk_time": None,
            "decay_status": "active",
        }
