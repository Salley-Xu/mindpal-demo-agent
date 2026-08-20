"""
RiskMemoryStore — 风险记忆独立存储（Phase 0）。

将风险记忆从 user_profile.preferences JSON blob 迁移到独立表：
  - risk_events: append-only 事件日志，解决并发覆盖 P0 问题
  - risk_baselines: 每用户一行基线，行级锁安全
  - risk_triggers: 触发词统计与衰减
  - protective_factors: 保护因素

所有 baseline 计算逻辑复用 RiskMemoryWriter 现有规则（算法一致，仅存储层不同）。
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiosqlite

from config import config
from models import RiskEvent, RiskBaseline, RiskTrigger, ProtectiveFactor
from utils import (
    generate_risk_event_id,
    generate_risk_trigger_id,
    generate_protective_factor_id,
)

logger = logging.getLogger(__name__)


class RiskMemoryStore:
    """风险记忆独立存储。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.SESSION_DB_PATH
        self._conn: Optional[aiosqlite.Connection] = None

    async def _open_conn(self) -> aiosqlite.Connection:
        """Open and configure a connection; callers own its lifecycle."""
        if self.db_path == ":memory:" and self._conn is not None:
            return self._conn
        conn = await aiosqlite.connect(self.db_path, timeout=30)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA busy_timeout = 30000")
        await conn.execute("PRAGMA foreign_keys = ON")
        await self._ensure_tables_on_conn(conn)
        if self.db_path == ":memory:":
            self._conn = conn
        return conn

    @asynccontextmanager
    async def _connection(self):
        """Keep in-memory state alive, but always close file-backed connections."""
        conn = await self._open_conn()
        try:
            yield conn
        finally:
            if self.db_path != ":memory:":
                await conn.close()

    async def _close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None

    @staticmethod
    async def _ensure_tables_on_conn(conn: aiosqlite.Connection):
        """在给定连接上创建表。"""
        await conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS risk_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT,
                turn_id TEXT,
                event_time TIMESTAMP NOT NULL,
                risk_level TEXT NOT NULL,
                risk_score REAL,
                subject TEXT NOT NULL DEFAULT 'self',
                topic TEXT,
                summary TEXT NOT NULL,
                evidence_snippet TEXT,
                safety_confirmed BOOLEAN DEFAULT FALSE,
                support_engaged BOOLEAN DEFAULT FALSE,
                decayed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_risk_user_time ON risk_events(user_id, event_time);
            CREATE INDEX IF NOT EXISTS idx_risk_user_level ON risk_events(user_id, risk_level);
            CREATE INDEX IF NOT EXISTS idx_risk_user_decayed ON risk_events(user_id, decayed);

            CREATE TABLE IF NOT EXISTS risk_baselines (
                user_id TEXT PRIMARY KEY,
                baseline TEXT NOT NULL DEFAULT 'low',
                baseline_score REAL,
                baseline_updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                decay_status TEXT DEFAULT 'active',
                last_high_risk_time TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS risk_triggers (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                trigger TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_seen_at TIMESTAMP,
                decayed BOOLEAN DEFAULT FALSE
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_risk_trigger_unique ON risk_triggers(user_id, trigger);

            CREATE TABLE IF NOT EXISTS protective_factors (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                factor TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 0.5,
                last_seen_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        )
        await conn.commit()

    # ============================================================
    # 风险事件（append-only）
    # ============================================================

    async def add_risk_event(self, event: RiskEvent) -> str:
        """追加写入风险事件。"""
        if not event.id:
            event.id = generate_risk_event_id()
        async with self._connection() as conn:
            await conn.execute(
                """
                INSERT INTO risk_events (
                    id, user_id, session_id, turn_id, event_time,
                    risk_level, risk_score, subject, topic, summary,
                    evidence_snippet, safety_confirmed, support_engaged, decayed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.user_id,
                    event.session_id,
                    event.turn_id,
                    (
                        event.event_time.isoformat()
                        if isinstance(event.event_time, datetime)
                        else event.event_time
                    ),
                    event.risk_level,
                    event.risk_score,
                    event.subject,
                    event.topic,
                    event.summary,
                    event.evidence_snippet,
                    int(event.safety_confirmed),
                    int(event.support_engaged),
                    int(event.decayed),
                ),
            )
            await conn.commit()
        return event.id

    async def add_risk_event_with_baseline_update(
        self, event: RiskEvent
    ) -> tuple[str, RiskBaseline]:
        """
        原子操作：写入风险事件 + 重算基线（单事务）。
        利用 SQLite 串行化写事务解决并发覆盖问题。
        """
        if not event.id:
            event.id = generate_risk_event_id()
        now = datetime.now()
        async with self._connection() as conn:
            # 1. 插入事件
            await conn.execute(
                """
            INSERT INTO risk_events (
                id, user_id, session_id, turn_id, event_time,
                risk_level, risk_score, subject, topic, summary,
                evidence_snippet, safety_confirmed, support_engaged, decayed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.id,
                    event.user_id,
                    event.session_id,
                    event.turn_id,
                    (
                        event.event_time.isoformat()
                        if isinstance(event.event_time, datetime)
                        else event.event_time
                    ),
                    event.risk_level,
                    event.risk_score,
                    event.subject,
                    event.topic,
                    event.summary,
                    event.evidence_snippet,
                    int(event.safety_confirmed),
                    int(event.support_engaged),
                    int(event.decayed),
                ),
            )

            # 2. 读取近期未衰减事件用于基线计算
            cursor = await conn.execute(
                """
            SELECT * FROM risk_events
            WHERE user_id = ? AND decayed = 0
            ORDER BY event_time DESC
            """,
                (event.user_id,),
            )
            rows = await cursor.fetchall()

            # 3. 内存中计算基线
            baseline_value = self._compute_baseline_from_rows(rows, now)

            # 4. 回写基线
            await conn.execute(
                """
            INSERT OR REPLACE INTO risk_baselines (
                user_id, baseline, baseline_updated_at, decay_status, last_high_risk_time
            ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    event.user_id,
                    baseline_value,
                    now.isoformat(),
                    "active",
                    now.isoformat() if baseline_value in ("high", "medium") else None,
                ),
            )
            await conn.commit()

        baseline = RiskBaseline(
            user_id=event.user_id,
            baseline=baseline_value,
            baseline_updated_at=now,
        )
        return event.id, baseline

    async def get_recent_events(
        self,
        user_id: str,
        days: int = 14,
        include_decayed: bool = False,
    ) -> List[RiskEvent]:
        """按时间窗口查询风险事件。"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        async with self._connection() as conn:
            if include_decayed:
                cursor = await conn.execute(
                    """
                SELECT * FROM risk_events
                WHERE user_id = ? AND event_time >= ?
                ORDER BY event_time DESC
                """,
                    (user_id, cutoff),
                )
            else:
                cursor = await conn.execute(
                    """
                SELECT * FROM risk_events
                WHERE user_id = ? AND event_time >= ? AND decayed = 0
                ORDER BY event_time DESC
                """,
                    (user_id, cutoff),
                )
            rows = await cursor.fetchall()
        return [self._row_to_risk_event(r) for r in rows]

    # ============================================================
    # 风险基线
    # ============================================================

    async def get_baseline(self, user_id: str) -> Optional[RiskBaseline]:
        """读取当前基线。"""
        async with self._connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM risk_baselines WHERE user_id = ?", (user_id,)
            )
            row = await cursor.fetchone()
        if not row:
            return None
        return RiskBaseline(
            user_id=row["user_id"],
            baseline=row["baseline"],
            baseline_score=row["baseline_score"],
            baseline_updated_at=row["baseline_updated_at"],
            decay_status=row["decay_status"],
            last_high_risk_time=row["last_high_risk_time"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def set_baseline(self, baseline: RiskBaseline) -> None:
        """写入/更新基线。"""
        now = datetime.now().isoformat()
        async with self._connection() as conn:
            await self._set_baseline_on_conn(conn, baseline, now)
            await conn.commit()

    @staticmethod
    async def _set_baseline_on_conn(conn, baseline: RiskBaseline, now: str) -> None:
        await conn.execute(
            """
            INSERT OR REPLACE INTO risk_baselines (
                user_id, baseline, baseline_score, baseline_updated_at,
                decay_status, last_high_risk_time, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                baseline.user_id,
                baseline.baseline,
                baseline.baseline_score,
                (
                    baseline.baseline_updated_at.isoformat()
                    if isinstance(baseline.baseline_updated_at, datetime)
                    else now
                ),
                baseline.decay_status,
                (
                    baseline.last_high_risk_time.isoformat()
                    if isinstance(baseline.last_high_risk_time, datetime)
                    else None
                ),
                json.dumps(baseline.metadata, ensure_ascii=False),
            ),
        )

    async def recompute_baseline(self, user_id: str) -> RiskBaseline:
        """从 risk_events 重算基线。"""
        now = datetime.now()
        async with self._connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM risk_events
                WHERE user_id = ? AND decayed = 0
                ORDER BY event_time DESC
                """,
                (user_id,),
            )
            rows = await cursor.fetchall()

            baseline_value = self._compute_baseline_from_rows(rows, now)
            baseline = RiskBaseline(
                user_id=user_id,
                baseline=baseline_value,
                baseline_updated_at=now,
            )
            await self._set_baseline_on_conn(conn, baseline, now.isoformat())
            await conn.commit()
        return baseline

    async def get_recent_risk_context(
        self, user_id: str, days: int = 14
    ) -> Dict[str, Any]:
        """获取近期风险上下文（供 MemoryInjector 使用）。"""
        events = await self.get_recent_events(user_id, days=days, include_decayed=False)
        baseline = await self.get_baseline(user_id)
        triggers = await self.get_active_triggers(user_id)
        return {
            "baseline": baseline.baseline if baseline else "low",
            "recent_event_count": len(events),
            "max_level_in_window": max(
                (e.risk_level for e in events), default="level_0"
            ),
            "common_triggers": [t.trigger for t in triggers],
            "has_recent_high_risk": any(
                self._level_index(e.risk_level) >= 2 for e in events
            ),
        }

    # ============================================================
    # 风险触发词
    # ============================================================

    async def add_risk_trigger(self, user_id: str, trigger: str) -> None:
        """Upsert 触发词，增加频次，更新最后出现时间。"""
        now = datetime.now().isoformat()
        async with self._connection() as conn:
            await conn.execute(
                """
            INSERT INTO risk_triggers (id, user_id, trigger, frequency, last_seen_at)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(user_id, trigger) DO UPDATE SET
                frequency = COALESCE(frequency, 0) + 1,
                last_seen_at = excluded.last_seen_at,
                decayed = FALSE
            """,
                (generate_risk_trigger_id(), user_id, trigger, now),
            )
            await conn.commit()

    async def get_active_triggers(self, user_id: str) -> List[RiskTrigger]:
        """获取未衰减的触发词。"""
        async with self._connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM risk_triggers
                WHERE user_id = ? AND decayed = 0
                ORDER BY frequency DESC
                LIMIT 10
                """,
                (user_id,),
            )
            rows = await cursor.fetchall()
        return [
            RiskTrigger(
                id=r["id"],
                user_id=r["user_id"],
                trigger=r["trigger"],
                frequency=r["frequency"],
                last_seen_at=r["last_seen_at"],
                decayed=bool(r["decayed"]),
            )
            for r in rows
        ]

    # ============================================================
    # 保护因素
    # ============================================================

    async def add_protective_factor(self, factor: ProtectiveFactor) -> str:
        """添加保护因素。"""
        if not factor.id:
            factor.id = generate_protective_factor_id()
        now = datetime.now().isoformat()
        async with self._connection() as conn:
            await conn.execute(
                """
            INSERT INTO protective_factors (id, user_id, factor, source, confidence, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    factor.id,
                    factor.user_id,
                    factor.factor,
                    factor.source,
                    factor.confidence,
                    now,
                ),
            )
            await conn.commit()
        return factor.id

    async def get_protective_factors(self, user_id: str) -> List[ProtectiveFactor]:
        """获取保护因素列表。"""
        async with self._connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM protective_factors WHERE user_id = ? ORDER BY confidence DESC",
                (user_id,),
            )
            rows = await cursor.fetchall()
        return [
            ProtectiveFactor(
                id=r["id"],
                user_id=r["user_id"],
                factor=r["factor"],
                source=r["source"],
                confidence=r["confidence"],
                last_seen_at=r["last_seen_at"],
            )
            for r in rows
        ]

    # ============================================================
    # 内部：基线计算（复用 RiskMemoryWriter 现有规则）
    # ============================================================

    def _compute_baseline_from_rows(
        self, rows: List[aiosqlite.Row], now: datetime
    ) -> str:
        """
        根据 risk_events 行数据计算基线。
        算法与 RiskMemoryWriter._recompute_baseline 保持一致。
        """
        from risk_levels import risk_level_index

        # 统一时间比较：使用 naive datetime 避免 offset-aware 比较错误
        if now.tzinfo is not None:
            now_naive = now.replace(tzinfo=None)
        else:
            now_naive = now

        events_7d = []
        events_14d = []

        for row in rows:
            try:
                et_str = row["event_time"]
                et = (
                    datetime.fromisoformat(et_str)
                    if isinstance(et_str, str)
                    else et_str
                )
                if et.tzinfo is not None:
                    et = et.replace(tzinfo=None)
            except (ValueError, TypeError):
                continue
            days_diff = (now_naive - et).days
            if days_diff < config.RISK_BASELINE_HIGH_WINDOW_DAYS:
                events_7d.append(row)
            if days_diff < config.RISK_BASELINE_MEDIUM_WINDOW_DAYS:
                events_14d.append(row)

        # 7 天内 Level 3 → high
        if any(risk_level_index(row["risk_level"]) >= 3 for row in events_7d):
            return "high"

        # 7 天内 >= 2 次 Level 2 → high
        l2_in_7d = sum(
            1 for row in events_7d if risk_level_index(row["risk_level"]) >= 2
        )
        if l2_in_7d >= 2:
            return "high"

        # 14 天内 Level 2 → medium
        if any(risk_level_index(row["risk_level"]) >= 2 for row in events_14d):
            return "medium"

        # 14 天内 >= 3 次 Level 1 → medium
        l1_in_14d = sum(
            1 for row in events_14d if risk_level_index(row["risk_level"]) >= 1
        )
        if l1_in_14d >= 3:
            return "medium"

        return "low"

    @staticmethod
    def _level_index(level: str) -> int:
        from risk_levels import risk_level_index as _rli

        return _rli(level)

    @staticmethod
    def _row_to_risk_event(row: aiosqlite.Row) -> RiskEvent:
        return RiskEvent(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            event_time=row["event_time"],
            risk_level=row["risk_level"],
            risk_score=row["risk_score"],
            subject=row["subject"],
            topic=row["topic"],
            summary=row["summary"],
            evidence_snippet=row["evidence_snippet"],
            safety_confirmed=bool(row["safety_confirmed"]),
            support_engaged=bool(row["support_engaged"]),
            decayed=bool(row["decayed"]),
        )


# 全局单例
risk_memory_store = RiskMemoryStore()
