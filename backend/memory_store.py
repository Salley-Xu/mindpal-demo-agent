"""
MemoryStore — 统一记忆存储接口（Phase 0：基础 CRUD）

提供 MemoryItem 的持久化读写，基于 SQLite memory_items 表。
Phase 0 只做基础 CRUD，Phase 3 起接入混合检索。
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite

from config import config
from models import MemoryItem, MemoryCandidate, MemoryQuery, MemorySearchResult
from utils import generate_memory_id

logger = logging.getLogger(__name__)


class MemoryStore:
    """统一记忆存储：MemoryItem 的 CRUD + 检索入口。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.SESSION_DB_PATH

    # ---------------------------------------------------------------
    # 连接管理
    # ---------------------------------------------------------------

    async def _connect(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        return conn

    # ---------------------------------------------------------------
    # 写入
    # ---------------------------------------------------------------

    async def write(self, item: MemoryItem) -> str:
        """写入单条记忆，返回 memory_id。"""
        if not item.id:
            item.id = generate_memory_id()
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT INTO memory_items (
                    id, user_id, session_id, turn_id, memory_type,
                    content, summary, source_text,
                    emotion, emotion_intensity, risk_level, stress_source, user_intent,
                    importance, confidence, sensitivity,
                    tags, metadata, source, scope, status,
                    access_count, created_at, updated_at, last_accessed_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id, item.user_id, item.session_id, item.turn_id, item.memory_type,
                    item.content, item.summary, item.source_text,
                    item.emotion, item.emotion_intensity, item.risk_level, item.stress_source, item.user_intent,
                    item.importance, item.confidence, item.sensitivity,
                    json.dumps(item.tags, ensure_ascii=False),
                    json.dumps(item.metadata, ensure_ascii=False),
                    item.source, item.scope, item.status,
                    item.access_count,
                    now, now, now,
                    item.expires_at.isoformat() if item.expires_at else None,
                ),
            )
            await conn.commit()
        return item.id

    async def batch_write(self, items: List[MemoryItem]) -> List[str]:
        """事务写入多条记忆。"""
        ids = []
        async with aiosqlite.connect(self.db_path) as conn:
            now = datetime.now(timezone.utc).isoformat()
            for item in items:
                if not item.id:
                    item.id = generate_memory_id()
                ids.append(item.id)
                await conn.execute(
                    """
                    INSERT INTO memory_items (
                        id, user_id, session_id, turn_id, memory_type,
                        content, summary, source_text,
                        emotion, emotion_intensity, risk_level, stress_source, user_intent,
                        importance, confidence, sensitivity,
                        tags, metadata, source, scope, status,
                        access_count, created_at, updated_at, last_accessed_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.id, item.user_id, item.session_id, item.turn_id, item.memory_type,
                        item.content, item.summary, item.source_text,
                        item.emotion, item.emotion_intensity, item.risk_level, item.stress_source, item.user_intent,
                        item.importance, item.confidence, item.sensitivity,
                        json.dumps(item.tags, ensure_ascii=False),
                        json.dumps(item.metadata, ensure_ascii=False),
                        item.source, item.scope, item.status,
                        item.access_count,
                        now, now, now,
                        item.expires_at.isoformat() if item.expires_at else None,
                    ),
                )
            await conn.commit()
        return ids

    # ---------------------------------------------------------------
    # 读取
    # ---------------------------------------------------------------

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """按 ID 获取单条记忆。"""
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM memory_items WHERE id = ?", (memory_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_item(row)

    async def get_active_items(
        self,
        user_id: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[MemoryItem]:
        """获取用户 active 状态的记忆条目。"""
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            if memory_types:
                placeholders = ",".join("?" for _ in memory_types)
                cursor = await conn.execute(
                    f"""
                    SELECT * FROM memory_items
                    WHERE user_id = ? AND status = 'active' AND memory_type IN ({placeholders})
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ?
                    """,
                    (user_id, *memory_types, limit),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT * FROM memory_items
                    WHERE user_id = ? AND status = 'active'
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                )
            rows = await cursor.fetchall()
            return [self._row_to_item(r) for r in rows]

    async def search(
        self, query: MemoryQuery, limit: int = 20
    ) -> List[MemorySearchResult]:
        """
        基础搜索（Phase 0：按 user_id + memory_type 过滤 + 关键词粗略匹配）。
        Phase 3 起将使用 MemoryRetriever 多路召回替换此方法。
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            conditions = ["user_id = ?", "status = 'active'"]
            params: List[Any] = [query.text]

            if query.memory_types:
                placeholders = ",".join("?" for _ in query.memory_types)
                conditions.append(f"memory_type IN ({placeholders})")
                params.extend(query.memory_types)

            if query.risk_level:
                conditions.append("risk_level = ?")
                params.append(query.risk_level)

            sql = f"""
                SELECT * FROM memory_items
                WHERE {' AND '.join(conditions)}
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """
            params.append(limit)
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            items = [self._row_to_item(r) for r in rows]
            return [
                MemorySearchResult(item=item, score=0.5, rank=i, retrieval_method="basic")
                for i, item in enumerate(items)
            ]

    # ---------------------------------------------------------------
    # 更新
    # ---------------------------------------------------------------

    async def update(self, memory_id: str, patch: Dict[str, Any]) -> None:
        """部分更新记忆字段。"""
        now = datetime.now(timezone.utc).isoformat()
        patch["updated_at"] = now
        set_clauses = ", ".join(f"{k} = ?" for k in patch)
        values = list(patch.values()) + [memory_id]
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                f"UPDATE memory_items SET {set_clauses} WHERE id = ?", values
            )
            await conn.commit()

    async def archive(self, memory_id: str, reason: str = "") -> None:
        """归档记忆（设置 status='archived'）。"""
        async with aiosqlite.connect(self.db_path) as conn:
            now = datetime.now(timezone.utc).isoformat()
            await conn.execute(
                "UPDATE memory_items SET status = 'archived', updated_at = ?, metadata = json_set(COALESCE(metadata, '{}'), '$.archive_reason', ?) WHERE id = ?",
                (now, reason, memory_id),
            )
            await conn.commit()

    async def delete(self, memory_id: str) -> None:
        """软删除（设置 status='deleted'）。"""
        async with aiosqlite.connect(self.db_path) as conn:
            now = datetime.now(timezone.utc).isoformat()
            await conn.execute(
                "UPDATE memory_items SET status = 'deleted', updated_at = ? WHERE id = ?",
                (now, memory_id),
            )
            await conn.commit()

    # ---------------------------------------------------------------
    # 辅助
    # ---------------------------------------------------------------

    @staticmethod
    def _row_to_item(row: aiosqlite.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            memory_type=row["memory_type"],
            content=row["content"],
            summary=row["summary"],
            source_text=row["source_text"],
            emotion=row["emotion"],
            emotion_intensity=row["emotion_intensity"],
            risk_level=row["risk_level"],
            stress_source=row["stress_source"],
            user_intent=row["user_intent"],
            importance=row["importance"],
            confidence=row["confidence"],
            sensitivity=row["sensitivity"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            source=row["source"],
            scope=row["scope"],
            status=row["status"],
            access_count=row["access_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            expires_at=row["expires_at"],
        )


# 全局单例
memory_store = MemoryStore()
