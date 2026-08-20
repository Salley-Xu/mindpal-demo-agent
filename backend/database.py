import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import json
import logging
from config import config
import threading
import asyncio
import aiosqlite
from risk_levels import normalize_risk_level

logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理器 - 处理会话持久化"""
    
    def __init__(self, db_path: str = None, pool_size: int = 5):
        self.db_path = db_path or config.SESSION_DB_PATH
        self.pool_size = pool_size
        self._connection_pool = []
        self._lock = threading.Lock()
        self._init_database()
        self._init_connection_pool()
    
    def _init_connection_pool(self):
        """初始化连接池"""
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            self._connection_pool.append(conn)
        logger.info(f"数据库连接池初始化完成，大小: {self.pool_size}")
    
    def _get_connection(self):
        """从连接池获取连接"""
        with self._lock:
            if not self._connection_pool:
                # 连接池为空，创建新连接
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                logger.warning("连接池为空，创建新连接")
                return conn
            return self._connection_pool.pop()
    
    def _return_connection(self, conn):
        """归还连接到连接池"""
        try:
            with self._lock:
                if len(self._connection_pool) < self.pool_size:
                    self._connection_pool.append(conn)
                else:
                    # 连接池已满，关闭多余连接
                    conn.close()
        except Exception as e:
            logger.error(f"归还连接失败: {e}")
            try:
                conn.close()
            except:
                pass
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接上下文管理器"""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            self._return_connection(conn)
    
    def _init_database(self):
        """初始化数据库表结构"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    conversation_stage TEXT DEFAULT 'initial',
                    key_concerns TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, session_id)
                )
            """)
            
            # 对话历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    turn_number INTEGER NOT NULL,
                    user_input TEXT NOT NULL,
                    detected_emotion TEXT NOT NULL,
                    context_emotion TEXT,
                    confidence REAL DEFAULT 0.5,
                    ai_response TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            
            # 用户画像表
            # ── 情绪存储职责 ──────────────────────────────────────────
            # conversation_history:  每轮 detected_emotion / context_emotion / confidence
            # mood_events:           跨会话丰富情绪事件（含 emotion_type / intensity 等）
            # 注意：不再使用独立的 emotion_timeline 表——历史情绪查询走 conversation_history，
            #       跨会话分析走 mood_events。
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    user_id TEXT PRIMARY KEY,
                    risk_level TEXT DEFAULT 'normal',
                    preferences TEXT DEFAULT '{}',
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 情绪事件表（跨会话的情绪追踪）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mood_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    source TEXT DEFAULT 'conversation',
                    text_snippet TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    turn_number INTEGER NOT NULL,
                    recommend_type TEXT NOT NULL,
                    item_ids TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    content_id TEXT NOT NULL,
                    feedback TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, content_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_lookup ON sessions(user_id, session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_session ON conversation_history(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_active ON sessions(last_active)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_profile ON user_profile(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood_events_user ON mood_events(user_id, created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_events_session ON recommendation_events(session_id, created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_feedback_session ON recommendation_feedback(session_id, updated_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_feedback_user ON recommendation_feedback(user_id, updated_at)")

            # ============================================================
            # Memory System v2.0 新表（Phase 0）
            # ============================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    turn_id TEXT,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT,
                    source_text TEXT,
                    emotion TEXT,
                    emotion_intensity REAL,
                    risk_level TEXT,
                    stress_source TEXT,
                    user_intent TEXT,
                    importance REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 0.5,
                    sensitivity TEXT DEFAULT 'normal',
                    tags TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    source TEXT DEFAULT 'inferred',
                    scope TEXT DEFAULT 'long_term',
                    status TEXT DEFAULT 'active',
                    access_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed_at TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_user_type ON memory_items(user_id, memory_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_user_status ON memory_items(user_id, status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory_items(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_expires_at ON memory_items(expires_at)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_vector BLOB,
                    vector_index_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(memory_id) REFERENCES memory_items(id)
                )
            """)

            cursor.execute("""
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
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_user_time ON risk_events(user_id, event_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_user_level ON risk_events(user_id, risk_level)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_user_decayed ON risk_events(user_id, decayed)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_baselines (
                    user_id TEXT PRIMARY KEY,
                    baseline TEXT NOT NULL DEFAULT 'low',
                    baseline_score REAL,
                    baseline_updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    decay_status TEXT DEFAULT 'active',
                    last_high_risk_time TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_triggers (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_seen_at TIMESTAMP,
                    decayed BOOLEAN DEFAULT FALSE
                )
            """)
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_risk_trigger_unique ON risk_triggers(user_id, trigger)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS protective_factors (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    factor TEXT NOT NULL,
                    source TEXT,
                    confidence REAL DEFAULT 0.5,
                    last_seen_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_outbox (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    retry_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self._run_schema_migrations(cursor)

            logger.info("数据库表结构初始化完成")

    def _run_schema_migrations(self, cursor):
        """以兼容方式补充新增字段，避免破坏已有数据库。"""
        self._ensure_column_exists(cursor, "mood_events", "emotion_type", "TEXT")
        self._ensure_column_exists(cursor, "mood_events", "emotion_intensity", "REAL DEFAULT 0.5")
        self._ensure_column_exists(cursor, "mood_events", "stress_source", "TEXT")
        self._ensure_column_exists(cursor, "mood_events", "user_intent", "TEXT")
        self._ensure_column_exists(cursor, "mood_events", "event_summary", "TEXT")
        self._ensure_column_exists(cursor, "mood_events", "risk_level", "TEXT DEFAULT 'low'")
        # 推荐追踪字段（v4.4）
        self._ensure_column_exists(cursor, "recommendation_events", "gate_score", "REAL")
        self._ensure_column_exists(cursor, "recommendation_events", "gate_threshold", "REAL")
        self._ensure_column_exists(cursor, "recommendation_events", "reason_codes", "TEXT DEFAULT '[]'")
        self._ensure_column_exists(cursor, "recommendation_events", "cooldown_remaining", "INTEGER DEFAULT 0")
        self._ensure_column_exists(cursor, "recommendation_events", "safety_overridden", "INTEGER DEFAULT 0")

    def _ensure_column_exists(self, cursor, table_name: str, column_name: str, definition: str):
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row["name"] for row in cursor.fetchall()}
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
            )
            logger.info("已为 %s 添加字段 %s", table_name, column_name)

    def _normalize_risk_level(self, risk_level: Optional[str]) -> str:
        return normalize_risk_level(risk_level)
    
    def create_or_update_session(self, user_id: str, session_id: str, 
                                conversation_stage: str = 'initial',
                                key_concerns: List[str] = None) -> int:
        """创建或更新会话，返回session_id"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            concerns_json = json.dumps(key_concerns or [], ensure_ascii=False)
            
            cursor.execute("""
                INSERT INTO sessions (user_id, session_id, conversation_stage, key_concerns, last_active)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, session_id) 
                DO UPDATE SET 
                    conversation_stage = excluded.conversation_stage,
                    key_concerns = excluded.key_concerns,
                    last_active = CURRENT_TIMESTAMP
            """, (user_id, session_id, conversation_stage, concerns_json))
            
            # 获取会话ID
            cursor.execute("SELECT id FROM sessions WHERE user_id = ? AND session_id = ?", 
                         (user_id, session_id))
            result = cursor.fetchone()
            return result['id'] if result else None
    
    def add_conversation_turn(self, session_db_id: int, turn_number: int,
                             user_input: str, detected_emotion: str,
                             context_emotion: str, confidence: float,
                             ai_response: str):
        """添加对话轮次"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversation_history 
                (session_id, turn_number, user_input, detected_emotion, 
                 context_emotion, confidence, ai_response)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_db_id, turn_number, user_input, detected_emotion,
                  context_emotion, confidence, ai_response))
    
    # 情绪时间线已合并到 conversation_history.detected_emotion，不再使用独立的 emotion_timeline 表。
    # 跨会话丰富情绪事件走 mood_events。
    
    def get_session_data(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话完整数据"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 获取会话基本信息
            cursor.execute("""
                SELECT id, user_id, session_id, conversation_stage, 
                       key_concerns, created_at, last_active
                FROM sessions 
                WHERE user_id = ? AND session_id = ?
            """, (user_id, session_id))
            
            session_row = cursor.fetchone()
            if not session_row:
                return None
            
            session_db_id = session_row['id']
            
            # 获取对话历史
            cursor.execute("""
                SELECT turn_number, user_input, detected_emotion, 
                       context_emotion, confidence, ai_response, timestamp
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY turn_number ASC
            """, (session_db_id,))
            
            history_rows = cursor.fetchall()
            history = [
                {
                    'turn_number': row['turn_number'],
                    'user_input': row['user_input'],
                    'detected_emotion': row['detected_emotion'],
                    'context_emotion': row['context_emotion'],
                    'confidence': row['confidence'],
                    'ai_response': row['ai_response'],
                    'timestamp': row['timestamp']
                }
                for row in history_rows
            ]
            
            # 获取情绪时间线（从 conversation_history 构建，替代已移除的 emotion_timeline 表）
            cursor.execute("""
                SELECT detected_emotion AS emotion, user_input AS text_snippet, timestamp
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY turn_number ASC
            """, (session_db_id,))
            
            emotion_rows = cursor.fetchall()
            emotion_timeline = [
                {
                    'emotion': row['emotion'],
                    'text_snippet': row['text_snippet'],
                    'timestamp': row['timestamp']
                }
                for row in emotion_rows
            ]

            cursor.execute("""
                SELECT turn_number, recommend_type, item_ids, created_at
                FROM recommendation_events
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
            """, (session_db_id,))
            recommendation_event_rows = cursor.fetchall()
            recommendation_events = [
                {
                    "turn_number": row["turn_number"],
                    "recommend_type": row["recommend_type"],
                    "item_ids": json.loads(row["item_ids"] or "[]"),
                    "timestamp": row["created_at"],
                }
                for row in recommendation_event_rows
            ]

            cursor.execute("""
                SELECT content_id, feedback, created_at, updated_at
                FROM recommendation_feedback
                WHERE session_id = ?
                ORDER BY updated_at ASC, id ASC
            """, (session_db_id,))
            recommendation_feedback_rows = cursor.fetchall()
            recommendation_feedback = {
                row["content_id"]: row["feedback"]
                for row in recommendation_feedback_rows
            }
            
            return {
                'id': session_db_id,
                'user_id': session_row['user_id'],
                'session_id': session_row['session_id'],
                'conversation_stage': session_row['conversation_stage'],
                'key_concerns': json.loads(session_row['key_concerns']),
                'created_at': session_row['created_at'],
                'last_active': session_row['last_active'],
                'history': history,
                'emotion_timeline': emotion_timeline,
                'recommendation_events': recommendation_events,
                'recommendation_feedback': recommendation_feedback,
            }
    
    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取用户的所有会话列表"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_id, session_id, conversation_stage,
                       key_concerns, created_at, last_active
                FROM sessions
                WHERE user_id = ?
                ORDER BY last_active DESC
                LIMIT ?
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            return [
                {
                    'id': row['id'],
                    'user_id': row['user_id'],
                    'session_id': row['session_id'],
                    'conversation_stage': row['conversation_stage'],
                    'key_concerns': json.loads(row['key_concerns']),
                    'created_at': row['created_at'],
                    'last_active': row['last_active']
                }
                for row in rows
            ]
    
    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除会话"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM sessions 
                WHERE user_id = ? AND session_id = ?
            """, (user_id, session_id))
            return cursor.rowcount > 0
    
    def cleanup_expired_sessions(self, days: int = 30) -> int:
        """清理过期会话"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM sessions 
                WHERE last_active < ?
            """, (cutoff_date.isoformat(),))
            deleted_count = cursor.rowcount
            logger.info(f"清理了 {deleted_count} 个过期会话")
            return deleted_count
    
    def get_session_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取会话统计信息"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if user_id:
                # 单用户统计
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        AVG(CAST((julianday('now') - julianday(last_active)) * 24 * 60 as REAL)) as avg_inactive_minutes,
                        MAX(last_active) as last_active
                    FROM sessions
                    WHERE user_id = ?
                """, (user_id,))
            else:
                # 全局统计
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(DISTINCT user_id) as total_users,
                        AVG(CAST((julianday('now') - julianday(last_active)) * 24 * 60 as REAL)) as avg_inactive_minutes
                    FROM sessions
                """)
            
            row = cursor.fetchone()
            
            # 获取情绪分布（从 conversation_history 查询，替代已移除的 emotion_timeline 表）
            if user_id:
                cursor.execute("""
                    SELECT ch.detected_emotion AS emotion, COUNT(*) as count
                    FROM conversation_history ch
                    JOIN sessions s ON ch.session_id = s.id
                    WHERE s.user_id = ?
                    GROUP BY ch.detected_emotion
                    ORDER BY count DESC
                    LIMIT 5
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT detected_emotion AS emotion, COUNT(*) as count
                    FROM conversation_history
                    GROUP BY detected_emotion
                    ORDER BY count DESC
                    LIMIT 5
                """)
            
            emotion_dist = cursor.fetchall()
            
            return {
                'total_sessions': row['total_sessions'] if row else 0,
                'total_users': row.get('total_users', 0) if row else 0,
                'avg_inactive_minutes': row['avg_inactive_minutes'] if row else 0,
                'last_active': row.get('last_active') if row else None,
                'top_emotions': [
                    {'emotion': e['emotion'], 'count': e['count']}
                    for e in emotion_dist
                ]
            }

    # ---------- 用户画像与情绪事件（同步接口，主要用于管理/调试） ----------

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, risk_level, preferences, last_updated
                FROM user_profile
                WHERE user_id = ?
                """,
                (user_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "user_id": row["user_id"],
                "risk_level": self._normalize_risk_level(row["risk_level"]),
                "preferences": json.loads(row["preferences"] or "{}"),
                "last_updated": row["last_updated"],
            }

    def upsert_user_profile(
        self, user_id: str, risk_level: Optional[str], preferences_patch: Dict[str, Any]
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            existing = self.get_user_profile(user_id)
            if existing:
                merged_preferences = existing["preferences"]
                merged_preferences.update(preferences_patch or {})
                final_risk = self._normalize_risk_level(risk_level or existing["risk_level"])
            else:
                merged_preferences = preferences_patch or {}
                final_risk = self._normalize_risk_level(risk_level or "low")

            cursor.execute(
                """
                INSERT INTO user_profile (user_id, risk_level, preferences, last_updated)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    risk_level = excluded.risk_level,
                    preferences = excluded.preferences,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (user_id, final_risk, json.dumps(merged_preferences, ensure_ascii=False)),
            )

    def add_mood_event(
        self,
        user_id: str,
        session_id: str,
        emotion: str,
        source: str,
        text_snippet: str,
        emotion_type: Optional[str] = None,
        emotion_intensity: Optional[float] = None,
        stress_source: Optional[str] = None,
        user_intent: Optional[str] = None,
        event_summary: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> str:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO mood_events (
                    user_id, session_id, emotion, source, text_snippet,
                    emotion_type, emotion_intensity, stress_source, user_intent,
                    event_summary, risk_level
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    emotion,
                    source,
                    text_snippet[:100],
                    emotion_type or emotion,
                    emotion_intensity if emotion_intensity is not None else 0.5,
                    stress_source,
                    user_intent,
                    event_summary[:200] if event_summary else None,
                    self._normalize_risk_level(risk_level),
                ),
            )
            cursor.execute("SELECT created_at FROM mood_events WHERE id = last_insert_rowid()")
            row = cursor.fetchone()
            return row["created_at"] if row else datetime.now().isoformat()

    def add_recommendation_event(
        self,
        session_db_id: int,
        turn_number: int,
        recommend_type: str,
        item_ids: Optional[List[str]] = None,
        trace_data: Optional[Dict[str, Any]] = None,
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if trace_data:
                cursor.execute(
                    """
                    INSERT INTO recommendation_events
                        (session_id, turn_number, recommend_type, item_ids,
                         gate_score, gate_threshold, reason_codes, cooldown_remaining, safety_overridden)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_db_id,
                        turn_number,
                        recommend_type,
                        json.dumps(item_ids or [], ensure_ascii=False),
                        trace_data.get("gate_score"),
                        trace_data.get("gate_threshold"),
                        json.dumps(trace_data.get("reason_codes", []), ensure_ascii=False),
                        trace_data.get("cooldown_remaining"),
                        1 if trace_data.get("safety_overridden") else 0,
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO recommendation_events (session_id, turn_number, recommend_type, item_ids)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        session_db_id,
                        turn_number,
                        recommend_type,
                        json.dumps(item_ids or [], ensure_ascii=False),
                    ),
                )

    def upsert_recommendation_feedback(
        self,
        session_db_id: int,
        user_id: str,
        content_id: str,
        feedback: str,
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO recommendation_feedback (
                    session_id, user_id, content_id, feedback, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id, content_id) DO UPDATE SET
                    feedback = excluded.feedback,
                    user_id = excluded.user_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_db_id, user_id, content_id, feedback),
            )

    def get_recent_mood_events(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, session_id, emotion, source, text_snippet, created_at,
                       emotion_type, emotion_intensity, stress_source, user_intent,
                       event_summary, risk_level
                FROM mood_events
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            rows = cursor.fetchall()
            return [
                {
                    "user_id": r["user_id"],
                    "session_id": r["session_id"],
                    "emotion": r["emotion"],
                    "source": r["source"],
                    "text_snippet": r["text_snippet"],
                    "emotion_type": r["emotion_type"] or r["emotion"],
                    "emotion_intensity": r["emotion_intensity"] if r["emotion_intensity"] is not None else 0.5,
                    "stress_source": r["stress_source"],
                    "user_intent": r["user_intent"],
                    "event_summary": r["event_summary"],
                    "risk_level": self._normalize_risk_level(r["risk_level"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

# 创建全局数据库实例
db_manager = DatabaseManager()

class AsyncDatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.SESSION_DB_PATH
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self):
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        conversation_stage TEXT DEFAULT 'initial',
                        key_concerns TEXT DEFAULT '[]',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, session_id)
                    )
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        turn_number INTEGER NOT NULL,
                        user_input TEXT NOT NULL,
                        detected_emotion TEXT NOT NULL,
                        context_emotion TEXT,
                        confidence REAL DEFAULT 0.5,
                        ai_response TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    )
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_profile (
                        user_id TEXT PRIMARY KEY,
                        risk_level TEXT DEFAULT 'normal',
                        preferences TEXT DEFAULT '{}',
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS mood_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        emotion TEXT NOT NULL,
                        source TEXT DEFAULT 'conversation',
                        text_snippet TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS recommendation_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        turn_number INTEGER NOT NULL,
                        recommend_type TEXT NOT NULL,
                        item_ids TEXT DEFAULT '[]',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    )
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS recommendation_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        user_id TEXT NOT NULL,
                        content_id TEXT NOT NULL,
                        feedback TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(session_id, content_id),
                        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    )
                """)
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_session_lookup ON sessions(user_id, session_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_history_session ON conversation_history(session_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_last_active ON sessions(last_active)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_profile ON user_profile(user_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_mood_events_user ON mood_events(user_id, created_at)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_events_session ON recommendation_events(session_id, created_at)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_feedback_session ON recommendation_feedback(session_id, updated_at)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_feedback_user ON recommendation_feedback(user_id, updated_at)")

                # ============================================================
                # Memory System v2.0 新表（Phase 0）
                # ============================================================
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_items (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_id TEXT,
                        turn_id TEXT,
                        memory_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        summary TEXT,
                        source_text TEXT,
                        emotion TEXT,
                        emotion_intensity REAL,
                        risk_level TEXT,
                        stress_source TEXT,
                        user_intent TEXT,
                        importance REAL DEFAULT 0.5,
                        confidence REAL DEFAULT 0.5,
                        sensitivity TEXT DEFAULT 'normal',
                        tags TEXT DEFAULT '[]',
                        metadata TEXT DEFAULT '{}',
                        source TEXT DEFAULT 'inferred',
                        scope TEXT DEFAULT 'long_term',
                        status TEXT DEFAULT 'active',
                        access_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_accessed_at TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_user_type ON memory_items(user_id, memory_type)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_user_status ON memory_items(user_id, status)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory_items(created_at)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_expires_at ON memory_items(expires_at)")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_embeddings (
                        memory_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        embedding_model TEXT NOT NULL,
                        embedding_vector BLOB,
                        vector_index_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(memory_id) REFERENCES memory_items(id)
                    )
                """)
                await conn.execute("""
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
                    )
                """)
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_user_time ON risk_events(user_id, event_time)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_user_level ON risk_events(user_id, risk_level)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_user_decayed ON risk_events(user_id, decayed)")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS risk_baselines (
                        user_id TEXT PRIMARY KEY,
                        baseline TEXT NOT NULL DEFAULT 'low',
                        baseline_score REAL,
                        baseline_updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        decay_status TEXT DEFAULT 'active',
                        last_high_risk_time TIMESTAMP,
                        metadata TEXT DEFAULT '{}'
                    )
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS risk_triggers (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        trigger TEXT NOT NULL,
                        frequency INTEGER DEFAULT 1,
                        last_seen_at TIMESTAMP,
                        decayed BOOLEAN DEFAULT FALSE
                    )
                """)
                await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_risk_trigger_unique ON risk_triggers(user_id, trigger)")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS protective_factors (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        factor TEXT NOT NULL,
                        source TEXT,
                        confidence REAL DEFAULT 0.5,
                        last_seen_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_outbox (
                        id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        retry_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                await self._run_schema_migrations(conn)
                await conn.commit()
            self._initialized = True

    async def _run_schema_migrations(self, conn):
        await self._ensure_column_exists(conn, "mood_events", "emotion_type", "TEXT")
        await self._ensure_column_exists(conn, "mood_events", "emotion_intensity", "REAL DEFAULT 0.5")
        await self._ensure_column_exists(conn, "mood_events", "stress_source", "TEXT")
        await self._ensure_column_exists(conn, "mood_events", "user_intent", "TEXT")
        await self._ensure_column_exists(conn, "mood_events", "event_summary", "TEXT")
        await self._ensure_column_exists(conn, "mood_events", "risk_level", "TEXT DEFAULT 'low'")
        # 推荐追踪字段（v4.4）
        await self._ensure_column_exists(conn, "recommendation_events", "gate_score", "REAL")
        await self._ensure_column_exists(conn, "recommendation_events", "gate_threshold", "REAL")
        await self._ensure_column_exists(conn, "recommendation_events", "reason_codes", "TEXT DEFAULT '[]'")
        await self._ensure_column_exists(conn, "recommendation_events", "cooldown_remaining", "INTEGER DEFAULT 0")
        await self._ensure_column_exists(conn, "recommendation_events", "safety_overridden", "INTEGER DEFAULT 0")

    async def _ensure_column_exists(self, conn, table_name: str, column_name: str, definition: str):
        cursor = await conn.execute(f"PRAGMA table_info({table_name})")
        rows = await cursor.fetchall()
        existing_columns = {row[1] for row in rows}
        if column_name not in existing_columns:
            await conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
            )
            logger.info("已为 %s 添加字段 %s", table_name, column_name)

    def _normalize_risk_level(self, risk_level: Optional[str]) -> str:
        return normalize_risk_level(risk_level)

    async def create_or_update_session(self, user_id: str, session_id: str,
                                       conversation_stage: str = 'initial',
                                       key_concerns: List[str] = None) -> int:
        await self._ensure_initialized()
        concerns_json = json.dumps(key_concerns or [], ensure_ascii=False)
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT INTO sessions (user_id, session_id, conversation_stage, key_concerns, last_active)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, session_id) 
                DO UPDATE SET 
                    conversation_stage = excluded.conversation_stage,
                    key_concerns = excluded.key_concerns,
                    last_active = CURRENT_TIMESTAMP
            """, (user_id, session_id, conversation_stage, concerns_json))
            cursor = await conn.execute("SELECT id FROM sessions WHERE user_id = ? AND session_id = ?", (user_id, session_id))
            row = await cursor.fetchone()
            await conn.commit()
            return row[0] if row else None

    async def add_conversation_turn(self, session_db_id: int, turn_number: int,
                                    user_input: str, detected_emotion: str,
                                    context_emotion: str, confidence: float,
                                    ai_response: str):
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("""
                INSERT INTO conversation_history 
                (session_id, turn_number, user_input, detected_emotion, 
                 context_emotion, confidence, ai_response)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_db_id, turn_number, user_input, detected_emotion,
                  context_emotion, confidence, ai_response))
            await conn.commit()

    # 情绪时间线已合并到 conversation_history.detected_emotion，不再使用独立的 emotion_timeline 表。
    # 跨会话丰富情绪事件走 mood_events。

    async def get_session_data(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("""
                SELECT id, user_id, session_id, conversation_stage, 
                       key_concerns, created_at, last_active
                FROM sessions 
                WHERE user_id = ? AND session_id = ?
            """, (user_id, session_id))
            session_row = await cursor.fetchone()
            if not session_row:
                return None
            session_db_id = session_row["id"]
            cursor = await conn.execute("""
                SELECT turn_number, user_input, detected_emotion, 
                       context_emotion, confidence, ai_response, timestamp
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY turn_number ASC
            """, (session_db_id,))
            history_rows = await cursor.fetchall()
            history = [
                {
                    'turn_number': r["turn_number"],
                    'user_input': r["user_input"],
                    'detected_emotion': r["detected_emotion"],
                    'context_emotion': r["context_emotion"],
                    'confidence': r["confidence"],
                    'ai_response': r["ai_response"],
                    'timestamp': r["timestamp"]
                }
                for r in history_rows
            ]
            cursor = await conn.execute("""
                SELECT detected_emotion AS emotion, user_input AS text_snippet, timestamp
                FROM conversation_history
                WHERE session_id = ?
                ORDER BY turn_number ASC
            """, (session_db_id,))
            emotion_rows = await cursor.fetchall()
            emotion_timeline = [
                {
                    'emotion': r["emotion"],
                    'text_snippet': r["text_snippet"],
                    'timestamp': r["timestamp"]
                }
                for r in emotion_rows
            ]
            cursor = await conn.execute("""
                SELECT turn_number, recommend_type, item_ids, created_at
                FROM recommendation_events
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
            """, (session_db_id,))
            recommendation_event_rows = await cursor.fetchall()
            recommendation_events = [
                {
                    'turn_number': r["turn_number"],
                    'recommend_type': r["recommend_type"],
                    'item_ids': json.loads(r["item_ids"] or "[]"),
                    'timestamp': r["created_at"]
                }
                for r in recommendation_event_rows
            ]
            cursor = await conn.execute("""
                SELECT content_id, feedback
                FROM recommendation_feedback
                WHERE session_id = ?
                ORDER BY updated_at ASC, id ASC
            """, (session_db_id,))
            recommendation_feedback_rows = await cursor.fetchall()
            recommendation_feedback = {
                r["content_id"]: r["feedback"]
                for r in recommendation_feedback_rows
            }
            return {
                'id': session_db_id,
                'user_id': session_row["user_id"],
                'session_id': session_row["session_id"],
                'conversation_stage': session_row["conversation_stage"],
                'key_concerns': json.loads(session_row["key_concerns"]),
                'created_at': session_row["created_at"],
                'last_active': session_row["last_active"],
                'history': history,
                'emotion_timeline': emotion_timeline,
                'recommendation_events': recommendation_events,
                'recommendation_feedback': recommendation_feedback,
            }

    async def cleanup_expired_sessions(self, days: int = 30) -> int:
        await self._ensure_initialized()
        cutoff_date = datetime.now() - timedelta(days=days)
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("""
                DELETE FROM sessions 
                WHERE last_active < ?
            """, (cutoff_date.isoformat(),))
            await conn.commit()
            return cursor.rowcount
    
    async def get_session_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            if user_id:
                cursor = await conn.execute("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        AVG(CAST((julianday('now') - julianday(last_active)) * 24 * 60 as REAL)) as avg_inactive_minutes,
                        MAX(last_active) as last_active
                    FROM sessions
                    WHERE user_id = ?
                """, (user_id,))
            else:
                cursor = await conn.execute("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(DISTINCT user_id) as total_users,
                        AVG(CAST((julianday('now') - julianday(last_active)) * 24 * 60 as REAL)) as avg_inactive_minutes
                    FROM sessions
                """)
            row = await cursor.fetchone()
            if user_id:
                emo_cur = await conn.execute("""
                    SELECT ch.detected_emotion AS emotion, COUNT(*) as count
                    FROM conversation_history ch
                    JOIN sessions s ON ch.session_id = s.id
                    WHERE s.user_id = ?
                    GROUP BY ch.detected_emotion
                    ORDER BY count DESC
                    LIMIT 5
                """, (user_id,))
            else:
                emo_cur = await conn.execute("""
                    SELECT detected_emotion AS emotion, COUNT(*) as count
                    FROM conversation_history
                    GROUP BY detected_emotion
                    ORDER BY count DESC
                    LIMIT 5
                """)
            emotion_dist = await emo_cur.fetchall()
            return {
                'total_sessions': row['total_sessions'] if row else 0,
                'total_users': (row['total_users'] if 'total_users' in row.keys() else 0) if row else 0,
                'avg_inactive_minutes': row['avg_inactive_minutes'] if row else 0,
                'last_active': (row['last_active'] if 'last_active' in row.keys() else None) if row else None,
                'top_emotions': [
                    {'emotion': e['emotion'], 'count': e['count']}
                    for e in emotion_dist
                ]
            }

    # ---------- 用户画像与情绪事件（异步接口，供 Agent 工具使用） ----------

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                """
                SELECT user_id, risk_level, preferences, last_updated
                FROM user_profile
                WHERE user_id = ?
                """,
                (user_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "user_id": row["user_id"],
                "risk_level": self._normalize_risk_level(row["risk_level"]),
                "preferences": json.loads(row["preferences"] or "{}"),
                "last_updated": row["last_updated"],
            }

    async def upsert_user_profile(
        self, user_id: str, risk_level: Optional[str], preferences_patch: Dict[str, Any]
    ):
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            # 读取现有
            cursor = await conn.execute(
                """
                SELECT risk_level, preferences FROM user_profile WHERE user_id = ?
                """,
                (user_id,),
            )
            row = await cursor.fetchone()
            if row:
                merged_preferences = json.loads(row["preferences"] or "{}")
                merged_preferences.update(preferences_patch or {})
                final_risk = self._normalize_risk_level(risk_level or row["risk_level"])
            else:
                merged_preferences = preferences_patch or {}
                final_risk = self._normalize_risk_level(risk_level or "low")

            await conn.execute(
                """
                INSERT INTO user_profile (user_id, risk_level, preferences, last_updated)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    risk_level = excluded.risk_level,
                    preferences = excluded.preferences,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (user_id, final_risk, json.dumps(merged_preferences, ensure_ascii=False)),
            )
            await conn.commit()

    async def add_mood_event(
        self,
        user_id: str,
        session_id: str,
        emotion: str,
        source: str,
        text_snippet: str,
        emotion_type: Optional[str] = None,
        emotion_intensity: Optional[float] = None,
        stress_source: Optional[str] = None,
        user_intent: Optional[str] = None,
        event_summary: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> str:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT INTO mood_events (
                    user_id, session_id, emotion, source, text_snippet,
                    emotion_type, emotion_intensity, stress_source, user_intent,
                    event_summary, risk_level
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    emotion,
                    source,
                    text_snippet[:100],
                    emotion_type or emotion,
                    emotion_intensity if emotion_intensity is not None else 0.5,
                    stress_source,
                    user_intent,
                    event_summary[:200] if event_summary else None,
                    self._normalize_risk_level(risk_level),
                ),
            )
            cursor = await conn.execute(
                "SELECT created_at FROM mood_events WHERE id = last_insert_rowid()"
            )
            row = await cursor.fetchone()
            await conn.commit()
            return row[0] if row else datetime.now().isoformat()

    async def get_recent_mood_events(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                """
                SELECT user_id, session_id, emotion, source, text_snippet, created_at,
                       emotion_type, emotion_intensity, stress_source, user_intent,
                       event_summary, risk_level
                FROM mood_events
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "user_id": r["user_id"],
                    "session_id": r["session_id"],
                    "emotion": r["emotion"],
                    "source": r["source"],
                    "text_snippet": r["text_snippet"],
                    "emotion_type": r["emotion_type"] or r["emotion"],
                    "emotion_intensity": r["emotion_intensity"] if r["emotion_intensity"] is not None else 0.5,
                    "stress_source": r["stress_source"],
                    "user_intent": r["user_intent"],
                    "event_summary": r["event_summary"],
                    "risk_level": self._normalize_risk_level(r["risk_level"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    async def add_recommendation_event(
        self,
        session_db_id: int,
        turn_number: int,
        recommend_type: str,
        item_ids: Optional[List[str]] = None,
        trace_data: Optional[Dict[str, Any]] = None,
    ):
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as conn:
            if trace_data:
                await conn.execute(
                    """
                    INSERT INTO recommendation_events
                        (session_id, turn_number, recommend_type, item_ids,
                         gate_score, gate_threshold, reason_codes, cooldown_remaining, safety_overridden)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_db_id,
                        turn_number,
                        recommend_type,
                        json.dumps(item_ids or [], ensure_ascii=False),
                        trace_data.get("gate_score"),
                        trace_data.get("gate_threshold"),
                        json.dumps(trace_data.get("reason_codes", []), ensure_ascii=False),
                        trace_data.get("cooldown_remaining"),
                        1 if trace_data.get("safety_overridden") else 0,
                    ),
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO recommendation_events (session_id, turn_number, recommend_type, item_ids)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        session_db_id,
                        turn_number,
                        recommend_type,
                        json.dumps(item_ids or [], ensure_ascii=False),
                    ),
                )
            await conn.commit()

    async def upsert_recommendation_feedback(
        self,
        session_db_id: int,
        user_id: str,
        content_id: str,
        feedback: str,
    ):
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """
                INSERT INTO recommendation_feedback (
                    session_id, user_id, content_id, feedback, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id, content_id) DO UPDATE SET
                    feedback = excluded.feedback,
                    user_id = excluded.user_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_db_id, user_id, content_id, feedback),
            )
            await conn.commit()
    
    async def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("""
                SELECT id, user_id, session_id, conversation_stage,
                       key_concerns, created_at, last_active
                FROM sessions
                WHERE user_id = ?
                ORDER BY last_active DESC
                LIMIT ?
            """, (user_id, limit))
            rows = await cursor.fetchall()
            return [
                {
                    'id': row['id'],
                    'user_id': row['user_id'],
                    'session_id': row['session_id'],
                    'conversation_stage': row['conversation_stage'],
                    'key_concerns': json.loads(row['key_concerns']),
                    'created_at': row['created_at'],
                    'last_active': row['last_active']
                }
                for row in rows
            ]
    
    async def delete_session(self, user_id: str, session_id: str) -> bool:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as conn:
            cursor = await conn.execute("""
                DELETE FROM sessions 
                WHERE user_id = ? AND session_id = ?
            """, (user_id, session_id))
            await conn.commit()
            return cursor.rowcount > 0

adb_manager = AsyncDatabaseManager()
