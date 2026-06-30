import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel
from risk_levels import LEVEL_0

from content_db import content_db
from content_recommender import content_recommender
from database import db_manager, adb_manager

logger = logging.getLogger(__name__)


class KnowledgeBaseQuery(BaseModel):
    query: str
    limit: int = 5


class KnowledgeBaseDocument(BaseModel):
    id: str
    title: str
    snippet: str
    metadata: Dict[str, Any] = {}


class KnowledgeBaseResult(BaseModel):
    documents: List[KnowledgeBaseDocument]


class UserProfile(BaseModel):
    """用户长期画像

    为保持对数据库结构的兼容性，所有偏好字段最终都会被写入/读取自 preferences(JSON)：
    - preferred_types:    偏好内容类型，例如 ["article", "audio"]
    - preferred_categories: 偏好主题/类别，例如 ["stress", "academic"]
    - preferred_difficulty: 偏好难度，"beginner" / "intermediate" / "advanced"
    - preferred_duration_range: 可选的时长范围配置，如 {"min": 5, "max": 20}
    """

    user_id: str
    risk_level: str = LEVEL_0
    preferences: Dict[str, Any] = {}
    # 结构化偏好字段（方便上游/下游直接使用）
    preferred_types: List[str] = []
    preferred_categories: List[str] = []
    preferred_difficulty: str = "beginner"
    preferred_duration_range: Optional[Dict[str, Any]] = None
    preferred_support_style: Optional[str] = None
    avoid_style: List[str] = []
    main_stress_sources: List[str] = []
    recommendation_feedback: Dict[str, str] = {}
    last_updated: datetime


class MoodEvent(BaseModel):
    user_id: str
    session_id: str
    emotion: str
    emotion_type: Optional[str] = None
    emotion_intensity: float = 0.5
    stress_source: Optional[str] = None
    user_intent: Optional[str] = None
    event_summary: Optional[str] = None
    risk_level: str = LEVEL_0
    source: str
    text_snippet: str
    created_at: datetime


class EscalationResult(BaseModel):
    level: str
    message: str
    suggestions: List[str]
    resources: Optional[List[Dict[str, Any]]] = None


class ToolError(Exception):
    """统一的工具错误类型"""


class KnowledgeBaseTool:
    """简单基于 content_db 的知识库检索工具"""

    @staticmethod
    async def search(query: KnowledgeBaseQuery) -> KnowledgeBaseResult:
        try:
            items = content_db.search_content(query.query, limit=query.limit)
            docs: List[KnowledgeBaseDocument] = []
            for item in items:
                docs.append(
                    KnowledgeBaseDocument(
                        id=item.id,
                        title=item.title,
                        snippet=item.description[:120],
                        metadata={
                            "type": item.type,
                            "category": item.category,
                            "tags": item.tags,
                        },
                    )
                )
            return KnowledgeBaseResult(documents=docs)
        except Exception as e:
            logger.error(f"KnowledgeBaseTool.search 失败: {e}", exc_info=True)
            raise ToolError(str(e))


class UserProfileTool:
    """用户长期画像读写"""

    @staticmethod
    async def get_profile(user_id: str) -> Optional[UserProfile]:
        try:
            row = await adb_manager.get_user_profile(user_id)
            if not row:
                return None

            
            raw_preferences = row.get("preferences") or {}
            if isinstance(raw_preferences, str):
                try:
                    import json
                    preferences: Dict[str, Any] = json.loads(raw_preferences) or {}
                except Exception:
                    preferences = {}
            else:
                preferences = dict(raw_preferences) if isinstance(raw_preferences, dict) else {}

            return UserProfile(
                user_id=row["user_id"],
                risk_level=row.get("risk_level", "low"),
                preferences=preferences,
                preferred_types=preferences.get("preferred_types", []) or [],
                preferred_categories=preferences.get("preferred_categories", []) or [],
                preferred_difficulty=preferences.get("preferred_difficulty", "beginner"),
                preferred_duration_range=preferences.get("preferred_duration_range"),
                preferred_support_style=preferences.get("preferred_support_style"),
                avoid_style=preferences.get("avoid_style", []) or [],
                main_stress_sources=preferences.get("main_stress_sources", []) or [],
                recommendation_feedback=preferences.get("recommendation_feedback", {}) or {},
                last_updated=datetime.fromisoformat(row["last_updated"]),
            )
        except Exception as e:
            logger.error(f"UserProfileTool.get_profile 失败: {e}", exc_info=True)
            raise ToolError(str(e))

    @staticmethod
    async def upsert_profile(
        user_id: str,
        risk_level: Optional[str] = None,
        preferences_patch: Optional[Dict[str, Any]] = None,
        preferred_types: Optional[List[str]] = None,
        preferred_categories: Optional[List[str]] = None,
        preferred_difficulty: Optional[str] = None,
        preferred_duration_range: Optional[Dict[str, Any]] = None,
        preferred_support_style: Optional[str] = None,
        avoid_style: Optional[List[str]] = None,
        main_stress_sources: Optional[List[str]] = None,
        recommendation_feedback: Optional[Dict[str, str]] = None,
    ) -> UserProfile:
        try:
            # 将显式偏好字段合并进 preferences_patch，写回统一的 JSON 结构中
            merged_patch: Dict[str, Any] = dict(preferences_patch or {})
            if preferred_types is not None:
                merged_patch["preferred_types"] = preferred_types
            if preferred_categories is not None:
                merged_patch["preferred_categories"] = preferred_categories
            if preferred_difficulty is not None:
                merged_patch["preferred_difficulty"] = preferred_difficulty
            if preferred_duration_range is not None:
                merged_patch["preferred_duration_range"] = preferred_duration_range
            if preferred_support_style is not None:
                merged_patch["preferred_support_style"] = preferred_support_style
            if avoid_style is not None:
                merged_patch["avoid_style"] = avoid_style
            if main_stress_sources is not None:
                merged_patch["main_stress_sources"] = main_stress_sources
            if recommendation_feedback is not None:
                merged_patch["recommendation_feedback"] = recommendation_feedback

            await adb_manager.upsert_user_profile(user_id, risk_level, merged_patch)
            row = await adb_manager.get_user_profile(user_id)

            raw_preferences = row.get("preferences") or {}
            if isinstance(raw_preferences, str):
                try:
                    import json
                    preferences: Dict[str, Any] = json.loads(raw_preferences) or {}
                except Exception:
                    preferences = {}
            else:
                preferences = dict(raw_preferences) if isinstance(raw_preferences, dict) else {}

            return UserProfile(
                user_id=row["user_id"],
                risk_level=row.get("risk_level", "low"),
                preferences=preferences,
                preferred_types=preferences.get("preferred_types", []) or [],
                preferred_categories=preferences.get("preferred_categories", []) or [],
                preferred_difficulty=preferences.get("preferred_difficulty", "beginner"),
                preferred_duration_range=preferences.get("preferred_duration_range"),
                preferred_support_style=preferences.get("preferred_support_style"),
                avoid_style=preferences.get("avoid_style", []) or [],
                main_stress_sources=preferences.get("main_stress_sources", []) or [],
                recommendation_feedback=preferences.get("recommendation_feedback", {}) or {},
                last_updated=datetime.fromisoformat(row["last_updated"]),
            )
        except Exception as e:
            logger.error(f"UserProfileTool.upsert_profile 失败: {e}", exc_info=True)
            raise ToolError(str(e))


class MoodTrackingTool:
    """情绪追踪工具"""

    @staticmethod
    async def log_event(
        user_id: str,
        session_id: str,
        emotion: str,
        source: str,
        text_snippet: str,
        emotion_type: Optional[str] = None,
        emotion_intensity: float = 0.5,
        stress_source: Optional[str] = None,
        user_intent: Optional[str] = None,
        event_summary: Optional[str] = None,
        risk_level: str = "low",
    ) -> MoodEvent:
        try:
            created_at = await adb_manager.add_mood_event(
                user_id=user_id,
                session_id=session_id,
                emotion=emotion,
                source=source,
                text_snippet=text_snippet[:100],
                emotion_type=emotion_type,
                emotion_intensity=emotion_intensity,
                stress_source=stress_source,
                user_intent=user_intent,
                event_summary=event_summary,
                risk_level=risk_level,
            )
            return MoodEvent(
                user_id=user_id,
                session_id=session_id,
                emotion=emotion,
                emotion_type=emotion_type or emotion,
                emotion_intensity=emotion_intensity,
                stress_source=stress_source,
                user_intent=user_intent,
                event_summary=event_summary,
                risk_level=risk_level,
                source=source,
                text_snippet=text_snippet[:100],
                created_at=datetime.fromisoformat(created_at),
            )
        except Exception as e:
            logger.error(f"MoodTrackingTool.log_event 失败: {e}", exc_info=True)
            raise ToolError(str(e))

    @staticmethod
    async def get_recent_trend(
        user_id: str,
        limit: int = 20,
    ) -> List[MoodEvent]:
        try:
            rows = await adb_manager.get_recent_mood_events(user_id, limit)
            events: List[MoodEvent] = []
            for row in rows:
                events.append(
                    MoodEvent(
                        user_id=row["user_id"],
                        session_id=row["session_id"],
                        emotion=row["emotion"],
                        emotion_type=row.get("emotion_type"),
                        emotion_intensity=row.get("emotion_intensity", 0.5),
                        stress_source=row.get("stress_source"),
                        user_intent=row.get("user_intent"),
                        event_summary=row.get("event_summary"),
                        risk_level=row.get("risk_level", "low"),
                        source=row.get("source", "conversation"),
                        text_snippet=row.get("text_snippet", ""),
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )
            return events
        except Exception as e:
            logger.error(f"MoodTrackingTool.get_recent_trend 失败: {e}", exc_info=True)
            raise ToolError(str(e))


class ContentRecommendTool:
    """内容推荐工具（复用现有推荐器）"""

    @staticmethod
    async def recommend(
        user_input: str,
        current_emotion: str,
        conversation_summary: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None,
        limit: int = 2,
    ) -> Tuple[List[Any], str, Dict[str, float]]:
        try:
            return await content_recommender.recommend_content(
                user_input=user_input,
                current_emotion=current_emotion,
                conversation_summary=conversation_summary,
                user_profile=user_profile or {},
                limit=limit,
            )
        except Exception as e:
            logger.error(f"ContentRecommendTool.recommend 失败: {e}", exc_info=True)
            raise ToolError(str(e))


# 工具定义（OpenAI Function Calling 格式）
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索心理学知识库，获取相关文章或建议",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量限制",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "获取用户的长期画像（偏好、风险等级等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "用户ID (自动填充)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": "更新用户的长期画像信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户ID (自动填充)"},
                    "risk_level": {
                        "type": "string",
                        "enum": ["level_0", "level_1", "level_2", "level_3", "low", "medium", "high"],
                    },
                    "preferred_types": {"type": "array", "items": {"type": "string"}},
                    "preferred_categories": {"type": "array", "items": {"type": "string"}},
                    "preferred_difficulty": {"type": "string"},
                    "preferred_support_style": {"type": "string"},
                    "avoid_style": {"type": "array", "items": {"type": "string"}},
                    "main_stress_sources": {"type": "array", "items": {"type": "string"}}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "log_mood_event",
            "description": "记录用户当前的情绪事件",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "自动填充"},
                    "session_id": {"type": "string", "description": "自动填充"},
                    "emotion": {"type": "string", "description": "检测到的情绪"},
                    "emotion_type": {"type": "string", "description": "标准化情绪类型"},
                    "emotion_intensity": {"type": "number", "description": "情绪强度 0-1"},
                    "stress_source": {"type": "string", "description": "压力来源"},
                    "user_intent": {"type": "string", "description": "当前用户意图"},
                    "event_summary": {"type": "string", "description": "事件摘要"},
                    "risk_level": {
                        "type": "string",
                        "enum": ["level_0", "level_1", "level_2", "level_3", "low", "medium", "high"],
                    },
                    "source": {"type": "string", "default": "agent"},
                    "text_snippet": {"type": "string", "description": "相关的用户输入片段"}
                },
                "required": ["emotion", "text_snippet"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_mood_trend",
            "description": "获取用户最近的情绪变化趋势",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "自动填充"},
                    "limit": {"type": "integer", "description": "返回的记录数量", "default": 10}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_content",
            "description": "根据用户当前状态推荐相关内容（冥想、文章等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "推荐数量", "default": 2}
                },
                "required": []
            }
        }
    }
]
