#conversation_manager.py
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import logging
from database import db_manager, adb_manager
from config import config
import asyncio

logger = logging.getLogger(__name__)

class ConversationManager:
    """管理对话上下文和情绪演变（支持持久化）"""
    
    def __init__(self, use_persistence: bool = None):
        self.sessions = {}  # session_id -> 对话数据（内存缓存）
        self.max_history = config.MAX_HISTORY
        self.use_persistence = use_persistence if use_persistence is not None else config.SESSION_PERSISTENCE_ENABLED
        self.session_db_ids = {}  # 缓存session_id到数据库ID的映射
        self.context_cache = {}  # 上下文压缩缓存
        self.max_context_tokens = config.MAX_CONTEXT_TOKENS
        self.context_soft_threshold = config.CONTEXT_SOFT_THRESHOLD
        self.context_hard_threshold = config.CONTEXT_HARD_THRESHOLD
        self.context_min_recent_turns = config.CONTEXT_MIN_RECENT_TURNS
        self._token_encoder = self._init_token_encoder()
        
        if self.use_persistence:
            logger.info(f"会话持久化已启用，数据库路径: {config.SESSION_DB_PATH}")
        else:
            logger.info("会话持久化未启用，使用内存存储")

    def _init_token_encoder(self):
        try:
            import tiktoken

            try:
                return tiktoken.encoding_for_model(config.CHAT_MODEL)
            except Exception:
                return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    def _estimate_tokens(self, text: Any) -> int:
        if text in (None, ""):
            return 0
        normalized = str(text)
        if self._token_encoder:
            try:
                return len(self._token_encoder.encode(normalized))
            except Exception:
                pass
        return max(1, len(normalized) // 4)

    def _estimate_history_tokens(self, history: List[Dict[str, Any]]) -> int:
        total = 0
        for entry in history:
            total += self._estimate_tokens(entry.get("user_input", ""))
            total += self._estimate_tokens(entry.get("ai_response", ""))
            total += 12
        return total

    def _get_context_thresholds(self) -> Dict[str, int]:
        soft_limit = max(128, int(self.max_context_tokens * self.context_soft_threshold))
        hard_limit = max(soft_limit + 1, int(self.max_context_tokens * self.context_hard_threshold))
        return {"soft_limit": soft_limit, "hard_limit": hard_limit}

    def _derive_feedback_state(self, feedback_map: Optional[Dict[str, str]]) -> Dict[str, List[str]]:
        feedback_map = feedback_map or {}
        accepted = [
            content_id
            for content_id, feedback in feedback_map.items()
            if feedback in {"accepted", "preferred", "helpful"}
        ]
        rejected = [
            content_id
            for content_id, feedback in feedback_map.items()
            if feedback in {"rejected", "not_helpful", "avoid"}
        ]
        return {
            "accepted_recommendations": accepted[-10:],
            "rejected_recommendations": rejected[-10:],
        }

    def _schedule_persistence(self, coro):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)
    
    def get_or_create_session(self, user_id: str, session_id: str) -> Dict:
        """获取或创建对话会话"""
        key = f"{user_id}_{session_id}"
        
        # 先检查内存缓存
        if key in self.sessions:
            self.sessions[key]['last_active'] = datetime.now()
            return self.sessions[key]
        
        # 如果启用了持久化，尝试从数据库加载
        if self.use_persistence:
            session_data = db_manager.get_session_data(user_id, session_id)
            if session_data:
                profile_data = db_manager.get_user_profile(user_id) or {}
                recommendation_feedback = session_data.get("recommendation_feedback", {}) or {}
                feedback_state = self._derive_feedback_state(recommendation_feedback)
                # 加载到内存缓存
                self.sessions[key] = {
                    'id': session_data['id'],
                    'history': session_data['history'],
                    'emotion_timeline': session_data['emotion_timeline'],
                    'key_concerns': session_data['key_concerns'],
                    'conversation_stage': session_data['conversation_stage'],
                    'created_at': datetime.fromisoformat(session_data['created_at']),
                    'last_active': datetime.fromisoformat(session_data['last_active']),
                    'long_term_profile': profile_data.get('preferences', {}),
                    'recommendation_events': session_data.get('recommendation_events', [])[-5:],
                    'recommendation_feedback': recommendation_feedback,
                    'accepted_recommendations': feedback_state["accepted_recommendations"],
                    'rejected_recommendations': feedback_state["rejected_recommendations"],
                    'emotion_cache': {
                        'main_emotion': '中性',
                        'overall_emotion': '中性',
                        'confidence': 0.5,
                        'last_analyzed_turn': len(session_data['history']),
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                }
                self.session_db_ids[key] = session_data['id']
                logger.info(f"从数据库加载会话: {key}")
                return self.sessions[key]
        
        # 创建新会话
        if self.use_persistence:
            session_db_id = db_manager.create_or_update_session(
                user_id, session_id, 'initial', []
            )
            self.session_db_ids[key] = session_db_id
        
        self.sessions[key] = {
            'id': self.session_db_ids.get(key),
            'history': [],
            'emotion_timeline': [],
            'key_concerns': [],
            'conversation_stage': 'initial',
            'created_at': datetime.now(),
            'last_active': datetime.now(),
            'long_term_profile': {},
            'recommendation_events': [],
            'recommendation_feedback': {},
            'accepted_recommendations': [],
            'rejected_recommendations': [],
            'emotion_cache': {
                'main_emotion': '中性',
                'overall_emotion': '中性',
                'confidence': 0.5,
                'last_analyzed_turn': 0,
                'analysis_timestamp': None
            }
        }
        
        # 初始化上下文缓存
        self.context_cache[key] = {
            'compressed_context': '',
            'last_compressed_turn': 0,
            'context_quality': 1.0,
            'token_count': 0,
            'compression_applied': False,
            'compression_level': 'none',
        }
        
        return self.sessions[key]
    
    async def get_or_create_session_async(self, user_id: str, session_id: str) -> Dict:
        key = f"{user_id}_{session_id}"
        if key in self.sessions:
            self.sessions[key]['last_active'] = datetime.now()
            return self.sessions[key]
        if self.use_persistence:
            session_data = await adb_manager.get_session_data(user_id, session_id)
            if session_data:
                profile_data = await adb_manager.get_user_profile(user_id) or {}
                recommendation_feedback = session_data.get("recommendation_feedback", {}) or {}
                feedback_state = self._derive_feedback_state(recommendation_feedback)
                self.sessions[key] = {
                    'id': session_data['id'],
                    'history': session_data['history'],
                    'emotion_timeline': session_data['emotion_timeline'],
                    'key_concerns': session_data['key_concerns'],
                    'conversation_stage': session_data['conversation_stage'],
                    'created_at': datetime.fromisoformat(session_data['created_at']),
                    'last_active': datetime.fromisoformat(session_data['last_active']),
                    'long_term_profile': profile_data.get('preferences', {}),
                    'recommendation_events': session_data.get('recommendation_events', [])[-5:],
                    'recommendation_feedback': recommendation_feedback,
                    'accepted_recommendations': feedback_state["accepted_recommendations"],
                    'rejected_recommendations': feedback_state["rejected_recommendations"],
                    'emotion_cache': {
                        'main_emotion': '中性',
                        'overall_emotion': '中性',
                        'confidence': 0.5,
                        'last_analyzed_turn': len(session_data['history']),
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                }
                self.session_db_ids[key] = session_data['id']
                return self.sessions[key]
        if self.use_persistence:
            session_db_id = await adb_manager.create_or_update_session(user_id, session_id, 'initial', [])
            self.session_db_ids[key] = session_db_id
        self.sessions[key] = {
            'id': self.session_db_ids.get(key),
            'history': [],
            'emotion_timeline': [],
            'key_concerns': [],
            'conversation_stage': 'initial',
            'created_at': datetime.now(),
            'last_active': datetime.now(),
            'long_term_profile': {},
            'recommendation_events': [],
            'recommendation_feedback': {},
            'accepted_recommendations': [],
            'rejected_recommendations': [],
            'emotion_cache': {
                'main_emotion': '中性',
                'overall_emotion': '中性',
                'confidence': 0.5,
                'last_analyzed_turn': 0,
                'analysis_timestamp': None
            }
        }
        self.context_cache[key] = {
            'compressed_context': '',
            'last_compressed_turn': 0,
            'context_quality': 1.0,
            'token_count': 0,
            'compression_applied': False,
            'compression_level': 'none',
        }
        return self.sessions[key]

    async def session_exists_async(self, user_id: str, session_id: str) -> bool:
        """只检查会话是否存在，避免只读查询隐式创建空会话。"""
        key = f"{user_id}_{session_id}"
        if key in self.sessions:
            return True
        if not self.use_persistence:
            return False
        session_data = await adb_manager.get_session_data(user_id, session_id)
        return session_data is not None
    
    def add_interaction(self, user_id: str, session_id: str, 
                        user_input: str, emotion: str, context_emotion: str = None,
                        confidence: float = 0.5, ai_response: str = "",
                        emotion_state: Optional[Dict[str, Any]] = None,
                        risk_state: Optional[Dict[str, Any]] = None):
        """添加一次完整交互"""
        session = self.get_or_create_session(user_id, session_id)
        key = f"{user_id}_{session_id}"
        emotion_state = emotion_state or {}
        risk_state = risk_state or {}
        
        # 计算轮次号
        turn_number = len(session['history']) + 1
        
        # 添加到历史
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'detected_emotion': emotion,
            'context_emotion': context_emotion,
            'confidence': confidence,
            'emotion_type': emotion_state.get('emotion_type'),
            'emotion_intensity': emotion_state.get('emotion_intensity', confidence),
            'stress_source': emotion_state.get('stress_source'),
            'user_intent': emotion_state.get('user_intent'),
            'negative_trend': emotion_state.get('negative_trend', False),
            'risk_level': risk_state.get('level', 'low'),
            'ai_response': ai_response
        }
        session['history'].append(history_entry)
        
        # 保持历史长度
        if len(session['history']) > self.max_history:
            session['history'] = session['history'][-self.max_history:]
        
        # 更新情绪时间线
        emotion_entry = {
            'time': datetime.now().isoformat(),
            'emotion': emotion,
            'emotion_type': emotion_state.get('emotion_type', emotion),
            'emotion_intensity': emotion_state.get('emotion_intensity', confidence),
            'stress_source': emotion_state.get('stress_source'),
            'user_intent': emotion_state.get('user_intent'),
            'risk_level': risk_state.get('level', 'low'),
            'text_snippet': user_input[:50]
        }
        session['emotion_timeline'].append(emotion_entry)
        
        # 分析对话阶段
        self._analyze_conversation_stage(session)
        
        # 提取关键关切点
        self._extract_key_concerns(session, user_input)
        self._update_long_term_signals(session, user_input, emotion_state, risk_state)
        
        # 基于上下文估算结果决定是否压缩短期记忆
        self._refresh_context_cache(user_id, session_id)
        
        session['last_active'] = datetime.now()
        
        # 持久化到数据库
        if self.use_persistence and key in self.session_db_ids:
            session_db_id = self.session_db_ids[key]
            async def _persist():
                await adb_manager.add_conversation_turn(
                    session_db_id, turn_number, user_input, emotion,
                    context_emotion or emotion, confidence, ai_response
                )
                await adb_manager.add_emotion_event(session_db_id, emotion, user_input[:50])
                await adb_manager.create_or_update_session(
                    user_id, session_id, 
                    session['conversation_stage'], 
                    session['key_concerns']
                )
                await adb_manager.add_mood_event(
                    user_id=user_id,
                    session_id=session_id,
                    emotion=emotion,
                    source="conversation",
                    text_snippet=user_input[:100],
                    emotion_type=emotion_state.get('emotion_type'),
                    emotion_intensity=emotion_state.get('emotion_intensity', confidence),
                    stress_source=emotion_state.get('stress_source'),
                    user_intent=emotion_state.get('user_intent'),
                    event_summary=self._build_event_summary(user_input, emotion_state),
                    risk_level=risk_state.get('level', 'low'),
                )
                await adb_manager.upsert_user_profile(
                    user_id=user_id,
                    risk_level=risk_state.get('level'),
                    preferences_patch=session.get('long_term_profile', {}),
                )
            try:
                self._schedule_persistence(_persist())
            except Exception:
                raise
        
        return session
    
    def _analyze_conversation_stage(self, session: Dict):
        """分析当前对话阶段"""
        history_len = len(session['history'])
        
        if history_len <= 2:
            session['conversation_stage'] = 'initial'
        elif history_len <= 5:
            session['conversation_stage'] = 'exploring'
        elif history_len <= 10:
            session['conversation_stage'] = 'deepening'
        else:
            session['conversation_stage'] = 'resolving'
    
    def _extract_key_concerns(self, session: Dict, user_input: str):
        """提取关键关切点"""
        # 简单关键词提取，实际可更复杂
        concern_keywords = {
            'relationship': ['对象', '男朋友', '女朋友', '室友', '朋友', '关系'],
            'academic': ['考试', '学习', '论文', '毕业', '成绩', '复习'],
            'future': ['将来', '未来', '以后', '规划', '方向'],
            'self': ['我', '自己', '个人', '性格', '习惯']
        }
        
        for concern_type, keywords in concern_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                if concern_type not in session['key_concerns']:
                    session['key_concerns'].append(concern_type)
        
        # 保持最多5个关切点
        session['key_concerns'] = session['key_concerns'][:5]
    
    def _refresh_context_cache(self, user_id: str, session_id: str):
        key = f"{user_id}_{session_id}"
        if key not in self.sessions:
            return

        session = self.sessions[key]
        history = session['history']

        token_count = self._estimate_history_tokens(history)
        thresholds = self._get_context_thresholds()
        if token_count < thresholds["soft_limit"]:
            self.context_cache[key] = {
                'compressed_context': '',
                'last_compressed_turn': 0,
                'context_quality': 1.0,
                'token_count': token_count,
                'compression_applied': False,
                'compression_level': 'none',
            }
            return

        self._compress_conversation_context(
            user_id=user_id,
            session_id=session_id,
            token_count=token_count,
            soft_limit=thresholds["soft_limit"],
            hard_limit=thresholds["hard_limit"],
        )

    def _compress_conversation_context(
        self,
        user_id: str,
        session_id: str,
        token_count: Optional[int] = None,
        soft_limit: Optional[int] = None,
        hard_limit: Optional[int] = None,
    ):
        """压缩对话上下文"""
        key = f"{user_id}_{session_id}"
        if key not in self.sessions:
            return

        session = self.sessions[key]
        history = session['history']
        token_count = token_count if token_count is not None else self._estimate_history_tokens(history)
        thresholds = self._get_context_thresholds()
        soft_limit = soft_limit if soft_limit is not None else thresholds["soft_limit"]
        hard_limit = hard_limit if hard_limit is not None else thresholds["hard_limit"]
        compression_level = "hard" if token_count >= hard_limit else "soft"
        recent_token_budget = max(80, int(self.max_context_tokens * (0.25 if compression_level == "hard" else 0.4)))

        recent_history_reversed = []
        recent_token_count = 0
        for entry in reversed(history):
            entry_token_count = (
                self._estimate_tokens(entry.get("user_input", ""))
                + self._estimate_tokens(entry.get("ai_response", ""))
                + 12
            )
            if (
                recent_history_reversed
                and recent_token_count + entry_token_count > recent_token_budget
                and len(recent_history_reversed) >= self.context_min_recent_turns
            ):
                break
            recent_history_reversed.append(entry)
            recent_token_count += entry_token_count

        recent_history = list(reversed(recent_history_reversed))
        older_history = history[:-len(recent_history)] if len(recent_history) < len(history) else []

        compressed_parts = []
        for entry in older_history[-6:]:
            emotion = entry.get('detected_emotion', '中性')
            stress_source = entry.get('stress_source')
            source_text = f"({stress_source})" if stress_source else ""
            compressed_parts.append(
                f"[{emotion}{source_text}] 用户提到: {entry['user_input'][:40]}..."
            )

        compressed_context = "历史对话摘要: "
        compressed_context += " ".join(compressed_parts[:4]) if compressed_parts else "暂无早期摘要。"
        compressed_context += "。最近对话: "

        for entry in recent_history:
            compressed_context += f"用户: {entry['user_input'][:50]}... 助手: {entry['ai_response'][:50]}... "

        self.context_cache[key] = {
            'compressed_context': compressed_context,
            'last_compressed_turn': len(history),
            'context_quality': 0.85 if compression_level == "soft" else 0.75,
            'token_count': token_count,
            'compression_applied': True,
            'compression_level': compression_level,
        }

        logger.debug(
            "对话上下文已压缩: %s, tokens=%s, level=%s, 压缩后长度=%s",
            key,
            token_count,
            compression_level,
            len(compressed_context),
        )

    def _build_event_summary(self, user_input: str, emotion_state: Dict[str, Any]) -> str:
        emotion_label = emotion_state.get("emotion_type") or emotion_state.get("current_emotion") or "neutral"
        stress_source = emotion_state.get("stress_source")
        if stress_source:
            return f"用户围绕{stress_source}表达了{emotion_label}相关困扰：{user_input[:60]}"
        return f"用户表达了{emotion_label}相关困扰：{user_input[:60]}"

    def _count_recent_matches(self, history: List[Dict[str, Any]], key: str, value: Any, limit: int = 6) -> int:
        recent_history = history[-limit:]
        return sum(1 for entry in recent_history if entry.get(key) == value)

    def _extract_support_preferences(self, user_input: str) -> Dict[str, Any]:
        text = user_input or ""
        direct_patterns = [
            "直接告诉我怎么做",
            "最好直接告诉我怎么做",
            "给我具体方法",
            "给我具体建议",
            "简单说重点",
            "直接一点",
            "我该怎么办",
        ]
        avoid_empty_comfort_patterns = [
            "不要空泛安慰",
            "别安慰我",
            "不喜欢空泛安慰",
            "不要讲大道理",
            "别说大道理",
        ]
        return {
            "preferred_support_style": "direct_actionable"
            if any(pattern in text for pattern in direct_patterns)
            else None,
            "avoid_style": ["empty_comfort"]
            if any(pattern in text for pattern in avoid_empty_comfort_patterns)
            else [],
        }

    def _update_long_term_signals(
        self,
        session: Dict[str, Any],
        user_input: str,
        emotion_state: Dict[str, Any],
        risk_state: Dict[str, Any],
    ):
        profile = session.setdefault(
            "long_term_profile",
            {
                "main_stress_sources": [],
                "preferred_support_style": None,
                "avoid_style": [],
                "recommendation_feedback": {},
            },
        )
        # 兼容历史会话或旧结构，避免缺少画像键时主链路报错。
        profile.setdefault("main_stress_sources", [])
        profile.setdefault("preferred_support_style", None)
        profile.setdefault("avoid_style", [])
        profile.setdefault("recommendation_feedback", {})

        stress_source = emotion_state.get("stress_source")
        stress_source_count = self._count_recent_matches(
            session.get("history", []),
            "stress_source",
            stress_source,
        ) if stress_source else 0
        if (
            stress_source
            and stress_source_count >= 2
            and stress_source not in profile["main_stress_sources"]
        ):
            profile["main_stress_sources"].append(stress_source)
            profile["main_stress_sources"] = profile["main_stress_sources"][:5]

        preference_signal = self._extract_support_preferences(user_input)
        explicit_support_style = preference_signal.get("preferred_support_style")
        if explicit_support_style:
            profile["preferred_support_style"] = explicit_support_style

        for avoid_style in preference_signal.get("avoid_style", []):
            if avoid_style not in profile["avoid_style"]:
                profile["avoid_style"].append(avoid_style)
        profile["avoid_style"] = profile["avoid_style"][-5:]

        user_intent = emotion_state.get("user_intent")
        high_action_intent_count = sum(
            1
            for entry in session.get("history", [])[-6:]
            if entry.get("user_intent") in {"planning", "seeking_help"}
        )
        if (
            user_intent in {"planning", "seeking_help"}
            and high_action_intent_count >= 2
            and not profile.get("preferred_support_style")
        ):
            profile["preferred_support_style"] = "direct_actionable"

        if risk_state.get("level"):
            profile["risk_level"] = risk_state["level"]

    def mark_recommendation(
        self,
        user_id: str,
        session_id: str,
        recommend_type: str,
        item_ids: Optional[List[str]] = None,
    ):
        session = self.get_or_create_session(user_id, session_id)
        session.setdefault("recommendation_events", [])
        turn_number = len(session["history"])
        session["recommendation_events"].append(
            {
                "turn_number": turn_number,
                "recommend_type": recommend_type,
                "item_ids": item_ids or [],
                "timestamp": datetime.now().isoformat(),
            }
        )
        session["recommendation_events"] = session["recommendation_events"][-5:]
        key = f"{user_id}_{session_id}"
        if self.use_persistence and key in self.session_db_ids:
            session_db_id = self.session_db_ids[key]

            async def _persist():
                await adb_manager.add_recommendation_event(
                    session_db_id=session_db_id,
                    turn_number=turn_number,
                    recommend_type=recommend_type,
                    item_ids=item_ids or [],
                )

            self._schedule_persistence(_persist())

    def record_recommendation_feedback(
        self,
        user_id: str,
        session_id: str,
        content_id: str,
        feedback: str,
    ):
        session = self.get_or_create_session(user_id, session_id)
        session.setdefault("recommendation_feedback", {})
        session["recommendation_feedback"][content_id] = feedback
        profile = session.setdefault("long_term_profile", {})
        profile.setdefault("recommendation_feedback", {})
        profile["recommendation_feedback"][content_id] = feedback

        if feedback in {"accepted", "preferred", "helpful"}:
            session.setdefault("accepted_recommendations", [])
            session.setdefault("rejected_recommendations", [])
            session["rejected_recommendations"] = [
                item for item in session["rejected_recommendations"] if item != content_id
            ]
            if content_id not in session["accepted_recommendations"]:
                session["accepted_recommendations"].append(content_id)
            session["accepted_recommendations"] = session["accepted_recommendations"][-10:]
        elif feedback in {"rejected", "not_helpful", "avoid"}:
            session.setdefault("accepted_recommendations", [])
            session["accepted_recommendations"] = [
                item for item in session["accepted_recommendations"] if item != content_id
            ]
            session.setdefault("rejected_recommendations", [])
            if content_id not in session["rejected_recommendations"]:
                session["rejected_recommendations"].append(content_id)
            session["rejected_recommendations"] = session["rejected_recommendations"][-10:]

        key = f"{user_id}_{session_id}"
        if self.use_persistence and key in self.session_db_ids:
            session_db_id = self.session_db_ids[key]

            async def _persist():
                await adb_manager.upsert_recommendation_feedback(
                    session_db_id=session_db_id,
                    user_id=user_id,
                    content_id=content_id,
                    feedback=feedback,
                )

            self._schedule_persistence(_persist())
    
    
    async def get_conversation_summary_async(self, user_id: str, session_id: str):
        session = await self.get_or_create_session_async(user_id, session_id)
        key = f"{user_id}_{session_id}"
        if not session['history']:
            return {
                'conversation_stage': 'initial',
                'primary_emotion': '中性',
                'emotion_trend': 'new',
                'key_concerns': [],
                'turn_count': 0,
                'recent_emotions': [],
                'current_topic': '初始对话',
                'recent_intents': [],
                'stress_sources': [],
                'recent_risk_levels': [],
                'risk_expressions': False,
                'accepted_recommendations': [],
                'rejected_recommendations': [],
                'recent_recommendation_turns': [],
                'compressed_context': ''
            }
        emotions = [h['detected_emotion'] for h in session['history'][-5:]]
        primary_emotion = max(set(emotions), key=emotions.count) if emotions else '中性'
        trend = 'stable'
        if len(emotions) >= 3:
            recent = emotions[-3:]
            if all(e in ['焦虑', '压力', '愤怒'] for e in recent):
                trend = 'escalating'
            elif all(e in ['平静', '中性', '快乐'] for e in recent):
                trend = 'improving'
        current_topic = self._infer_current_topic(session['key_concerns'])
        recent_intents = [
            h.get('user_intent') for h in session['history'][-3:] if h.get('user_intent')
        ]
        stress_sources = []
        for entry in session['history'][-5:]:
            source = entry.get('stress_source')
            if source and source not in stress_sources:
                stress_sources.append(source)
        recent_risk_levels = [
            h.get('risk_level', 'low') for h in session['history'][-3:]
        ]
        risk_expressions = any(level in {'medium', 'high'} for level in recent_risk_levels)
        recent_recommendation_turns = [
            event.get("turn_number")
            for event in session.get("recommendation_events", [])[-3:]
            if event.get("turn_number") is not None
        ]
        compressed_context = ''
        if key in self.context_cache:
            compressed_context = self.context_cache[key]['compressed_context']
        return {
            'conversation_stage': session['conversation_stage'],
            'primary_emotion': primary_emotion,
            'emotion_trend': trend,
            'key_concerns': session['key_concerns'],
            'turn_count': len(session['history']),
            'recent_emotions': emotions[-3:] if len(emotions) >= 3 else emotions,
            'current_topic': current_topic,
            'recent_intents': recent_intents,
            'stress_sources': stress_sources,
            'recent_risk_levels': recent_risk_levels,
            'risk_expressions': risk_expressions,
            'accepted_recommendations': session.get('accepted_recommendations', []),
            'rejected_recommendations': session.get('rejected_recommendations', []),
            'recent_recommendation_turns': recent_recommendation_turns,
            'compressed_context': compressed_context
        }

    def _infer_current_topic(self, key_concerns: List[str]) -> str:
        mapping = {
            'academic': '学业与求职',
            'relationship': '人际关系',
            'future': '未来规划',
            'self': '自我评价',
        }
        if not key_concerns:
            return '日常交流'
        return mapping.get(key_concerns[-1], '日常交流')
    
    
    async def delete_session_async(self, user_id: str, session_id: str) -> bool:
        key = f"{user_id}_{session_id}"
        if key in self.sessions:
            del self.sessions[key]
        if key in self.context_cache:
            del self.context_cache[key]
        if self.use_persistence:
            if key in self.session_db_ids:
                del self.session_db_ids[key]
            return await adb_manager.delete_session(user_id, session_id)
        return True
    
    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取用户的所有会话列表"""
        if self.use_persistence:
            return db_manager.get_user_sessions(user_id, limit)
        else:
            # 从内存中获取
            sessions = []
            for key, session in self.sessions.items():
                if key.startswith(f"{user_id}_"):
                    session_id = key.split('_', 1)[1]
                    sessions.append({
                        'user_id': user_id,
                        'session_id': session_id,
                        'conversation_stage': session['conversation_stage'],
                        'key_concerns': session['key_concerns'],
                        'created_at': session.get('created_at', datetime.now()).isoformat(),
                        'last_active': session['last_active'].isoformat()
                    })
            return sorted(sessions, key=lambda x: x['last_active'], reverse=True)[:limit]
    
    
    async def cleanup_expired_sessions_async(self, days: int = 30) -> int:
        if not self.use_persistence:
            return 0
        deleted_count = await adb_manager.cleanup_expired_sessions(days)
        cutoff_date = datetime.now() - timedelta(days=days)
        keys_to_delete = []
        for key, session in list(self.sessions.items()):
            if session['last_active'] < cutoff_date:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            if key in self.sessions:
                del self.sessions[key]
            if key in self.session_db_ids:
                del self.session_db_ids[key]
            if key in self.context_cache:
                del self.context_cache[key]
        return deleted_count
    
    def update_emotion_cache(self, user_id: str, session_id: str, 
                           main_emotion: str, overall_emotion: str, 
                           confidence: float, turn_count: int):
        """更新情绪缓存"""
        key = f"{user_id}_{session_id}"
        if key in self.sessions:
            self.sessions[key]['emotion_cache'] = {
                'main_emotion': main_emotion,
                'overall_emotion': overall_emotion,
                'confidence': confidence,
                'last_analyzed_turn': turn_count,
                'analysis_timestamp': datetime.now().isoformat()
            }
            logger.debug(f"情绪缓存已更新: {key}, 情绪={main_emotion}")
    
    def get_emotion_cache(self, user_id: str, session_id: str) -> dict:
        """获取情绪缓存"""
        key = f"{user_id}_{session_id}"
        if key in self.sessions:
            return self.sessions[key].get('emotion_cache', {
                'main_emotion': '中性',
                'overall_emotion': '中性',
                'confidence': 0.5,
                'last_analyzed_turn': 0,
                'analysis_timestamp': None
            })
        return {
            'main_emotion': '中性',
            'overall_emotion': '中性',
            'confidence': 0.5,
            'last_analyzed_turn': 0,
            'analysis_timestamp': None
        }

conversation_manager = ConversationManager()
