#conversation_manager.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from database import db_manager, adb_manager
from config import config
import asyncio

logger = logging.getLogger(__name__)

class ConversationManager:
    """管理对话上下文和情绪演变（支持持久化）"""
    
    def __init__(self, use_persistence: bool = None):
        self.sessions = {}  # session_id -> 对话数据（内存缓存）
        self.max_history = 20  # 最大对话轮次
        self.use_persistence = use_persistence if use_persistence is not None else config.SESSION_PERSISTENCE_ENABLED
        self.session_db_ids = {}  # 缓存session_id到数据库ID的映射
        self.context_cache = {}  # 上下文压缩缓存
        
        if self.use_persistence:
            logger.info(f"会话持久化已启用，数据库路径: {config.SESSION_DB_PATH}")
        else:
            logger.info("会话持久化未启用，使用内存存储")
    
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
                # 加载到内存缓存
                self.sessions[key] = {
                    'id': session_data['id'],
                    'history': session_data['history'],
                    'emotion_timeline': session_data['emotion_timeline'],
                    'key_concerns': session_data['key_concerns'],
                    'conversation_stage': session_data['conversation_stage'],
                    'created_at': datetime.fromisoformat(session_data['created_at']),
                    'last_active': datetime.fromisoformat(session_data['last_active']),
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
            'context_quality': 1.0
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
                self.sessions[key] = {
                    'id': session_data['id'],
                    'history': session_data['history'],
                    'emotion_timeline': session_data['emotion_timeline'],
                    'key_concerns': session_data['key_concerns'],
                    'conversation_stage': session_data['conversation_stage'],
                    'created_at': datetime.fromisoformat(session_data['created_at']),
                    'last_active': datetime.fromisoformat(session_data['last_active']),
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
            'context_quality': 1.0
        }
        return self.sessions[key]
    
    def add_interaction(self, user_id: str, session_id: str, 
                        user_input: str, emotion: str, context_emotion: str = None,
                        confidence: float = 0.5, ai_response: str = ""):
        """添加一次完整交互"""
        session = self.get_or_create_session(user_id, session_id)
        key = f"{user_id}_{session_id}"
        
        # 计算轮次号
        turn_number = len(session['history']) + 1
        
        # 添加到历史
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'detected_emotion': emotion,
            'context_emotion': context_emotion,
            'confidence': confidence,
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
            'text_snippet': user_input[:50]
        }
        session['emotion_timeline'].append(emotion_entry)
        
        # 分析对话阶段
        self._analyze_conversation_stage(session)
        
        # 提取关键关切点
        self._extract_key_concerns(session, user_input)
        
        # 压缩对话上下文
        if len(session['history']) >= 5:
            self._compress_conversation_context(user_id, session_id)
        
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
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_persist())
            except RuntimeError:
                asyncio.run(_persist())
        
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
    
    def _compress_conversation_context(self, user_id: str, session_id: str):
        """压缩对话上下文"""
        key = f"{user_id}_{session_id}"
        if key not in self.sessions:
            return
        
        session = self.sessions[key]
        history = session['history']
        
        # 只压缩超过5轮的对话
        if len(history) <= 5:
            return
        
        # 提取最近3轮完整对话
        recent_history = history[-3:]
        
        # 压缩更早的对话
        compressed_parts = []
        for i, entry in enumerate(history[:-3]):
            if i % 2 == 0:  # 每2轮压缩一次
                compressed_parts.append(f"用户提到: {entry['user_input'][:30]}...")
        
        # 构建压缩上下文
        compressed_context = "历史对话摘要: "
        compressed_context += " ".join(compressed_parts[:3])  # 最多保留3个压缩部分
        compressed_context += "。最近对话: "
        
        for i, entry in enumerate(recent_history):
            compressed_context += f"用户: {entry['user_input'][:50]}... 助手: {entry['ai_response'][:50]}... "
        
        # 更新上下文缓存
        self.context_cache[key] = {
            'compressed_context': compressed_context,
            'last_compressed_turn': len(history),
            'context_quality': 0.9 if len(recent_history) >= 3 else 0.7
        }
        
        logger.debug(f"对话上下文已压缩: {key}, 压缩后长度: {len(compressed_context)}")
    
    
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
        current_topic = '日常交流'
        recent_intents = []
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
            'compressed_context': compressed_context
        }
    
    
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
