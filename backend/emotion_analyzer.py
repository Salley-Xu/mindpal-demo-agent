import logging
from typing import Optional, Dict, Tuple
from openai import AsyncOpenAI
from config import config
import hashlib
import time
import re

logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """情绪分析器"""
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        self.client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.API_BASE_URL
        )
        self.model = config.CHAT_MODEL
        self.cache_size = cache_size  # 缓存大小
        self.cache_ttl = cache_ttl  # 缓存过期时间（秒）
        self._emotion_cache = {}  # 情绪分析缓存
        self._cache_lru = []  # LRU缓存管理
        logger.info(f"情绪分析缓存初始化完成，大小: {cache_size}, TTL: {cache_ttl}秒")
    
    def _generate_cache_key(self, text: str, conversation_summary: Optional[Dict] = None) -> str:
        """生成缓存键"""
        # 基于文本内容和对话摘要生成唯一键
        key_material = text[:500]  # 限制文本长度
        if conversation_summary:
            # 添加对话摘要的关键信息
            key_material += str(conversation_summary.get('turn_count', 0))
            key_material += str(conversation_summary.get('conversation_stage', ''))
            key_material += ','.join(conversation_summary.get('key_concerns', []))
        return hashlib.md5(key_material.encode('utf-8')).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Tuple[str, str, float]]:
        """检查缓存"""
        if cache_key in self._emotion_cache:
            cached_data = self._emotion_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                # 缓存未过期
                self._update_lru(cache_key)
                logger.debug(f"使用缓存的情绪分析结果: {cached_data['result']}")
                return cached_data['result']
            else:
                # 缓存过期，删除
                del self._emotion_cache[cache_key]
                if cache_key in self._cache_lru:
                    self._cache_lru.remove(cache_key)
        return None
    
    def _update_cache(self, cache_key: str, result: Tuple[str, str, float]):
        """更新缓存"""
        # 检查缓存大小
        if len(self._emotion_cache) >= self.cache_size:
            # 删除最久未使用的缓存
            oldest_key = self._cache_lru.pop(0)
            if oldest_key in self._emotion_cache:
                del self._emotion_cache[oldest_key]
        
        # 更新缓存
        self._emotion_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        self._update_lru(cache_key)
    
    def _update_lru(self, cache_key: str):
        """更新LRU列表"""
        if cache_key in self._cache_lru:
            self._cache_lru.remove(cache_key)
        self._cache_lru.append(cache_key)
    
    async def analyze_with_context_async(self, text: str, 
                            conversation_summary: Optional[Dict] = None) -> Tuple[str, str, float]:
        try:
            cache_key = self._generate_cache_key(text, conversation_summary)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                return cached_result
            current_emotion = await self._analyze_base_emotion_async(text)
            if conversation_summary and conversation_summary.get('turn_count', 0) > 0:
                context_emotion = await self._analyze_context_emotion_async(text, current_emotion, conversation_summary)
            else:
                context_emotion = current_emotion
            confidence = self._calculate_confidence(text, current_emotion)
            result = (current_emotion, context_emotion, confidence)
            self._update_cache(cache_key, result)
            logger.info(f"情绪分析(异步): 当前={current_emotion}, 深层={context_emotion}, 置信度={confidence}")
            return result
        except Exception as e:
            logger.error(f"情绪分析失败(异步): {e}")
            return "中性", "中性", 0.5
    
    async def _call_llm(self, messages: list, temperature: float = 0.1, max_tokens: int = 100) -> str:
        """调用LLM API"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    async def _analyze_base_emotion_async(self, text: str) -> str:
        prompt = """分析以下文本的主要情绪（从选项中选择最贴切的）：
        选项：学业压力、焦虑、抑郁、愤怒、压力、人际矛盾、困惑、不确定、中性、快乐、平静、放松、其他

        文本："{}"
        情绪标签："""
        return await self._call_llm(
            messages=[
                {"role": "system", "content": "只返回情绪标签"},
                {"role": "user", "content": prompt.format(text)}
            ],
            temperature=0.1,
            max_tokens=10
        )
    
    async def _analyze_context_emotion_async(self, text: str, base_emotion: str, 
                               conversation_summary: Dict) -> str:
        prompt = f"""你是一位专业的心理咨询师，正在分析一位用户的情绪状态。
        
        用户当前输入："{text}"
        当前检测到的表层情绪是：{base_emotion}
        
        对话上下文信息：
        - 对话阶段：{conversation_summary.get('conversation_stage', 'initial')}
        - 主要关切点：{', '.join(conversation_summary.get('key_concerns', []))}
        - 近期情绪变化：{', '.join(conversation_summary.get('recent_emotions', []))}
        
        请分析用户的深层情绪。深层情绪可能是用户没有直接表达，但隐藏在话语背后的情绪。
        深层情绪类型：
        1. 表层情绪（就是当前表达的情绪）
        2. 深层焦虑（表面情绪下隐藏的焦虑）
        3. 关系困扰（与人际关系相关的深层困扰）
        4. 自我怀疑（对自身能力或价值的怀疑）
        5. 未来迷茫（对未来的不确定和迷茫）
        6. 学业压力（与学业相关的深层压力）
        7. 情绪压抑（未能表达的负面情绪积压）
        8. 家庭压力（来自家庭的压力）
        9. 社交恐惧（对社交场合的恐惧）
        
        请以以下格式回答：
        深层情绪：[你的选择]
        解释：[简要解释]"""
        result = await self._call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        # Use regex for more robust parsing
        match = re.search(r"深层情绪[:：]\s*(.+)", result)
        if match:
            context_emotion = match.group(1).strip()
        else:
            context_emotion = base_emotion
            
        logger.debug(f"深层情绪分析(异步): {base_emotion} -> {context_emotion}")
        return context_emotion
    
    def _calculate_confidence(self, text: str, emotion: str) -> float:
        """计算置信度"""
        base_confidence = 0.85
        
        # 根据文本长度调整置信度
        if len(text) < 10:
            base_confidence -= 0.2
        elif len(text) > 100:
            base_confidence += 0.1
        
        # 中性情绪置信度较低
        if emotion == '中性':
            base_confidence -= 0.1
        
        return max(0.5, min(1.0, base_confidence))
    
    async def analyze_conversation_emotions_async(self, conversation_history: list) -> Tuple[str, str, float]:
        try:
            if not conversation_history:
                return "中性", "中性", 0.5
            history_text = ""
            for i, entry in enumerate(conversation_history[-5:]):
                history_text += f"轮次{i+1}:\n"
                history_text += f"用户: {entry.get('user_input', '')[:100]}\n"
                history_text += f"情绪: {entry.get('detected_emotion', '未知')}\n\n"
            prompt = f"""你是一位专业的心理咨询师，需要基于多轮对话历史分析用户的综合情绪状态。
            
            对话历史:\n{history_text}
            
            请分析：
            1. 用户的主要情绪（从选项中选择）：学业压力、焦虑、抑郁、愤怒、压力、人际矛盾、困惑、不确定、中性、快乐、平静、放松、其他
            2. 综合情绪状态（简要描述用户的整体情绪状态和变化趋势）
            3. 情绪分析的置信度(0-1)
            
            请以以下格式回答：
            主要情绪：[情绪标签]
            综合情绪：[描述]
            置信度：[数值]"""
            result = await self._call_llm(
                messages=[
                    {"role": "system", "content": "请严格按照指定格式回答"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            main_emotion = "中性"
            overall_emotion = "中性"
            confidence = 0.7
            
            # Regex parsing for robustness
            main_match = re.search(r"主要情绪[:：]\s*(.+)", result)
            if main_match:
                main_emotion = main_match.group(1).strip()
                
            overall_match = re.search(r"综合情绪[:：]\s*(.+)", result)
            if overall_match:
                overall_emotion = overall_match.group(1).strip()
                
            conf_match = re.search(r"置信度[:：]\s*(\d+(\.\d+)?)", result)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                except:
                    pass
                    
            logger.info(f"综合情绪分析(异步): 主要={main_emotion}, 综合={overall_emotion}, 置信度={confidence}")
            return main_emotion, overall_emotion, confidence
        except Exception as e:
            logger.error(f"综合情绪分析失败(异步): {e}")
            return "中性", "中性", 0.5
    
    def should_analyze_emotion(self, turn_count: int, config: dict = None) -> bool:
        """
        判断是否需要进行情绪分析
        基于对话轮次和配置
        """
        # 默认配置：每3轮分析一次，前2轮每次都分析
        analysis_interval = config.get('analysis_interval', 3) if config else 3
        
        if turn_count <= 2:
            return True
        
        return turn_count % analysis_interval == 0

emotion_analyzer = EmotionAnalyzer()