# urgent_detector.py
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
from config import config
from risk_levels import (
    LEVEL_0,
    LEVEL_1,
    LEVEL_2,
    LEVEL_3,
    is_emergency_risk,
    is_non_low_risk,
    normalize_risk_level,
    risk_level_band,
    risk_level_index,
    risk_level_label,
)

logger = logging.getLogger(__name__)

class UrgentDetector:
    """紧急情况检测器"""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.API_BASE_URL
        )
        self.model = config.CHAT_MODEL
        
        # 紧急关键词 - 需要立即干预
        self.urgent_keywords = [
            '自杀', '不想活了', '结束生命', '绝望', '活够了',
            '想死', '死掉', '离开世界', '生命没意义', '自我了断',
            '跳楼', '割腕', '服毒', '上吊', '烧炭', '安乐死'
        ]
        
        # 警告关键词 - 需要密切关注
        self.warning_keywords = [
            '活不下去', '没意思', '太痛苦', '撑不住', '崩溃',
            '想消失', '人间不值得', '好累', '绝望', '无助',
            '没人理解', '孤独', '被抛弃', '没有希望', '想放弃'
        ]
        
        # 情绪增强因子
        self.emotion_enhancers = {
            '抑郁': 2.0,
            '焦虑': 1.5,
            '愤怒': 1.2,
            '压力': 1.3
        }
    
    def detect(self, text: str, emotion: str) -> Dict[str, Any]:
        """
        检测用户输入中的紧急情况
        返回: {
            'level': 'level_0'/'level_1'/'level_2'/'level_3',
            'message': str,
            'suggestions': List[str],
            'triggers': List[str],
            'risk_score': float
        }
        """
        text_lower = text.lower()
        
        found_urgent = self._find_keywords(text_lower, self.urgent_keywords)
        found_warning = self._find_keywords(text_lower, self.warning_keywords)
        
        return self._evaluate_urgency_level(found_urgent, found_warning, emotion)
    
    def _find_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """查找关键词"""
        found = []
        for keyword in keywords:
            if keyword in text:
                found.append(keyword)
        return found
    
    def _evaluate_urgency_level(self, urgent_keywords: List[str], 
                               warning_keywords: List[str], emotion: str) -> Dict[str, Any]:
        """评估紧急级别"""
        if urgent_keywords:
            return self._create_high_risk_response(urgent_keywords)
        
        elif warning_keywords:
            return self._create_medium_risk_response(warning_keywords, emotion)
        
        else:
            return self._create_low_risk_response()
    
    def _normalize_level(self, level: Optional[str]) -> str:
        return normalize_risk_level(level)

    def _create_high_risk_response(self, triggers: List[str]) -> Dict[str, Any]:
        """创建高风险响应"""
        return {
            'level': LEVEL_3,
            'legacy_level': risk_level_band(LEVEL_3),
            'level_index': risk_level_index(LEVEL_3),
            'level_label': risk_level_label(LEVEL_3),
            'message': '检测到紧急情况，请立即寻求专业帮助！',
            'suggestions': [
                '立即拨打心理援助热线(如:400-161-9995)',
                '联系学校心理咨询中心',
                '告诉信任的家人或朋友',
                '前往最近医院的急诊科'
            ],
            'triggers': triggers,
            'resources': [
                {'name': '全国心理援助热线', 'phone': '400-161-9995', 'hours': '24小时'},
                {'name': '北京心理援助热线', 'phone': '010-82951332', 'hours': '24小时'},
                {'name': '希望24热线', 'phone': '400-161-9995', 'hours': '24小时'}
            ],
            'risk_score': 10.0
        }
    
    def _create_medium_risk_response(self, triggers: List[str], emotion: str) -> Dict[str, Any]:
        """创建中风险响应"""
        severity = len(triggers)
        
        # 考虑情绪增强因子
        if emotion in self.emotion_enhancers:
            severity *= self.emotion_enhancers[emotion]
        
        risk_score = min(severity * 2.0, 9.9)  # 风险评分0-9.9
        
        if severity >= 3:
            return {
                'level': LEVEL_2,
                'legacy_level': risk_level_band(LEVEL_2),
                'level_index': risk_level_index(LEVEL_2),
                'level_label': risk_level_label(LEVEL_2),
                'message': '检测到较高风险，建议尽快寻求帮助',
                'suggestions': [
                    '建议联系学校心理咨询师',
                    '可以拨打心理援助热线',
                    '与信任的人谈谈你的感受',
                    '尝试一些放松技巧（深呼吸、冥想）'
                ],
                'triggers': triggers,
                'risk_score': risk_score
            }
        else:
            return {
                'level': LEVEL_1,
                'legacy_level': risk_level_band(LEVEL_1),
                'level_index': risk_level_index(LEVEL_1),
                'level_label': risk_level_label(LEVEL_1),
                'message': '检测到潜在风险，需要关注',
                'suggestions': [
                    '建议寻求专业支持',
                    '可以联系信任的人谈谈',
                    '尝试记录情绪日记'
                ],
                'triggers': triggers,
                'risk_score': risk_score
            }
    
    def _create_low_risk_response(self) -> Dict[str, Any]:
        """创建低风险响应"""
        return {
            'level': LEVEL_0,
            'legacy_level': risk_level_band(LEVEL_0),
            'level_index': risk_level_index(LEVEL_0),
            'level_label': risk_level_label(LEVEL_0),
            'message': '',
            'suggestions': [],
            'triggers': [],
            'risk_score': 0.0
        }
    
    async def generate_crisis_response_async(self, user_input: str, urgent_issue: Dict, 
                               conversation_summary: Dict) -> str:
        """针对紧急情况生成特殊回应"""
        level = self._normalize_level((urgent_issue or {}).get('level'))
        if is_emergency_risk(level):
            # Level 3 must not depend on network/model latency or availability.
            return self._get_default_urgent_response()
        elif is_non_low_risk(level):
            return await self._generate_warning_response_async(user_input, urgent_issue)
        return None

    async def generate_third_party_support_response_async(
        self,
        user_input: str,
        urgent_issue: Dict,
        conversation_summary: Dict,
    ) -> str:
        """为第三方危机求助生成专门回应。"""
        # Third-party crisis guidance is also deterministic: the requester
        # should receive actionable instructions even when the LLM is down.
        return self._get_default_third_party_support_response()
    
    async def _generate_urgent_response_async(self, user_input: str, urgent_issue: Dict) -> str:
        """生成紧急情况回应"""
        prompt = f"""用户表达了严重困扰："{user_input}"
        
        检测到关键词：{', '.join(urgent_issue['triggers'])}
        
        请生成紧急回应：
        1. 表达共情和理解(简短)
        2. 强调寻求专业帮助的重要性
        3. 提供具体可用的帮助资源
        4. 鼓励立即行动
        5. 保持冷静和支持性的语气
        
        回应要求：
        - 不超过150字
        - 直接、明确
        - 提供具体的联系方式
        - 表达持续的支持
        
        现在生成回应："""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位心理危机干预助手，正在处理紧急情况."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"生成紧急回应失败: {e}")
            return self._get_default_urgent_response()
    
    async def _generate_warning_response_async(self, user_input: str, urgent_issue: Dict) -> str:
        """生成警告情况回应"""
        prompt = f"""用户表达了困扰："{user_input}"
        
        检测到可能需要关注的信号。
        
        请生成引导性回应：
        1. 表达共情和关心
        2. 询问更多关于感受的细节
        3. 建议寻求专业支持的选项
        4. 提供持续对话的空间
        
        回应要求：
        - 温暖、支持性
        - 引导但不强迫
        - 提供具体建议
        
        现在生成回应："""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位细心倾听的心理支持伙伴."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"生成警告回应失败: {e}")
            return self._get_default_warning_response()
    
    def _get_default_urgent_response(self) -> str:
        """获取默认紧急回应"""
        return """我听到你正在经历非常艰难的时刻，你的感受非常重要。请立即联系专业帮助：

1. 拨打心理援助热线:400-161-9995(24小时)
2. 联系学校心理咨询中心
3. 告诉信任的家人或朋友
4. 前往最近医院的急诊科

你并不孤单，请立即寻求帮助。我会在这里继续支持你。"""
    
    def _get_default_warning_response(self) -> str:
        """获取默认警告回应"""
        return "我能感受到你现在可能有些困扰。如果你愿意，可以多和我聊聊你的感受。寻求帮助是勇敢的表现，如果需要，我可以为你提供一些专业资源的建议。"

    def _get_default_third_party_support_response(self) -> str:
        return (
            "你愿意为对方求助很重要。现在先尽量不要让对方一个人待着，直接联系他身边可信任的家人、老师、辅导员或同住的人，"
            "并询问他现在是否安全、是否已经有具体计划；如果风险很近或已经失控，请立即联系当地紧急服务。"
        )


class UrgentLogger:
    """紧急情况日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def log_interaction(self, interaction_data: Dict):
        """记录紧急交互到文件"""
        log_entry = self._create_log_entry(interaction_data)
        
        log_file = f"{self.log_dir}/urgent_cases_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            logs = self._load_existing_logs(log_file)
            logs.append(log_entry)
            self._save_logs(log_file, logs)
            logger.info(f"紧急情况已记录: {log_file}")
        except Exception as e:
            logger.error(f"记录紧急情况失败: {e}")
    
    async def log_interaction_async(self, interaction_data: Dict):
        try:
            await asyncio.to_thread(self.log_interaction, interaction_data)
        except Exception as e:
            logger.error(f"记录紧急情况失败(异步): {e}")

    def _create_log_entry(self, interaction_data: Dict) -> Dict:
        """创建日志条目"""
        return {
            'timestamp': datetime.now().isoformat(),
            'user_id': interaction_data['user_id'],
            'session_id': interaction_data['session_id'],
            'urgent_level': normalize_risk_level(interaction_data['urgent_issue']['level']),
            'legacy_level': interaction_data['urgent_issue'].get(
                'legacy_level',
                risk_level_band(interaction_data['urgent_issue']['level']),
            ),
            'triggers': interaction_data['urgent_issue']['triggers'],
            'risk_score': interaction_data['urgent_issue'].get('risk_score', 0.0),
            'user_input': interaction_data['user_input'][:200],
            'emotion': interaction_data['emotion'],
            'ai_response_preview': interaction_data['ai_response'][:100]
        }
    
    def _load_existing_logs(self, log_file: str) -> List[Dict]:
        """加载现有日志"""
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_logs(self, log_file: str, logs: List[Dict]):
        """保存日志到文件"""
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    
    def get_recent_cases(self, days: int = 1, level: Optional[str] = None) -> Dict:
        """获取最近的紧急情况记录"""
        today = datetime.now()
        cases = []
        
        for i in range(days):
            date_str = (today - timedelta(days=i)).strftime('%Y%m%d')
            log_file = f"{self.log_dir}/urgent_cases_{date_str}.json"
            
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        day_cases = json.load(f)
                    
                    if level:
                        day_cases = [c for c in day_cases if c['urgent_level'] == level]
                    
                    cases.extend(day_cases)
                except Exception as e:
                    logger.error(f"读取日志文件失败 {log_file}: {e}")
        
        # 按时间倒序排序
        cases.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # 统计信息
        stats = self._calculate_statistics(cases, days)
        
        return {
            'statistics': stats,
            'cases': cases[:100]  # 返回最多100条记录
        }
    
    async def get_recent_cases_async(self, days: int = 1, level: Optional[str] = None) -> Dict:
        return await asyncio.to_thread(self.get_recent_cases, days, level)
    
    def _calculate_statistics(self, cases: List[Dict], days: int) -> Dict[str, Any]:
        """计算统计信息"""
        return {
            'total_cases': len(cases),
            'urgent_count': len(
                [c for c in cases if normalize_risk_level(c['urgent_level']) == LEVEL_3]
            ),
            'warning_high_count': len(
                [c for c in cases if normalize_risk_level(c['urgent_level']) == LEVEL_2]
            ),
            'warning_count': len(
                [c for c in cases if normalize_risk_level(c['urgent_level']) == LEVEL_1]
            ),
            'high_count': len([c for c in cases if normalize_risk_level(c['urgent_level']) == LEVEL_3]),
            'medium_count': len(
                [c for c in cases if normalize_risk_level(c['urgent_level']) in {LEVEL_1, LEVEL_2}]
            ),
            'low_count': len([c for c in cases if normalize_risk_level(c['urgent_level']) == LEVEL_0]),
            'period_days': days,
            'avg_risk_score': sum(c.get('risk_score', 0) for c in cases) / max(len(cases), 1)
        }


urgent_detector = UrgentDetector()
urgent_logger = UrgentLogger()
