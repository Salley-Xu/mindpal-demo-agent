import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from models import ContentItem
import logging
from pydantic.json import pydantic_encoder

logger = logging.getLogger(__name__)

class ContentDatabase:
    """内容数据库管理器"""
    
    def __init__(self, data_file: str = "data/content_db.json"):
        self.data_file = data_file
        self.content_items: Dict[str, ContentItem] = {}
        self._load_content()
    
    def _load_content(self):
        """加载内容数据"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item_data in data:
                        item_data = self._upgrade_item_schema(item_data)
                        # 处理datetime字符串
                        if 'created_at' in item_data and isinstance(item_data['created_at'], str):
                            try:
                                item_data['created_at'] = datetime.fromisoformat(item_data['created_at'])
                            except:
                                item_data['created_at'] = datetime.now()
                    
                        item = ContentItem(**item_data)
                        self.content_items[item.id] = item
                logger.info(f"已加载 {len(self.content_items)} 个内容项")
            else:
                # 初始化示例数据
                self._initialize_sample_content()
                logger.info("已初始化示例内容数据库")
        except json.JSONDecodeError as e:
            logger.error(f"内容数据库JSON格式错误: {e}")
            logger.info("重新初始化内容数据库...")
            self._initialize_sample_content()
        except Exception as e:
            logger.error(f"加载内容数据库失败: {e}")
            self._initialize_sample_content()

    def _upgrade_item_schema(self, item_data: Dict) -> Dict:
        """为旧版内容库补充新 schema 默认字段。"""
        upgraded = dict(item_data)
        defaults_by_id = {
            "article_001": {
                "content": "将任务拆成可执行的小步，先处理最紧急的一件，再安排复盘和休息时间。",
                "recommend_type": "hard",
                "priority": 0.88,
                "source": "mindpal_editorial",
                "actionability": 0.82,
                "contraindications": ["high_risk_crisis"],
            },
            "audio_001": {
                "content": "跟随音频进行呼吸放松和身体扫描，帮助快速降低紧张感。",
                "recommend_type": "soft",
                "priority": 0.84,
                "source": "mindpal_editorial",
                "actionability": 0.7,
                "contraindications": ["high_risk_crisis"],
            },
            "exercise_001": {
                "content": "记录情绪触发事件、当时想法、身体感受和你想采取的下一步。",
                "recommend_type": "soft",
                "priority": 0.72,
                "source": "mindpal_editorial",
                "actionability": 0.68,
                "contraindications": ["high_risk_crisis"],
            },
            "article_002": {
                "content": "尝试使用感受-需求-请求的表达框架，减少冲突中的指责感。",
                "recommend_type": "hard",
                "priority": 0.76,
                "source": "mindpal_editorial",
                "actionability": 0.73,
                "contraindications": ["high_risk_crisis"],
            },
            "audio_002": {
                "content": "进行 3 到 5 分钟的呼吸计数，把注意力带回当下。",
                "recommend_type": "hard",
                "priority": 0.9,
                "source": "mindpal_editorial",
                "actionability": 0.9,
                "contraindications": ["high_risk_crisis"],
            },
            "tool_001": {
                "content": "写下自动化想法、证据支持与反证，并重写更平衡的新想法。",
                "recommend_type": "hard",
                "priority": 0.78,
                "source": "mindpal_editorial",
                "actionability": 0.86,
                "contraindications": ["high_risk_crisis"],
            },
            "article_003": {
                "content": "把未来规划拆成探索、验证和行动三步，先验证最小方向而不是一次定终身。",
                "recommend_type": "soft",
                "priority": 0.8,
                "source": "mindpal_editorial",
                "actionability": 0.74,
                "contraindications": ["high_risk_crisis"],
            },
            "audio_003": {
                "content": "依次放松肩颈、手臂、腿部与呼吸，帮助身体进入睡眠准备状态。",
                "recommend_type": "hard",
                "priority": 0.77,
                "source": "mindpal_editorial",
                "actionability": 0.83,
                "contraindications": ["high_risk_crisis"],
            },
        }
        defaults = defaults_by_id.get(upgraded.get("id"), {})
        for key, value in defaults.items():
            if key not in upgraded:
                upgraded[key] = value
        upgraded.setdefault("risk_levels", ["low", "medium"])
        upgraded.setdefault("contraindications", [])
        upgraded.setdefault("recommend_type", "soft")
        upgraded.setdefault("priority", 0.5)
        upgraded.setdefault("source", "mindpal_editorial")
        upgraded.setdefault("actionability", 0.5)
        return upgraded
    
    def _initialize_sample_content(self):
        """初始化示例内容"""
        sample_content = [
            {
                "id": "article_001",
                "title": "如何应对学业压力:5个实用策略",
                "type": "article",
                "category": "academic",
                "description": "针对大学生常见的学业压力问题，提供具体的应对策略和心理调适方法。",
                "content": "将任务拆成可执行的小步，先处理最紧急的一件，再安排复盘和休息时间。",
                "url": "/articles/academic_stress_management.html",
                "tags": ["学业压力", "时间管理", "考试焦虑", "学习方法"],
                "emotion_tags": ["学业压力", "焦虑", "压力", "困惑"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "hard",
                "priority": 0.88,
                "source": "mindpal_editorial",
                "actionability": 0.82,
                "difficulty": "beginner"
            },
            {
                "id": "audio_001",
                "title": "10分钟放松冥想引导",
                "type": "audio",
                "category": "relaxation",
                "description": "专门为缓解焦虑设计的冥想音频，适合睡前或压力大时聆听。",
                "content": "跟随音频进行呼吸放松和身体扫描，帮助快速降低紧张感。",
                "url": "/audios/10min_relaxation.mp3",
                "duration_minutes": 10,
                "tags": ["冥想", "放松", "焦虑缓解", "睡眠"],
                "emotion_tags": ["焦虑", "压力", "失眠", "紧张"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.84,
                "source": "mindpal_editorial",
                "actionability": 0.7,
                "difficulty": "beginner"
            },
            {
                "id": "exercise_001",
                "title": "情绪日记练习",
                "type": "exercise",
                "category": "self_reflection",
                "description": "通过记录情绪日记，提高情绪觉察能力，了解自己的情绪模式。",
                "content": "记录情绪触发事件、当时想法、身体感受和你想采取的下一步。",
                "tags": ["情绪觉察", "日记", "自我反思", "情绪管理"],
                "emotion_tags": ["困惑", "不确定", "情绪压抑", "自我怀疑"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.72,
                "source": "mindpal_editorial",
                "actionability": 0.68,
                "difficulty": "beginner"
            },
            {
                "id": "article_002",
                "title": "改善人际关系的沟通技巧",
                "type": "article",
                "category": "relationship",
                "description": "学习有效的沟通方法，改善与朋友、家人和恋人的关系。",
                "content": "尝试使用感受-需求-请求的表达框架，减少冲突中的指责感。",
                "url": "/articles/communication_skills.html",
                "tags": ["人际关系", "沟通", "冲突解决", "社交技巧"],
                "emotion_tags": ["人际矛盾", "孤独", "被误解", "社交焦虑"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "hard",
                "priority": 0.76,
                "source": "mindpal_editorial",
                "actionability": 0.73,
                "difficulty": "intermediate"
            },
            {
                "id": "audio_002",
                "title": "正念呼吸练习",
                "type": "audio",
                "category": "mindfulness",
                "description": "简短的正念呼吸练习，帮助你在紧张时刻快速平静下来。",
                "content": "进行 3 到 5 分钟的呼吸计数，把注意力带回当下。",
                "url": "/audios/mindful_breathing.mp3",
                "duration_minutes": 5,
                "tags": ["正念", "呼吸", "专注", "当下"],
                "emotion_tags": ["焦虑", "压力", "注意力分散", "过度思考"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "hard",
                "priority": 0.9,
                "source": "mindpal_editorial",
                "actionability": 0.9,
                "difficulty": "beginner"
            },
            {
                "id": "tool_001",
                "title": "认知重构工作表",
                "type": "tool",
                "category": "cognitive_restructuring",
                "description": "识别并挑战负面思维模式，建立更健康的思考方式。",
                "content": "写下自动化想法、证据支持与反证，并重写更平衡的新想法。",
                "url": "/tools/cognitive_restructuring.pdf",
                "tags": ["认知行为疗法", "思维模式", "自动思维", "心理工具"],
                "emotion_tags": ["焦虑", "抑郁", "自我怀疑", "负面思维"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "hard",
                "priority": 0.78,
                "source": "mindpal_editorial",
                "actionability": 0.86,
                "difficulty": "intermediate"
            },
            {
                "id": "article_003",
                "title": "未来规划：如何应对职业迷茫",
                "type": "article",
                "category": "future",
                "description": "针对大学生常见的职业迷茫问题，提供实用的规划方法和心态调整建议。",
                "content": "把未来规划拆成探索、验证和行动三步，先验证最小方向而不是一次定终身。",
                "url": "/articles/career_confusion.html",
                "tags": ["职业规划", "未来迷茫", "就业焦虑", "自我探索"],
                "emotion_tags": ["未来迷茫", "不确定", "焦虑", "压力"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.8,
                "source": "mindpal_editorial",
                "actionability": 0.74,
                "difficulty": "intermediate"
            },
            {
                "id": "audio_003",
                "title": "改善睡眠的渐进式肌肉放松",
                "type": "audio",
                "category": "sleep",
                "description": "针对失眠问题的肌肉放松训练，帮助你更容易入睡。",
                "content": "依次放松肩颈、手臂、腿部与呼吸，帮助身体进入睡眠准备状态。",
                "url": "/audios/progressive_relaxation.mp3",
                "duration_minutes": 15,
                "tags": ["睡眠", "放松", "失眠", "身体扫描"],
                "emotion_tags": ["失眠", "焦虑", "压力", "身体紧张"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "hard",
                "priority": 0.77,
                "source": "mindpal_editorial",
                "actionability": 0.83,
                "difficulty": "beginner"
            }
        ]
        
        for item_data in sample_content:
            item = ContentItem(**item_data)
            self.content_items[item.id] = item
        
        # 保存到文件
        self._save_content()
    
    def _save_content(self):
        """保存内容到文件"""
        try:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            content_list = [item.dict() for item in self.content_items.values()]
        
            # 使用自定义的JSON编码器处理datetime
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(
                    content_list, 
                    f, 
                    ensure_ascii=False, 
                    indent=2,
                    default=self._json_serializer
                )
            logger.info(f"内容数据库已保存: {self.data_file}")
        except Exception as e:
            logger.error(f"保存内容数据库失败: {e}")
    
    def _json_serializer(self, obj):
        """自定义JSON序列化器"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        
    
    def get_all_content(self) -> List[ContentItem]:
        """获取所有内容"""
        return list(self.content_items.values())
    
    def get_content_by_id(self, content_id: str) -> Optional[ContentItem]:
        """根据ID获取内容"""
        return self.content_items.get(content_id)
    
    def search_content(self, query: str, limit: int = 10) -> List[ContentItem]:
        """搜索内容 (关键词匹配优化版)"""
        query_lower = query.lower()
        keywords = query_lower.split()
        results = []
        
        for item in self.content_items.values():
            # 计算匹配分数
            score = 0
            
            # 1. 精确短语匹配
            if query_lower in item.title.lower():
                score += 10
            elif query_lower in item.description.lower():
                score += 5
                
            # 2. 关键词匹配
            for kw in keywords:
                if len(kw) < 2: continue # 跳过太短的词
                
                if kw in item.title.lower():
                    score += 3
                if kw in item.description.lower():
                    score += 1
                
                # 标签匹配
                for tag in item.tags:
                    if kw in tag.lower():
                        score += 2
            
            if score > 0:
                results.append((score, item))
        
        # 按分数降序排序
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [item for score, item in results[:limit]]
    
    def increment_popularity(self, content_id: str):
        """增加内容热度"""
        if content_id in self.content_items:
            self.content_items[content_id].popularity += 1
            self._save_content()
    
    def add_content(self, content_item: ContentItem):
        """添加新内容"""
        self.content_items[content_item.id] = content_item
        self._save_content()
        logger.info(f"已添加内容: {content_item.title}")

# 全局内容数据库实例
content_db = ContentDatabase()
