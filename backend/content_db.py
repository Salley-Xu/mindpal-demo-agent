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
            },
            # === 新内容：覆盖抑郁 ===
            {
                "id": "article_004",
                "title": "走出抑郁低谷：日常情绪管理指南",
                "type": "article",
                "category": "mood_management",
                "description": "针对持续性情绪低落，提供温和可操作的日常管理策略，帮助逐步恢复内在平衡。",
                "content": "每天给自己三个小目标：一次外出、一件完成的小事、一次对自己说「这样就很好」。逐步积累正向体验。",
                "url": "/articles/mood_management_guide.html",
                "tags": ["抑郁", "情绪管理", "日常习惯", "自我关怀"],
                "emotion_tags": ["抑郁", "情绪低落", "自我怀疑", "疲惫"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.82,
                "source": "mindpal_editorial",
                "actionability": 0.76,
                "difficulty": "beginner"
            },
            {
                "id": "audio_004",
                "title": "温暖自我关怀冥想（12分钟）",
                "type": "audio",
                "category": "mood_management",
                "description": "以温和友善的态度引导你关注自己的内心，适合情绪低落或自我批评强烈时练习。",
                "content": "将手放在心上，对自己说：愿我平安，愿我快乐，愿我轻松。让温暖的感觉从掌心蔓延到全身。",
                "url": "/audios/self_compassion.mp3",
                "duration_minutes": 12,
                "tags": ["自我关怀", "冥想", "温和", "接纳"],
                "emotion_tags": ["抑郁", "焦虑", "自我怀疑", "疲惫"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.8,
                "source": "mindpal_editorial",
                "actionability": 0.65,
                "difficulty": "beginner"
            },
            {
                "id": "exercise_002",
                "title": "每日三件好事：积极心理练习",
                "type": "exercise",
                "category": "self_reflection",
                "description": "基于积极心理学的研究，通过记录每天的三件好事来改善情绪状态。",
                "content": "每天睡前写下今天发生的三件好事（可大可小），并写下它们为什么会发生。坚持21天。",
                "tags": ["积极心理", "感恩", "日记", "正向思维"],
                "emotion_tags": ["抑郁", "情绪低落", "自我怀疑", "困惑"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "hard",
                "priority": 0.75,
                "source": "mindpal_editorial",
                "actionability": 0.9,
                "difficulty": "beginner"
            },
            # === 新内容：覆盖愤怒 ===
            {
                "id": "article_005",
                "title": "愤怒管理：识别情绪触发器与健康表达",
                "type": "article",
                "category": "anger_management",
                "description": "帮助你理解愤怒背后的需求，学会在情绪爆发前识别信号，并以不伤害关系的方式表达。",
                "content": "愤怒来临时先做三次深呼吸，问自己：我真正需要的是什么？然后用「我感到…因为…我需要…」的句式表达。",
                "url": "/articles/anger_management.html",
                "tags": ["愤怒", "情绪管理", "沟通", "冲突解决"],
                "emotion_tags": ["愤怒", "人际矛盾", "压力", "烦躁"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "hard",
                "priority": 0.78,
                "source": "mindpal_editorial",
                "actionability": 0.72,
                "difficulty": "intermediate"
            },
            {
                "id": "audio_005",
                "title": "3分钟快速冷静呼吸练习",
                "type": "audio",
                "category": "emotional_regulation",
                "description": "极简的呼吸冷静练习，适合愤怒或极度焦虑时快速平复情绪。",
                "content": "吸气4秒→屏息4秒→呼气6秒。重复5轮，专注于呼吸的节奏，让身体慢慢放松下来。",
                "url": "/audios/quick_calm_breathing.mp3",
                "duration_minutes": 3,
                "tags": ["呼吸", "冷静", "情绪调节", "快速放松"],
                "emotion_tags": ["愤怒", "焦虑", "紧张", "烦躁"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "hard",
                "priority": 0.88,
                "source": "mindpal_editorial",
                "actionability": 0.92,
                "difficulty": "beginner"
            },
            # === 新内容：覆盖不确定/困惑 ===
            {
                "id": "article_006",
                "title": "在不确定中找到方向：决策焦虑应对指南",
                "type": "article",
                "category": "decision_making",
                "description": "面对人生选择和不确定的未来时，提供一套可操作的决策框架来减轻焦虑。",
                "content": "列出每个选项的利弊，想象每个选择后1年的自己是什么状态，然后问自己：哪个选择更接近我想要的生活。",
                "url": "/articles/decision_anxiety.html",
                "tags": ["决策", "不确定", "焦虑", "选择困难"],
                "emotion_tags": ["不确定", "困惑", "焦虑", "未来迷茫"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.8,
                "source": "mindpal_editorial",
                "actionability": 0.74,
                "difficulty": "beginner"
            },
            {
                "id": "tool_002",
                "title": "决策平衡单：可视化选择分析工具",
                "type": "tool",
                "category": "decision_making",
                "description": "通过多维度打分矩阵，将模糊的决策困境转化为可视化的比较数据。",
                "content": "列出你的选项，定义5-8个评估维度（如收入、成长、幸福感等），每个维度按1-10打分，加权求和后比较。",
                "url": "/tools/decision_balance.html",
                "tags": ["决策工具", "分析", "规划", "理性思考"],
                "emotion_tags": ["不确定", "困惑", "未来迷茫", "焦虑"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.76,
                "source": "mindpal_editorial",
                "actionability": 0.85,
                "difficulty": "intermediate"
            },
            # === 新内容：覆盖自我怀疑 ===
            {
                "id": "article_007",
                "title": "打破自我怀疑：建立健康的自我评价体系",
                "type": "article",
                "category": "self_esteem",
                "description": "识别内在批评者的声音，学习用更客观友善的视角看待自己的价值。",
                "content": "当自我怀疑出现时，写下三个证据证明你「不够好」，再写下三个反例。你往往会发现反例更真实。",
                "url": "/articles/overcome_self_doubt.html",
                "tags": ["自我怀疑", "自信", "自尊", "自我接纳"],
                "emotion_tags": ["自我怀疑", "抑郁", "困惑", "焦虑"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.8,
                "source": "mindpal_editorial",
                "actionability": 0.7,
                "difficulty": "intermediate"
            },
            {
                "id": "audio_006",
                "title": "自我肯定与内在力量冥想（10分钟）",
                "type": "audio",
                "category": "self_esteem",
                "description": "通过积极肯定的引导语，帮助你建立内在的安全感和自我价值感。",
                "content": "深呼吸，对自己说：我足够好，我有自己的独特价值，我值得被尊重和爱。让这些话语渗透到内心。",
                "url": "/audios/self_affirmation.mp3",
                "duration_minutes": 10,
                "tags": ["自我肯定", "冥想", "自信", "内在力量"],
                "emotion_tags": ["自我怀疑", "焦虑", "压力", "抑郁"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.82,
                "source": "mindpal_editorial",
                "actionability": 0.68,
                "difficulty": "beginner"
            },
            # === 新内容：覆盖孤独 ===
            {
                "id": "article_008",
                "title": "连接的力量：如何应对孤独感",
                "type": "article",
                "category": "social_skills",
                "description": "孤独是人类的共通体验，本文提供从自我连接开始，逐步建立社交联系的方法。",
                "content": "从每周一次与一位朋友简短聊天开始，参与一个兴趣小组，尝试在安全的环境中分享真实的感受。",
                "url": "/articles/dealing_with_loneliness.html",
                "tags": ["孤独", "社交", "连接", "人际关系"],
                "emotion_tags": ["孤独", "人际矛盾", "困惑", "社交焦虑"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.76,
                "source": "mindpal_editorial",
                "actionability": 0.71,
                "difficulty": "beginner"
            },
            {
                "id": "exercise_003",
                "title": "社交连接计划：从日常小事开始",
                "type": "exercise",
                "category": "relationship",
                "description": "通过结构化的小练习，逐步建立和维护有意义的社交联系。",
                "content": "本周目标：给一个老朋友发条消息；参加一个集体活动；对一位陌生人微笑或打招呼。记录每次的感受。",
                "tags": ["社交", "连接", "人际关系", "小步骤"],
                "emotion_tags": ["孤独", "人际矛盾", "社交焦虑"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.7,
                "source": "mindpal_editorial",
                "actionability": 0.82,
                "difficulty": "intermediate"
            },
            # === 新内容：失眠补充 ===
            {
                "id": "article_009",
                "title": "睡眠卫生指南：科学改善睡眠质量",
                "type": "article",
                "category": "sleep",
                "description": "基于睡眠科学的实用建议，帮助你建立健康的睡眠习惯。",
                "content": "固定作息时间，睡前1小时远离屏幕，保持卧室凉爽黑暗，避免午后摄入咖啡因，睡前不做剧烈运动。",
                "url": "/articles/sleep_hygiene.html",
                "tags": ["睡眠", "失眠", "作息", "健康习惯"],
                "emotion_tags": ["失眠", "焦虑", "压力", "疲惫"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "hard",
                "priority": 0.78,
                "source": "mindpal_editorial",
                "actionability": 0.8,
                "difficulty": "beginner"
            },
            # === 新内容：高级难度 ===
            {
                "id": "article_010",
                "title": "深层认知重构：改变自动化思维模式",
                "type": "article",
                "category": "cognitive_restructuring",
                "description": "面向有一定心理学基础的读者，深入探讨如何识别并重建深层的核心信念。",
                "content": "追踪反复出现的负性思维主题，识别其背后的核心信念（如「我不够好」），收集长期反证，逐步建立替代信念。",
                "url": "/articles/deep_cognitive_restructuring.html",
                "tags": ["认知重构", "核心信念", "CBT", "高级心理技巧"],
                "emotion_tags": ["焦虑", "抑郁", "自我怀疑", "负面思维"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.72,
                "source": "mindpal_editorial",
                "actionability": 0.62,
                "difficulty": "advanced"
            },
            # === 新内容：高风险适配（部分 risk_levels 含 "high"） ===
            {
                "id": "tool_003",
                "title": "情绪急救箱：危机应对计划模板",
                "type": "tool",
                "category": "crisis_management",
                "description": "当情绪快失控时有一份具体的行动计划，帮助你在风暴中锚定自己。",
                "content": "写下：1)我的预警信号 2)立即可以做的3件安抚小事 3)可以联系的人 4)让我感到安全的地方 5)专业求助热线。",
                "url": "/tools/emotional_first_aid.html",
                "tags": ["危机应对", "情绪急救", "安全计划", "自我照顾"],
                "emotion_tags": ["焦虑", "抑郁", "压力", "愤怒"],
                "risk_levels": ["low", "medium", "high"],
                "contraindications": [],
                "recommend_type": "soft",
                "priority": 0.85,
                "source": "mindpal_editorial",
                "actionability": 0.88,
                "difficulty": "beginner"
            },
            {
                "id": "audio_007",
                "title": "5-4-3-2-1 接地练习：应急情绪稳定（5分钟）",
                "type": "audio",
                "category": "emotional_regulation",
                "description": "经典的 grounding 技术，通过调动五种感官快速回到当下，适合情绪即将失控时使用。",
                "content": "说出你看到的5样东西、摸到的4样东西、听到的3种声音、闻到的2种气味、尝到的1种味道。重复直到平静。",
                "url": "/audios/grounding_54321.mp3",
                "duration_minutes": 5,
                "tags": ["接地", "情绪稳定", "焦虑急救", "正念"],
                "emotion_tags": ["焦虑", "愤怒", "紧张", "恐慌"],
                "risk_levels": ["low", "medium", "high"],
                "contraindications": [],
                "recommend_type": "hard",
                "priority": 0.9,
                "source": "mindpal_editorial",
                "actionability": 0.95,
                "difficulty": "beginner"
            },
            # === 新内容：正念与放松补充 ===
            {
                "id": "article_011",
                "title": "正念入门：在日常生活中培养觉察力",
                "type": "article",
                "category": "mindfulness",
                "description": "不需要盘腿打坐，在日常活动中即可练习的正念方法。",
                "content": "选择一件每天必做的事（刷牙、走路、喝水），做的时候全神贯注于感官体验。每次偏离就温和地拉回来。",
                "url": "/articles/mindfulness_basics.html",
                "tags": ["正念", "觉察", "活在当下", "冥想入门"],
                "emotion_tags": ["焦虑", "压力", "困惑", "注意力分散"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.77,
                "source": "mindpal_editorial",
                "actionability": 0.73,
                "difficulty": "beginner"
            },
            {
                "id": "video_001",
                "title": "身体扫描放松练习（引导视频）",
                "type": "video",
                "category": "relaxation",
                "description": "15分钟全身放松引导，从头到脚逐步释放紧绷，适合压力大或入睡前。",
                "content": "从头顶开始，逐步将注意力移到额头、面部、颈部、肩膀……每个部位停留3个呼吸，感受放松的感觉。",
                "url": "/videos/body_scan.mp4",
                "duration_minutes": 15,
                "tags": ["身体扫描", "放松", "减压", "入睡"],
                "emotion_tags": ["焦虑", "压力", "失眠", "紧张"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "hard",
                "priority": 0.8,
                "source": "mindpal_editorial",
                "actionability": 0.85,
                "difficulty": "beginner"
            },
            # === 新内容：情绪韧性 ===
            {
                "id": "article_012",
                "title": "建立情绪韧性：心理健康的长期养护",
                "type": "article",
                "category": "stress_management",
                "description": "情绪韧性是应对生活挑战的心理免疫力，本文提供系统化的培养方案。",
                "content": "从四个维度养护：身体（睡眠/运动/饮食）、情绪（允许感受/自我关怀）、社交（支持网络）、意义（价值感/目标）。",
                "url": "/articles/emotional_resilience.html",
                "tags": ["韧性", "心理健康", "长期养护", "自我成长"],
                "emotion_tags": ["压力", "焦虑", "疲惫", "职业倦怠"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "soft",
                "priority": 0.73,
                "source": "mindpal_editorial",
                "actionability": 0.68,
                "difficulty": "intermediate"
            },
            # === 新内容：考试焦虑 ===
            {
                "id": "article_013",
                "title": "考前心理调适：应对考试焦虑的实用策略",
                "type": "article",
                "category": "academic",
                "description": "针对备考和考试期间的高焦虑状态，提供考前、考中、考后的全流程应对方法。",
                "content": "考前：制定复习计划留出缓冲。考中：紧张时做4-7-8呼吸。考后：不对答案，给自己一个放松活动。",
                "url": "/articles/exam_anxiety.html",
                "tags": ["考试焦虑", "考前准备", "学习方法", "心理调适"],
                "emotion_tags": ["学业压力", "焦虑", "紧张", "压力"],
                "risk_levels": ["low", "medium"],
                "contraindications": ["high_risk_crisis"],
                "recommend_type": "hard",
                "priority": 0.83,
                "source": "mindpal_editorial",
                "actionability": 0.78,
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
