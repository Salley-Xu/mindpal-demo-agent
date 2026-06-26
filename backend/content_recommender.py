import logging
from typing import List, Dict, Any, Tuple
import re
from datetime import datetime
from openai import AsyncOpenAI
from config import config
from models import ContentItem
from conversation_manager import ConversationManager
from content_db import content_db

logger = logging.getLogger(__name__)

class ContentRecommender:
    """个性化内容推荐引擎"""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.API_BASE_URL
        )
        self.model = config.CHAT_MODEL
        # 推荐策略配置
        self.enable_ai_rerank: bool = getattr(config, "ENABLE_RECOMMEND_AI_RERANK", True)
        self.rerank_candidate_size: int = getattr(config, "RECOMMEND_RERANK_CANDIDATE_SIZE", 10)
        
        # 情绪到内容的映射权重
        self.emotion_weights = {
            "学业压力": ["academic", "stress_management"],
            "焦虑": ["relaxation", "mindfulness", "anxiety"],
            "抑郁": ["self_reflection", "mood_management"],
            "愤怒": ["anger_management", "emotional_regulation"],
            "压力": ["stress_management", "relaxation"],
            "人际矛盾": ["relationship", "communication"],
            "困惑": ["self_reflection", "decision_making"],
            "不确定": ["future", "decision_making"],
            "未来迷茫": ["future", "career_planning"],
            "自我怀疑": ["self_esteem", "self_reflection"],
            "孤独": ["relationship", "social_skills"],
            "失眠": ["sleep", "relaxation"]
        }
        
        # 对话阶段到内容深度的映射
        self.stage_depth_mapping = {
            "initial": "beginner",
            "exploring": "beginner",
            "deepening": "intermediate",
            "resolving": ["intermediate", "advanced"]
        }
    
    async def recommend_content(self, 
                         user_input: str,
                         current_emotion: str,
                         conversation_summary: Dict[str, Any],
                         user_profile: Dict[str, Any] = None,
                         content_types: List[str] = None,
                         limit: int = 3) -> Tuple[List[ContentItem], str, Dict[str, float]]:
        """
        推荐个性化内容
        
        返回: (推荐内容列表, 推荐理由, 匹配度分数)
        """
        try:
            # 构造/归一化用户画像，保证下游逻辑稳定
            normalized_profile = self._normalize_user_profile(user_profile or {})

            # 策略1: 基于情绪、对话上下文 + 用户画像的规则推荐
            # 为了给 AI rerank 预留余地，这里先取 Top-N 候选，再根据配置裁剪到最终上限
            rule_limit = max(limit, self.rerank_candidate_size) if self.enable_ai_rerank else limit
            rule_based_recs = self._rule_based_recommendation(
                user_input, current_emotion, conversation_summary, normalized_profile, rule_limit
            )
            
            # 策略2: 使用 AI 对规则候选集进行 rerank（可通过配置关闭）
            if self.enable_ai_rerank and rule_based_recs:
                ai_reranked = await self._ai_based_recommendation(
                    user_input,
                    current_emotion,
                    conversation_summary,
                    rule_based_recs,
                    normalized_profile,
                    limit,
                )
                # AI 只对规则候选做排序/筛选，若失败则回退到规则结果
                recommended_items = ai_reranked[:limit] if ai_reranked else rule_based_recs[:limit]
            else:
                recommended_items = rule_based_recs[:limit]
            
            # 生成推荐理由
            rationale = self._generate_rationale(
                recommended_items, user_input, current_emotion, conversation_summary
            )
            
            # 计算匹配度分数
            match_scores = self._calculate_match_scores(
                recommended_items, user_input, current_emotion, conversation_summary, normalized_profile
            )
            
            return recommended_items, rationale, match_scores
            
        except Exception as e:
            logger.error(f"内容推荐失败: {e}")
            # 返回默认推荐
            default_recs = content_db.search_content(current_emotion, limit=limit)
            return default_recs, "根据你的当前状态推荐以下内容", {"default": 0.7}
    
    def _rule_based_recommendation(self,
                                 user_input: str,
                                 current_emotion: str,
                                 conversation_summary: Dict[str, Any],
                                 user_profile: Dict[str, Any],
                                 limit: int) -> List[ContentItem]:
        """基于规则的推荐"""
        all_content = content_db.get_all_content()
        scored_items = []
        
        # 提取关键词
        keywords = self._extract_keywords(user_input)
        
        for item in all_content:
            score = 0.0
            
            # 1. 情绪匹配（权重最高）
            if current_emotion in item.emotion_tags:
                score += 3.0
            for emotion_tag in item.emotion_tags:
                if emotion_tag in self.emotion_weights:
                    if current_emotion in self.emotion_weights[emotion_tag]:
                        score += 2.0
            
            # 2. 关键词匹配
            for keyword in keywords:
                if keyword in item.title.lower() or keyword in ' '.join(item.tags).lower():
                    score += 2.0
            
            # 3. 关切点匹配
            key_concerns = conversation_summary.get('key_concerns', [])
            for concern in key_concerns:
                if concern in item.tags or concern in item.category:
                    score += 1.5
            
            # 4. 对话阶段匹配（难度适配）
            stage = conversation_summary.get('conversation_stage', 'initial')
            depth = self.stage_depth_mapping.get(stage, 'beginner')
            if isinstance(depth, list):
                if item.difficulty in depth:
                    score += 1.0
            elif item.difficulty == depth:
                score += 1.0
            
            # 5. 热度加权
            score += item.popularity * 0.01

            # 6. 画像偏好加权
            score = self._adjust_score_with_profile(score, item, user_profile)
            
            if score > 0:
                scored_items.append((score, item))
        
        # 按分数排序
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_items[:limit]]
    
    async def _ai_based_recommendation(self,
                                user_input: str,
                                current_emotion: str,
                                conversation_summary: Dict[str, Any],
                                candidates: List[ContentItem],
                                user_profile: Dict[str, Any],
                                limit: int) -> List[ContentItem]:
        """基于AI对规则候选集进行 rerank 的智能推荐"""
        try:
            # 构建系统提示词
            system_prompt = """你是一个心理内容推荐专家。请根据用户的情况,从以下内容库中选择最合适的3个推荐。
            考虑因素：
            1. 用户的当前情绪状态
            2. 用户的表达内容
            3. 对话阶段和深度
            4. 内容的匹配度和实用性
            
            请返回内容ID列表的JSON格式，例如: {"ids": ["id1", "id2", "id3"]}"""
            
            # 仅对规则候选集进行 rerank，避免一次性传入全部内容
            rerank_candidates = candidates[: self.rerank_candidate_size]
            content_descriptions = []
            for item in rerank_candidates:
                desc = (
                    f"ID: {item.id} | 标题: {item.title} | 类型: {item.type} | 分类: {item.category} | "
                    f"难度: {item.difficulty or 'unknown'} | 时长: {item.duration_minutes or 'unknown'} 分钟 | "
                    f"描述: {item.description} | 标签: {', '.join(item.tags)}"
                )
                content_descriptions.append(desc)
            
            # 用户画像摘要，帮助模型进行个性化排序
            profile_summary = f"""
            用户风险等级: {user_profile.get('risk_level', 'normal')}
            偏好内容类型: {', '.join(user_profile.get('preferred_types', []) or [])}
            偏好主题: {', '.join(user_profile.get('preferred_categories', []) or [])}
            偏好难度: {user_profile.get('preferred_difficulty', 'beginner')}
            偏好时长范围: {user_profile.get('preferred_duration_range') or '未指定'}
            """

            user_prompt = f"""用户输入: {user_input}
            当前情绪: {current_emotion}
            对话阶段: {conversation_summary.get('conversation_stage', 'initial')}
            关切点: {', '.join(conversation_summary.get('key_concerns', []))}
            用户画像: {profile_summary}
            
            可用内容:
            {'\\n'.join(content_descriptions)}  # 已在候选层面做了数量控制
            
            请推荐最合适的3个内容ID:"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=100,
                response_format={ "type": "json_object" }
            )
            
            # 解析响应
            response_text = response.choices[0].message.content.strip()
            logger.info(f"AI推荐响应: {response_text}")
            
            # 提取内容ID，并保持在候选集合内
            id_to_item = {item.id: item for item in rerank_candidates}
            ranked_items: List[ContentItem] = []
            try:
                import json
                data = json.loads(response_text)
                # 尝试从不同的键中获取ID列表，兼容AI可能返回的各种格式
                ids_list = data.get("ids") or data.get("content_ids") or data.get("recommendations") or []
                
                # 如果是直接返回列表的情况
                if isinstance(data, list):
                    ids_list = data
                    
                # 确保是列表
                if not isinstance(ids_list, list):
                    logger.warning(f"AI返回格式未能解析为列表: {data}")
                    ids_list = []
                
                for content_id in ids_list:
                    content_id_str = str(content_id)
                    content_item = id_to_item.get(content_id_str)
                    if content_item and content_item not in ranked_items:
                        ranked_items.append(content_item)
            except json.JSONDecodeError:
                logger.error(f"AI响应JSON解析失败: {response_text}")
                # 降级策略：尝试正则提取
                import re
                id_pattern = r'["\']([a-zA-Z0-9_]+)["\']'
                matches = re.findall(id_pattern, response_text)
                for content_id in matches:
                    content_item = id_to_item.get(content_id)
                    if content_item and content_item not in ranked_items:
                        ranked_items.append(content_item)
            
            # 若模型返回为空或解析失败，调用方会回退到规则打分结果
            return ranked_items[:limit] if ranked_items else []
            
        except Exception as e:
            logger.error(f"AI推荐失败: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简单的中文关键词提取
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        
        # 心理相关关键词增强
        psych_keywords = [
            "压力", "焦虑", "抑郁", "情绪", "学习", "考试", "工作",
            "关系", "朋友", "家人", "未来", "迷茫", "自我", "自信",
            "睡眠", "饮食", "运动", "放松", "冥想", "正念"
        ]
        
        keywords = list(set(chinese_words))
        
        # 添加匹配的心理关键词
        for kw in psych_keywords:
            if kw in text and kw not in keywords:
                keywords.append(kw)
        
        return keywords
    
    def _generate_rationale(self,
                           recommended_items: List[ContentItem],
                           user_input: str,
                           current_emotion: str,
                           conversation_summary: Dict[str, Any]) -> str:
        """生成推荐理由"""
        if not recommended_items:
            return "暂时没有找到特别匹配的内容。"
        
        # 根据推荐内容类型生成理由
        content_types = [item.type for item in recommended_items]
        stage = conversation_summary.get('conversation_stage', 'initial')
        
        rationale_templates = {
            "initial": "根据你提到的内容，这些资源可能对你有帮助：",
            "exploring": "在探索阶段，这些内容可以帮助你更深入地理解自己：",
            "deepening": "这些专业资源可以帮助你进一步分析问题：",
            "resolving": "这些实用工具和策略可以帮助你采取行动："
        }
        
        base_rationale = rationale_templates.get(stage, "根据你的情况推荐以下内容：")
        
        # 添加具体理由
        specific_reasons = []
        for item in recommended_items[:2]:  # 只取前两个详细说明
            if current_emotion in item.emotion_tags:
                specific_reasons.append(f"《{item.title}》特别适合处理{current_emotion}状态")
            elif any(tag in item.tags for tag in conversation_summary.get('key_concerns', [])):
                specific_reasons.append(f"《{item.title}》与你关注的方面相关")
        
        if specific_reasons:
            return f"{base_rationale} {'；'.join(specific_reasons)}"
        
        return base_rationale
    
    def _calculate_match_scores(self,
                               recommended_items: List[ContentItem],
                               user_input: str,
                               current_emotion: str,
                               conversation_summary: Dict[str, Any],
                               user_profile: Dict[str, Any]) -> Dict[str, float]:
        """计算匹配度分数"""
        scores = {}
        
        for item in recommended_items:
            item_score = 0.0
            
            # 情绪匹配度
            if current_emotion in item.emotion_tags:
                item_score += 0.4
            
            # 关切点匹配度
            key_concerns = conversation_summary.get('key_concerns', [])
            for concern in key_concerns:
                if concern in item.tags or concern in item.category:
                    item_score += 0.3
                    break
            
            # 对话阶段适配度
            stage = conversation_summary.get('conversation_stage', 'initial')
            depth = self.stage_depth_mapping.get(stage, 'beginner')
            if item.difficulty == depth or (isinstance(depth, list) and item.difficulty in depth):
                item_score += 0.2
            
            # 内容热度
            item_score += min(item.popularity * 0.01, 0.1)

            # 个性化匹配度（用户画像维度，最高 0.2）
            personalization_score = 0.0
            preferred_types = user_profile.get("preferred_types", []) or []
            preferred_categories = user_profile.get("preferred_categories", []) or []
            preferred_difficulty = user_profile.get("preferred_difficulty", "beginner")

            if preferred_types and item.type in preferred_types:
                personalization_score += 0.1
            if preferred_categories and item.category in preferred_categories:
                personalization_score += 0.1
            if item.difficulty and item.difficulty == preferred_difficulty:
                personalization_score += 0.05

            # 风险等级对某些内容的总体降/升权（这里保持简单：高风险时，过高难度内容略微降权）
            risk_level = user_profile.get("risk_level", "normal")
            if risk_level == "high" and item.difficulty == "advanced":
                personalization_score -= 0.05

            item_score += max(min(personalization_score, 0.2), -0.1)
            
            scores[item.id] = min(item_score, 1.0)
        
        return scores

    def _normalize_user_profile(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """确保用户画像结构完整，提供合理默认值"""
        profile = dict(user_profile or {})
        profile.setdefault("risk_level", "normal")
        profile.setdefault("preferred_types", [])
        profile.setdefault("preferred_categories", [])
        profile.setdefault("preferred_difficulty", "beginner")
        # preferred_duration_range 可以为空，表示不做时长强约束
        if "preferred_duration_range" not in profile:
            profile["preferred_duration_range"] = None
        return profile

    def _adjust_score_with_profile(
        self, score: float, item: ContentItem, user_profile: Dict[str, Any]
    ) -> float:
        """根据用户画像对规则分数进行微调"""
        adjusted = score
        preferred_types = user_profile.get("preferred_types", []) or []
        preferred_categories = user_profile.get("preferred_categories", []) or []
        preferred_difficulty = user_profile.get("preferred_difficulty", "beginner")
        preferred_duration_range = user_profile.get("preferred_duration_range")
        risk_level = user_profile.get("risk_level", "normal")

        # 类型偏好
        if preferred_types and item.type in preferred_types:
            adjusted += 0.8

        # 主题/类别偏好
        if preferred_categories and item.category in preferred_categories:
            adjusted += 0.8

        # 难度偏好
        if item.difficulty:
            if item.difficulty == preferred_difficulty:
                adjusted += 0.5
            # 高风险用户，过高难度内容略微降权
            if risk_level == "high" and item.difficulty == "advanced":
                adjusted -= 0.5

        # 时长偏好（如有配置）
        if preferred_duration_range and item.duration_minutes is not None:
            try:
                min_dur = preferred_duration_range.get("min")
                max_dur = preferred_duration_range.get("max")
                if min_dur is not None and item.duration_minutes < min_dur:
                    adjusted -= 0.2
                if max_dur is not None and item.duration_minutes > max_dur:
                    adjusted -= 0.2
                if (
                    min_dur is not None
                    and max_dur is not None
                    and min_dur <= item.duration_minutes <= max_dur
                ):
                    adjusted += 0.3
            except Exception:
                # 偏好结构异常时不影响主流程
                pass

        return adjusted

# 全局推荐器实例
content_recommender = ContentRecommender()
