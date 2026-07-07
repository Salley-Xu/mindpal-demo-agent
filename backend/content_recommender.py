import logging
from typing import List, Dict, Any, Tuple
import re

from openai import AsyncOpenAI

from config import config
from content_db import content_db
from hybrid_retriever import hybrid_retriever
from models import ContentItem
from risk_levels import LEVEL_2, LEVEL_3, normalize_risk_level, risk_level_band

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
            recent_risk_levels = conversation_summary.get("recent_risk_levels", []) or []
            current_risk_level = (
                recent_risk_levels[-1]
                if recent_risk_levels
                else normalized_profile.get("risk_level", "low")
            )
            canonical_risk_level = normalize_risk_level(current_risk_level)

            if canonical_risk_level == LEVEL_3:
                return [], "", {}

            # 策略1: 先混合召回候选，再做规则个性化重排
            candidate_limit = max(limit, self.rerank_candidate_size) if self.enable_ai_rerank else limit
            retrieved_candidates = self._retrieve_candidates(
                user_input=user_input,
                current_emotion=current_emotion,
                conversation_summary=conversation_summary,
                user_profile=normalized_profile,
                limit=max(candidate_limit, 8),
            )
            rule_based_recs = self._rank_candidates(
                candidates=retrieved_candidates,
                user_input=user_input,
                current_emotion=current_emotion,
                conversation_summary=conversation_summary,
                user_profile=normalized_profile,
                limit=candidate_limit,
            )
            if canonical_risk_level == LEVEL_2:
                rule_based_recs = [
                    item for item in rule_based_recs
                    if self._is_item_allowed_for_risk(item, canonical_risk_level)
                ]
            
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
    
    def _retrieve_candidates(
        self,
        user_input: str,
        current_emotion: str,
        conversation_summary: Dict[str, Any],
        user_profile: Dict[str, Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        all_content = content_db.get_all_content()
        query_variants = self._build_query_variants(
            user_input=user_input,
            current_emotion=current_emotion,
            conversation_summary=conversation_summary,
            user_profile=user_profile,
        )
        aggregated: Dict[str, Dict[str, Any]] = {}

        for variant in query_variants:
            hybrid_results = hybrid_retriever.retrieve(
                query=variant["query"],
                items=all_content,
                limit=limit,
                extra_terms=variant["extra_terms"],
            )
            for rank, (item, score) in enumerate(hybrid_results, start=1):
                candidate = aggregated.setdefault(
                    item.id,
                    {
                        "item": item,
                        "retrieval_score": 0.0,
                        "matched_queries": [],
                        "retrieval_sources": ["bm25", "vector", "rrf"],
                        "best_rank": rank,
                    },
                )
                candidate["retrieval_score"] += score * variant["weight"]
                candidate["best_rank"] = min(candidate["best_rank"], rank)
                if variant["label"] not in candidate["matched_queries"]:
                    candidate["matched_queries"].append(variant["label"])

        if aggregated:
            ranked_candidates = sorted(
                aggregated.values(),
                key=lambda candidate: candidate["retrieval_score"],
                reverse=True,
            )
            return ranked_candidates[:limit]

        # 混合检索没有结果时，回退到原始关键词检索
        fallback_items = content_db.search_content(f"{user_input} {current_emotion}", limit=limit)
        return [
            {
                "item": item,
                "retrieval_score": 0.2,
                "matched_queries": ["fallback_keyword"],
                "retrieval_sources": ["keyword_fallback"],
                "best_rank": index + 1,
            }
            for index, item in enumerate(fallback_items)
        ]

    def _rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        user_input: str,
        current_emotion: str,
        conversation_summary: Dict[str, Any],
        user_profile: Dict[str, Any],
        limit: int,
    ) -> List[ContentItem]:
        """对召回候选做规则个性化重排。

        各维度子分均为 0-1 归一化值，通过加权公式合成 final_score。
        """
        scored_items = []
        keywords = self._extract_keywords(user_input)

        recent_risk_levels = conversation_summary.get("recent_risk_levels", []) or []
        current_risk_level = (
            recent_risk_levels[-1]
            if recent_risk_levels
            else user_profile.get("risk_level", "low")
        )
        current_risk_band = risk_level_band(current_risk_level)
        recent_recommendation_turns = conversation_summary.get("recent_recommendation_turns", []) or []
        recent_item_ids = conversation_summary.get("recent_recommendation_item_ids", []) or []

        for candidate in candidates:
            item = candidate["item"]
            if not self._is_item_allowed_for_risk(item, current_risk_level):
                continue

            # ---- 归一化子分 ----
            retrieval_score = self._normalize_retrieval_score(candidate.get("retrieval_score", 0.0))
            emotion_match_score = self._calc_emotion_match(item, current_emotion)
            keyword_match_score = self._calc_keyword_match(item, keywords)
            concern_match_score = self._calc_concern_match(item, conversation_summary)
            stage_match_score = self._calc_stage_match(item, conversation_summary)
            risk_match_score = self._calc_risk_match(item, current_risk_band)
            repetition_penalty = 0.4 if recent_recommendation_turns else 0.0

            # 画像偏好（_preference_match_score 0-1 + profile_boost 0-0.4 + duration_match -0.05-0.1，上限 1.0）
            preference_score = min(
                self._preference_match_score(item, user_profile)
                + self._calc_profile_boost(item, user_profile)
                + self._calc_duration_match(item, user_profile),
                1.0,
            )

            # 内容去重：过去 N 轮推荐过同样内容 => 额外惩罚
            dedup_penalty = 0.6 if item.id in recent_item_ids else 0.0

            # ---- 加权合成 ----
            final_score = (
                0.25 * retrieval_score
                + 0.18 * emotion_match_score
                + 0.12 * keyword_match_score
                + 0.10 * concern_match_score
                + 0.07 * stage_match_score
                + 0.10 * risk_match_score
                + 0.10 * preference_score
                + 0.10 * item.actionability
                - 0.05 * repetition_penalty
                - 0.03 * dedup_penalty
                + 0.02 * min(item.popularity * 0.01, 0.1)
                + 0.02 * item.priority
            )

            if final_score > 0:
                enriched_item = item.model_copy(
                    update={
                        "retrieval_metadata": {
                            "retrieval_score": round(candidate.get("retrieval_score", 0.0), 4),
                            "final_score": round(final_score, 4),
                            "matched_queries": candidate.get("matched_queries", []),
                            "retrieval_sources": candidate.get("retrieval_sources", []),
                            "best_rank": candidate.get("best_rank"),
                        }
                    }
                )
                scored_items.append((final_score, enriched_item))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_items[:limit]]

    # ---- 归一化子分辅助方法 ----

    def _calc_emotion_match(self, item: ContentItem, current_emotion: str) -> float:
        """情绪匹配子分：精确匹配 1.0，权重映射匹配 0.8，无匹配 0.0"""
        if current_emotion in item.emotion_tags:
            return 1.0
        for et in item.emotion_tags:
            if et in self.emotion_weights and current_emotion in self.emotion_weights[et]:
                return 0.8
        return 0.0

    def _calc_keyword_match(self, item: ContentItem, keywords: List[str]) -> float:
        """关键词匹配子分：每命中关键词 +0.33，上限 1.0"""
        if not keywords:
            return 0.0
        matches = sum(
            1 for kw in keywords
            if kw in item.title.lower() or kw in " ".join(item.tags).lower()
        )
        return min(matches * 0.33, 1.0)

    def _calc_concern_match(self, item: ContentItem, conversation_summary: Dict[str, Any]) -> float:
        """关切点匹配子分：任意关切点命中 tag/category 得 1.0"""
        concerns = conversation_summary.get("key_concerns", [])
        for concern in concerns:
            if concern in item.tags or concern in item.category:
                return 1.0
        return 0.0

    def _calc_stage_match(self, item: ContentItem, conversation_summary: Dict[str, Any]) -> float:
        """对话阶段匹配子分：难度与阶段适配得 1.0"""
        stage = conversation_summary.get("conversation_stage", "initial")
        depth = self.stage_depth_mapping.get(stage, "beginner")
        if isinstance(depth, list):
            return 1.0 if item.difficulty in depth else 0.0
        return 1.0 if item.difficulty == depth else 0.0

    def _calc_risk_match(self, item: ContentItem, current_risk_band: str) -> float:
        """风险匹配子分：适配得 1.0，禁忌得 -1.0，默认 0.2"""
        if current_risk_band in item.risk_levels:
            return 1.0
        if current_risk_band == "high" and "high_risk_crisis" in item.contraindications:
            return -1.0
        return 0.2

    def _calc_profile_boost(self, item: ContentItem, user_profile: Dict[str, Any]) -> float:
        """画像额外加分（类型/类别/难度偏好），上限 0.4"""
        boost = 0.0
        preferred_types = user_profile.get("preferred_types", []) or []
        preferred_categories = user_profile.get("preferred_categories", []) or []
        preferred_difficulty = user_profile.get("preferred_difficulty", "beginner")

        if preferred_types and item.type in preferred_types:
            boost += 0.15
        if preferred_categories and item.category in preferred_categories:
            boost += 0.15
        if item.difficulty and item.difficulty == preferred_difficulty:
            boost += 0.10

        risk_band = risk_level_band(user_profile.get("risk_level", "low"))
        if risk_band == "high" and item.difficulty == "advanced":
            boost -= 0.10

        return max(boost, 0.0)

    def _is_item_allowed_for_risk(self, item: ContentItem, risk_level: str) -> bool:
        canonical_level = normalize_risk_level(risk_level)
        risk_band = risk_level_band(canonical_level)
        if canonical_level == LEVEL_3:
            return False
        if canonical_level == LEVEL_2:
            return (
                item.recommend_type == "soft"
                and (item.difficulty in {None, "beginner"})
                and item.actionability >= 0.5
                and (
                    item.duration_minutes is None
                    or item.duration_minutes <= 10
                )
            )
        return True

    def _build_query_variants(
        self,
        user_input: str,
        current_emotion: str,
        conversation_summary: Dict[str, Any],
        user_profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        key_concerns = conversation_summary.get("key_concerns", []) or []
        stress_sources = conversation_summary.get("stress_sources", []) or []
        recent_intents = conversation_summary.get("recent_intents", []) or []
        main_sources = user_profile.get("main_stress_sources", []) or []

        variants = [
            {
                "label": "raw_input",
                "query": user_input,
                "extra_terms": [current_emotion],
                "weight": 1.0,
            }
        ]
        if current_emotion:
            variants.append(
                {
                    "label": "emotion_enhanced",
                    "query": f"{current_emotion} {' '.join(key_concerns)} 缓解方法",
                    "extra_terms": [current_emotion] + key_concerns,
                    "weight": 0.8,
                }
            )
        if stress_sources:
            variants.append(
                {
                    "label": "stress_source",
                    "query": f"{' '.join(stress_sources)} {' '.join(key_concerns)} 行动建议",
                    "extra_terms": stress_sources + key_concerns,
                    "weight": 0.7,
                }
            )
        if recent_intents:
            intent_text = recent_intents[-1]
            variants.append(
                {
                    "label": "intent_enhanced",
                    "query": f"{intent_text} 具体方法 可执行建议",
                    "extra_terms": [intent_text],
                    "weight": 0.65,
                }
            )
        if main_sources:
            variants.append(
                {
                    "label": "long_memory",
                    "query": f"{' '.join(main_sources[:2])} 偏好支持",
                    "extra_terms": main_sources[:2],
                    "weight": 0.55,
                }
            )
        return variants

    def _normalize_retrieval_score(self, score: float) -> float:
        return max(0.0, min(score * 5, 1.0))

    def _preference_match_score(self, item: ContentItem, user_profile: Dict[str, Any]) -> float:
        score = 0.0
        preferred_types = user_profile.get("preferred_types", []) or []
        preferred_categories = user_profile.get("preferred_categories", []) or []
        preferred_difficulty = user_profile.get("preferred_difficulty", "beginner")

        if preferred_types and item.type in preferred_types:
            score += 0.45
        if preferred_categories and item.category in preferred_categories:
            score += 0.35
        if item.difficulty and item.difficulty == preferred_difficulty:
            score += 0.2
        return min(score, 1.0)
    
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
        """生成推荐理由（优先引用 retrieval_metadata.final_score）"""
        if not recommended_items:
            return "暂时没有找到特别匹配的内容。"

        stage = conversation_summary.get('conversation_stage', 'initial')

        rationale_templates = {
            "initial": "根据你提到的内容，这些资源可能对你有帮助：",
            "exploring": "在探索阶段，这些内容可以帮助你更深入地理解自己：",
            "deepening": "这些专业资源可以帮助你进一步分析问题：",
            "resolving": "这些实用工具和策略可以帮助你采取行动："
        }

        base_rationale = rationale_templates.get(stage, "根据你的情况推荐以下内容：")

        # 收集具体理由（基于分数 + 情绪/关切点匹配）
        specific_reasons = []
        for item in recommended_items[:2]:
            final_score = None
            if item.retrieval_metadata and "final_score" in item.retrieval_metadata:
                final_score = item.retrieval_metadata["final_score"]

            # 优先用分数说明匹配度
            if final_score is not None and final_score >= 0.3:
                score_pct = min(int(final_score * 100), 99)
                specific_reasons.append(f"《{item.title}》匹配度达 {score_pct}%")
            elif current_emotion in item.emotion_tags:
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
        """计算匹配度分数（优先使用 retrieval_metadata.final_score）"""
        scores = {}

        for item in recommended_items:
            # 优先使用 _rank_candidates 产出的 final_score
            final_score = item.retrieval_metadata.get("final_score") if item.retrieval_metadata else None
            if final_score is not None:
                # 归一化到 0-1 范围显示
                scores[item.id] = min(max(final_score, 0.0), 1.0)
                continue

            # 兜底：对于没有 metadata 的 item 做轻量重算
            item_score = 0.0
            if current_emotion in item.emotion_tags:
                item_score += 0.4
            key_concerns = conversation_summary.get('key_concerns', [])
            for concern in key_concerns:
                if concern in item.tags or concern in item.category:
                    item_score += 0.3
                    break
            stage = conversation_summary.get('conversation_stage', 'initial')
            depth = self.stage_depth_mapping.get(stage, 'beginner')
            if item.difficulty == depth or (isinstance(depth, list) and item.difficulty in depth):
                item_score += 0.2
            item_score += min(item.popularity * 0.01, 0.1)
            scores[item.id] = min(item_score, 1.0)

        return scores

    def _normalize_user_profile(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """确保用户画像结构完整，提供合理默认值"""
        profile = dict(user_profile or {})
        profile.setdefault("risk_level", "low")
        profile.setdefault("preferred_types", [])
        profile.setdefault("preferred_categories", [])
        profile.setdefault("preferred_difficulty", "beginner")
        # preferred_duration_range 可以为空，表示不做时长强约束
        if "preferred_duration_range" not in profile:
            profile["preferred_duration_range"] = None
        return profile

    def _calc_duration_match(self, item: ContentItem, user_profile: Dict[str, Any]) -> float:
        """时长偏好匹配：范围内 +0.1，范围外 -0.05，无配置 0"""
        duration_range = user_profile.get("preferred_duration_range")
        if duration_range and item.duration_minutes is not None:
            try:
                min_dur = duration_range.get("min")
                max_dur = duration_range.get("max")
                if min_dur is not None and max_dur is not None and min_dur <= item.duration_minutes <= max_dur:
                    return 0.1
                if (min_dur is not None and item.duration_minutes < min_dur) or \
                   (max_dur is not None and item.duration_minutes > max_dur):
                    return -0.05
            except Exception:
                pass
        return 0.0

# 全局推荐器实例
content_recommender = ContentRecommender()
