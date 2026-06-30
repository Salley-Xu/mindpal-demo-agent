# api_endpoints.py - 完整路由版本
from fastapi import APIRouter, HTTPException
from typing import Optional
import logging
import time

from models import (
    TextInput,
    EmotionResponse,
    ChatRequest,
    ChatResponse,
    AgentRunRequest,
    AgentRunResponse,
    ContentRecommendRequest,
    ContentRecommendResponse,
    RecommendationFeedbackRequest,
    RecommendationFeedbackResponse,
)
from conversation_manager import conversation_manager
from emotion_analyzer import emotion_analyzer
from urgent_detector import urgent_detector, urgent_logger
from content_recommender import content_recommender
from content_db import content_db
from utils import validate_user_input
from agent_orchestrator import agent_orchestrator
from agent_tools import UserProfileTool
from risk_levels import is_non_low_risk

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

# ==================== 首页和健康检查 ====================
@router.get("/")
async def root():
    """首页"""
    from database import adb_manager
    
    # 获取数据库统计
    db_stats = await adb_manager.get_session_statistics()
    
    return {
        "message": "MindPal Pro 后端服务运行中",
        "version": "3.2",
        "feature": "上下文感知对话系统 + 个性化推荐 + 会话持久化",
        "active_sessions": len(conversation_manager.sessions),
        "database": {
            "enabled": True,
            "total_sessions": db_stats.get('total_sessions', 0),
            "total_users": db_stats.get('total_users', 0)
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "emotion_analysis": "/emotion/analyze",
            "chat": "/chat/intelligent",
            "content_recommend": "/content/recommend",
            "session_list": "/session/{user_id}/list",
            "session_history": "/session/{user_id}/{session_id}/history",
            "session_stats": "/session/statistics"
        }
    }

@router.get("/health")
async def health_check():
    """健康检查"""
    from database import adb_manager
    stats = await adb_manager.get_session_statistics()
    return {
        "status": "ok",
        "model": "deepseek-chat",
        "conversation_manager": "active",
        "session_count": len(conversation_manager.sessions),
        "total_sessions": stats.get("total_sessions", 0),
        "total_users": stats.get("total_users", 0),
        "content_items": len(content_db.content_items),
        "timestamp": time.time()
    }

# ==================== 情绪分析API ====================
@router.post("/emotion/analyze", response_model=EmotionResponse)
async def analyze_emotion(input_data: TextInput):
    """情绪分析API"""
    if not validate_user_input(input_data.text):
        raise HTTPException(status_code=400, detail="输入文本无效")
    
    logger.info(f"情绪分析请求: user_id={input_data.user_id}")
    
    # 获取对话摘要
    conversation_summary = None
    if input_data.session_id:
        conversation_summary = await conversation_manager.get_conversation_summary_async(
            input_data.user_id, input_data.session_id
        )
    
    # 分析情绪
    current_emotion, context_emotion, confidence = await emotion_analyzer.analyze_with_context_async(
        input_data.text, conversation_summary
    )
    
    # 检测紧急情况
    urgent_issue = urgent_detector.detect(input_data.text, current_emotion)
    
    # 分析趋势
    trend = "new"
    if conversation_summary and conversation_summary.get('recent_emotions'):
        recent = conversation_summary['recent_emotions']
        if current_emotion in recent:
            trend = "consistent"
        elif current_emotion in ['焦虑', '压力', '愤怒'] and '平静' in recent:
            trend = "escalating"
        elif current_emotion in ['平静', '中性', '快乐'] and '焦虑' in recent:
            trend = "calming"
    
    # 记录紧急情况
    if is_non_low_risk(urgent_issue.get("level")):
        logger.warning(f"紧急情况: {urgent_issue['level']}, 用户: {input_data.user_id}")
    
    return EmotionResponse(
        text=input_data.text,
        emotion=current_emotion,
        confidence=confidence,
        context_emotion=context_emotion,
        trend=trend,
        urgent_issue=urgent_issue
    )

# ==================== 智能对话API ====================
@router.post("/chat/intelligent", response_model=ChatResponse)
async def intelligent_chat(chat_request: ChatRequest):
    """智能对话API(基于 Agent 编排器的轻量封装)"""
    if not validate_user_input(chat_request.text):
        raise HTTPException(status_code=400, detail="输入文本无效")

    logger.info(f"智能对话请求: user_id={chat_request.user_id}, session_id={chat_request.session_id}")

    try:
        agent_request = AgentRunRequest(
            text=chat_request.text,
            user_id=chat_request.user_id,
            session_id=chat_request.session_id,
            return_steps=False,
        )
        agent_response = await agent_orchestrator.run_agent(agent_request)
        return agent_response.chat
    except Exception as e:
        logger.error(f"智能对话处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.post("/agent/run", response_model=AgentRunResponse)
async def agent_run(request: AgentRunRequest):
    """统一 Agent 入口：返回对话回复 + 可选步骤与工具调用摘要"""
    if not validate_user_input(request.text):
        raise HTTPException(status_code=400, detail="输入文本无效")

    try:
        return await agent_orchestrator.run_agent(request)
    except Exception as e:
        logger.error(f"/agent/run 处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# ==================== 会话管理API ====================
@router.get("/session/{user_id}/{session_id}/summary")
async def get_session_summary(user_id: str, session_id: str):
    """获取会话摘要"""
    session_exists = await conversation_manager.session_exists_async(user_id, session_id)
    if not session_exists:
        raise HTTPException(status_code=404, detail="会话不存在")

    summary = await conversation_manager.get_conversation_summary_async(user_id, session_id)
    session = await conversation_manager.get_or_create_session_async(user_id, session_id)
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "summary": summary,
        "recent_history": session['history'][-5:] if session['history'] else [],
        "active": True
    }

@router.get("/session/{user_id}/{session_id}/history")
async def get_session_history(user_id: str, session_id: str, limit: int = 50):
    """获取会话完整历史记录"""
    session = await conversation_manager.get_or_create_session_async(user_id, session_id)
    
    if not session.get('history'):
        raise HTTPException(status_code=404, detail="会话不存在或无历史记录")
    
    history = session['history'][-limit:] if limit else session['history']
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "total_turns": len(session['history']),
        "history": history
    }

@router.get("/session/{user_id}/list")
async def get_user_sessions(user_id: str, limit: int = 10):
    """获取用户的所有会话列表"""
    from database import adb_manager
    sessions = await adb_manager.get_user_sessions(user_id, limit)
    
    return {
        "user_id": user_id,
        "total_sessions": len(sessions),
        "sessions": sessions
    }

@router.delete("/session/{user_id}/{session_id}")
async def clear_session(user_id: str, session_id: str):
    """清除会话"""
    success = await conversation_manager.delete_session_async(user_id, session_id)
    
    if success:
        logger.info(f"会话已清除: {user_id}_{session_id}")
        return {"message": "会话已清除"}
    else:
        raise HTTPException(status_code=404, detail="会话不存在")

@router.post("/session/cleanup")
async def cleanup_sessions(days: int = 30):
    """清理过期会话（管理员接口）"""
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="天数必须在1-365之间")
    
    deleted_count = await conversation_manager.cleanup_expired_sessions_async(days)
    
    return {
        "message": f"已清理 {deleted_count} 个过期会话",
        "days": days,
        "deleted_count": deleted_count
    }

@router.get("/session/statistics")
async def get_session_statistics(user_id: Optional[str] = None):
    """获取会话统计信息"""
    from database import adb_manager
    stats = await adb_manager.get_session_statistics(user_id)
    
    return {
        "statistics": stats,
        "user_id": user_id
    }

# ==================== 紧急情况管理API ====================
@router.get("/urgent/cases")
async def get_recent_urgent_cases(days: int = 1, level: Optional[str] = None):
    """获取最近的紧急情况记录"""
    if days > 30:  # 限制查询天数
        days = 30
    
    result = await urgent_logger.get_recent_cases_async(days, level)
    return result

@router.get("/resources/emergency")
async def get_emergency_resources():
    """获取紧急求助资源"""
    return {
        'resources': [
            {
                'name': '全国心理援助热线',
                'phone': '400-161-9995',
                'hours': '24小时',
                'description': '专业的心理援助服务'
            },
            {
                'name': '北京心理援助热线',
                'phone': '010-82951332',
                'hours': '24小时',
                'description': '北京市心理援助热线'
            },
            {
                'name': '简单心理',
                'url': 'https://www.jiandanxinli.com',
                'description': '在线心理咨询平台'
            }
        ],
        'tips': [
            '你不是一个人，很多人愿意帮助你',
            '寻求帮助是勇敢和明智的选择',
            '紧急情况下请立即联系专业人员',
            '你的感受是重要的，值得被倾听'
        ]
    }

# ==================== 内容推荐API ====================
@router.post("/content/recommend", response_model=ContentRecommendResponse)
async def recommend_content(request: ContentRecommendRequest):
    """个性化内容推荐API"""
    try:
        key_concerns = request.key_concerns
        if isinstance(key_concerns, str):
            key_concerns_list = [item.strip() for item in key_concerns.split(",") if item.strip()]
        else:
            key_concerns_list = key_concerns or []

        # 构建对话摘要
        conversation_summary = {
            'conversation_stage': request.conversation_stage,
            'key_concerns': key_concerns_list,
            'turn_count': 1,
            'recent_emotions': [request.current_emotion]
        }

        # 可选：根据 user_id 加载用户画像，用于个性化推荐
        user_profile_context = {}
        if request.user_id:
            try:
                profile = await UserProfileTool.get_profile(request.user_id)
                if profile:
                    user_profile_context = {
                        "user_id": profile.user_id,
                        "risk_level": profile.risk_level,
                        "preferred_types": profile.preferred_types,
                        "preferred_categories": profile.preferred_categories,
                        "preferred_difficulty": profile.preferred_difficulty,
                        "preferred_duration_range": profile.preferred_duration_range,
                        "preferred_support_style": profile.preferred_support_style,
                        "avoid_style": profile.avoid_style,
                        "main_stress_sources": profile.main_stress_sources,
                        "recommendation_feedback": profile.recommendation_feedback,
                        "last_updated": profile.last_updated.isoformat(),
                    }
            except Exception as e:
                logger.error(f"/content/recommend 加载用户画像失败: {e}", exc_info=True)
        
        recommendations, rationale, match_scores = await content_recommender.recommend_content(
            user_input=request.user_input,
            current_emotion=request.current_emotion,
            conversation_summary=conversation_summary,
            user_profile=user_profile_context,
            limit=request.limit
        )
        
        return ContentRecommendResponse(
            recommendations=recommendations,
            rationale=rationale,
            match_scores=match_scores,
        )
        
    except Exception as e:
        logger.error(f"内容推荐API失败: {e}")
        raise HTTPException(status_code=500, detail="内容推荐失败")


@router.post("/content/feedback", response_model=RecommendationFeedbackResponse)
async def submit_recommendation_feedback(request: RecommendationFeedbackRequest):
    """记录用户对推荐内容的反馈"""
    valid_feedback = {"accepted", "preferred", "helpful", "neutral", "rejected", "not_helpful", "avoid"}
    if request.feedback not in valid_feedback:
        raise HTTPException(status_code=400, detail="反馈值无效")

    try:
        profile = await UserProfileTool.get_profile(request.user_id)
        existing_feedback = profile.recommendation_feedback if profile else {}
        updated_feedback = dict(existing_feedback or {})
        updated_feedback[request.content_id] = request.feedback

        await UserProfileTool.upsert_profile(
            user_id=request.user_id,
            recommendation_feedback=updated_feedback,
        )
        conversation_manager.record_recommendation_feedback(
            user_id=request.user_id,
            session_id=request.session_id,
            content_id=request.content_id,
            feedback=request.feedback,
        )
        return RecommendationFeedbackResponse(
            message="推荐反馈已记录",
            content_id=request.content_id,
            feedback=request.feedback,
        )
    except Exception as e:
        logger.error(f"推荐反馈记录失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="推荐反馈记录失败")

@router.get("/content/search")
async def search_content(q: str, limit: int = 10):
    """搜索内容"""
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="搜索关键词太短")
    
    results = content_db.search_content(q, limit)
    return {
        "query": q,
        "results": results,
        "count": len(results)
    }

@router.get("/content/{content_id}")
async def get_content_detail(content_id: str):
    """获取内容详情"""
    content_item = content_db.get_content_by_id(content_id)
    if not content_item:
        raise HTTPException(status_code=404, detail="内容不存在")
    
    # 增加热度
    content_db.increment_popularity(content_id)
    
    return content_item

@router.get("/content/stats")
async def get_content_stats():
    """获取内容统计信息"""
    all_content = content_db.get_all_content()
    
    stats = {
        "total_count": len(all_content),
        "by_type": {},
        "by_category": {},
        "top_popular": []
    }
    
    # 按类型统计
    for item in all_content:
        stats["by_type"][item.type] = stats["by_type"].get(item.type, 0) + 1
        stats["by_category"][item.category] = stats["by_category"].get(item.category, 0) + 1
    
    # 热门内容
    sorted_by_popularity = sorted(all_content, key=lambda x: x.popularity, reverse=True)[:10]
    stats["top_popular"] = [
        {"id": item.id, "title": item.title, "popularity": item.popularity}
        for item in sorted_by_popularity
    ]
    
    return stats
