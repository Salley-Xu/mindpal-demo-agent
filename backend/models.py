#models.py
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

class TextInput(BaseModel):
    text: str
    user_id: str
    session_id: Optional[str] = None

class EmotionResponse(BaseModel):
    text: str
    emotion: str
    confidence: float
    context_emotion: Optional[str] = None  # 基于上下文的情绪
    trend: Optional[str] = None  # 情绪趋势
    urgent_issue: Optional[Dict[str, Any]] = None # 紧急问题识别结果

class ChatRequest(BaseModel):
    text: str
    user_id: str
    session_id: str
    context_summary: Optional[Dict[str, Any]] = None


class EmotionState(BaseModel):
    current_emotion: str
    emotion_type: str
    context_emotion: Optional[str] = None
    emotion_intensity: float = 0.5
    stress_source: Optional[str] = None
    user_intent: str = "sharing"
    negative_trend: bool = False
    confidence: float = 0.5
    emotion_trend: Optional[str] = None


class RiskState(BaseModel):
    level: str
    message: str = ""
    suggestions: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    risk_score: float = 0.0


class SessionSummary(BaseModel):
    conversation_stage: str = "initial"
    key_concerns: List[str] = Field(default_factory=list)
    turn_count: int = 0
    emotion_trend: Optional[str] = None
    primary_emotion: Optional[str] = None


class RecommendationDecision(BaseModel):
    should_recommend: bool = False
    recommend_type: str = "none"
    score: float = 0.0
    threshold: float = 0.0
    reason_codes: List[str] = Field(default_factory=list)
    cooldown_remaining: int = 0

class ContentItem(BaseModel):
    """内容项模型"""
    id: str
    title: str
    type: str  # article, audio, video, exercise, tool
    category: str  # 分类：stress, anxiety, relationship, academic, etc.
    description: str
    content: Optional[str] = None
    url: Optional[str] = None
    duration_minutes: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    emotion_tags: List[str] = Field(default_factory=list)  # 适合的情绪
    risk_levels: List[str] = Field(default_factory=lambda: ["low", "medium"])
    contraindications: List[str] = Field(default_factory=list)
    recommend_type: str = "soft"
    priority: float = 0.5
    source: Optional[str] = None
    actionability: float = 0.5
    retrieval_metadata: Optional[Dict[str, Any]] = None
    difficulty: Optional[str] = None  # beginner, intermediate, advanced
    created_at: datetime = datetime.now()
    popularity: int = 0  # 热度


class ContentRecommendRequest(BaseModel):
    user_input: str
    current_emotion: str
    conversation_stage: str
    key_concerns: Optional[Union[List[str], str]] = None
    user_id: Optional[str] = None
    limit: int = 3


class ContentRecommendResponse(BaseModel):
    recommendations: List[ContentItem] = Field(default_factory=list)
    rationale: str
    match_scores: Dict[str, float] = Field(default_factory=dict)


class RecommendationFeedbackRequest(BaseModel):
    user_id: str
    session_id: str
    content_id: str
    feedback: str


class RecommendationFeedbackResponse(BaseModel):
    message: str
    content_id: str
    feedback: str


class RecommendationRequest(BaseModel):
    """推荐请求"""
    user_input: str
    current_emotion: str
    conversation_summary: Dict[str, Any]
    content_types: Optional[List[str]] = None  # 指定类型
    limit: int = 3


class RecommendationResponse(BaseModel):
    """推荐响应"""
    recommendations: List[ContentItem]
    rationale: str  # 推荐理由
    match_scores: Dict[str, float]  # 匹配度分数


class ChatResponse(BaseModel):
    response: str
    emotion_state: Optional[EmotionState] = None
    risk_state: Optional[RiskState] = None
    session_summary: Optional[SessionSummary] = None
    recommendation_decision: Optional[RecommendationDecision] = None
    emotion_summary: Optional[Dict[str, Any]] = None  # 兼容旧前端读取逻辑
    urgent_issue: Optional[Dict[str, Any]] = None  # 兼容旧前端读取逻辑
    recommendations: Optional[List[ContentItem]] = None  # 新增：推荐内容
    recommendation_rationale: Optional[str] = None  # 新增：推荐理由


class AgentToolCall(BaseModel):
    """单次工具调用的输入输出摘要"""
    name: str
    input: Dict[str, Any]
    output_summary: Dict[str, Any]
    success: bool = True
    error_message: Optional[str] = None


class AgentStep(BaseModel):
    """Agent 执行过程中的一个阶段"""
    name: str
    description: Optional[str] = None
    started_at: datetime
    finished_at: datetime
    tool_calls: List[AgentToolCall] = Field(default_factory=list)


class AgentRunRequest(BaseModel):
    """Agent 统一入口请求模型"""
    text: str
    user_id: str
    session_id: str
    return_steps: bool = False  # 是否返回步骤与工具调用摘要


class AgentRunResponse(BaseModel):
    """Agent 统一入口响应模型"""
    run_id: str
    chat: ChatResponse
    steps: Optional[List[AgentStep]] = None
