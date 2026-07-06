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
    level: str = "level_0"
    legacy_level: str = "low"
    level_index: int = 0
    level_label: str = "Level 0"
    message: str = ""
    suggestions: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    risk_score: float = 0.0
    raw_score: float = 0.0
    risk_dimensions: Dict[str, int] = Field(default_factory=dict)
    risk_evidence: Dict[str, Any] = Field(default_factory=dict)
    escalation_reasons: List[str] = Field(default_factory=list)
    risk_context: Dict[str, Any] = Field(default_factory=dict)


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


# ============================================================
# Memory System v2.0 数据模型
# ============================================================

class MemoryItem(BaseModel):
    """统一长期记忆条目"""
    id: str = ""
    user_id: str
    session_id: Optional[str] = None
    turn_id: Optional[str] = None
    memory_type: str = "mood_event"  # preference / avoidance / stress_source / mood_event / risk_event / coping_strategy / recommendation_feedback / conversation_summary / personal_fact
    content: str
    summary: Optional[str] = None
    source_text: Optional[str] = None
    emotion: Optional[str] = None
    emotion_intensity: Optional[float] = None
    risk_level: Optional[str] = None
    stress_source: Optional[str] = None
    user_intent: Optional[str] = None
    importance: float = 0.5
    confidence: float = 0.5
    sensitivity: str = "normal"  # normal / sensitive / high
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = "inferred"  # explicit / inferred / risk_detector / tool / recommender
    scope: str = "long_term"  # current_turn / session / long_term
    status: str = "active"  # active / archived / decayed / deleted
    access_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_accessed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class MemoryCandidate(BaseModel):
    """写入 pipeline 候选记忆"""
    user_id: str
    session_id: Optional[str] = None
    turn_id: Optional[str] = None
    memory_type: str
    content: str
    summary: Optional[str] = None
    source_text: Optional[str] = None
    emotion: Optional[str] = None
    emotion_intensity: Optional[float] = None
    risk_level: Optional[str] = None
    stress_source: Optional[str] = None
    user_intent: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = "inferred"


class MemoryQuery(BaseModel):
    """记忆检索查询"""
    text: str = ""
    emotion: Optional[str] = None
    risk_level: Optional[str] = None
    stress_source: Optional[str] = None
    intent: Optional[str] = None
    memory_types: Optional[List[str]] = None
    need_risk_context: bool = False
    need_preference: bool = True
    top_k: int = 10


class MemorySearchResult(BaseModel):
    """记忆检索结果"""
    item: MemoryItem
    score: float = 0.0
    rank: int = 0
    retrieval_method: str = ""  # profile / lexical / semantic / risk


class TurnContext(BaseModel):
    """单轮对话上下文（用于写 pipeline）"""
    user_id: str
    session_id: str
    turn_id: Optional[str] = None
    user_input: str
    ai_response: Optional[str] = None
    emotion_state: Optional[Dict[str, Any]] = None
    risk_state: Optional[Dict[str, Any]] = None
    conversation_summary: Optional[Dict[str, Any]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


class InjectedMemoryContext(BaseModel):
    """记忆注入结果"""
    text: str = ""
    used_tokens: int = 0
    included_memory_ids: List[str] = Field(default_factory=list)
    dropped_memory_ids: List[str] = Field(default_factory=list)
    drop_reasons: Dict[str, str] = Field(default_factory=dict)


# ============================================================
# Risk Memory v2.0 数据模型
# ============================================================

class RiskEvent(BaseModel):
    """风险事件（append-only 存储）"""
    id: str = ""
    user_id: str
    session_id: Optional[str] = None
    turn_id: Optional[str] = None
    event_time: datetime = Field(default_factory=datetime.now)
    risk_level: str = "level_0"
    risk_score: Optional[float] = None
    subject: str = "self"
    topic: Optional[str] = None
    summary: str = ""
    evidence_snippet: Optional[str] = None
    safety_confirmed: bool = False
    support_engaged: bool = False
    decayed: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class RiskBaseline(BaseModel):
    """用户风险基线"""
    user_id: str
    baseline: str = "low"  # low / medium / high
    baseline_score: Optional[float] = None
    baseline_updated_at: datetime = Field(default_factory=datetime.now)
    decay_status: str = "active"
    last_high_risk_time: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskTrigger(BaseModel):
    """风险触发因素"""
    id: str = ""
    user_id: str
    trigger: str
    frequency: int = 1
    last_seen_at: Optional[datetime] = None
    decayed: bool = False


class ProtectiveFactor(BaseModel):
    """保护因素"""
    id: str = ""
    user_id: str
    factor: str
    source: Optional[str] = None
    confidence: float = 0.5
    last_seen_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
