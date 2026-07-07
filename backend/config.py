# config.py - 配置文件
import os
from typing import List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """应用配置类"""
    
    # 应用信息
    APP_NAME: str = os.getenv("APP_NAME", "MindPal Pro Backend")
    APP_VERSION: str = os.getenv("APP_VERSION", "3.2")
    
    # API 配置
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "deepseek-chat")
    API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.deepseek.com/v1")
    
    # 服务器配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # 对话配置
    MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "20"))
    SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT", "30"))
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "2400"))
    CONTEXT_SOFT_THRESHOLD: float = float(os.getenv("CONTEXT_SOFT_THRESHOLD", "0.65"))
    CONTEXT_HARD_THRESHOLD: float = float(os.getenv("CONTEXT_HARD_THRESHOLD", "0.85"))
    CONTEXT_MIN_RECENT_TURNS: int = int(os.getenv("CONTEXT_MIN_RECENT_TURNS", "2"))
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    
    # CORS 配置
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    # 内容数据库配置
    CONTENT_DB_FILE: str = os.getenv("CONTENT_DB_FILE", "data/content_db.json")
    
    # 会话持久化配置
    SESSION_PERSISTENCE_ENABLED: bool = os.getenv("SESSION_PERSISTENCE_ENABLED", "true").lower() == "true"
    SESSION_DB_PATH: str = os.getenv("SESSION_DB_PATH", "data/sessions.db")
    SESSION_CLEANUP_DAYS: int = int(os.getenv("SESSION_CLEANUP_DAYS", "30"))

    # API 安全配置（Bearer Token 认证，空字符串=不启用）
    API_AUTH_TOKEN: str = os.getenv("API_AUTH_TOKEN", "")

    # 推荐配置（规则 + 画像 + 可选 AI rerank）
    ENABLE_RECOMMEND_AI_RERANK: bool = os.getenv("ENABLE_RECOMMEND_AI_RERANK", "true").lower() == "true"
    RECOMMEND_RERANK_CANDIDATE_SIZE: int = int(os.getenv("RECOMMEND_RERANK_CANDIDATE_SIZE", "10"))
    # RecommendGate 加权因子与阈值（可通过 .env 覆盖调参）
    RECOMMEND_GATE_EMOTION_WEIGHT: float = float(os.getenv("RECOMMEND_GATE_EMOTION_WEIGHT", "0.30"))
    RECOMMEND_GATE_RISK_WEIGHT: float = float(os.getenv("RECOMMEND_GATE_RISK_WEIGHT", "0.23"))
    RECOMMEND_GATE_INTENT_WEIGHT: float = float(os.getenv("RECOMMEND_GATE_INTENT_WEIGHT", "0.23"))
    RECOMMEND_GATE_TREND_WEIGHT: float = float(os.getenv("RECOMMEND_GATE_TREND_WEIGHT", "0.14"))
    RECOMMEND_GATE_PREFERENCE_WEIGHT: float = float(os.getenv("RECOMMEND_GATE_PREFERENCE_WEIGHT", "0.10"))
    RECOMMEND_GATE_HARD_THRESHOLD: float = float(os.getenv("RECOMMEND_GATE_HARD_THRESHOLD", "0.58"))
    RECOMMEND_GATE_SOFT_THRESHOLD: float = float(os.getenv("RECOMMEND_GATE_SOFT_THRESHOLD", "0.22"))

    # RecommendationTrace 配置
    RECOMMEND_TRACE_ENABLED: bool = os.getenv("RECOMMEND_TRACE_ENABLED", "true").lower() == "true"

    # BERT 风险预测配置（风险评定已完全基于 BERT 模型）
    BERT_MODEL_PATH: str = os.getenv("BERT_MODEL_PATH", "bert_data/models/v4_3_coral/best_model")
    BERT_DEVICE: str = os.getenv("BERT_DEVICE", "cpu")
    BERT_BINARY_THRESHOLD: float = float(os.getenv("BERT_BINARY_THRESHOLD", "0.50"))

    # BERT 情绪分类配置
    EMOTION_MODEL_PATH: str = os.getenv("EMOTION_MODEL_PATH", "bert_data/models/emotion_v1/best_model")
    EMOTION_DEVICE: str = os.getenv("EMOTION_DEVICE", "cpu")
    EMOTION_CONFIDENCE_THRESHOLD: float = float(os.getenv("EMOTION_CONFIDENCE_THRESHOLD", "0.6"))
    # 设为 true 使用 BERT，false 使用 LLM（回退/对比用）
    USE_BERT_EMOTION: bool = os.getenv("USE_BERT_EMOTION", "true").lower() == "true"

    # ============================================================
    # Memory System v2.0 配置
    # ============================================================
    MEMORY_ENABLED: bool = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
    MEMORY_EMBEDDING_ENABLED: bool = os.getenv("MEMORY_EMBEDDING_ENABLED", "false").lower() == "true"
    MEMORY_INJECTION_BUDGET_RATIO: float = float(os.getenv("MEMORY_INJECTION_BUDGET_RATIO", "0.15"))
    MEMORY_MAX_ITEMS_PER_USER: int = int(os.getenv("MEMORY_MAX_ITEMS_PER_USER", "500"))
    MEMORY_RRF_K: int = int(os.getenv("MEMORY_RRF_K", "60"))
    MEMORY_DECAY_DAYS_MOOD_EVENT: int = int(os.getenv("MEMORY_DECAY_DAYS_MOOD_EVENT", "90"))
    MEMORY_DECAY_DAYS_STRESS_SOURCE: int = int(os.getenv("MEMORY_DECAY_DAYS_STRESS_SOURCE", "180"))
    MEMORY_DECAY_DAYS_RISK_EVENT: int = int(os.getenv("MEMORY_DECAY_DAYS_RISK_EVENT", "30"))

    # Risk memory v2.0 基线配置
    RISK_BASELINE_HIGH_WINDOW_DAYS: int = int(os.getenv("RISK_BASELINE_HIGH_WINDOW_DAYS", "7"))
    RISK_BASELINE_MEDIUM_WINDOW_DAYS: int = int(os.getenv("RISK_BASELINE_MEDIUM_WINDOW_DAYS", "14"))
    RISK_EVENT_DECAY_DAYS: int = int(os.getenv("RISK_EVENT_DECAY_DAYS", "30"))

    def validate(self):
        """验证配置"""
        if not self.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
        
        # 确保必要的目录存在
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.CONTENT_DB_FILE), exist_ok=True)
        
        return self

# 创建配置实例
config = Config().validate()
