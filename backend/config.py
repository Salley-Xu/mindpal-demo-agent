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

    # 推荐配置（规则 + 画像 + 可选 AI rerank）
    ENABLE_RECOMMEND_AI_RERANK: bool = os.getenv("ENABLE_RECOMMEND_AI_RERANK", "true").lower() == "true"
    RECOMMEND_RERANK_CANDIDATE_SIZE: int = int(os.getenv("RECOMMEND_RERANK_CANDIDATE_SIZE", "10"))
    
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