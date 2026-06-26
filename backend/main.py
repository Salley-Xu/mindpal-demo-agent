# main.py
import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from api_endpoints import router
from conversation_manager import conversation_manager
from config import config
from error_handler import setup_error_handlers

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ 初始化FastAPI ------------------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用生命周期"""
    # 启动时执行
    logger.info("应用启动中...")
    
    try:
        import aiosqlite  # 依赖检查
        logger.info("依赖检查通过: aiosqlite 可用")
    except Exception as e:
        logger.warning(f"依赖检查: aiosqlite 不可用，建议安装 pip install aiosqlite。错误: {e}")
    
    # 启动定时任务（每天凌晨3点执行）
    scheduler.add_job(
        func=cleanup_expired_sessions,
        trigger="cron",
        hour=3,
        minute=0,
        id="cleanup_sessions"
    )
    scheduler.start()
    logger.info(f"定时任务已启动：每天凌晨3点清理 {config.SESSION_CLEANUP_DAYS} 天前的会话")
    
    yield
    
    # 关闭时执行
    logger.info("应用关闭中...")
    scheduler.shutdown()
    logger.info("定时任务已停止")

app = FastAPI(
    title="MindPal Pro Backend", 
    version="3.2",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加响应压缩中间件
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,  # 最小压缩大小，小于此值的响应不压缩
)

# 注册所有路由
app.include_router(router)

# 设置错误处理器
setup_error_handlers(app)

# ------------------ 定时任务：清理过期会话 ------------------
def cleanup_expired_sessions():
    """清理过期会话的定时任务"""
    try:
        import asyncio
        deleted_count = asyncio.run(conversation_manager.cleanup_expired_sessions_async(config.SESSION_CLEANUP_DAYS))
        if deleted_count > 0:
            logger.info(f"定时清理任务：删除了 {deleted_count} 个过期会话")
    except Exception as e:
        logger.error(f"清理过期会话失败: {e}")

# 创建后台调度器
scheduler = BackgroundScheduler()


# ------------------ 启动入口 ------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MindPal Pro 后端服务 v3.2 启动中...")
    print("✨ 功能：上下文感知对话系统 + 个性化推荐")
    print("🔗 API地址: http://localhost:8000")
    print("📝 接口文档: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
