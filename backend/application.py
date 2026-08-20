"""FastAPI application composition and process lifecycle management.

Keeping the composition root here makes importing domain modules side-effect free and
gives tests a single place to control infrastructure such as schedulers.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import APIRouter, FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from config import config
from conversation_manager import ConversationManager, conversation_manager
from error_handler import setup_error_handlers
from middleware import setup_middlewares

logger = logging.getLogger(__name__)


def _run_session_cleanup(manager: ConversationManager, days: int) -> None:
    """Bridge APScheduler's synchronous worker to the async persistence API."""
    try:
        deleted_count = asyncio.run(manager.cleanup_expired_sessions_async(days))
        if deleted_count:
            logger.info("定时清理任务：删除了 %s 个过期会话", deleted_count)
    except Exception:
        logger.exception("清理过期会话失败")


def _build_lifespan(
    manager: ConversationManager,
    scheduler_factory: Callable[[], BackgroundScheduler],
):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("应用启动中...")
        scheduler: Optional[BackgroundScheduler] = None
        try:
            import aiosqlite  # noqa: F401 - startup dependency check

            logger.info("依赖检查通过: aiosqlite 可用")
        except Exception as exc:
            logger.warning(
                "依赖检查: aiosqlite 不可用，建议安装 pip install aiosqlite。错误: %s",
                exc,
            )

        try:
            scheduler = scheduler_factory()
            scheduler.add_job(
                func=_run_session_cleanup,
                args=[manager, config.SESSION_CLEANUP_DAYS],
                trigger="cron",
                hour=config.SESSION_CLEANUP_HOUR,
                minute=config.SESSION_CLEANUP_MINUTE,
                id="cleanup_sessions",
                replace_existing=True,
            )
            scheduler.start()
            app.state.scheduler = scheduler
            logger.info(
                "定时任务已启动：每天 %02d:%02d 清理 %s 天前的会话",
                config.SESSION_CLEANUP_HOUR,
                config.SESSION_CLEANUP_MINUTE,
                config.SESSION_CLEANUP_DAYS,
            )
            logger.info("BERT 风险预测模型将在首次请求时自动加载")
            yield
        finally:
            # Persistence tasks belong to the application lifecycle. Draining them
            # prevents SQLite writes from leaking into a closing event loop.
            await manager.wait_for_persistence()
            if scheduler is not None and scheduler.running:
                scheduler.shutdown(wait=True)
            logger.info("应用已关闭")

    return lifespan


def create_app(
    router: APIRouter,
    *,
    manager: ConversationManager = conversation_manager,
    scheduler_factory: Callable[[], BackgroundScheduler] = BackgroundScheduler,
) -> FastAPI:
    """Create a fully configured application with injectable infrastructure."""
    app = FastAPI(
        title="MindPal Pro Backend",
        version="3.2",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_build_lifespan(manager, scheduler_factory),
    )
    setup_middlewares(app)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.include_router(router)
    setup_error_handlers(app)
    return app
