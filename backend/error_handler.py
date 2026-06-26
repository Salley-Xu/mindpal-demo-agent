import logging
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional

logger = logging.getLogger(__name__)

class AppException(Exception):
    """应用自定义异常"""
    def __init__(self, status_code: int, detail: str, error_code: Optional[str] = None):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        super().__init__(detail)

async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    if isinstance(exc, HTTPException):
        # 处理FastAPI内置HTTP异常
        logger.warning(f"HTTP异常: {exc.detail}, 状态码: {exc.status_code}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTPException",
                "detail": exc.detail,
                "status_code": exc.status_code
            }
        )
    elif isinstance(exc, AppException):
        # 处理应用自定义异常
        logger.warning(f"应用异常: {exc.detail}, 错误码: {exc.error_code}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "AppException",
                "detail": exc.detail,
                "error_code": exc.error_code,
                "status_code": exc.status_code
            }
        )
    else:
        # 处理其他未捕获的异常
        logger.error(f"未捕获的异常: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "detail": "服务器内部错误",
                "status_code": 500
            }
        )

def setup_error_handlers(app):
    """设置错误处理器"""
    app.add_exception_handler(Exception, global_exception_handler)
    logger.info("全局错误处理器已设置")
