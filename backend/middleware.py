# middleware.py - 中间件配置

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from config import config


class AuthMiddleware(BaseHTTPMiddleware):
    """API 认证中间件（Bearer Token）"""

    # 不需要认证的开放路径
    OPEN_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        # 未配置 token 时放行所有请求（开发模式）
        if not config.API_AUTH_TOKEN:
            return await call_next(request)

        # 开放路径免认证
        if request.url.path in self.OPEN_PATHS:
            return await call_next(request)

        # 校验 Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "未提供有效的认证令牌"},
            )

        token = auth_header.removeprefix("Bearer ").strip()
        if token != config.API_AUTH_TOKEN:
            return JSONResponse(
                status_code=401,
                content={"detail": "未提供有效的认证令牌"},
            )

        return await call_next(request)


def setup_cors(app: FastAPI):
    """设置CORS中间件"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_middlewares(app: FastAPI):
    """设置所有中间件（顺序：CORS → Auth → GZip）"""
    setup_cors(app)
    app.add_middleware(AuthMiddleware)
