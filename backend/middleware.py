# middleware.py - 中间件配置

import hmac
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from config import config


@dataclass(frozen=True)
class AuthPrincipal:
    role: str
    user_id: Optional[str] = None


def authorize_user(request: Optional[Request], user_id: str) -> None:
    """Enforce tenant ownership when authentication is configured."""
    principal = getattr(getattr(request, "state", None), "auth_principal", None)
    if principal is None:
        if not config.API_AUTH_REQUIRED and not config.API_AUTH_TOKEN and not config.user_token_map():
            return
        raise HTTPException(status_code=401, detail="未认证")
    if principal.role in {"admin", "development"}:
        return
    if principal.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问其他用户的数据")


def require_admin(request: Optional[Request]) -> None:
    principal = getattr(getattr(request, "state", None), "auth_principal", None)
    if principal is None:
        if not config.API_AUTH_REQUIRED and not config.API_AUTH_TOKEN and not config.user_token_map():
            return
        raise HTTPException(status_code=401, detail="未认证")
    if principal.role not in {"admin", "development"}:
        raise HTTPException(status_code=403, detail="需要管理员权限")


class AuthMiddleware(BaseHTTPMiddleware):
    """API 认证中间件（Bearer Token）"""

    # 不需要认证的开放路径
    OPEN_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        user_tokens = config.user_token_map()
        has_credentials = bool(config.API_AUTH_TOKEN or user_tokens)

        # 未配置 token 时仅允许显式开发模式。
        if not has_credentials and not config.API_AUTH_REQUIRED:
            request.state.auth_principal = AuthPrincipal(role="development")
            return await call_next(request)

        if not has_credentials:
            return JSONResponse(status_code=503, content={"detail": "服务认证尚未配置"})

        # 开放路径免认证
        if request.url.path in self.OPEN_PATHS:
            request.state.auth_principal = AuthPrincipal(role="anonymous")
            return await call_next(request)

        # 校验 Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "未提供有效的认证令牌"},
            )

        token = auth_header.removeprefix("Bearer ").strip()
        if config.API_AUTH_TOKEN and hmac.compare_digest(token, config.API_AUTH_TOKEN):
            request.state.auth_principal = AuthPrincipal(role="admin")
            return await call_next(request)

        matched_user_id = next(
            (
                user_id
                for candidate, user_id in user_tokens.items()
                if hmac.compare_digest(token, candidate)
            ),
            None,
        )
        if matched_user_id is None:
            return JSONResponse(status_code=401, content={"detail": "未提供有效的认证令牌"})

        request.state.auth_principal = AuthPrincipal(role="user", user_id=matched_user_id)
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
