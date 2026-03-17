# -*- coding: utf-8 -*-
"""Agent API 的 Bearer 鉴权中间件。"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from agent_stock.config import get_config


class AgentAuthMiddleware(BaseHTTPMiddleware):
    """为受保护接口校验静态 Bearer Token。"""

    def __init__(self, app):
        """初始化鉴权中间件并缓存当前配置。"""
        super().__init__(app)
        self._config = get_config()

    async def dispatch(self, request: Request, call_next):
        """放行健康检查，并拦截未携带正确令牌的请求。"""
        path = request.url.path
        if path in {"/api/health/live", "/api/health/ready"}:
            return await call_next(request)

        expected_token = str(getattr(self._config, "agent_service_auth_token", "") or "")
        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()

        if not expected_token or token != expected_token:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Missing or invalid bearer token",
                },
            )

        return await call_next(request)
