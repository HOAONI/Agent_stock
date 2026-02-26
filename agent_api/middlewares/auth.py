# -*- coding: utf-8 -*-
"""Bearer auth middleware for Agent API."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_config


class AgentAuthMiddleware(BaseHTTPMiddleware):
    """Validate static bearer token for API calls."""

    def __init__(self, app):
        super().__init__(app)
        self._config = get_config()

    async def dispatch(self, request: Request, call_next):
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
