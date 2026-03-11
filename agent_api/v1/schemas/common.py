# -*- coding: utf-8 -*-
"""Common schemas for Agent API."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Error response payload."""

    error: str
    message: str


class HealthResponse(BaseModel):
    """Health response payload."""

    status: str
    timestamp: str
    detail: str | None = None


class RuntimeLlmDefaultResponse(BaseModel):
    """Internal response exposing the current built-in default LLM metadata."""

    available: bool
    source: str = "agent_env"
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    has_token: bool = False
