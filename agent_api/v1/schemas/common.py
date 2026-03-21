# -*- coding: utf-8 -*-
"""Agent API 的通用数据模型。"""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """错误响应载荷。"""

    error: str
    message: str


class HealthResponse(BaseModel):
    """健康检查响应载荷。"""

    status: str
    timestamp: str
    detail: str | None = None


class RuntimeLlmDefaultResponse(BaseModel):
    """暴露当前内置默认 LLM 元数据的内部响应。"""

    available: bool
    source: str = "agent_env"
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    has_token: bool = False


class RuntimeMarketSourceOptionResponse(BaseModel):
    """内部市场源候选项。"""

    code: str
    label: str
    description: str
    available: bool
    reason: str | None = None


class RuntimeMarketSourcesResponse(BaseModel):
    """内部市场源候选列表。"""

    options: list[RuntimeMarketSourceOptionResponse]
