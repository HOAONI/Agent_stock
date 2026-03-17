# -*- coding: utf-8 -*-
"""内部 Backtrader 运行时接口的数据模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BacktraderInternalRequest(BaseModel):
    """`Backend_stock` 使用的通用请求包装。"""

    model_config = ConfigDict(extra="allow")

    user_id: int = Field(ge=1)
    broker_account_id: int = Field(ge=1)
    environment: str | None = None
    account_uid: str | None = None
    account_display_name: str | None = None
    provider_code: str | None = None
    provider_name: str | None = None
    credentials: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] | None = None
    idempotency_key: str | None = None


class BacktraderInternalEnvelope(BaseModel):
    """内部响应包装。"""

    ok: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
