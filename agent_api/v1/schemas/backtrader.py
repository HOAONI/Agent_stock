# -*- coding: utf-8 -*-
"""Internal Backtrader API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BacktraderInternalRequest(BaseModel):
    """Common request envelope used by Backend_stock."""

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
    """Internal response envelope."""

    ok: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
