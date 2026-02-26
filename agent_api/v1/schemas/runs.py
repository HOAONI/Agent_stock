# -*- coding: utf-8 -*-
"""Run schemas for Agent API."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RuntimeAccountRequest(BaseModel):
    """Request-level account configuration."""

    model_config = ConfigDict(extra="forbid")

    account_name: str = Field(min_length=1, max_length=128)
    initial_cash: float = Field(gt=0)
    account_display_name: str | None = Field(default=None, max_length=128)

    @field_validator("account_name")
    @classmethod
    def _normalize_account_name(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("account_name must not be empty")
        return text


class RuntimeLlmRequest(BaseModel):
    """Request-level LLM configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "deepseek", "custom"]
    base_url: str = Field(min_length=1, max_length=1024)
    model: str = Field(min_length=1, max_length=128)
    api_token: str | None = Field(default=None, min_length=1, max_length=4096)
    has_token: bool = False

    @field_validator("base_url", "model")
    @classmethod
    def _trim_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("must not be empty")
        return text

    @model_validator(mode="after")
    def _derive_has_token(self) -> "RuntimeLlmRequest":
        if self.api_token and not self.has_token:
            self.has_token = True
        return self


class RuntimeStrategyRequest(BaseModel):
    """Request-level strategy configuration."""

    model_config = ConfigDict(extra="forbid")

    position_max_pct: float = Field(ge=0, le=100)
    stop_loss_pct: float = Field(ge=0, le=100)
    take_profit_pct: float = Field(ge=0, le=100)


class RuntimeExecutionRequest(BaseModel):
    """Request-level execution configuration."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["paper", "broker"]
    has_ticket: bool = False
    credential_ticket: str | None = Field(default=None, min_length=1, max_length=4096)
    ticket_id: int | None = Field(default=None, ge=1)
    broker_account_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_broker_ticket(self) -> "RuntimeExecutionRequest":
        if self.credential_ticket and not self.has_ticket:
            self.has_ticket = True
        if self.mode == "broker" and not self.credential_ticket:
            raise ValueError("credential_ticket is required when execution.mode=broker")
        return self


class RuntimeConfigRequest(BaseModel):
    """Request-level runtime override payload."""

    model_config = ConfigDict(extra="forbid")

    account: RuntimeAccountRequest | None = None
    llm: RuntimeLlmRequest | None = None
    strategy: RuntimeStrategyRequest | None = None
    execution: RuntimeExecutionRequest | None = None


class RunCreateRequest(BaseModel):
    """Create run request payload."""

    model_config = ConfigDict(extra="forbid")

    stock_codes: list[str] = Field(default_factory=list)
    async_mode: bool = False
    request_id: str | None = None
    account_name: str | None = Field(default=None, min_length=1, max_length=128)
    runtime_config: RuntimeConfigRequest | None = None


class RunPayload(BaseModel):
    """Run payload from persistence."""

    run_id: str
    mode: str | None = None
    trade_date: str | None = None
    stock_codes: list[str] = Field(default_factory=list)
    account_name: str | None = None
    status: str | None = None
    data_snapshot: dict[str, Any] = Field(default_factory=dict)
    signal_snapshot: dict[str, Any] = Field(default_factory=dict)
    risk_snapshot: dict[str, Any] = Field(default_factory=dict)
    execution_snapshot: dict[str, Any] = Field(default_factory=dict)
    account_snapshot: dict[str, Any] = Field(default_factory=dict)
    report_path: str | None = None
    error_message: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    created_at: str | None = None


class RunListResponse(BaseModel):
    """Run list response payload."""

    total: int
    runs: list[RunPayload]
