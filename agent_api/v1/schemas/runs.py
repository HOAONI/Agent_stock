# -*- coding: utf-8 -*-
"""Agent API 运行请求与响应的数据模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_stock.config import ALLOWED_MARKET_SOURCES
from agent_stock.runtime_config import ALLOWED_EXECUTION_MODES, ALLOWED_LLM_PROVIDERS

_ALLOWED_LLM_PROVIDER_SET = set(ALLOWED_LLM_PROVIDERS)
_ALLOWED_EXECUTION_MODE_SET = set(ALLOWED_EXECUTION_MODES)
_ALLOWED_MARKET_SOURCE_SET = set(ALLOWED_MARKET_SOURCES)


class RuntimeAccountRequest(BaseModel):
    """请求级账户配置。"""

    model_config = ConfigDict(extra="forbid")

    account_name: str = Field(min_length=1, max_length=128)
    initial_cash: float = Field(gt=0)
    account_display_name: str | None = Field(default=None, max_length=128)

    @field_validator("account_name")
    @classmethod
    def _normalize_account_name(cls, value: str) -> str:
        """清理并校验账户名长度。"""
        text = value.strip()
        if not text:
            raise ValueError("account_name must not be empty")
        return text


class RuntimeLlmRequest(BaseModel):
    """请求级 LLM 配置。"""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    base_url: str = Field(min_length=1, max_length=1024)
    model: str = Field(min_length=1, max_length=128)
    api_token: str | None = Field(default=None, min_length=1, max_length=4096)
    has_token: bool = False

    @field_validator("base_url", "model")
    @classmethod
    def _trim_text(cls, value: str) -> str:
        """去掉字符串字段两端空白。"""
        text = value.strip()
        if not text:
            raise ValueError("must not be empty")
        return text

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        """校验请求级 provider 是否在允许列表中。"""
        text = value.strip()
        if text not in _ALLOWED_LLM_PROVIDER_SET:
            raise ValueError(f"provider must be one of {sorted(_ALLOWED_LLM_PROVIDER_SET)}")
        return text

    @model_validator(mode="after")
    def _derive_has_token(self) -> "RuntimeLlmRequest":
        """根据 `api_token` 自动补全 `has_token` 语义。"""
        if self.api_token and not self.has_token:
            self.has_token = True
        return self


class RuntimeStrategyRequest(BaseModel):
    """请求级策略配置。"""

    model_config = ConfigDict(extra="forbid")

    position_max_pct: float = Field(ge=0, le=100)
    stop_loss_pct: float = Field(ge=0, le=100)
    take_profit_pct: float = Field(ge=0, le=100)


class RuntimeExecutionRequest(BaseModel):
    """请求级执行配置。"""

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(min_length=1)
    has_ticket: bool = False
    broker_account_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _normalize_execution(self) -> "RuntimeExecutionRequest":
        """规范执行模式并校验 broker 模式必填字段。"""
        self.mode = self.mode.strip()
        if self.mode not in _ALLOWED_EXECUTION_MODE_SET:
            raise ValueError(f"execution.mode must be one of {sorted(_ALLOWED_EXECUTION_MODE_SET)}")
        # 保留 `has_ticket` 字段以兼容旧载荷结构；`broker`
        # 保留给受信任的 Backend -> Agent 内部模拟执行流程。
        if self.mode == "broker" and self.broker_account_id is None:
            raise ValueError("execution.broker_account_id is required when mode=broker")
        return self


class RuntimeDataSourceRequest(BaseModel):
    """请求级行情源配置。"""

    model_config = ConfigDict(extra="forbid")

    market_source: str = Field(min_length=1)

    @field_validator("market_source")
    @classmethod
    def _validate_market_source(cls, value: str) -> str:
        """校验请求级行情源是否在允许列表中。"""
        text = value.strip()
        if text not in _ALLOWED_MARKET_SOURCE_SET:
            raise ValueError(f"market_source must be one of {sorted(_ALLOWED_MARKET_SOURCE_SET)}")
        return text


class RuntimeContextRequest(BaseModel):
    """由 Backend 透传的请求级账户上下文。"""

    model_config = ConfigDict(extra="forbid")

    account_snapshot: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    positions: list[dict[str, Any]] | None = None


class RuntimeConfigRequest(BaseModel):
    """请求级运行时覆盖配置。"""

    model_config = ConfigDict(extra="forbid")

    account: RuntimeAccountRequest | None = None
    llm: RuntimeLlmRequest | None = None
    strategy: RuntimeStrategyRequest | None = None
    execution: RuntimeExecutionRequest | None = None
    data_source: RuntimeDataSourceRequest | None = None
    context: RuntimeContextRequest | None = None


class RunCreateRequest(BaseModel):
    """创建运行任务的请求体。"""

    model_config = ConfigDict(extra="forbid")

    stock_codes: list[str] = Field(default_factory=list)
    async_mode: bool = False
    request_id: str | None = None
    account_name: str | None = Field(default=None, min_length=1, max_length=128)
    runtime_config: RuntimeConfigRequest | None = None


class RunPayload(BaseModel):
    """持久化后的运行结果载荷。"""

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
    """运行记录列表响应。"""

    total: int
    runs: list[RunPayload]
