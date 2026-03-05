# -*- coding: utf-8 -*-
"""Internal backtest API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BacktestRunCandidate(BaseModel):
    """Candidate analysis record for backtest evaluation."""

    model_config = ConfigDict(extra="allow")

    analysis_history_id: int = Field(ge=1)
    owner_user_id: int | None = None
    code: str = Field(min_length=1, max_length=32)
    created_at: str | None = None
    context_snapshot: str | None = None
    operation_advice: str | None = None
    stop_loss: float | None = None
    take_profit: float | None = None


class BacktestRunRequest(BaseModel):
    """Run request payload."""

    model_config = ConfigDict(extra="allow")

    code: str | None = None
    force: bool = False
    eval_window_days: int = Field(default=10, ge=1, le=120)
    min_age_days: int | None = Field(default=None, ge=0, le=365)
    limit: int | None = Field(default=None, ge=1, le=5000)
    engine_version: str = Field(default="v1", min_length=1, max_length=32)
    neutral_band_pct: float = Field(default=2.0, ge=0, le=100)
    candidates: list[BacktestRunCandidate] = Field(default_factory=list)


class BacktestSummaryRow(BaseModel):
    """Summary input row."""

    model_config = ConfigDict(extra="allow")

    eval_status: str
    position_recommendation: str | None = None
    outcome: str | None = None
    direction_correct: bool | None = None
    stock_return_pct: float | None = None
    simulated_return_pct: float | None = None
    hit_stop_loss: bool | None = None
    hit_take_profit: bool | None = None
    first_hit: str | None = None
    first_hit_trading_days: int | None = None
    operation_advice: str | None = None


class BacktestSummaryRequest(BaseModel):
    """Summary request payload."""

    model_config = ConfigDict(extra="allow")

    scope: Literal["overall", "stock"] = "overall"
    code: str | None = None
    eval_window_days: int = Field(default=10, ge=1, le=120)
    engine_version: str = Field(default="v1", min_length=1, max_length=32)
    rows: list[BacktestSummaryRow] = Field(default_factory=list)


class BacktestCurveRow(BaseModel):
    """Curve input row."""

    model_config = ConfigDict(extra="allow")

    analysis_date: str | None = None
    evaluated_at: str | None = None
    simulated_return_pct: float | None = None
    stock_return_pct: float | None = None
    eval_status: str


class BacktestCurvesRequest(BaseModel):
    """Curves request payload."""

    model_config = ConfigDict(extra="allow")

    scope: Literal["overall", "stock"] = "overall"
    code: str | None = None
    eval_window_days: int = Field(default=10, ge=1, le=120)
    rows: list[BacktestCurveRow] = Field(default_factory=list)


class BacktestDistributionRequest(BaseModel):
    """Distribution request payload."""

    model_config = ConfigDict(extra="allow")

    scope: Literal["overall", "stock"] = "overall"
    code: str | None = None
    eval_window_days: int = Field(default=10, ge=1, le=120)
    engine_version: str = Field(default="v1", min_length=1, max_length=32)
    rows: list[BacktestSummaryRow] = Field(default_factory=list)


class BacktestCompareRow(BaseModel):
    """Compare input row."""

    model_config = ConfigDict(extra="allow")

    code: str | None = None
    analysis_date: str | None = None
    evaluated_at: str | None = None
    simulated_return_pct: float | None = None
    stock_return_pct: float | None = None
    eval_status: str
    position_recommendation: str | None = None
    operation_advice: str | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    context_snapshot: str | None = None


class BacktestCompareRequest(BaseModel):
    """Compare request payload."""

    model_config = ConfigDict(extra="allow")

    eval_window_days_list: list[int] = Field(default_factory=list)
    strategy_codes: list[str] | None = None
    neutral_band_pct: float = Field(default=2.0, ge=0, le=100)
    rows_by_window: dict[str, list[BacktestCompareRow]] = Field(default_factory=dict)


class BacktestInternalEnvelope(BaseModel):
    """Standard internal response envelope."""

    ok: bool = True
    data: dict[str, Any] = Field(default_factory=dict)


class StrategyRangeRunRequest(BaseModel):
    """Date-range strategy backtest request."""

    model_config = ConfigDict(extra="allow")

    code: str = Field(min_length=1, max_length=32)
    start_date: str = Field(min_length=10, max_length=32)
    end_date: str = Field(min_length=10, max_length=32)
    strategy_codes: list[str] | None = None
    initial_capital: float | None = Field(default=None, gt=0)
    commission_rate: float | None = Field(default=None, ge=0, le=1)
    slippage_bps: float | None = Field(default=None, ge=0, le=1000)
