# -*- coding: utf-8 -*-
"""内部回测 API 使用的数据模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_api.v1.schemas.runs import RuntimeLlmRequest


class BacktestRunCandidate(BaseModel):
    """用于回测评估的候选分析记录。"""

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
    """运行请求载荷。"""

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
    """汇总输入行。"""

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
    """汇总请求载荷。"""

    model_config = ConfigDict(extra="allow")

    scope: Literal["overall", "stock"] = "overall"
    code: str | None = None
    eval_window_days: int = Field(default=10, ge=1, le=120)
    engine_version: str = Field(default="v1", min_length=1, max_length=32)
    rows: list[BacktestSummaryRow] = Field(default_factory=list)


class BacktestCurveRow(BaseModel):
    """曲线输入行。"""

    model_config = ConfigDict(extra="allow")

    analysis_date: str | None = None
    evaluated_at: str | None = None
    simulated_return_pct: float | None = None
    stock_return_pct: float | None = None
    eval_status: str


class BacktestCurvesRequest(BaseModel):
    """曲线请求载荷。"""

    model_config = ConfigDict(extra="allow")

    scope: Literal["overall", "stock"] = "overall"
    code: str | None = None
    eval_window_days: int = Field(default=10, ge=1, le=120)
    rows: list[BacktestCurveRow] = Field(default_factory=list)


class BacktestDistributionRequest(BaseModel):
    """分布请求载荷。"""

    model_config = ConfigDict(extra="allow")

    scope: Literal["overall", "stock"] = "overall"
    code: str | None = None
    eval_window_days: int = Field(default=10, ge=1, le=120)
    engine_version: str = Field(default="v1", min_length=1, max_length=32)
    rows: list[BacktestSummaryRow] = Field(default_factory=list)


class BacktestCompareRow(BaseModel):
    """对比输入行。"""

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
    """对比请求载荷。"""

    model_config = ConfigDict(extra="allow")

    eval_window_days_list: list[int] = Field(default_factory=list)
    strategy_codes: list[str] | None = None
    neutral_band_pct: float = Field(default=2.0, ge=0, le=100)
    rows_by_window: dict[str, list[BacktestCompareRow]] = Field(default_factory=dict)


class BacktestInternalEnvelope(BaseModel):
    """标准内部响应包装。"""

    ok: bool = True
    data: dict[str, Any] = Field(default_factory=dict)


class BacktestInterpretationDateRange(BaseModel):
    """回测解释用的日期区间。"""

    model_config = ConfigDict(extra="allow")

    start_date: str | None = Field(default=None, max_length=32)
    end_date: str | None = Field(default=None, max_length=32)


class BacktestInterpretationItemRequest(BaseModel):
    """单条回测解释请求。"""

    model_config = ConfigDict(extra="allow")

    item_key: str = Field(min_length=1, max_length=128)
    item_type: Literal["strategy", "agent"] = "strategy"
    label: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=1, max_length=32)
    requested_range: BacktestInterpretationDateRange = Field(default_factory=BacktestInterpretationDateRange)
    effective_range: BacktestInterpretationDateRange = Field(default_factory=BacktestInterpretationDateRange)
    metrics: dict[str, Any] = Field(default_factory=dict)
    benchmark: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class BacktestInterpretRequest(BaseModel):
    """批量回测解释请求。"""

    model_config = ConfigDict(extra="allow")

    language: Literal["zh-CN"] = "zh-CN"
    runtime_llm: RuntimeLlmRequest | None = None
    items: list[BacktestInterpretationItemRequest] = Field(default_factory=list)


class StrategyTemplateRunDefinition(BaseModel):
    """由 Backend 解析后的具体策略定义。"""

    model_config = ConfigDict(extra="allow")

    strategy_id: int | None = Field(default=None, ge=1)
    strategy_name: str = Field(min_length=1, max_length=64)
    template_code: Literal["ma_cross", "rsi_threshold"]
    params: dict[str, float] = Field(default_factory=dict)


class StrategyRangeRunRequest(BaseModel):
    """区间策略回测请求。"""

    model_config = ConfigDict(extra="allow")

    code: str = Field(min_length=1, max_length=32)
    start_date: str = Field(min_length=10, max_length=32)
    end_date: str = Field(min_length=10, max_length=32)
    strategies: list[StrategyTemplateRunDefinition] | None = None
    strategy_codes: list[str] | None = None
    initial_capital: float | None = Field(default=None, gt=0)
    commission_rate: float | None = Field(default=None, ge=0, le=1)
    slippage_bps: float | None = Field(default=None, ge=0, le=1000)


class AgentHistoricalRuntimeStrategy(BaseModel):
    """历史回放的运行时策略覆盖项。"""

    model_config = ConfigDict(extra="allow")

    position_max_pct: float | None = Field(default=None, ge=0, le=100)
    stop_loss_pct: float | None = Field(default=None, ge=0, le=100)
    take_profit_pct: float | None = Field(default=None, ge=0, le=500)


class AgentHistoricalCachedSnapshot(BaseModel):
    """由 Backend 传入的缓存信号快照。"""

    model_config = ConfigDict(extra="allow")

    trade_date: str = Field(min_length=10, max_length=32)
    decision_source: str | None = None
    llm_used: bool = False
    confidence: float | None = None
    factor_payload: dict[str, Any] = Field(default_factory=dict)
    archived_news_payload: list[dict[str, Any]] = Field(default_factory=list)
    signal_payload: dict[str, Any] = Field(default_factory=dict)
    ai_overlay: dict[str, Any] = Field(default_factory=dict)


class AgentHistoricalRunRequest(BaseModel):
    """Agent 历史回放回测请求。"""

    model_config = ConfigDict(extra="allow")

    code: str = Field(min_length=1, max_length=32)
    start_date: str = Field(min_length=10, max_length=32)
    end_date: str = Field(min_length=10, max_length=32)
    phase: Literal["fast", "refine"] = "fast"
    initial_capital: float = Field(default=100000.0, gt=0)
    commission_rate: float = Field(default=0.0003, ge=0, le=1)
    slippage_bps: float = Field(default=2.0, ge=0, le=1000)
    runtime_strategy: AgentHistoricalRuntimeStrategy | None = None
    runtime_llm: RuntimeLlmRequest | None = None
    signal_profile_hash: str | None = None
    snapshot_version: int = Field(default=1, ge=1, le=1000)
    archived_news_by_date: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    cached_snapshots: list[AgentHistoricalCachedSnapshot] = Field(default_factory=list)
