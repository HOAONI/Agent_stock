# -*- coding: utf-8 -*-
"""提供给 Backend_stock 的内部回测接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent_api.deps import (
    get_agent_historical_backtest_service_dep,
    get_backtest_service_dep,
    get_strategy_backtest_service_dep,
)
from agent_api.v1.schemas.backtest import (
    AgentHistoricalRunRequest,
    BacktestCompareRequest,
    BacktestCurvesRequest,
    BacktestDistributionRequest,
    BacktestInternalEnvelope,
    BacktestRunRequest,
    BacktestSummaryRequest,
    StrategyRangeRunRequest,
)
from agent_stock.services.agent_historical_backtest_service import AgentHistoricalBacktestService
from agent_stock.services.backtest_service import BacktestService
from agent_stock.services.strategy_backtest_service import StrategyBacktestService
from agent_stock.config import redact_sensitive_text

router = APIRouter()


def _handle_error(exc: Exception) -> HTTPException:
    """将服务层异常统一转换为 HTTP 错误响应。"""
    safe_message = redact_sensitive_text(str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(
            status_code=400,
            detail={"error": "validation_error", "message": safe_message},
        )
    return HTTPException(
        status_code=500,
        detail={"error": "internal_error", "message": safe_message},
    )


@router.post("/run", response_model=BacktestInternalEnvelope)
def run_backtest(
    request: BacktestRunRequest,
    service: BacktestService = Depends(get_backtest_service_dep),
) -> BacktestInternalEnvelope:
    """执行单批候选记录的内部回测。"""
    try:
        data = service.run(request.model_dump(exclude_none=True))
        return BacktestInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/summary", response_model=BacktestInternalEnvelope)
def summary_backtest(
    request: BacktestSummaryRequest,
    service: BacktestService = Depends(get_backtest_service_dep),
) -> BacktestInternalEnvelope:
    """汇总一组回测结果的统计指标。"""
    try:
        data = service.summary(request.model_dump(exclude_none=True))
        return BacktestInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/curves", response_model=BacktestInternalEnvelope)
def curves_backtest(
    request: BacktestCurvesRequest,
    service: BacktestService = Depends(get_backtest_service_dep),
) -> BacktestInternalEnvelope:
    """计算回测结果的收益曲线数据。"""
    try:
        data = service.curves(request.model_dump(exclude_none=True))
        return BacktestInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/distribution", response_model=BacktestInternalEnvelope)
def distribution_backtest(
    request: BacktestDistributionRequest,
    service: BacktestService = Depends(get_backtest_service_dep),
) -> BacktestInternalEnvelope:
    """统计回测结果中的仓位与胜负分布。"""
    try:
        data = service.distribution(request.model_dump(exclude_none=True))
        return BacktestInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/compare", response_model=BacktestInternalEnvelope)
def compare_backtest(
    request: BacktestCompareRequest,
    service: BacktestService = Depends(get_backtest_service_dep),
) -> BacktestInternalEnvelope:
    """比较多种策略模板在不同窗口下的表现。"""
    try:
        data = service.compare(request.model_dump(exclude_none=True))
        return BacktestInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/strategy/run", response_model=BacktestInternalEnvelope)
def run_strategy_backtest(
    request: StrategyRangeRunRequest,
    service: StrategyBacktestService = Depends(get_strategy_backtest_service_dep),
) -> BacktestInternalEnvelope:
    """执行基于模板策略的区间回测。"""
    try:
        data = service.run(request.model_dump(exclude_none=True))
        return BacktestInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/agent/run", response_model=BacktestInternalEnvelope)
def run_agent_historical_backtest(
    request: AgentHistoricalRunRequest,
    service: AgentHistoricalBacktestService = Depends(get_agent_historical_backtest_service_dep),
) -> BacktestInternalEnvelope:
    """执行 Agent 信号回放型历史回测。"""
    try:
        data = service.run(request.model_dump(exclude_none=True))
        return BacktestInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc
