# -*- coding: utf-8 -*-
"""Internal backtest endpoints for Backend_stock."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent_api.deps import get_backtest_service_dep, get_strategy_backtest_service_dep
from agent_api.v1.schemas.backtest import (
    BacktestCompareRequest,
    BacktestCurvesRequest,
    BacktestDistributionRequest,
    BacktestInternalEnvelope,
    BacktestRunRequest,
    BacktestSummaryRequest,
    StrategyRangeRunRequest,
)
from agent_stock.services.backtest_service import BacktestService
from agent_stock.services.strategy_backtest_service import StrategyBacktestService
from src.config import redact_sensitive_text

router = APIRouter()


def _handle_error(exc: Exception) -> HTTPException:
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
    try:
        data = service.run(request.model_dump(exclude_none=True))
        return BacktestInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc
