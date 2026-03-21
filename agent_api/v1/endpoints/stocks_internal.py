# -*- coding: utf-8 -*-
"""提供给 Backend_stock 的内部股票行情接口。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from agent_api.deps import get_runtime_market_service_dep
from agent_stock.services.runtime_market_service import RuntimeMarketService

router = APIRouter()


def _rethrow_market_error(action: str, error: Exception) -> None:
    message = str(error).strip() or f"{action} failed"
    if "unsupported market source" in message or "stock_code is required" in message:
        raise HTTPException(status_code=422, detail={"error": "validation_error", "message": message}) from error
    if "No available daily bar" in message:
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": message}) from error
    raise HTTPException(status_code=502, detail={"error": "upstream_error", "message": message}) from error


@router.get("/{stock_code}/quote")
def get_stock_quote(
    stock_code: str,
    market_source: Annotated[str, Query(min_length=1)],
    service: RuntimeMarketService = Depends(get_runtime_market_service_dep),
) -> dict:
    try:
        return service.get_quote(stock_code, market_source)
    except Exception as error:  # pragma: no cover - 内部异常统一映射
        _rethrow_market_error("get quote", error)


@router.get("/{stock_code}/history")
def get_stock_history(
    stock_code: str,
    market_source: Annotated[str, Query(min_length=1)],
    period: Annotated[str, Query()] = "daily",
    days: Annotated[int, Query(ge=1, le=365)] = 30,
    service: RuntimeMarketService = Depends(get_runtime_market_service_dep),
) -> dict:
    if period != "daily":
        raise HTTPException(status_code=422, detail={"error": "validation_error", "message": "period must be daily"})
    try:
        return service.get_history(stock_code, days, market_source)
    except Exception as error:  # pragma: no cover - 内部异常统一映射
        _rethrow_market_error("get history", error)


@router.get("/{stock_code}/indicators")
def get_stock_indicators(
    stock_code: str,
    market_source: Annotated[str, Query(min_length=1)],
    period: Annotated[str, Query()] = "daily",
    days: Annotated[int, Query(ge=1, le=365)] = 120,
    windows: Annotated[str, Query()] = "5,10,20,60",
    service: RuntimeMarketService = Depends(get_runtime_market_service_dep),
) -> dict:
    if period != "daily":
        raise HTTPException(status_code=422, detail={"error": "validation_error", "message": "period must be daily"})
    try:
        parsed_windows = [
            int(item.strip())
            for item in str(windows).split(",")
            if item.strip()
        ]
        return service.get_indicators(stock_code, days, parsed_windows, market_source)
    except Exception as error:  # pragma: no cover - 内部异常统一映射
        _rethrow_market_error("get indicators", error)


@router.get("/{stock_code}/factors")
def get_stock_factors(
    stock_code: str,
    market_source: Annotated[str, Query(min_length=1)],
    date: str | None = None,
    service: RuntimeMarketService = Depends(get_runtime_market_service_dep),
) -> dict:
    try:
        return service.get_factors(stock_code, market_source, target_date=date)
    except Exception as error:  # pragma: no cover - 内部异常统一映射
        _rethrow_market_error("get factors", error)
