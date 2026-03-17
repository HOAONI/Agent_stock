# -*- coding: utf-8 -*-
"""提供给 Backend_stock 的内部模拟交易运行时接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent_api.deps import get_backtrader_runtime_service_dep
from agent_api.v1.schemas.backtrader import BacktraderInternalEnvelope, BacktraderInternalRequest
from agent_stock.services.backtrader_runtime_service import BacktraderRuntimeService
from agent_stock.config import redact_sensitive_text

router = APIRouter()


def _handle_error(exc: Exception) -> HTTPException:
    """将运行时异常统一转换为 HTTP 错误响应。"""
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


@router.post("/accounts/provision", response_model=BacktraderInternalEnvelope)
def provision_account(
    request: BacktraderInternalRequest,
    service: BacktraderRuntimeService = Depends(get_backtrader_runtime_service_dep),
) -> BacktraderInternalEnvelope:
    """为指定券商账户创建或校验本地模拟账户。"""
    try:
        data = service.provision_account(request.model_dump(exclude_none=True))
        return BacktraderInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/account-summary", response_model=BacktraderInternalEnvelope)
def get_account_summary(
    request: BacktraderInternalRequest,
    service: BacktraderRuntimeService = Depends(get_backtrader_runtime_service_dep),
) -> BacktraderInternalEnvelope:
    """返回本地模拟账户的汇总资产信息。"""
    try:
        data = service.get_account_summary(request.model_dump(exclude_none=True))
        return BacktraderInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/positions", response_model=BacktraderInternalEnvelope)
def get_positions(
    request: BacktraderInternalRequest,
    service: BacktraderRuntimeService = Depends(get_backtrader_runtime_service_dep),
) -> BacktraderInternalEnvelope:
    """返回本地模拟账户的持仓列表。"""
    try:
        data = {"items": service.get_positions(request.model_dump(exclude_none=True))}
        return BacktraderInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/orders", response_model=BacktraderInternalEnvelope)
def get_orders(
    request: BacktraderInternalRequest,
    service: BacktraderRuntimeService = Depends(get_backtrader_runtime_service_dep),
) -> BacktraderInternalEnvelope:
    """返回本地模拟账户的订单列表。"""
    try:
        data = {"items": service.get_orders(request.model_dump(exclude_none=True))}
        return BacktraderInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/trades", response_model=BacktraderInternalEnvelope)
def get_trades(
    request: BacktraderInternalRequest,
    service: BacktraderRuntimeService = Depends(get_backtrader_runtime_service_dep),
) -> BacktraderInternalEnvelope:
    """返回本地模拟账户的成交列表。"""
    try:
        data = {"items": service.get_trades(request.model_dump(exclude_none=True))}
        return BacktraderInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/place-order", response_model=BacktraderInternalEnvelope)
def place_order(
    request: BacktraderInternalRequest,
    service: BacktraderRuntimeService = Depends(get_backtrader_runtime_service_dep),
) -> BacktraderInternalEnvelope:
    """按市价模拟下单并立即撮合成交。"""
    try:
        data = service.place_order(request.model_dump(exclude_none=True))
        return BacktraderInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/cancel-order", response_model=BacktraderInternalEnvelope)
def cancel_order(
    request: BacktraderInternalRequest,
    service: BacktraderRuntimeService = Depends(get_backtrader_runtime_service_dep),
) -> BacktraderInternalEnvelope:
    """尝试取消本地模拟订单。"""
    try:
        data = service.cancel_order(request.model_dump(exclude_none=True))
        return BacktraderInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc


@router.post("/add-funds", response_model=BacktraderInternalEnvelope)
def add_funds(
    request: BacktraderInternalRequest,
    service: BacktraderRuntimeService = Depends(get_backtrader_runtime_service_dep),
) -> BacktraderInternalEnvelope:
    """向本地模拟账户追加资金。"""
    try:
        data = service.add_funds(request.model_dump(exclude_none=True))
        return BacktraderInternalEnvelope(ok=True, data=data)
    except Exception as exc:
        raise _handle_error(exc) from exc
