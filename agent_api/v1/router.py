# -*- coding: utf-8 -*-
"""API 路由聚合与挂载定义。"""

from __future__ import annotations

from fastapi import APIRouter

from agent_api.v1.endpoints import accounts, backtest_internal, backtrader_internal, health, runs, runtime_internal, tasks

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(runs.router, prefix="/runs", tags=["Runs"])
api_v1_router.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])
api_v1_router.include_router(accounts.router, prefix="/accounts", tags=["Accounts"])

health_router = APIRouter(prefix="/api/health")
health_router.include_router(health.router, tags=["Health"])

internal_backtrader_router = APIRouter(prefix="/internal/v1/backtrader")
internal_backtrader_router.include_router(backtrader_internal.router, tags=["BacktraderInternal"])

internal_backtest_router = APIRouter(prefix="/internal/v1/backtest")
internal_backtest_router.include_router(backtest_internal.router, tags=["BacktestInternal"])

internal_runtime_router = APIRouter(prefix="/internal/v1/runtime")
internal_runtime_router.include_router(runtime_internal.router, tags=["RuntimeInternal"])
