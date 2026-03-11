# -*- coding: utf-8 -*-
"""Dependency providers for Agent API."""

from __future__ import annotations

from fastapi import Depends

from agent_stock.services.backtest_service import BacktestService, get_backtest_service
from agent_stock.services.agent_historical_backtest_service import (
    AgentHistoricalBacktestService,
    get_agent_historical_backtest_service,
)
from agent_stock.services.strategy_backtest_service import StrategyBacktestService, get_strategy_backtest_service
from agent_stock.services.agent_task_service import AgentTaskService, get_agent_task_service
from agent_stock.services.backtrader_runtime_service import BacktraderRuntimeService, get_backtrader_runtime_service
from src.config import Config, get_config


def get_config_dep() -> Config:
    """Get config singleton."""
    return get_config()


def get_task_service_dep(config: Config = Depends(get_config_dep)) -> AgentTaskService:
    """Get task service singleton."""
    return get_agent_task_service(config=config)


def get_backtrader_runtime_service_dep() -> BacktraderRuntimeService:
    """Get Backtrader runtime singleton."""
    return get_backtrader_runtime_service()


def get_backtest_service_dep() -> BacktestService:
    """Get Backtest service singleton."""
    return get_backtest_service()


def get_strategy_backtest_service_dep() -> StrategyBacktestService:
    """Get strategy date-range backtest service singleton."""
    return get_strategy_backtest_service()


def get_agent_historical_backtest_service_dep() -> AgentHistoricalBacktestService:
    """Get agent historical replay backtest singleton."""
    return get_agent_historical_backtest_service()
