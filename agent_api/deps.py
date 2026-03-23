# -*- coding: utf-8 -*-
"""Agent API 的依赖注入提供器。"""

from __future__ import annotations

from fastapi import Depends

from agent_stock.services.backtest_service import BacktestService, get_backtest_service
from agent_stock.services.agent_historical_backtest_service import (
    AgentHistoricalBacktestService,
    get_agent_historical_backtest_service,
)
from agent_stock.services.backtest_interpretation_service import (
    BacktestInterpretationService,
    get_backtest_interpretation_service,
)
from agent_stock.services.strategy_backtest_service import StrategyBacktestService, get_strategy_backtest_service
from agent_stock.services.agent_task_service import AgentTaskService, get_agent_task_service
from agent_stock.services.backtrader_runtime_service import BacktraderRuntimeService, get_backtrader_runtime_service
from agent_stock.services.runtime_market_service import RuntimeMarketService, get_runtime_market_service
from agent_stock.config import Config, get_config


def get_config_dep() -> Config:
    """返回配置单例。"""
    return get_config()


def get_task_service_dep(config: Config = Depends(get_config_dep)) -> AgentTaskService:
    """返回任务服务单例。"""
    return get_agent_task_service(config=config)


def get_backtrader_runtime_service_dep() -> BacktraderRuntimeService:
    """返回本地模拟交易运行时服务单例。"""
    return get_backtrader_runtime_service()


def get_runtime_market_service_dep(config: Config = Depends(get_config_dep)) -> RuntimeMarketService:
    """返回内部市场服务单例。"""
    return get_runtime_market_service(config=config)


def get_backtest_service_dep() -> BacktestService:
    """返回内部回测服务单例。"""
    return get_backtest_service()


def get_backtest_interpretation_service_dep() -> BacktestInterpretationService:
    """返回回测自然语言解读服务单例。"""
    return get_backtest_interpretation_service()


def get_strategy_backtest_service_dep() -> StrategyBacktestService:
    """返回策略区间回测服务单例。"""
    return get_strategy_backtest_service()


def get_agent_historical_backtest_service_dep() -> AgentHistoricalBacktestService:
    """返回 Agent 历史回放回测服务单例。"""
    return get_agent_historical_backtest_service()
