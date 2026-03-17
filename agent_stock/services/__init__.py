# -*- coding: utf-8 -*-
"""延迟导出服务对象，避免智能体模块间循环导入。"""

from __future__ import annotations

__all__ = [
    "AgentService",
    "AgentHistoricalBacktestService",
    "AgentTaskService",
    "BacktestService",
    "StrategyBacktestService",
    "get_agent_historical_backtest_service",
    "get_backtest_service",
    "get_strategy_backtest_service",
    "get_agent_task_service",
    "reset_agent_task_service",
]


def __getattr__(name: str):
    """按需导入服务实现，减少启动时耦合。"""
    if name == "AgentService":
        from agent_stock.services.agent_service import AgentService

        return AgentService
    if name in {"AgentHistoricalBacktestService", "get_agent_historical_backtest_service"}:
        from agent_stock.services.agent_historical_backtest_service import (
            AgentHistoricalBacktestService,
            get_agent_historical_backtest_service,
        )

        return {
            "AgentHistoricalBacktestService": AgentHistoricalBacktestService,
            "get_agent_historical_backtest_service": get_agent_historical_backtest_service,
        }[name]
    if name in {"AgentTaskService", "get_agent_task_service", "reset_agent_task_service"}:
        from agent_stock.services.agent_task_service import AgentTaskService, get_agent_task_service, reset_agent_task_service

        return {
            "AgentTaskService": AgentTaskService,
            "get_agent_task_service": get_agent_task_service,
            "reset_agent_task_service": reset_agent_task_service,
        }[name]
    if name in {"BacktestService", "get_backtest_service"}:
        from agent_stock.services.backtest_service import BacktestService, get_backtest_service

        return {
            "BacktestService": BacktestService,
            "get_backtest_service": get_backtest_service,
        }[name]
    if name in {"StrategyBacktestService", "get_strategy_backtest_service"}:
        from agent_stock.services.strategy_backtest_service import StrategyBacktestService, get_strategy_backtest_service

        return {
            "StrategyBacktestService": StrategyBacktestService,
            "get_strategy_backtest_service": get_strategy_backtest_service,
        }[name]
    raise AttributeError(name)
