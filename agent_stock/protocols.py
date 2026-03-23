# -*- coding: utf-8 -*-
"""项目内常用的结构化协议类型。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd

if TYPE_CHECKING:
    from agent_stock.agents.contracts import (
        AgentRunResult,
        DataAgentOutput,
        ExecutionAgentOutput,
        RiskAgentOutput,
        SignalAgentOutput,
    )


class SupportsDailyDataFetcher(Protocol):
    """支持按代码拉取日线数据的最小接口。"""

    def get_daily_data(self, *args: Any, **kwargs: Any) -> tuple[pd.DataFrame | None, str | None]:
        """返回 `(日线 DataFrame, 数据源名)`。"""
        ...


class SupportsRealtimeQuoteFetcher(SupportsDailyDataFetcher, Protocol):
    """支持日线与实时行情拉取的最小接口。"""

    def get_realtime_quote(self, *args: Any, **kwargs: Any) -> Any:
        """返回实时行情对象或字典。"""
        ...


class SupportsTrendAnalyzer(Protocol):
    """支持趋势分析的最小接口。"""

    def analyze(self, *args: Any, **kwargs: Any) -> Any:
        """执行趋势分析。"""
        ...


class SupportsDataAgent(Protocol):
    """支持数据阶段执行的最小接口。"""

    def run(self, *args: Any, **kwargs: Any) -> DataAgentOutput:
        """执行数据阶段。"""
        ...


class SupportsSignalAgent(Protocol):
    """支持信号阶段执行的最小接口。"""

    def run(self, *args: Any, **kwargs: Any) -> SignalAgentOutput:
        """执行信号阶段。"""
        ...


class SupportsRiskAgent(Protocol):
    """支持风控阶段执行的最小接口。"""

    def run(self, *args: Any, **kwargs: Any) -> RiskAgentOutput:
        """执行风控阶段。"""
        ...


class SupportsExecutionAgent(Protocol):
    """支持执行阶段执行的最小接口。"""

    def run(self, *args: Any, **kwargs: Any) -> ExecutionAgentOutput:
        """执行交易/意图生成阶段。"""
        ...


class SupportsBacktraderRuntimeService(Protocol):
    """支持模拟券商运行时的最小接口。"""

    def get_account_summary(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """返回账户摘要。"""
        ...

    def get_positions(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """返回持仓列表。"""
        ...

    def place_order(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """下单并返回成交结果。"""
        ...


class SupportsAgentOrchestrator(Protocol):
    """支持 Agent 服务调用的编排器最小接口。"""

    def run_once(self, *args: Any, **kwargs: Any) -> AgentRunResult:
        """执行单次运行。"""
        ...

    def run_realtime(self, *args: Any, **kwargs: Any) -> list[AgentRunResult]:
        """执行实时循环。"""
        ...


class SupportsMarketSessionGuard(Protocol):
    """支持实时循环交易时段判断的最小接口。"""

    timezone: Any

    def is_market_open(self, now: datetime | None = None) -> bool:
        """判断当前是否处于交易时段。"""
        ...
