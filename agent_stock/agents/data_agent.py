# -*- coding: utf-8 -*-
"""数据智能体。

它负责把“原始行情抓取”转换成后续阶段能直接消费的标准化输入，包括：
1. 拉取并缓存近 60 个交易日的日线数据
2. 尝试补充实时行情
3. 组装 `analysis_context` 供 Signal/Analyzer 使用
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from data_provider import DataFetcherManager
from data_provider.base import canonical_stock_code
from agent_stock.agents.contracts import AgentState, DataAgentOutput
from agent_stock.config import AgentRuntimeConfig, Config, get_config
from agent_stock.protocols import SupportsRealtimeQuoteFetcher
from agent_stock.storage import DatabaseManager, get_db

logger = logging.getLogger(__name__)


class DataAgent:
    """抓取日线与实时行情，并准备分析所需上下文。"""

    def __init__(
        self,
        config: Config | None = None,
        fetcher_manager: SupportsRealtimeQuoteFetcher | None = None,
        db_manager: DatabaseManager | None = None,
    ) -> None:
        """初始化数据抓取器与存储依赖。"""
        self.config = config or get_config()
        self.fetcher_manager = fetcher_manager or DataFetcherManager()
        self.db = db_manager or get_db()

    @staticmethod
    def _analyze_ma_status(today: dict) -> str:
        close = float(today.get("close") or 0)
        ma5 = float(today.get("ma5") or 0)
        ma10 = float(today.get("ma10") or 0)
        ma20 = float(today.get("ma20") or 0)

        if close > ma5 > ma10 > ma20 > 0:
            return "多头排列 📈"
        if close < ma5 < ma10 < ma20 and ma20 > 0:
            return "空头排列 📉"
        if close > ma5 and ma5 > ma10:
            return "短期向好 🔼"
        return "震荡整理"

    def _build_analysis_context_from_daily_frame(self, code: str, daily_df) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for _, row in daily_df.tail(60).iterrows():
            rows.append(
                {
                    "date": str(row.get("date"))[:10],
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                    "amount": row.get("amount"),
                    "pct_chg": row.get("pct_chg"),
                    "ma5": row.get("ma5"),
                    "ma10": row.get("ma10"),
                    "ma20": row.get("ma20"),
                    "volume_ratio": row.get("volume_ratio"),
                    "data_source": row.get("data_source"),
                }
            )

        if not rows:
            return {}

        today = dict(rows[-1])
        yesterday = dict(rows[-2]) if len(rows) > 1 else None
        # 分析上下文里既有行情原始字典，也有衍生出的数值字段，因此这里统一放宽为 Any。
        context: dict[str, Any] = {
            "code": code,
            "date": str(today.get("date") or ""),
            "today": today,
            "raw_data": rows,
        }

        if yesterday is not None:
            context["yesterday"] = yesterday
            yesterday_volume = float(yesterday.get("volume") or 0)
            yesterday_close = float(yesterday.get("close") or 0)
            today_volume = float(today.get("volume") or 0)
            today_close = float(today.get("close") or 0)
            if yesterday_volume > 0:
                context["volume_change_ratio"] = round(today_volume / yesterday_volume, 2)
            if yesterday_close > 0:
                context["price_change_ratio"] = round((today_close - yesterday_close) / yesterday_close * 100, 2)
            context["ma_status"] = self._analyze_ma_status(today)

        return context

    def run(self, code: str, *, runtime_config: AgentRuntimeConfig | None = None) -> DataAgentOutput:
        """为单只股票执行一次数据采集。"""
        normalized_code = canonical_stock_code(code)
        trade_date = date.today()
        state = AgentState.READY
        error_message = None
        data_source = None
        daily_df = None
        fixed_source = runtime_config.data_source.market_source if runtime_config and runtime_config.data_source else None

        try:
            # 日线是后续趋势分析和回测口径的基础，因此优先保证这部分数据可用。
            if fixed_source:
                daily_df, data_source = self.fetcher_manager.get_daily_data(
                    normalized_code,
                    days=60,
                    fixed_source=fixed_source,
                )
            else:
                daily_df, data_source = self.fetcher_manager.get_daily_data(normalized_code, days=60)
            if daily_df is not None and not daily_df.empty:
                self.db.save_daily_data(daily_df, normalized_code, fixed_source or data_source or "unknown")
        except Exception as exc:
            state = AgentState.FAILED
            error_message = f"daily data fetch failed: {exc}"
            logger.warning("[%s] %s", normalized_code, error_message)
            if fixed_source:
                return DataAgentOutput(
                    code=normalized_code,
                    trade_date=trade_date,
                    state=state,
                    analysis_context={},
                    realtime_quote={},
                    data_source=fixed_source,
                    error_message=error_message,
                )

        realtime_quote_dict = {}
        try:
            if fixed_source:
                quote = self.fetcher_manager.get_realtime_quote(normalized_code, fixed_source=fixed_source)
            else:
                quote = self.fetcher_manager.get_realtime_quote(normalized_code)
            if quote is not None:
                realtime_quote_dict = quote.to_dict() if hasattr(quote, "to_dict") else dict(quote.__dict__)
        except Exception as exc:
            logger.warning("[%s] realtime quote fetch failed: %s", normalized_code, exc)
            if fixed_source:
                state = AgentState.FAILED
                error_message = f"realtime quote fetch failed: {exc}"

        # 优先复用数据库中已经规整好的上下文；缺失时再用本次日线结果临时拼装。
        if fixed_source and daily_df is not None and not daily_df.empty:
            analysis_context = self._build_analysis_context_from_daily_frame(normalized_code, daily_df)
        else:
            analysis_context = self.db.get_analysis_context(normalized_code, history_days=60) or {}
            if "raw_data" not in analysis_context and daily_df is not None and not daily_df.empty:
                analysis_context = self._build_analysis_context_from_daily_frame(normalized_code, daily_df)
        if not analysis_context and state == AgentState.FAILED:
            return DataAgentOutput(
                code=normalized_code,
                trade_date=trade_date,
                state=AgentState.FAILED,
                analysis_context={},
                realtime_quote=realtime_quote_dict,
                data_source=data_source,
                error_message=error_message,
            )

        return DataAgentOutput(
            code=normalized_code,
            trade_date=trade_date,
            state=state,
            analysis_context=analysis_context,
            realtime_quote=realtime_quote_dict,
            data_source=fixed_source or data_source,
            error_message=error_message,
        )
