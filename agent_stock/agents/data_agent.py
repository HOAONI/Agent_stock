# -*- coding: utf-8 -*-
"""Data Agent: fetches market data and builds normalized snapshots."""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from data_provider import DataFetcherManager
from data_provider.base import canonical_stock_code
from agent_stock.agents.contracts import AgentState, DataAgentOutput
from src.config import Config, get_config
from agent_stock.storage import DatabaseManager, get_db

logger = logging.getLogger(__name__)


class DataAgent:
    """Fetch daily bars + realtime quote and prepare analysis context."""

    def __init__(
        self,
        config: Optional[Config] = None,
        fetcher_manager: Optional[DataFetcherManager] = None,
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        self.config = config or get_config()
        self.fetcher_manager = fetcher_manager or DataFetcherManager()
        self.db = db_manager or get_db()

    def run(self, code: str) -> DataAgentOutput:
        """Run data collection for one stock code."""
        normalized_code = canonical_stock_code(code)
        trade_date = date.today()
        state = AgentState.READY
        error_message = None
        data_source = None
        daily_df = None

        try:
            daily_df, data_source = self.fetcher_manager.get_daily_data(normalized_code, days=60)
            if daily_df is not None and not daily_df.empty:
                self.db.save_daily_data(daily_df, normalized_code, data_source or "unknown")
        except Exception as exc:
            state = AgentState.FAILED
            error_message = f"daily data fetch failed: {exc}"
            logger.warning("[%s] %s", normalized_code, error_message)

        realtime_quote_dict = {}
        try:
            quote = self.fetcher_manager.get_realtime_quote(normalized_code)
            if quote is not None:
                realtime_quote_dict = quote.to_dict() if hasattr(quote, "to_dict") else dict(quote.__dict__)
        except Exception as exc:
            logger.warning("[%s] realtime quote fetch failed: %s", normalized_code, exc)

        analysis_context = self.db.get_analysis_context(normalized_code, history_days=60) or {}
        if "raw_data" not in analysis_context and daily_df is not None and not daily_df.empty:
            raw_data = []
            for _, row in daily_df.tail(60).iterrows():
                raw_data.append(
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
                    }
                )
            analysis_context["raw_data"] = raw_data
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
            data_source=data_source,
            error_message=error_message,
        )
