# -*- coding: utf-8 -*-
"""Strategy date-range backtest service tests."""

from __future__ import annotations

from datetime import date
import math

import pandas as pd
import pytest

import agent_stock.services.strategy_backtest_service as strategy_module
from agent_stock.services.strategy_backtest_service import StrategyBacktestService


class _StubFetcher:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def get_daily_data(self, code, start_date=None, end_date=None, days=0):  # noqa: ARG002
        return self._frame.copy(), "stub"


class _RecordingFetcher(_StubFetcher):
    def __init__(self, frame: pd.DataFrame):
        super().__init__(frame)
        self.calls = []

    def get_daily_data(self, code, start_date=None, end_date=None, days=0):  # noqa: ARG002
        self.calls.append(
            {
                "code": code,
                "start_date": start_date,
                "end_date": end_date,
                "days": days,
            }
        )
        return self._frame.copy(), "stub"


def _sample_frame() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01")
    price = 10.0
    for i in range(180):
        day = start + pd.Timedelta(days=i)
        if day.weekday() >= 5:
            continue
        drift = 0.06 if i < 90 else -0.04
        price = max(1.0, price + drift)
        rows.append(
            {
                "date": day.date().isoformat(),
                "open": round(price * 0.998, 4),
                "high": round(price * 1.01, 4),
                "low": round(price * 0.99, 4),
                "close": round(price, 4),
                "volume": 1000000,
            }
        )
    return pd.DataFrame(rows)


def _volatile_gap_frame() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01")
    for i in range(320):
        day = start + pd.Timedelta(days=i)
        if day.weekday() >= 5:
            continue

        close = max(2.0, 10.0 + 0.012 * i + 1.5 * math.sin(i / 8))
        gap = 1.04 if i % 19 == 0 else (0.985 if i % 23 == 0 else 1.0)
        open_price = max(0.01, close * gap)
        rows.append(
            {
                "date": day.date().isoformat(),
                "open": round(open_price, 4),
                "high": round(max(open_price, close) * 1.01, 4),
                "low": round(min(open_price, close) * 0.99, 4),
                "close": round(close, 4),
                "volume": 1000000,
            }
        )
    return pd.DataFrame(rows)


def _flat_frame() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01")
    for i in range(220):
        day = start + pd.Timedelta(days=i)
        if day.weekday() >= 5:
            continue
        rows.append(
            {
                "date": day.date().isoformat(),
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 1000000,
            }
        )
    return pd.DataFrame(rows)


def _state_entry_frame() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01")
    for i in range(180):
        day = start + pd.Timedelta(days=i)
        if day.weekday() >= 5:
            continue
        close = 10.0 + 0.08 * i
        rows.append(
            {
                "date": day.date().isoformat(),
                "open": round(close * 0.998, 4),
                "high": round(close * 1.01, 4),
                "low": round(close * 0.99, 4),
                "close": round(close, 4),
                "volume": 1000000,
            }
        )
    return pd.DataFrame(rows)


def test_strategy_backtest_requires_backtrader(monkeypatch):
    service = StrategyBacktestService(fetcher_manager=_StubFetcher(_sample_frame()))
    monkeypatch.setattr(strategy_module, "bt", None)

    with pytest.raises(ValueError, match="backtrader_not_available"):
        service.run(
            {
                "code": "600519",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            }
        )


@pytest.mark.skipif(strategy_module.bt is None, reason="backtrader not installed")
def test_strategy_backtest_returns_range_and_items():
    service = StrategyBacktestService(fetcher_manager=_StubFetcher(_sample_frame()))
    result = service.run(
        {
            "code": "600519",
            "start_date": date(2024, 1, 1).isoformat(),
            "end_date": date(2024, 12, 31).isoformat(),
            "strategy_codes": ["ma20_trend", "rsi14_mean_reversion"],
            "initial_capital": 100000,
            "commission_rate": 0.0003,
            "slippage_bps": 2,
        }
    )

    assert result["code"] == "600519"
    assert result["requested_range"]["start_date"] == "2024-01-01"
    assert result["requested_range"]["end_date"] == "2024-12-31"
    assert result["effective_range"]["start_date"] >= "2024-01-01"
    assert result["effective_range"]["end_date"] <= "2024-12-31"

    items = result["items"]
    assert len(items) == 2
    assert {item["strategy_code"] for item in items} == {"ma20_trend", "rsi14_mean_reversion"}
    assert all(isinstance(item["equity"], list) and len(item["equity"]) > 0 for item in items)


@pytest.mark.skipif(strategy_module.bt is None, reason="backtrader not installed")
def test_strategy_backtest_short_range_does_not_crash():
    service = StrategyBacktestService(fetcher_manager=_StubFetcher(_sample_frame()))
    result = service.run(
        {
            "code": "600519",
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "strategy_codes": ["ma20_trend", "rsi14_mean_reversion"],
        }
    )

    assert result["requested_range"]["start_date"] == "2024-01-01"
    assert result["requested_range"]["end_date"] == "2024-01-10"
    assert len(result["items"]) == 2
    assert all("metrics" in item for item in result["items"])
    assert all(item["metrics"]["completed_trading_days"] > 0 for item in result["items"])
    assert all(len(item["equity"]) > 0 for item in result["items"])


@pytest.mark.skipif(strategy_module.bt is None, reason="backtrader not installed")
def test_strategy_backtest_handles_margin_without_homogenizing_results():
    service = StrategyBacktestService(fetcher_manager=_StubFetcher(_volatile_gap_frame()))
    result = service.run(
        {
            "code": "600519",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "strategy_codes": ["ma20_trend", "rsi14_mean_reversion"],
            "initial_capital": 100000,
            "commission_rate": 0.0003,
            "slippage_bps": 2,
        }
    )

    items = {item["strategy_code"]: item for item in result["items"]}
    ma20_metrics = items["ma20_trend"]["metrics"]
    rsi_metrics = items["rsi14_mean_reversion"]["metrics"]

    assert ma20_metrics["margin_rejections"] >= 0
    assert rsi_metrics["margin_rejections"] >= 0
    assert ma20_metrics["total_trades"] > 0 or rsi_metrics["total_trades"] > 0
    assert (
        ma20_metrics["total_trades"] != rsi_metrics["total_trades"]
        or abs(float(ma20_metrics["total_return_pct"]) - float(rsi_metrics["total_return_pct"])) > 1e-6
    )


@pytest.mark.skipif(strategy_module.bt is None, reason="backtrader not installed")
def test_strategy_backtest_no_entry_signal_reason_is_exposed():
    service = StrategyBacktestService(fetcher_manager=_StubFetcher(_flat_frame()))
    result = service.run(
        {
            "code": "600519",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "strategy_codes": ["ma20_trend", "rsi14_mean_reversion"],
            "initial_capital": 100000,
            "commission_rate": 0.0003,
            "slippage_bps": 2,
        }
    )

    for item in result["items"]:
        metrics = item["metrics"]
        assert metrics["total_trades"] == 0
        assert metrics["entry_signal_count"] == 0
        assert metrics["exit_signal_count"] == 0
        assert metrics["margin_rejections"] == 0
        assert metrics["no_trade_reason"] == "no_entry_signal"
        assert isinstance(metrics["no_trade_reason_detail"], str)
        assert len(metrics["no_trade_reason_detail"]) > 0


@pytest.mark.skipif(strategy_module.bt is None, reason="backtrader not installed")
def test_strategy_backtest_ma20_supports_window_start_state_entry_with_warmup():
    fetcher = _RecordingFetcher(_state_entry_frame())
    service = StrategyBacktestService(fetcher_manager=fetcher)
    result = service.run(
        {
            "code": "600519",
            "start_date": "2024-03-01",
            "end_date": "2024-03-29",
            "strategy_codes": ["ma20_trend"],
            "initial_capital": 100000,
            "commission_rate": 0.0003,
            "slippage_bps": 2,
        }
    )

    assert fetcher.calls
    call = fetcher.calls[0]
    assert call["start_date"] == "2023-11-02"
    assert call["end_date"] == "2024-03-29"

    item = result["items"][0]
    metrics = item["metrics"]
    assert metrics["entry_signal_count"] >= 1
    assert metrics["total_trades"] >= 1

    first_trade = item["trades"][0]
    assert first_trade["entry_date"] == "2024-03-04"
    assert first_trade["exit_date"] == "2024-03-29"
    assert first_trade["exit_reason"] == "window_end"
