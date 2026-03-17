# -*- coding: utf-8 -*-
"""Backtrader 回放引擎测试。"""

from __future__ import annotations

from datetime import date

import pytest

import agent_stock.services.backtrader_backtest_engine as engine_module
from agent_stock.services.backtrader_backtest_engine import BacktraderBacktestEngine, ReplayBar


def test_simulate_long_trade_fallback_uses_net_return():
    engine = BacktraderBacktestEngine(commission_rate=0.001, slippage_bps=10)
    engine.available = False

    result = engine.simulate_long_trade(
        analysis_date=date(2026, 1, 1),
        start_price=100.0,
        forward_bars=[
            ReplayBar(day=date(2026, 1, 2), high=106.0, low=99.0, close=105.0),
        ],
        stop_loss=None,
        take_profit=None,
    )

    buy_price = 100.0 * (1 + 10 / 10000)
    sell_price = 105.0 * (1 - 10 / 10000)
    expected_return = ((sell_price - buy_price - buy_price * 0.001 - sell_price * 0.001) / buy_price) * 100

    assert result["used_backtrader"] is False
    assert result["return_pct"] == pytest.approx(round(expected_return, 4), abs=1e-6)


@pytest.mark.skipif(engine_module.bt is None, reason="backtrader not installed")
def test_simulate_long_trade_backtrader_path_uses_net_return():
    engine = BacktraderBacktestEngine(commission_rate=0.001, slippage_bps=0)

    result = engine.simulate_long_trade(
        analysis_date=date(2026, 1, 1),
        start_price=100.0,
        forward_bars=[
            ReplayBar(day=date(2026, 1, 2), high=111.0, low=99.0, close=110.0),
        ],
        stop_loss=None,
        take_profit=None,
    )

    assert result["used_backtrader"] is True
    assert result["entry_price"] is not None
    assert result["exit_price"] is not None
    assert result["return_pct"] is not None

    entry_price = float(result["entry_price"])
    exit_price = float(result["exit_price"])
    expected_return = ((exit_price - entry_price - entry_price * 0.001 - exit_price * 0.001) / entry_price) * 100
    assert result["return_pct"] == pytest.approx(round(expected_return, 4), abs=1e-3)
