# -*- coding: utf-8 -*-
"""策略区间回测服务测试。

这些测试主要验证三类事情：
1. 参数和结果结构是否稳定
2. 短区间、跳空、资金不足等边界场景是否会回归
3. 旧策略编码映射到新模板编码后，是否仍保持可用
"""

from __future__ import annotations

from datetime import date
import math

import pandas as pd
import pytest

import agent_stock.services.strategy_backtest_service as strategy_module
from agent_stock.services.strategy_backtest_service import StrategyBacktestService


class _StubFetcher:
    """最小化行情抓取桩，只返回预构造的 DataFrame。"""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def get_daily_data(self, code, start_date=None, end_date=None, days=0) -> tuple[pd.DataFrame | None, str | None]:  # noqa: ARG002
        return self._frame.copy(), "stub"


class _RecordingFetcher(_StubFetcher):
    """在桩对象基础上记录调用参数，方便断言取数窗口。"""

    def __init__(self, frame: pd.DataFrame):
        super().__init__(frame)
        self.calls = []

    def get_daily_data(self, code, start_date=None, end_date=None, days=0) -> tuple[pd.DataFrame | None, str | None]:  # noqa: ARG002
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
    """构造先涨后跌的平滑样本，适合验证基础区间回测流程。"""
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
    """构造含跳空的波动样本，用于测试保证金拒单与策略差异。"""
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
    """构造横盘样本，用于验证“无信号/无收益”场景。"""
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
    """构造单边上涨样本，用于验证窗口起点建仓逻辑。"""
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


def _dsl_combo_frame() -> pd.DataFrame:
    """构造先超卖反弹、再跌破短均线的样本，用于验证组合规则 DSL。"""
    rows = []
    start = pd.Timestamp("2024-01-01")
    price = 22.0
    trading_index = 0
    for i in range(220):
        day = start + pd.Timedelta(days=i)
        if day.weekday() >= 5:
            continue

        if trading_index < 45:
            price -= 0.22
        elif trading_index < 65:
            price += 0.42
        elif trading_index < 85:
            price += 0.12
        else:
            price -= 0.28

        price = max(4.0, price)
        rows.append(
            {
                "date": day.date().isoformat(),
                "open": round(price * 0.997, 4),
                "high": round(price * 1.015, 4),
                "low": round(price * 0.985, 4),
                "close": round(price, 4),
                "volume": 1000000,
            }
        )
        trading_index += 1
    return pd.DataFrame(rows)


def test_strategy_backtest_requires_backtrader(monkeypatch):
    """未安装 Backtrader 时应明确报错，而不是默默返回空结果。"""
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
    """标准路径下要返回请求区间、有效区间和逐策略结果列表。"""
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
    assert {item["strategy_code"] for item in items} == {"ma_cross", "rsi_threshold"}
    assert {item["template_code"] for item in items} == {"ma_cross", "rsi_threshold"}
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
    """不同策略遇到资金约束后，不应被错误地“抹平”为同一份结果。"""
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
    ma20_metrics = items["ma_cross"]["metrics"]
    rsi_metrics = items["rsi_threshold"]["metrics"]

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
    assert item["strategy_code"] == "ma_cross"
    assert item["template_code"] == "ma_cross"
    metrics = item["metrics"]
    assert metrics["entry_signal_count"] >= 1
    assert metrics["total_trades"] >= 1

    first_trade = item["trades"][0]
    assert first_trade["entry_date"] == "2024-03-04"
    assert first_trade["exit_date"] == "2024-03-29"
    assert first_trade["exit_reason"] == "window_end"


@pytest.mark.skipif(strategy_module.bt is None, reason="backtrader not installed")
def test_strategy_backtest_supports_parameterized_template_payload():
    service = StrategyBacktestService(fetcher_manager=_StubFetcher(_sample_frame()))
    result = service.run(
        {
            "code": "600519",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "strategies": [
                {
                    "strategy_id": 11,
                    "strategy_name": "Fast MA",
                    "template_code": "ma_cross",
                    "params": {"maWindow": 10},
                },
                {
                    "strategy_id": 12,
                    "strategy_name": "Wide RSI",
                    "template_code": "rsi_threshold",
                    "params": {"rsiPeriod": 10, "oversoldThreshold": 25, "overboughtThreshold": 75},
                },
                {
                    "strategy_id": 13,
                    "strategy_name": "MACD 金叉",
                    "template_code": "macd_cross",
                    "params": {"macdFast": 8, "macdSlow": 21, "macdSignal": 5},
                },
            ],
        }
    )

    assert [item["strategy_id"] for item in result["items"]] == [11, 12, 13]
    assert [item["strategy_name"] for item in result["items"]] == ["Fast MA", "Wide RSI", "MACD 金叉"]
    assert result["items"][0]["params"]["maWindow"] == 10
    assert result["items"][1]["params"]["rsiPeriod"] == 10
    assert result["items"][1]["params"]["oversoldThreshold"] == 25
    assert result["items"][1]["params"]["overboughtThreshold"] == 75
    assert result["items"][2]["params"]["macdFast"] == 8
    assert result["items"][2]["params"]["macdSlow"] == 21
    assert result["items"][2]["params"]["macdSignal"] == 5


@pytest.mark.skipif(strategy_module.bt is None, reason="backtrader not installed")
def test_strategy_backtest_supports_rule_dsl_payload():
    service = StrategyBacktestService(fetcher_manager=_StubFetcher(_dsl_combo_frame()))
    result = service.run(
        {
            "code": "600519",
            "start_date": "2024-02-01",
            "end_date": "2024-08-30",
            "strategies": [
                {
                    "strategy_id": 21,
                    "strategy_name": "MACD+RSI 组合止损",
                    "template_code": "rule_dsl",
                    "params": {
                        "entry": {
                            "operator": "and",
                            "conditions": [
                                {
                                    "kind": "macd_cross",
                                    "direction": "bullish",
                                    "fast": 12,
                                    "slow": 26,
                                    "signal": 9,
                                },
                                {
                                    "kind": "rsi_threshold",
                                    "period": 14,
                                    "operator": "lt",
                                    "threshold": 45,
                                },
                            ],
                        },
                        "exit": {
                            "operator": "or",
                            "conditions": [
                                {
                                    "kind": "price_ma_relation",
                                    "maWindow": 5,
                                    "relation": "cross_below",
                                }
                            ],
                        },
                    },
                }
            ],
        }
    )

    item = result["items"][0]
    assert item["strategy_id"] == 21
    assert item["strategy_code"] == "rule_dsl"
    assert item["template_code"] == "rule_dsl"
    assert item["params"]["entry"]["conditions"][0]["kind"] == "macd_cross"
    assert item["params"]["exit"]["conditions"][0]["relation"] == "cross_below"
    assert item["params"]["dsl_summary"].startswith("入场：")
    assert item["metrics"]["entry_signal_count"] >= 1
    assert item["metrics"]["total_trades"] >= 1
    assert any("MA5" in str(trade["exit_reason"]) for trade in item["trades"])


def test_strategy_backtest_rejects_invalid_template_params():
    service = StrategyBacktestService(fetcher_manager=_StubFetcher(_sample_frame()))

    with pytest.raises(ValueError, match="oversoldThreshold must be between 1 and 49"):
        service.run(
            {
                "code": "600519",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "strategies": [
                    {
                        "strategy_name": "Bad RSI",
                        "template_code": "rsi_threshold",
                        "params": {"rsiPeriod": 14, "oversoldThreshold": 80, "overboughtThreshold": 70},
                    }
                ],
            }
        )
