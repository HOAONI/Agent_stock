# -*- coding: utf-8 -*-
"""策略区间回测服务测试。

这些测试主要验证三类事情：
1. 参数和结果结构是否稳定
2. 短区间、跳空、资金不足等边界场景是否会回归
3. 旧策略编码映射到新模板编码后，是否仍保持可用
"""

from __future__ import annotations

import io
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


def _byd_short_effective_window_frame() -> pd.DataFrame:
    """复现真实缓存窗口从首个可用 bar 开始时的指标预热边界。"""
    return pd.read_csv(
        io.StringIO(
            """date,open,high,low,close,volume
2025-11-05,96.66,96.66,94.73,95.8,58721510
2025-11-06,95.58,97.75,95.4,97.52,40200367
2025-11-07,97,98.15,96.83,97.2,32256073
2025-11-10,97.35,100.35,97.35,99.39,55487690
2025-11-11,99.5,99.78,98.14,98.71,31985788
2025-11-12,98.8,98.99,97.54,97.77,26064335
2025-11-13,97.01,100.5,97.01,99.83,579990
2025-11-14,99,100.5,98.33,98.37,342985
2025-11-17,97.8,98.59,97.25,98.08,20724759
2025-11-18,97.54,97.85,96.02,96.49,30475645
2025-11-19,96.4,96.66,95.01,95.17,34829235
2025-11-20,95.17,95.5,93.5,93.61,39181087
2025-11-21,92.49,94.39,91.7,92.7,45171313
2025-11-24,93.1,94.41,92.2,93.93,32730213
2025-11-25,94,95.38,93.7,93.91,27748887
2025-11-26,94.75,95.44,94.45,95.09,30657697
2025-11-27,94.6,95.16,94.22,94.29,20227996
2025-11-28,94.6,95.23,94.2,95.17,21935062
2025-12-01,95.39,96.3,95.14,95.77,23371272
2025-12-02,96.42,97.8,96.25,96.62,35921746
2025-12-03,96.64,96.65,94.89,95.05,23157284
2025-12-04,94.92,95.29,94.2,95.24,16614164
2025-12-05,95.29,96.22,94.71,95.98,21161847
2025-12-08,96.21,97.15,96.21,96.53,24780498
2025-12-09,96.21,96.66,95.7,96.03,16322702
2025-12-10,95.6,96.65,94.9,96.47,19840224
2025-12-11,96.98,97.12,95.85,96.23,25090767
2025-12-12,96.2,97.7,95.65,97,55462088
2025-12-15,96.55,97.16,95.53,95.53,29330099
2025-12-16,95.5,95.7,94.14,94.25,25876771
2025-12-17,94.5,95.7,93.68,95.21,27282500
2025-12-18,94.39,94.6,93.5,93.53,25279734
2025-12-19,94,95.27,93.73,94.23,27206794
2025-12-22,94.7,95.36,94.25,94.37,20623482
2025-12-23,94.5,95.5,94.5,94.81,23661741
2025-12-24,94.8,94.8,94.05,94.42,19280995
2025-12-25,94.43,95.23,94.12,94.84,18220163
2025-12-26,95.28,101.45,95.25,100.01,102095895
2025-12-29,100.5,101.3,99.6,100.21,51916696
2025-12-30,99.26,100.15,98.74,99.75,31875638
2025-12-31,99.98,100.2,97.38,97.72,40285887
2026-01-05,98.4,99.48,97.9,98.11,38262521
2026-01-06,98.22,100.5,98.12,99.99,52178486
2026-01-07,99.96,99.96,97.06,97.6,56084881
2026-01-08,97.2,97.2,96.33,96.88,30065475
2026-01-09,97.05,97.95,96.9,97.01,35034058
2026-01-12,97.01,98,96.3,97.47,42525719
2026-01-13,99,99.6,96.88,97.19,49872272
2026-01-14,96.81,97.74,95.51,96.1,51474973
2026-01-15,96.08,96.64,95.6,95.67,29563546
2026-01-16,96.7,97.44,95.6,95.86,38458802
2026-01-19,96.81,97.3,95.77,96.19,33492715
2026-01-20,96.22,96.26,94.68,94.74,38914947
2026-01-21,94.01,94.94,94,94.1,30833393
2026-01-22,94.72,95.13,93.89,94.12,29101425
2026-01-23,94.24,94.36,93.51,93.65,38330272
2026-01-26,93.65,93.66,92.5,92.63,37545961
2026-01-27,92.67,92.99,91.8,91.81,36298092
2026-01-28,91.8,93.5,90.01,93.34,71556618
2026-01-29,92.66,92.92,91.61,92.31,37182481
2026-01-30,92.27,92.27,90.7,90.89,32917560
2026-02-02,88,89.09,86.97,87.05,64697385
2026-02-03,87.2,87.4,85.88,87.37,37782685
2026-02-04,86.99,89.33,86.45,89.14,37650471
2026-02-05,88.5,91.23,88.33,90.11,35424031
2026-02-06,89.39,90.66,89,89.82,24708779
2026-02-09,90.5,90.65,89.83,90.06,20817951
2026-02-10,90,91.57,89.84,90.81,26760481
2026-02-11,90.88,92.95,90.88,92.28,39694959
2026-02-12,92.29,92.35,91.08,91.16,23031332
2026-02-13,90.05,91.09,89.88,90.27,21733887
2026-02-24,91.98,91.99,90.61,90.87,20644458
2026-02-25,90.89,92.79,90.89,91.45,32478010
2026-02-26,91.4,91.4,89.61,89.85,29101894
2026-02-27,89.4,89.94,89.1,89.32,21934083
2026-03-02,88,97.25,87.72,96.79,143513688
2026-03-03,96.39,96.99,93.99,95.21,86574145
2026-03-04,95.2,96.8,93.7,95.99,70096892
2026-03-05,96.8,96.82,93.89,94.47,55243997
2026-03-06,96,96,93.25,93.62,50354037
2026-03-09,93.62,98,93,97.52,109340899
2026-03-10,97.18,97.49,96.08,96.6,50746243
2026-03-11,96.75,100.34,95.67,99.22,90374375
2026-03-12,99.23,100.69,98.7,99.1,50252357
2026-03-13,99,100.79,98.3,99.67,51976482
2026-03-16,99.67,105,98.79,104.62,108781426
2026-03-17,104.63,106.66,102.71,102.87,84138860
2026-03-18,102.82,102.82,100.05,101.69,61799648
2026-03-19,100.99,103.88,100.47,102.31,53105939
2026-03-20,102,104.45,101.56,103.03,62878324
2026-03-23,103.87,111.82,103.18,107.63,178244551
2026-03-24,108.4,108.4,104.93,106.64,92144617
2026-03-25,106.72,107.66,105.28,106.6,66266489
2026-03-26,106.62,107.68,102.7,103.14,82812067
2026-03-27,103.99,105.89,102.75,105.3,65845464
2026-03-30,103.51,108,103.25,106.05,72755723
2026-03-31,107,108.49,105.1,105.25,58865901
2026-04-01,105.9,106.5,102.37,102.65,71519811
2026-04-02,102.49,103.48,100.5,101.65,60434947
2026-04-03,101.8,102.3,99,99.01,49174516
"""
        )
    )


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


@pytest.mark.skipif(strategy_module.bt is None, reason="backtrader not installed")
def test_strategy_backtest_rule_dsl_handles_short_effective_window_without_index_error():
    """真实缓存只覆盖近几个月时，也不应因指标 buffer 未就绪而抛 500。"""
    service = StrategyBacktestService(fetcher_manager=_StubFetcher(_byd_short_effective_window_frame()))
    result = service.run(
        {
            "code": "002594",
            "start_date": "2025-04-09",
            "end_date": "2026-04-09",
            "strategies": [
                {
                    "strategy_name": "MACD(12,26,9)金叉 且 RSI14 < 30 / 跌破 MA5",
                    "template_code": "rule_dsl",
                    "params": {
                        "dslVersion": "rule_v1",
                        "sourceText": "如果我在比亚迪每次MACD 金叉且 RSI<30 且跌破 5 日线止损，过去一年收益怎样",
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
                                    "threshold": 30,
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
            "initial_capital": 100000,
        }
    )

    assert result["code"] == "002594"
    assert result["effective_range"]["start_date"] == "2025-11-05"
    assert result["effective_range"]["end_date"] == "2026-04-03"

    item = result["items"][0]
    assert item["strategy_code"] == "rule_dsl"
    assert item["template_code"] == "rule_dsl"
    assert item["params"]["entry"]["conditions"][0]["kind"] == "macd_cross"
    assert item["params"]["entry"]["conditions"][1]["threshold"] == 30
    assert item["params"]["exit"]["conditions"][0]["relation"] == "cross_below"
    assert item["metrics"]["completed_trading_days"] > 0


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
