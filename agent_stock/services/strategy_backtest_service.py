# -*- coding: utf-8 -*-
"""Date-range strategy backtest service based on backtrader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from data_provider import DataFetcherManager
from data_provider.base import canonical_stock_code, normalize_stock_code

try:
    import backtrader as bt  # type: ignore
except Exception:  # pragma: no cover
    bt = None

LEGACY_STRATEGY_CODES = ["ma20_trend", "rsi14_mean_reversion"]
LEGACY_STRATEGY_NAMES: Dict[str, str] = {
    "ma20_trend": "MA20 Trend",
    "rsi14_mean_reversion": "RSI14 Mean Reversion",
}
TEMPLATE_NAMES: Dict[str, str] = {
    "ma_cross": "MA Cross",
    "rsi_threshold": "RSI Threshold",
}
LEGACY_STRATEGY_TEMPLATE_MAP: Dict[str, Dict[str, Any]] = {
    "ma20_trend": {
        "template_code": "ma_cross",
        "strategy_name": LEGACY_STRATEGY_NAMES["ma20_trend"],
        "params": {
            "maWindow": 20,
        },
    },
    "rsi14_mean_reversion": {
        "template_code": "rsi_threshold",
        "strategy_name": LEGACY_STRATEGY_NAMES["rsi14_mean_reversion"],
        "params": {
            "rsiPeriod": 14,
            "oversoldThreshold": 30,
            "overboughtThreshold": 70,
        },
    },
}
WARMUP_CALENDAR_DAYS = 120


@dataclass(frozen=True)
class StrategyRunParams:
    code: str
    start_date: date
    end_date: date
    initial_capital: float
    commission_rate: float
    slippage_bps: float


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: Optional[int]
    strategy_name: str
    template_code: str
    template_name: str
    params: Dict[str, float]


class _BaseTrackedStrategy(bt.Strategy if bt is not None else object):
    """Common tracking for deterministic strategy output."""

    params = dict(initial_capital=100000.0, trade_start=None, trade_end=None)
    _BUY_RATIOS: tuple[float, ...] = (0.95, 0.85, 0.75, 0.65)

    def __init__(self):  # type: ignore[override]
        self.pending_order = None
        self.pending_exit_reason = None
        self.current_entry = None
        self.trades: List[Dict[str, Any]] = []
        self.equity_points: List[Dict[str, Any]] = []
        self.margin_rejections = 0
        self.entry_signal_count = 0
        self.exit_signal_count = 0
        self._buy_ratio_index = 0

    def _next_open_price(self) -> float:
        value = float(self.data.open[0])
        if not math.isfinite(value) or value <= 0:
            value = float(self.data.close[0])
        return max(0.01, value)

    def _track_equity(self) -> None:
        if not self._is_in_trade_window():
            return
        current_date = bt.num2date(self.data.datetime[0]).date().isoformat()
        self.equity_points.append(
            {
                "trade_date": current_date,
                "equity": float(self.broker.getvalue()),
            }
        )

    def _current_trade_day(self) -> date:
        return bt.num2date(self.data.datetime[0]).date()

    def _trade_window_bounds(self) -> tuple[Optional[date], Optional[date]]:
        trade_start = getattr(self.p, "trade_start", None)
        trade_end = getattr(self.p, "trade_end", None)
        if isinstance(trade_start, datetime):
            trade_start = trade_start.date()
        if isinstance(trade_end, datetime):
            trade_end = trade_end.date()
        return (
            trade_start if isinstance(trade_start, date) else None,
            trade_end if isinstance(trade_end, date) else None,
        )

    def _is_in_trade_window(self) -> bool:
        day = self._current_trade_day()
        trade_start, trade_end = self._trade_window_bounds()
        if trade_start is not None and day < trade_start:
            return False
        if trade_end is not None and day > trade_end:
            return False
        return True

    def _is_window_start_bar(self) -> bool:
        trade_start, _ = self._trade_window_bounds()
        if trade_start is None:
            return False
        return self._current_trade_day() == trade_start

    def _estimate_order_unit_cost(self, open_price: float) -> float:
        commission_rate = 0.0
        try:
            commission_info = self.broker.getcommissioninfo(self.data)
            commission_rate = max(0.0, float(getattr(commission_info.p, "commission", 0.0) or 0.0))
        except Exception:
            commission_rate = 0.0

        broker_params = getattr(self.broker, "p", None)
        slippage_rate = max(0.0, float(getattr(broker_params, "slip_perc", 0.0) or 0.0))
        estimated = open_price * (1.0 + commission_rate) * (1.0 + slippage_rate)
        if not math.isfinite(estimated) or estimated <= 0:
            return open_price
        return estimated

    def _submit_full_position_buy(self) -> None:
        if self._buy_ratio_index >= len(self._BUY_RATIOS):
            return

        open_price = self._next_open_price()
        cash = float(self.broker.getcash())
        if not math.isfinite(cash) or cash <= 0:
            return

        unit_cost = self._estimate_order_unit_cost(open_price)
        ratio = float(self._BUY_RATIOS[self._buy_ratio_index])
        size = int((cash * ratio) // unit_cost)
        if size > 0:
            self.pending_order = self.buy(size=size)

    def _submit_full_position_sell(self, reason: str) -> None:
        if not self.position:
            return
        self.pending_exit_reason = reason
        self.pending_order = self.sell(size=self.position.size)

    def _should_force_window_end_close(self) -> bool:
        if not self._is_in_trade_window():
            return False
        last_close_trigger_index = max(1, int(self.data.buflen()) - 1)
        return len(self.data) >= last_close_trigger_index

    def _should_buy(self) -> bool:
        raise NotImplementedError

    def _should_sell(self) -> bool:
        raise NotImplementedError

    def _signal_exit_reason(self) -> str:
        return "signal_exit"

    def next(self):  # type: ignore[override]
        self._track_equity()

        if self.pending_order is not None:
            return

        if not self._is_in_trade_window():
            return

        if not self.position:
            if self._should_buy():
                self.entry_signal_count += 1
                self._submit_full_position_buy()
            return

        if self._should_sell():
            self.exit_signal_count += 1
            self._submit_full_position_sell(self._signal_exit_reason())
            return

        if self._should_force_window_end_close():
            # last bar safeguard
            self.pending_exit_reason = "window_end"
            self.pending_order = self.close(exectype=bt.Order.Close)

    def notify_order(self, order):  # type: ignore[override]
        if order.status in [order.Submitted, order.Accepted]:
            return

        margin_status = getattr(order, "Margin", None)
        if margin_status is not None and order.status == margin_status and order.isbuy():
            self.margin_rejections += 1
            self.pending_order = None
            if not self.position and self._buy_ratio_index + 1 < len(self._BUY_RATIOS):
                self._buy_ratio_index += 1
                self._submit_full_position_buy()
            return

        if order.status != order.Completed:
            self.pending_order = None
            return

        executed_date = bt.num2date(order.executed.dt).date().isoformat()
        quantity = int(abs(order.executed.size) or 0)
        price = float(order.executed.price)
        commission = float(order.executed.comm or 0.0)

        if order.isbuy():
            self._buy_ratio_index = 0
            self.current_entry = {
                "entry_date": executed_date,
                "entry_price": price,
                "qty": quantity,
                "entry_fees": commission,
            }
            self.pending_order = None
            return

        if order.issell() and self.current_entry is not None:
            entry_price = float(self.current_entry["entry_price"])
            qty = int(self.current_entry["qty"])
            entry_fees = float(self.current_entry.get("entry_fees") or 0.0)
            exit_fees = commission
            notional = entry_price * qty
            gross_return_pct = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
            net_pnl = (price * qty) - (entry_price * qty) - entry_fees - exit_fees
            net_return_pct = (net_pnl / notional) * 100 if notional > 0 else 0.0

            self.trades.append(
                {
                    "entry_date": self.current_entry["entry_date"],
                    "exit_date": executed_date,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(price, 4),
                    "qty": qty,
                    "gross_return_pct": round(gross_return_pct, 4),
                    "net_return_pct": round(net_return_pct, 4),
                    "fees": round(entry_fees + exit_fees, 6),
                    "exit_reason": str(self.pending_exit_reason or "signal_exit"),
                }
            )
            self.current_entry = None

        self.pending_order = None
        self.pending_exit_reason = None


class _MACrossStrategy(_BaseTrackedStrategy):
    """Moving-average cross strategy with configurable window."""

    params = dict(initial_capital=100000.0, trade_start=None, trade_end=None, ma_window=20)

    def __init__(self):  # type: ignore[override]
        super().__init__()
        self.ma_window = max(5, int(getattr(self.p, "ma_window", 20) or 20))
        self.moving_average = bt.indicators.SimpleMovingAverage(self.data.close, period=self.ma_window)
        self.cross_up = bt.indicators.CrossUp(self.data.close, self.moving_average)
        self.cross_down = bt.indicators.CrossDown(self.data.close, self.moving_average)

    def _should_buy(self) -> bool:
        if len(self) < self.ma_window:
            return False

        ma_value = float(self.moving_average[0])
        if not math.isfinite(ma_value):
            return False

        if self._is_window_start_bar():
            close_value = float(self.data.close[0])
            return math.isfinite(close_value) and close_value > ma_value

        return len(self) >= self.ma_window + 1 and float(self.cross_up[0]) > 0

    def _should_sell(self) -> bool:
        return len(self) >= self.ma_window + 1 and float(self.cross_down[0]) > 0

    def _signal_exit_reason(self) -> str:
        return f"ma{self.ma_window}_cross_down"


class _RSIThresholdStrategy(_BaseTrackedStrategy):
    """RSI threshold strategy with configurable period and thresholds."""

    params = dict(
        initial_capital=100000.0,
        trade_start=None,
        trade_end=None,
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70,
    )

    def __init__(self):  # type: ignore[override]
        super().__init__()
        self.rsi_period = max(5, int(getattr(self.p, "rsi_period", 14) or 14))
        self.oversold_threshold = float(getattr(self.p, "oversold_threshold", 30) or 30)
        self.overbought_threshold = float(getattr(self.p, "overbought_threshold", 70) or 70)
        self.rsi = bt.indicators.RSI_Safe(self.data.close, period=self.rsi_period)

    def _should_buy(self) -> bool:
        if len(self) < self.rsi_period + 1:
            return False
        value = float(self.rsi[0])
        return math.isfinite(value) and value < self.oversold_threshold

    def _should_sell(self) -> bool:
        if len(self) < self.rsi_period + 1:
            return False
        value = float(self.rsi[0])
        return math.isfinite(value) and value > self.overbought_threshold

    def _signal_exit_reason(self) -> str:
        return f"rsi_gt_{int(self.overbought_threshold)}"


class StrategyBacktestService:
    """Run MA20/RSI14 date-range backtests with strict backtrader dependency."""

    def __init__(self, fetcher_manager: Optional[DataFetcherManager] = None):
        self.fetcher = fetcher_manager or DataFetcherManager()

    @staticmethod
    def _round(value: float, digits: int = 4) -> float:
        factor = 10 ** digits
        return math.floor(value * factor + 0.5) / factor

    @staticmethod
    def _to_date(value: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        text = str(value).strip()
        if not text:
            return None
        try:
            return date.fromisoformat(text[:10])
        except Exception:
            return None

    @staticmethod
    def _to_float(value: Any, fallback: float) -> float:
        try:
            number = float(value)
        except Exception:
            return fallback
        if not math.isfinite(number):
            return fallback
        return number

    @staticmethod
    def _normalize_code(raw: Any) -> str:
        code = canonical_stock_code(str(raw or ""))
        return normalize_stock_code(code)

    @staticmethod
    def _normalize_template_params(template_code: str, raw_params: Any) -> Dict[str, float]:
        source = raw_params if isinstance(raw_params, dict) else {}

        if template_code == "ma_cross":
            ma_window = int(StrategyBacktestService._to_float(source.get("maWindow"), 20))
            if ma_window < 5 or ma_window > 120:
                raise ValueError("validation_error: maWindow must be between 5 and 120")
            return {
                "maWindow": float(ma_window),
            }

        if template_code == "rsi_threshold":
            rsi_period = int(StrategyBacktestService._to_float(source.get("rsiPeriod"), 14))
            oversold_threshold = int(StrategyBacktestService._to_float(source.get("oversoldThreshold"), 30))
            overbought_threshold = int(StrategyBacktestService._to_float(source.get("overboughtThreshold"), 70))

            if rsi_period < 5 or rsi_period > 60:
                raise ValueError("validation_error: rsiPeriod must be between 5 and 60")
            if oversold_threshold < 1 or oversold_threshold > 49:
                raise ValueError("validation_error: oversoldThreshold must be between 1 and 49")
            if overbought_threshold < 51 or overbought_threshold > 99:
                raise ValueError("validation_error: overboughtThreshold must be between 51 and 99")
            if oversold_threshold >= overbought_threshold:
                raise ValueError("validation_error: oversoldThreshold must be less than overboughtThreshold")

            return {
                "rsiPeriod": float(rsi_period),
                "oversoldThreshold": float(oversold_threshold),
                "overboughtThreshold": float(overbought_threshold),
            }

        raise ValueError(f"validation_error: unsupported template_code={template_code}")

    def _normalize_strategy_definitions(self, payload: Dict[str, Any]) -> List[StrategyDefinition]:
        raw_strategies = payload.get("strategies")
        if isinstance(raw_strategies, list) and raw_strategies:
            definitions: List[StrategyDefinition] = []
            seen: set[tuple[Optional[int], str, str]] = set()
            for item in raw_strategies:
                if not isinstance(item, dict):
                    raise ValueError("validation_error: strategies must be objects")
                template_code = str(item.get("template_code") or "").strip()
                if template_code not in TEMPLATE_NAMES:
                    raise ValueError(f"validation_error: unsupported template_code={template_code}")
                strategy_name = str(item.get("strategy_name") or "").strip() or TEMPLATE_NAMES[template_code]
                strategy_id_raw = item.get("strategy_id")
                strategy_id = int(strategy_id_raw) if strategy_id_raw is not None else None
                if strategy_id is not None and strategy_id <= 0:
                    raise ValueError("validation_error: strategy_id must be positive integer")
                params = self._normalize_template_params(template_code, item.get("params"))
                dedupe_key = (strategy_id, strategy_name, template_code)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                definitions.append(
                    StrategyDefinition(
                        strategy_id=strategy_id,
                        strategy_name=strategy_name[:64],
                        template_code=template_code,
                        template_name=TEMPLATE_NAMES[template_code],
                        params=params,
                    )
                )

            if definitions:
                return definitions

        raw_codes = payload.get("strategy_codes")
        values = raw_codes if isinstance(raw_codes, list) else LEGACY_STRATEGY_CODES
        selected = [str(item).strip() for item in values if str(item).strip() in LEGACY_STRATEGY_TEMPLATE_MAP]
        deduped = list(dict.fromkeys(selected))
        if not deduped:
            deduped = list(LEGACY_STRATEGY_CODES)

        definitions = []
        for strategy_code in deduped:
            resolved = LEGACY_STRATEGY_TEMPLATE_MAP[strategy_code]
            definitions.append(
                StrategyDefinition(
                    strategy_id=None,
                    strategy_name=str(resolved["strategy_name"]),
                    template_code=str(resolved["template_code"]),
                    template_name=TEMPLATE_NAMES[str(resolved["template_code"])],
                    params={key: float(value) for key, value in dict(resolved["params"]).items()},
                )
            )
        return definitions

    def _validate_params(self, payload: Dict[str, Any]) -> StrategyRunParams:
        if bt is None:
            raise ValueError("backtrader_not_available: install backtrader to run strategy backtest")

        code = self._normalize_code(payload.get("code"))
        if not code:
            raise ValueError("validation_error: code is required")

        start_date = self._to_date(payload.get("start_date"))
        end_date = self._to_date(payload.get("end_date"))
        if start_date is None or end_date is None:
            raise ValueError("validation_error: start_date and end_date are required")
        if start_date > end_date:
            raise ValueError("validation_error: start_date must be <= end_date")

        initial_capital = max(1.0, self._to_float(payload.get("initial_capital"), 100000.0))
        commission_rate = max(0.0, self._to_float(payload.get("commission_rate"), 0.0003))
        slippage_bps = max(0.0, self._to_float(payload.get("slippage_bps"), 2.0))

        return StrategyRunParams(
            code=code,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_bps=slippage_bps,
        )

    def _load_daily_frame(self, params: StrategyRunParams) -> tuple[pd.DataFrame, pd.DataFrame]:
        warmup_start = params.start_date - timedelta(days=WARMUP_CALENDAR_DAYS)
        frame, _ = self.fetcher.get_daily_data(
            params.code,
            start_date=warmup_start.isoformat(),
            end_date=params.end_date.isoformat(),
            days=5000,
        )

        if frame is None or frame.empty:
            raise ValueError("insufficient_data: no daily bars in requested range")

        rows: List[Dict[str, Any]] = []
        for _, row in frame.iterrows():
            day = self._to_date(row.get("date"))
            if day is None:
                continue
            open_price = self._to_float(row.get("open"), float("nan"))
            close_price = self._to_float(row.get("close"), float("nan"))
            high_price = self._to_float(row.get("high"), float("nan"))
            low_price = self._to_float(row.get("low"), float("nan"))
            volume = self._to_float(row.get("volume"), 1.0)

            if not math.isfinite(close_price) or close_price <= 0:
                continue
            if not math.isfinite(open_price) or open_price <= 0:
                open_price = close_price
            if not math.isfinite(high_price) or high_price <= 0:
                high_price = max(open_price, close_price)
            if not math.isfinite(low_price) or low_price <= 0:
                low_price = min(open_price, close_price)

            rows.append(
                {
                    "date": pd.Timestamp(day),
                    "open": float(open_price),
                    "high": float(max(high_price, open_price, close_price)),
                    "low": float(min(low_price, open_price, close_price)),
                    "close": float(close_price),
                    "volume": float(max(volume, 1.0)),
                }
            )

        if not rows:
            raise ValueError("insufficient_data: no valid OHLC bars after normalization")

        normalized = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date")
        trade_window = normalized[
            (normalized["date"] >= pd.Timestamp(params.start_date))
            & (normalized["date"] <= pd.Timestamp(params.end_date))
        ]
        if trade_window.empty:
            raise ValueError("insufficient_data: no trading bars inside date range")

        trade_end = pd.Timestamp(trade_window["date"].iloc[-1])
        prepared = normalized[normalized["date"] <= trade_end]
        if prepared.empty:
            raise ValueError("insufficient_data: no bars available after warmup preparation")

        return prepared.set_index("date"), trade_window.set_index("date")

    @staticmethod
    def _compute_benchmark_equity(frame: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        first_close = float(frame["close"].iloc[0])
        shares = int(initial_capital // first_close) if first_close > 0 else 0
        cash = initial_capital - shares * first_close

        points: List[Dict[str, Any]] = []
        for idx, row in frame.iterrows():
            close_price = float(row["close"])
            equity = cash + shares * close_price
            points.append({"trade_date": idx.date().isoformat(), "benchmark_equity": equity})

        final_equity = points[-1]["benchmark_equity"] if points else initial_capital
        total_return_pct = ((final_equity / initial_capital) - 1) * 100 if initial_capital > 0 else 0.0
        return {
            "points": points,
            "initial_equity": initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
        }

    @staticmethod
    def _merge_equity_with_benchmark(
        equity_points: Iterable[Dict[str, Any]],
        benchmark_points: Iterable[Dict[str, Any]],
        initial_capital: float,
    ) -> List[Dict[str, Any]]:
        benchmark_rows = [
            {
                "trade_date": str(point.get("trade_date") or ""),
                "benchmark_equity": float(point.get("benchmark_equity") or 0.0),
            }
            for point in benchmark_points
            if str(point.get("trade_date") or "")
        ]
        benchmark_map = {row["trade_date"]: row["benchmark_equity"] for row in benchmark_rows}
        equity_map = {
            str(point.get("trade_date") or ""): float(point.get("equity") or 0.0)
            for point in equity_points
            if str(point.get("trade_date") or "")
        }

        merged: List[Dict[str, Any]] = []
        peak = None
        latest_equity = float(initial_capital)
        for point in benchmark_rows:
            trade_date = point["trade_date"]
            if trade_date in equity_map:
                latest_equity = float(equity_map[trade_date])
            equity = latest_equity
            peak = equity if peak is None else max(peak, equity)
            drawdown = ((equity / peak) - 1) * 100 if peak and peak > 0 else 0.0
            merged.append(
                {
                    "trade_date": trade_date,
                    "equity": round(equity, 4),
                    "drawdown_pct": round(drawdown, 4),
                    "benchmark_equity": round(float(benchmark_map.get(trade_date) or 0.0), 4),
                }
            )
        return merged

    @staticmethod
    def _compute_sharpe(equity_points: List[Dict[str, Any]]) -> Optional[float]:
        if len(equity_points) < 3:
            return None

        returns: List[float] = []
        previous = None
        for point in equity_points:
            equity = float(point.get("equity") or 0.0)
            if previous is not None and previous > 0:
                returns.append((equity / previous) - 1)
            previous = equity

        if len(returns) < 2:
            return None

        mean = sum(returns) / len(returns)
        variance = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        if std <= 0:
            return None

        return float((mean / std) * math.sqrt(252))

    def _run_single_strategy(
        self,
        strategy: StrategyDefinition,
        frame: pd.DataFrame,
        trade_window: pd.DataFrame,
        params: StrategyRunParams,
        benchmark: Dict[str, Any],
    ) -> Dict[str, Any]:
        if strategy.template_code == "ma_cross":
            strategy_cls = _MACrossStrategy
            strategy_kwargs = {
                "ma_window": int(strategy.params.get("maWindow", 20)),
            }
        elif strategy.template_code == "rsi_threshold":
            strategy_cls = _RSIThresholdStrategy
            strategy_kwargs = {
                "rsi_period": int(strategy.params.get("rsiPeriod", 14)),
                "oversold_threshold": float(strategy.params.get("oversoldThreshold", 30)),
                "overbought_threshold": float(strategy.params.get("overboughtThreshold", 70)),
            }
        else:
            raise ValueError(f"validation_error: unsupported template_code={strategy.template_code}")

        cerebro = bt.Cerebro(stdstats=False)
        data_feed = bt.feeds.PandasData(dataname=frame)
        cerebro.adddata(data_feed)
        trade_start = trade_window.index[0].date()
        trade_end = trade_window.index[-1].date()
        cerebro.addstrategy(
            strategy_cls,
            initial_capital=params.initial_capital,
            trade_start=trade_start,
            trade_end=trade_end,
            **strategy_kwargs,
        )
        cerebro.broker.setcash(float(params.initial_capital))
        cerebro.broker.setcommission(commission=float(params.commission_rate))
        if params.slippage_bps > 0:
            cerebro.broker.set_slippage_perc(
                perc=float(params.slippage_bps) / 10000.0,
                slip_open=True,
                slip_match=True,
                slip_limit=True,
            )

        # Run in step mode to avoid backtrader runonce short-sample index errors
        # when indicator period is longer than available bars.
        result = cerebro.run(runonce=False)
        if not result:
            raise ValueError(f"backtest_failed: empty result for template={strategy.template_code}")
        strategy_state = result[0]

        raw_points = strategy_state.equity_points if isinstance(strategy_state.equity_points, list) else []
        if raw_points:
            last_date = trade_window.index[-1].date().isoformat()
            if str(raw_points[-1].get("trade_date")) != last_date:
                raw_points.append(
                    {
                        "trade_date": last_date,
                        "equity": float(cerebro.broker.getvalue()),
                    }
                )
        merged_equity = self._merge_equity_with_benchmark(
            raw_points,
            benchmark.get("points") or [],
            params.initial_capital,
        )

        trades = strategy_state.trades if isinstance(strategy_state.trades, list) else []
        win_trades = sum(1 for trade in trades if float(trade.get("net_return_pct") or 0.0) > 0)
        loss_trades = sum(1 for trade in trades if float(trade.get("net_return_pct") or 0.0) < 0)
        avg_trade_return_pct = (
            sum(float(trade.get("net_return_pct") or 0.0) for trade in trades) / len(trades)
            if trades
            else None
        )

        final_equity = float(merged_equity[-1]["equity"]) if merged_equity else params.initial_capital
        total_return_pct = ((final_equity / params.initial_capital) - 1) * 100 if params.initial_capital > 0 else 0.0
        benchmark_return_pct = float(benchmark.get("total_return_pct") or 0.0)
        excess_return_pct = total_return_pct - benchmark_return_pct
        max_drawdown_pct = min((float(point.get("drawdown_pct") or 0.0) for point in merged_equity), default=0.0)
        sharpe_ratio = self._compute_sharpe(merged_equity)
        margin_rejections = int(getattr(strategy_state, "margin_rejections", 0) or 0)
        entry_signal_count = int(getattr(strategy_state, "entry_signal_count", 0) or 0)
        exit_signal_count = int(getattr(strategy_state, "exit_signal_count", 0) or 0)

        no_trade_reason = None
        no_trade_reason_detail = None
        if len(trades) == 0:
            if margin_rejections > 0:
                no_trade_reason = "entry_rejected_margin"
                no_trade_reason_detail = (
                    f"entry_signals={entry_signal_count}, margin_rejections={margin_rejections}"
                )
            elif entry_signal_count == 0:
                no_trade_reason = "no_entry_signal"
                no_trade_reason_detail = "no entry signal triggered in selected window"
            elif exit_signal_count == 0:
                no_trade_reason = "no_exit_signal"
                no_trade_reason_detail = (
                    f"entry_signals={entry_signal_count}, no exit signal before window end"
                )
            else:
                no_trade_reason = "no_completed_trade"
                no_trade_reason_detail = (
                    f"entry_signals={entry_signal_count}, exit_signals={exit_signal_count}, trades=0"
                )

        metrics = {
            "initial_capital": self._round(params.initial_capital, 2),
            "final_equity": self._round(final_equity, 4),
            "total_return_pct": self._round(total_return_pct, 4),
            "benchmark_return_pct": self._round(benchmark_return_pct, 4),
            "excess_return_pct": self._round(excess_return_pct, 4),
            "max_drawdown_pct": self._round(max_drawdown_pct, 4),
            "total_trades": len(trades),
            "win_trades": win_trades,
            "loss_trades": loss_trades,
            "win_rate_pct": self._round((win_trades / len(trades)) * 100, 2) if trades else None,
            "avg_trade_return_pct": self._round(avg_trade_return_pct, 4) if avg_trade_return_pct is not None else None,
            "sharpe_ratio": self._round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
            "completed_trading_days": len(merged_equity),
            "margin_rejections": margin_rejections,
            "entry_signal_count": entry_signal_count,
            "exit_signal_count": exit_signal_count,
            "no_trade_reason": no_trade_reason,
            "no_trade_reason_detail": no_trade_reason_detail,
        }

        params_payload = {
            "signal_profile": "template_v1",
            "template_code": strategy.template_code,
            **strategy.params,
            "initial_capital": self._round(params.initial_capital, 2),
            "commission_rate": self._round(params.commission_rate, 6),
            "slippage_bps": self._round(params.slippage_bps, 4),
            "entry_rule": (
                f"window_start: close[t] > ma{int(strategy.params.get('maWindow', 20))}[t] => t+1 open buy; otherwise cross above MA => t+1 open buy"
                if strategy.template_code == "ma_cross"
                else (
                    f"rsi{int(strategy.params.get('rsiPeriod', 14))}[t] < {int(strategy.params.get('oversoldThreshold', 30))}, t+1 open buy"
                )
            ),
            "exit_rule": (
                f"cross below MA{int(strategy.params.get('maWindow', 20))}, t+1 open sell"
                if strategy.template_code == "ma_cross"
                else (
                    f"rsi{int(strategy.params.get('rsiPeriod', 14))}[t] > {int(strategy.params.get('overboughtThreshold', 70))}, t+1 open sell"
                )
            ),
        }

        return {
            "strategy_id": strategy.strategy_id,
            "strategy_code": strategy.template_code,
            "strategy_name": strategy.strategy_name,
            "template_code": strategy.template_code,
            "template_name": strategy.template_name,
            "strategy_version": "v1",
            "params": params_payload,
            "metrics": metrics,
            "benchmark": {
                "initial_equity": self._round(float(benchmark.get("initial_equity") or params.initial_capital), 4),
                "final_equity": self._round(float(benchmark.get("final_equity") or params.initial_capital), 4),
                "total_return_pct": self._round(float(benchmark.get("total_return_pct") or 0.0), 4),
            },
            "trades": trades,
            "equity": merged_equity,
        }

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        strategy_definitions = self._normalize_strategy_definitions(payload)
        params = self._validate_params(payload)
        frame, trade_window = self._load_daily_frame(params)

        effective_start = trade_window.index[0].date().isoformat()
        effective_end = trade_window.index[-1].date().isoformat()

        benchmark = self._compute_benchmark_equity(trade_window, params.initial_capital)
        items = [self._run_single_strategy(strategy, frame, trade_window, params, benchmark) for strategy in strategy_definitions]

        return {
            "engine_version": "backtrader_v1",
            "code": params.code,
            "requested_range": {
                "start_date": params.start_date.isoformat(),
                "end_date": params.end_date.isoformat(),
            },
            "effective_range": {
                "start_date": effective_start,
                "end_date": effective_end,
            },
            "items": items,
        }


_strategy_backtest_service: Optional[StrategyBacktestService] = None


def get_strategy_backtest_service() -> StrategyBacktestService:
    global _strategy_backtest_service
    if _strategy_backtest_service is None:
        _strategy_backtest_service = StrategyBacktestService()
    return _strategy_backtest_service
