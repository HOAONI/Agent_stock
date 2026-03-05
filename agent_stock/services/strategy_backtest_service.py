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

STRATEGY_CODES = ["ma20_trend", "rsi14_mean_reversion"]
STRATEGY_NAMES: Dict[str, str] = {
    "ma20_trend": "MA20 Trend",
    "rsi14_mean_reversion": "RSI14 Mean Reversion",
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


class _MA20TrendStrategy(_BaseTrackedStrategy):
    """MA20 cross strategy: up-cross enter, down-cross exit."""

    def __init__(self):  # type: ignore[override]
        super().__init__()
        self.ma20 = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        self.cross_up = bt.indicators.CrossUp(self.data.close, self.ma20)
        self.cross_down = bt.indicators.CrossDown(self.data.close, self.ma20)

    def _should_buy(self) -> bool:
        if len(self) < 20:
            return False

        ma20_value = float(self.ma20[0])
        if not math.isfinite(ma20_value):
            return False

        if self._is_window_start_bar():
            close_value = float(self.data.close[0])
            return math.isfinite(close_value) and close_value > ma20_value

        return len(self) >= 21 and float(self.cross_up[0]) > 0

    def _should_sell(self) -> bool:
        return len(self) >= 21 and float(self.cross_down[0]) > 0

    def _signal_exit_reason(self) -> str:
        return "ma20_cross_down"


class _RSI14MeanReversionStrategy(_BaseTrackedStrategy):
    """RSI14 strategy: RSI<30 enter, RSI>70 exit."""

    def __init__(self):  # type: ignore[override]
        super().__init__()
        self.rsi14 = bt.indicators.RSI_Safe(self.data.close, period=14)

    def _should_buy(self) -> bool:
        if len(self) < 15:
            return False
        value = float(self.rsi14[0])
        return math.isfinite(value) and value < 30

    def _should_sell(self) -> bool:
        if len(self) < 15:
            return False
        value = float(self.rsi14[0])
        return math.isfinite(value) and value > 70

    def _signal_exit_reason(self) -> str:
        return "rsi14_gt_70"


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
    def _parse_strategy_codes(raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return list(STRATEGY_CODES)
        values = [str(item).strip() for item in raw]
        selected = [item for item in values if item in STRATEGY_CODES]
        dedup = list(dict.fromkeys(selected))
        return dedup if dedup else list(STRATEGY_CODES)

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
        strategy_code: str,
        frame: pd.DataFrame,
        trade_window: pd.DataFrame,
        params: StrategyRunParams,
        benchmark: Dict[str, Any],
    ) -> Dict[str, Any]:
        strategy_cls = _MA20TrendStrategy if strategy_code == "ma20_trend" else _RSI14MeanReversionStrategy

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
            raise ValueError(f"backtest_failed: empty result for strategy={strategy_code}")
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
            "signal_profile": "classic",
            "initial_capital": self._round(params.initial_capital, 2),
            "commission_rate": self._round(params.commission_rate, 6),
            "slippage_bps": self._round(params.slippage_bps, 4),
            "entry_rule": (
                "window_start: close[t] > ma20[t] => t+1 open buy; otherwise close[t-1] <= ma20[t-1] and close[t] > ma20[t], t+1 open buy"
                if strategy_code == "ma20_trend"
                else "rsi14[t] < 30, t+1 open buy"
            ),
            "exit_rule": (
                "close[t-1] >= ma20[t-1] and close[t] < ma20[t], t+1 open sell"
                if strategy_code == "ma20_trend"
                else "rsi14[t] > 70, t+1 open sell"
            ),
        }

        return {
            "strategy_code": strategy_code,
            "strategy_name": STRATEGY_NAMES[strategy_code],
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
        params = self._validate_params(payload)
        strategy_codes = self._parse_strategy_codes(payload.get("strategy_codes"))
        frame, trade_window = self._load_daily_frame(params)

        effective_start = trade_window.index[0].date().isoformat()
        effective_end = trade_window.index[-1].date().isoformat()

        benchmark = self._compute_benchmark_equity(trade_window, params.initial_capital)
        items = [self._run_single_strategy(code, frame, trade_window, params, benchmark) for code in strategy_codes]

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
