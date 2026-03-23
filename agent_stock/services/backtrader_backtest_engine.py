# pyright: reportOptionalMemberAccess=false, reportCallIssue=false
# -*- coding: utf-8 -*-
"""回测服务使用的单窗口交易回放引擎。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Iterable, cast


import pandas as pd

try:
    import backtrader as bt  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    bt = None

bt = cast(Any, bt)


def _new_cerebro() -> Any:
    """创建 Backtrader 引擎实例，隔离动态三方 API 的静态分析噪音。"""
    bt_module: Any = bt
    return bt_module.Cerebro(stdstats=False)


def _new_pandas_data_feed(frame: pd.DataFrame) -> Any:
    """基于 pandas DataFrame 创建 Backtrader 数据源。"""
    bt_module: Any = bt
    return bt_module.feeds.PandasData(dataname=frame)


def _strategy_bar_count(strategy: Any) -> int:
    """读取策略当前可见 bar 数量。"""
    return int(len(strategy))


@dataclass(frozen=True)
class ReplayBar:
    """回放引擎使用的标准化日线 OHLC 数据。"""

    day: date
    high: float | None
    low: float | None
    close: float | None


def _to_float(value: Any) -> float | None:
    """安全地将输入转换为有限浮点数。"""
    if value is None:
        return None
    try:
        num = float(value)
    except Exception:
        return None
    if not math.isfinite(num):
        return None
    return num


def _safe_bar_close(bar: ReplayBar) -> float | None:
    """读取收盘价，并统一做数值清洗。"""
    return _to_float(bar.close)


def _safe_bar_high(bar: ReplayBar) -> float | None:
    """读取最高价；若缺失则回退到收盘价。"""
    high = _to_float(bar.high)
    if high is not None:
        return high
    return _safe_bar_close(bar)


def _safe_bar_low(bar: ReplayBar) -> float | None:
    """读取最低价；若缺失则回退到收盘价。"""
    low = _to_float(bar.low)
    if low is not None:
        return low
    return _safe_bar_close(bar)


def _manual_target_exit(
    forward_bars: Iterable[ReplayBar],
    stop_loss: float | None,
    take_profit: float | None,
    fallback_close: float | None,
) -> tuple[float | None, str]:
    """不用 Backtrader 时，按止盈止损规则手工推导离场价格。"""
    if stop_loss is None and take_profit is None:
        return fallback_close, "window_end"

    exit_price = fallback_close
    exit_reason = "window_end"
    for bar in forward_bars:
        low = _safe_bar_low(bar)
        high = _safe_bar_high(bar)
        # 同一根 K 线同时命中止盈止损时，保守地按止损处理并标记为歧义。
        stop_hit = stop_loss is not None and low is not None and low <= stop_loss
        take_hit = take_profit is not None and high is not None and high >= take_profit
        if not stop_hit and not take_hit:
            continue
        if stop_hit and take_hit:
            return stop_loss, "ambiguous_stop_loss"
        if stop_hit:
            return stop_loss, "stop_loss"
        return take_profit, "take_profit"
    return exit_price, exit_reason


if bt is not None:
    class _SingleWindowStrategy(bt.Strategy):
        """在评估窗口内回放一笔只做多交易。"""

        params = dict(
            stop_loss=None,
            take_profit=None,
            total_bars=0,
        )

        def __init__(self):
            """初始化订单句柄和进出场记录。"""
            self.entry_order = None
            self.stop_order = None
            self.take_order = None
            self.final_close_order = None

            self.entry_price = None
            self.exit_price = None
            self.exit_reason = "window_end"

        def next(self):
            """在首根 bar 建仓，并在窗口结束时兜底平仓。"""
            bar_index = _strategy_bar_count(self) - 1
            if bar_index == 0 and self.entry_order is None and not self.position:
                self.entry_order = self.buy(size=1)
                return

            if not self.position:
                return

            if _strategy_bar_count(self) >= int(self.p.total_bars or 0):
                if self.stop_order is not None and self.stop_order.alive():
                    self.cancel(self.stop_order)
                if self.take_order is not None and self.take_order.alive():
                    self.cancel(self.take_order)
                if self.final_close_order is None:
                    self.exit_reason = "window_end"
                    self.final_close_order = self.close(size=1)

        def notify_order(self, order):
            """记录成交后的进出场价格与离场原因。"""
            if order.status != order.Completed:
                return

            if order.isbuy():
                self.entry_price = float(order.executed.price)
                if self.p.stop_loss is not None:
                    self.stop_order = self.sell(
                        size=1,
                        exectype=bt.Order.Stop,
                        price=float(self.p.stop_loss),
                    )
                if self.p.take_profit is not None:
                    self.take_order = self.sell(
                        size=1,
                        exectype=bt.Order.Limit,
                        price=float(self.p.take_profit),
                        oco=self.stop_order if self.stop_order is not None else None,
                    )
                return

            self.exit_price = float(order.executed.price)
            if order is self.stop_order:
                self.exit_reason = "stop_loss"
            elif order is self.take_order:
                self.exit_reason = "take_profit"
            elif order is self.final_close_order:
                self.exit_reason = "window_end"
            elif self.exit_reason not in {"stop_loss", "take_profit"}:
                self.exit_reason = "window_end"


class BacktraderBacktestEngine:
    """优先使用 Backtrader，失败时回退到确定性模拟的回放引擎。"""

    def __init__(self, *, commission_rate: float, slippage_bps: float):
        """初始化手续费、滑点参数，并记录 Backtrader 可用性。"""
        self.commission_rate = max(0.0, float(commission_rate))
        self.slippage_bps = max(0.0, float(slippage_bps))
        self.available = bt is not None

    @staticmethod
    def _round(value: float, digits: int = 4) -> float:
        """按传统四舍五入方式保留小数位。"""
        factor = 10 ** digits
        return math.floor(value * factor + 0.5) / factor

    @staticmethod
    def _normalize_forward(forward_bars: Iterable[Any]) -> list[ReplayBar]:
        """将任意前瞻 bar 序列标准化并按日期排序。"""
        bars: list[ReplayBar] = []
        for item in forward_bars:
            day = getattr(item, "day", None)
            if not isinstance(day, date):
                continue
            bars.append(
                ReplayBar(
                    day=day,
                    high=_to_float(getattr(item, "high", None)),
                    low=_to_float(getattr(item, "low", None)),
                    close=_to_float(getattr(item, "close", None)),
                ),
            )
        bars.sort(key=lambda row: row.day)
        return bars

    def _simulate_without_backtrader(
        self,
        *,
        start_price: float,
        forward_bars: list[ReplayBar],
        stop_loss: float | None,
        take_profit: float | None,
    ) -> dict[str, Any]:
        """在没有 Backtrader 时，用纯规则模拟一笔交易。"""
        end_close = _safe_bar_close(forward_bars[-1]) if forward_bars else None
        exit_price, exit_reason = _manual_target_exit(forward_bars, stop_loss, take_profit, end_close)
        if exit_price is None:
            return {
                "entry_price": start_price,
                "exit_price": None,
                "exit_reason": exit_reason,
                "return_pct": None,
                "used_backtrader": False,
            }

        # 手工回放也要统一计入滑点与双边手续费，避免和引擎路径口径不一致。
        slip_ratio = self.slippage_bps / 10000.0
        buy_price = start_price * (1.0 + slip_ratio)
        sell_price = max(0.0, float(exit_price) * (1.0 - slip_ratio))
        buy_fee = buy_price * self.commission_rate
        sell_fee = sell_price * self.commission_rate
        net_return_pct = ((sell_price - buy_price - buy_fee - sell_fee) / buy_price) * 100 if buy_price > 0 else None

        return {
            "entry_price": self._round(buy_price),
            "exit_price": self._round(sell_price),
            "exit_reason": exit_reason,
            "return_pct": self._round(net_return_pct) if net_return_pct is not None else None,
            "used_backtrader": False,
        }

    def _simulate_with_backtrader(
        self,
        *,
        analysis_date: date,
        start_price: float,
        forward_bars: list[ReplayBar],
        stop_loss: float | None,
        take_profit: float | None,
    ) -> dict[str, Any]:
        """使用 Backtrader 回放一笔交易，并在异常时回退到手工逻辑。"""
        if bt is None:
            return self._simulate_without_backtrader(
                start_price=start_price,
                forward_bars=forward_bars,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        # 在分析日补一根 合成 bar，使首笔买入能严格对齐分析日收盘建仓。
        rows = [
            {
                "date": pd.Timestamp(analysis_date),
                "open": start_price,
                "high": start_price,
                "low": start_price,
                "close": start_price,
                "volume": 1.0,
            }
        ]

        for bar in forward_bars:
            close = _safe_bar_close(bar)
            if close is None:
                continue
            high = _safe_bar_high(bar) or close
            low = _safe_bar_low(bar) or close
            rows.append(
                {
                    "date": pd.Timestamp(bar.day),
                    "open": close,
                    "high": max(high, close),
                    "low": min(low, close),
                    "close": close,
                    "volume": 1.0,
                },
            )

        if len(rows) <= 1:
            return self._simulate_without_backtrader(
                start_price=start_price,
                forward_bars=forward_bars,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        frame = pd.DataFrame(rows).set_index("date")
        cerebro = _new_cerebro()
        data_feed = _new_pandas_data_feed(frame)
        cerebro.adddata(data_feed)
        cerebro.addstrategy(
            _SingleWindowStrategy,
            stop_loss=stop_loss,
            take_profit=take_profit,
            total_bars=len(frame.index),
        )
        cerebro.broker.setcash(1_000_000.0)
        cerebro.broker.setcommission(commission=self.commission_rate)
        if self.slippage_bps > 0:
            cerebro.broker.set_slippage_perc(
                perc=self.slippage_bps / 10000.0,
                slip_open=True,
                slip_match=True,
                slip_limit=True,
            )
        # 强制在当前 bar 收盘执行市价单，和分析日收盘建仓的口径保持一致。
        cerebro.broker.set_coc(True)

        strategies = cerebro.run()
        if not strategies:
            return self._simulate_without_backtrader(
                start_price=start_price,
                forward_bars=forward_bars,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        strategy = strategies[0]
        entry_price = _to_float(getattr(strategy, "entry_price", None))
        exit_price = _to_float(getattr(strategy, "exit_price", None))
        exit_reason = str(getattr(strategy, "exit_reason", "window_end") or "window_end")

        if entry_price is None:
            entry_price = start_price
        if exit_price is None:
            # 如果 broker 没有按预期平仓，则回退到确定性目标选择逻辑。
            fallback = self._simulate_without_backtrader(
                start_price=start_price,
                forward_bars=forward_bars,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            return {
                **fallback,
                "used_backtrader": True,
            }

        buy_fee = entry_price * self.commission_rate
        sell_fee = exit_price * self.commission_rate
        return_pct = (
            ((exit_price - entry_price - buy_fee - sell_fee) / entry_price) * 100
            if entry_price > 0
            else None
        )
        return {
            "entry_price": self._round(entry_price),
            "exit_price": self._round(exit_price),
            "exit_reason": exit_reason,
            "return_pct": self._round(return_pct) if return_pct is not None else None,
            "used_backtrader": True,
        }

    def simulate_long_trade(
        self,
        *,
        analysis_date: date,
        start_price: float | None,
        forward_bars: Iterable[Any],
        stop_loss: float | None,
        take_profit: float | None,
    ) -> dict[str, Any]:
        """统一入口：回放一笔只做多交易并返回标准化结果。"""
        start = _to_float(start_price)
        if start is None or start <= 0:
            return {
                "entry_price": None,
                "exit_price": None,
                "exit_reason": "error",
                "return_pct": None,
                "used_backtrader": self.available,
            }

        normalized = self._normalize_forward(forward_bars)
        if not normalized:
            return {
                "entry_price": start,
                "exit_price": None,
                "exit_reason": "insufficient_data",
                "return_pct": None,
                "used_backtrader": self.available,
            }

        if self.available:
            return self._simulate_with_backtrader(
                analysis_date=analysis_date,
                start_price=start,
                forward_bars=normalized,
                stop_loss=_to_float(stop_loss),
                take_profit=_to_float(take_profit),
            )

        return self._simulate_without_backtrader(
            start_price=start,
            forward_bars=normalized,
            stop_loss=_to_float(stop_loss),
            take_profit=_to_float(take_profit),
        )
