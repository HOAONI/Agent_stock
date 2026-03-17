# -*- coding: utf-8 -*-
"""为 Backend_stock 提供内部回测计算能力的服务。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from data_provider import DataFetcherManager
from data_provider.base import canonical_stock_code, normalize_stock_code
from agent_stock.services.backtrader_backtest_engine import BacktraderBacktestEngine

OVERALL_SENTINEL_CODE = "__overall__"

BACKTEST_COMPARE_STRATEGY_CODES = ["agent_v1", "ma20_trend", "rsi14_mean_reversion"]
BACKTEST_COMPARE_STRATEGY_NAMES: Dict[str, str] = {
    "agent_v1": "Agent v1",
    "ma20_trend": "MA20 Trend",
    "rsi14_mean_reversion": "RSI14 Mean Reversion",
}


@dataclass(frozen=True)
class DailyBar:
    """单个交易日的 OHLC 快照。"""

    day: date
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]


class BacktestService:
    """为内部 API 执行确定性的回测计算。"""

    bullish_keywords = ["买入", "加仓", "强烈买入", "增持", "建仓", "strong buy", "buy", "add"]
    bearish_keywords = ["卖出", "减仓", "强烈卖出", "清仓", "strong sell", "sell", "reduce"]
    hold_keywords = ["持有", "hold"]
    wait_keywords = ["观望", "等待", "wait"]
    negation_patterns = ["not", "don't", "do not", "no", "never", "avoid", "不要", "不", "别", "勿", "没有"]

    def __init__(self, fetcher_manager: Optional[DataFetcherManager] = None):
        """初始化数据获取器与单笔交易回放引擎。"""
        self.fetcher = fetcher_manager or DataFetcherManager()
        commission_rate = float(os.getenv("BACKTEST_COMMISSION_RATE", os.getenv("BACKTRADER_DEFAULT_COMMISSION", "0.0003")))
        slippage_bps = float(os.getenv("BACKTEST_SLIPPAGE_BPS", os.getenv("BACKTRADER_DEFAULT_SLIPPAGE_BPS", "2")))
        self.replay_engine = BacktraderBacktestEngine(
            commission_rate=max(0.0, commission_rate),
            slippage_bps=max(0.0, slippage_bps),
        )

    @staticmethod
    def _round(value: float, digits: int = 2) -> float:
        """按传统四舍五入方式保留指定小数位。"""
        factor = 10 ** digits
        return math.floor(value * factor + 0.5) / factor

    @staticmethod
    def _to_number(value: Any) -> Optional[float]:
        """将输入安全转换为有限浮点数。"""
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            num = float(value)
        except Exception:
            return None
        if not math.isfinite(num):
            return None
        return num

    @classmethod
    def _normalize_text(cls, value: Optional[str]) -> str:
        """将建议文本规整为便于匹配的形式。"""
        return str(value or "").strip().lower()

    @classmethod
    def _is_negated(cls, prefix: str) -> bool:
        """判断关键词前缀是否带有否定语义。"""
        stripped = prefix.rstrip()
        return any(stripped.endswith(token) for token in cls.negation_patterns)

    @classmethod
    def _matches_intent(cls, text: str, keywords: Iterable[str]) -> bool:
        """判断文本是否表达了给定关键词集合对应的意图。"""
        if not text:
            return False

        for keyword in keywords:
            if text == keyword:
                return True

        for keyword in keywords:
            idx = text.find(keyword)
            if idx == -1:
                continue
            if not cls._is_negated(text[:idx]):
                return True

        return False

    @classmethod
    def infer_direction_expected(cls, operation_advice: Optional[str]) -> str:
        """根据操作建议推断预期方向。"""
        text = cls._normalize_text(operation_advice)
        if cls._matches_intent(text, cls.bearish_keywords):
            return "down"
        if cls._matches_intent(text, cls.wait_keywords):
            return "flat"
        if cls._matches_intent(text, cls.bullish_keywords):
            return "up"
        if cls._matches_intent(text, cls.hold_keywords):
            return "not_down"
        return "flat"

    @classmethod
    def infer_position_recommendation(cls, operation_advice: Optional[str]) -> str:
        """根据操作建议推断仓位建议是做多还是空仓。"""
        text = cls._normalize_text(operation_advice)
        if cls._matches_intent(text, cls.bearish_keywords) or cls._matches_intent(text, cls.wait_keywords):
            return "cash"
        if cls._matches_intent(text, cls.bullish_keywords) or cls._matches_intent(text, cls.hold_keywords):
            return "long"
        return "cash"

    @classmethod
    def classify_outcome(cls, stock_return_pct: Optional[float], direction_expected: str, neutral_band_pct: float) -> tuple[Optional[str], Optional[bool]]:
        """根据实际涨跌幅与预期方向，判定预测结果。"""
        if stock_return_pct is None:
            return None, None

        band = abs(neutral_band_pct)
        value = stock_return_pct

        if direction_expected == "up":
            if value >= band:
                return "win", True
            if value <= -band:
                return "loss", False
            return "neutral", None

        if direction_expected == "down":
            if value <= -band:
                return "win", True
            if value >= band:
                return "loss", False
            return "neutral", None

        if direction_expected == "not_down":
            if value >= 0:
                return "win", True
            if value <= -band:
                return "loss", False
            return "neutral", None

        if abs(value) <= band:
            return "win", True
        return "loss", False

    @classmethod
    def classify_trade_outcome(cls, simulated_return_pct: Optional[float], neutral_band_pct: float) -> Optional[str]:
        """根据模拟收益率判定交易结果。"""
        if simulated_return_pct is None:
            return None
        band = abs(neutral_band_pct)
        value = simulated_return_pct
        if value >= band:
            return "win"
        if value <= -band:
            return "loss"
        return "neutral"

    @classmethod
    def evaluate_targets(
        cls,
        position: str,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        window_bars: list[DailyBar],
        end_close: Optional[float],
    ) -> Dict[str, Any]:
        """评估止盈止损在观察窗口内是否被触发。"""
        if position != "long":
            return {
                "hit_stop_loss": None,
                "hit_take_profit": None,
                "first_hit": "not_applicable",
                "first_hit_date": None,
                "first_hit_trading_days": None,
                "simulated_exit_price": None,
                "simulated_exit_reason": "cash",
            }

        if stop_loss is None and take_profit is None:
            return {
                "hit_stop_loss": None,
                "hit_take_profit": None,
                "first_hit": "neither",
                "first_hit_date": None,
                "first_hit_trading_days": None,
                "simulated_exit_price": end_close,
                "simulated_exit_reason": "window_end",
            }

        hit_stop_loss: Optional[bool] = None if stop_loss is None else False
        hit_take_profit: Optional[bool] = None if take_profit is None else False
        first_hit = "neither"
        first_hit_date: Optional[date] = None
        first_hit_trading_days: Optional[int] = None
        simulated_exit_price = end_close
        simulated_exit_reason = "window_end"

        for idx, bar in enumerate(window_bars):
            stop_hit = stop_loss is not None and bar.low is not None and bar.low <= stop_loss
            take_hit = take_profit is not None and bar.high is not None and bar.high >= take_profit

            if stop_hit:
                hit_stop_loss = True
            if take_hit:
                hit_take_profit = True

            if not stop_hit and not take_hit:
                continue

            first_hit_date = bar.day
            first_hit_trading_days = idx + 1

            if stop_hit and take_hit:
                # 单根 K 线同时触发止盈止损时无法还原真实先后顺序，标记为歧义。
                first_hit = "ambiguous"
                simulated_exit_price = stop_loss
                simulated_exit_reason = "ambiguous_stop_loss"
                break

            if stop_hit:
                first_hit = "stop_loss"
                simulated_exit_price = stop_loss
                simulated_exit_reason = "stop_loss"
                break

            first_hit = "take_profit"
            simulated_exit_price = take_profit
            simulated_exit_reason = "take_profit"
            break

        return {
            "hit_stop_loss": hit_stop_loss,
            "hit_take_profit": hit_take_profit,
            "first_hit": first_hit,
            "first_hit_date": first_hit_date.isoformat() if first_hit_date else None,
            "first_hit_trading_days": first_hit_trading_days,
            "simulated_exit_price": simulated_exit_price,
            "simulated_exit_reason": simulated_exit_reason,
        }

    @classmethod
    def evaluate_single(
        cls,
        *,
        operation_advice: Optional[str],
        analysis_date: date,
        start_price: Optional[float],
        forward_bars: list[DailyBar],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        eval_window_days: int,
        neutral_band_pct: float,
        engine_version: str,
        replay_engine: Optional[BacktraderBacktestEngine] = None,
    ) -> Dict[str, Any]:
        """对单条分析记录执行一次完整回测评估。"""
        position_recommendation = cls.infer_position_recommendation(operation_advice)
        direction_expected = cls.infer_direction_expected(operation_advice)

        if start_price is None or not math.isfinite(start_price) or start_price <= 0:
            return {
                "analysis_date": analysis_date.isoformat(),
                "eval_window_days": eval_window_days,
                "engine_version": engine_version,
                "eval_status": "error",
                "operation_advice": operation_advice,
                "position_recommendation": position_recommendation,
                "direction_expected": direction_expected,
            }

        if len(forward_bars) < eval_window_days:
            return {
                "analysis_date": analysis_date.isoformat(),
                "eval_window_days": eval_window_days,
                "engine_version": engine_version,
                "eval_status": "insufficient_data",
                "operation_advice": operation_advice,
                "position_recommendation": position_recommendation,
                "direction_expected": direction_expected,
            }

        window_bars = list(forward_bars[:eval_window_days])
        end_close = window_bars[-1].close if window_bars else None

        highs = [item.high for item in window_bars if item.high is not None]
        lows = [item.low for item in window_bars if item.low is not None]
        max_high = max(highs) if highs else None
        min_low = min(lows) if lows else None

        stock_return_pct = None
        if end_close is not None:
            stock_return_pct = ((end_close - start_price) / start_price) * 100

        outcome, direction_correct = cls.classify_outcome(stock_return_pct, direction_expected, neutral_band_pct)

        targets = cls.evaluate_targets(
            position_recommendation,
            stop_loss,
            take_profit,
            window_bars,
            end_close,
        )

        simulated_entry_price = start_price if position_recommendation == "long" else None
        simulated_return_pct = 0.0
        simulated_exit_price = targets.get("simulated_exit_price")
        simulated_exit_reason = targets.get("simulated_exit_reason")
        if position_recommendation == "long":
            replay = replay_engine.simulate_long_trade(
                analysis_date=analysis_date,
                start_price=start_price,
                forward_bars=window_bars,
                stop_loss=stop_loss,
                take_profit=take_profit,
            ) if replay_engine is not None else None

            if replay is not None:
                # 优先采用交易回放引擎的结果，保证收益率口径与手续费、滑点一致。
                replay_entry = cls._to_number(replay.get("entry_price"))
                replay_exit = cls._to_number(replay.get("exit_price"))
                replay_return = cls._to_number(replay.get("return_pct"))
                replay_reason = str(replay.get("exit_reason") or "").strip()

                if replay_entry is not None:
                    simulated_entry_price = replay_entry
                if replay_exit is not None:
                    simulated_exit_price = replay_exit
                if replay_reason:
                    simulated_exit_reason = replay_reason
                if replay_return is not None:
                    simulated_return_pct = replay_return
                else:
                    exit_price = cls._to_number(simulated_exit_price)
                    if exit_price is None:
                        simulated_return_pct = None
                    else:
                        simulated_return_pct = ((exit_price - start_price) / start_price) * 100
            else:
                exit_price = cls._to_number(simulated_exit_price)
                if exit_price is None:
                    simulated_return_pct = None
                else:
                    simulated_return_pct = ((exit_price - start_price) / start_price) * 100

        return {
            "analysis_date": analysis_date.isoformat(),
            "eval_window_days": eval_window_days,
            "engine_version": engine_version,
            "eval_status": "completed",
            "operation_advice": operation_advice,
            "position_recommendation": position_recommendation,
            "start_price": start_price,
            "end_close": end_close,
            "max_high": max_high,
            "min_low": min_low,
            "stock_return_pct": stock_return_pct,
            "direction_expected": direction_expected,
            "direction_correct": direction_correct,
            "outcome": outcome,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "hit_stop_loss": targets.get("hit_stop_loss"),
            "hit_take_profit": targets.get("hit_take_profit"),
            "first_hit": targets.get("first_hit"),
            "first_hit_date": targets.get("first_hit_date"),
            "first_hit_trading_days": targets.get("first_hit_trading_days"),
            "simulated_entry_price": simulated_entry_price,
            "simulated_exit_price": simulated_exit_price,
            "simulated_exit_reason": simulated_exit_reason,
            "simulated_return_pct": simulated_return_pct,
        }

    @staticmethod
    def _average(values: Iterable[Optional[float]]) -> Optional[float]:
        """计算一组数值的平均值，并忽略空值与非法值。"""
        usable = [item for item in values if item is not None and math.isfinite(item)]
        if not usable:
            return None
        return sum(usable) / len(usable)

    @classmethod
    def _compute_advice_breakdown(cls, rows: list[Dict[str, Any]]) -> Dict[str, Any]:
        """按操作建议汇总胜负分布。"""
        mapping: Dict[str, Dict[str, int]] = {}
        for row in rows:
            advice = str(row.get("operation_advice") or "").strip() or "(unknown)"
            if advice not in mapping:
                mapping[advice] = {"total": 0, "win": 0, "loss": 0, "neutral": 0}
            mapping[advice]["total"] += 1
            if row.get("outcome") == "win":
                mapping[advice]["win"] += 1
            if row.get("outcome") == "loss":
                mapping[advice]["loss"] += 1
            if row.get("outcome") == "neutral":
                mapping[advice]["neutral"] += 1

        payload: Dict[str, Any] = {}
        for advice, metrics in mapping.items():
            denominator = metrics["win"] + metrics["loss"]
            payload[advice] = {
                **metrics,
                "win_rate_pct": cls._round((metrics["win"] / denominator) * 100, 2) if denominator else None,
            }

        return payload

    @staticmethod
    def _compute_diagnostics(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总评估状态与止盈止损触发诊断信息。"""
        eval_status: Dict[str, int] = {}
        first_hit: Dict[str, int] = {}

        for row in rows:
            status = str(row.get("eval_status") or "(unknown)")
            eval_status[status] = eval_status.get(status, 0) + 1
            hit = str(row.get("first_hit") or "(none)")
            first_hit[hit] = first_hit.get(hit, 0) + 1

        return {"eval_status": eval_status, "first_hit": first_hit}

    @classmethod
    def compute_summary(
        cls,
        *,
        rows: list[Dict[str, Any]],
        scope: str,
        code: Optional[str],
        eval_window_days: int,
        engine_version: str,
        neutral_band_pct: float = 2.0,
    ) -> Dict[str, Any]:
        """基于评估明细计算汇总指标。"""
        completed = [row for row in rows if row.get("eval_status") == "completed"]

        total_evaluations = len(rows)
        insufficient_count = sum(1 for row in rows if row.get("eval_status") == "insufficient_data")
        long_count = sum(1 for row in completed if row.get("position_recommendation") == "long")
        cash_count = sum(1 for row in completed if row.get("position_recommendation") == "cash")

        win_count = sum(1 for row in completed if row.get("outcome") == "win")
        loss_count = sum(1 for row in completed if row.get("outcome") == "loss")
        neutral_count = sum(1 for row in completed if row.get("outcome") == "neutral")

        direction_rows = [row for row in completed if row.get("direction_correct") is not None]
        direction_accuracy_pct = None
        if direction_rows:
            direction_accuracy_pct = cls._round(
                (sum(1 for row in direction_rows if row.get("direction_correct") is True) / len(direction_rows)) * 100,
                2,
            )

        prediction_win_loss_denominator = win_count + loss_count
        prediction_win_rate_pct = (
            cls._round((win_count / prediction_win_loss_denominator) * 100, 2)
            if prediction_win_loss_denominator
            else None
        )
        neutral_rate_pct = cls._round((neutral_count / len(completed)) * 100, 2) if completed else None

        trade_outcomes = [
            cls.classify_trade_outcome(cls._to_number(row.get("simulated_return_pct")), neutral_band_pct=neutral_band_pct)
            for row in completed
        ]
        trade_win_count = sum(1 for item in trade_outcomes if item == "win")
        trade_loss_count = sum(1 for item in trade_outcomes if item == "loss")
        trade_win_loss_denominator = trade_win_count + trade_loss_count
        trade_win_rate_pct = (
            cls._round((trade_win_count / trade_win_loss_denominator) * 100, 2)
            if trade_win_loss_denominator
            else None
        )

        avg_stock_return_pct = cls._average(cls._to_number(row.get("stock_return_pct")) for row in completed)
        avg_simulated_return_pct = cls._average(cls._to_number(row.get("simulated_return_pct")) for row in completed)

        stop_applicable = [
            row
            for row in completed
            if row.get("position_recommendation") == "long" and row.get("hit_stop_loss") is not None
        ]
        take_applicable = [
            row
            for row in completed
            if row.get("position_recommendation") == "long" and row.get("hit_take_profit") is not None
        ]
        target_applicable = [
            row
            for row in completed
            if row.get("position_recommendation") == "long"
            and (row.get("hit_stop_loss") is not None or row.get("hit_take_profit") is not None)
        ]

        stop_loss_trigger_rate = (
            cls._round((sum(1 for row in stop_applicable if row.get("hit_stop_loss") is True) / len(stop_applicable)) * 100, 2)
            if stop_applicable
            else None
        )
        take_profit_trigger_rate = (
            cls._round((sum(1 for row in take_applicable if row.get("hit_take_profit") is True) / len(take_applicable)) * 100, 2)
            if take_applicable
            else None
        )
        ambiguous_rate = (
            cls._round((sum(1 for row in target_applicable if row.get("first_hit") == "ambiguous") / len(target_applicable)) * 100, 2)
            if target_applicable
            else None
        )

        avg_days_to_first_hit = cls._average(
            cls._to_number(row.get("first_hit_trading_days"))
            for row in target_applicable
            if row.get("first_hit") in {"stop_loss", "take_profit", "ambiguous"}
        )

        return {
            "scope": scope,
            "code": code,
            "eval_window_days": eval_window_days,
            "engine_version": engine_version,
            "total_evaluations": total_evaluations,
            "completed_count": len(completed),
            "insufficient_count": insufficient_count,
            "long_count": long_count,
            "cash_count": cash_count,
            "win_count": win_count,
            "loss_count": loss_count,
            "neutral_count": neutral_count,
            "direction_accuracy_pct": direction_accuracy_pct,
            "prediction_win_rate_pct": prediction_win_rate_pct,
            "trade_win_rate_pct": trade_win_rate_pct,
            "win_rate_pct": prediction_win_rate_pct,
            "neutral_rate_pct": neutral_rate_pct,
            "avg_stock_return_pct": cls._round(avg_stock_return_pct, 4) if avg_stock_return_pct is not None else None,
            "avg_simulated_return_pct": cls._round(avg_simulated_return_pct, 4) if avg_simulated_return_pct is not None else None,
            "stop_loss_trigger_rate": stop_loss_trigger_rate,
            "take_profit_trigger_rate": take_profit_trigger_rate,
            "ambiguous_rate": ambiguous_rate,
            "avg_days_to_first_hit": cls._round(avg_days_to_first_hit, 4) if avg_days_to_first_hit is not None else None,
            "advice_breakdown": cls._compute_advice_breakdown(completed),
            "diagnostics": cls._compute_diagnostics(rows),
        }

    @classmethod
    def build_curves(cls, rows: list[Dict[str, Any]], mode: str = "sequential") -> List[Dict[str, Any]]:
        """根据回测明细构建顺序曲线或组合曲线。"""
        completed = [row for row in rows if row.get("eval_status") == "completed"]
        completed.sort(key=lambda item: cls._curve_sort_tuple(item))

        if mode == "portfolio":
            # 组合模式会先按日期聚合多只股票，再把当日平均收益滚入权益曲线。
            grouped: Dict[str, Dict[str, Any]] = {}
            for row in completed:
                label = str(row.get("analysis_date") or row.get("evaluated_at") or "")
                if label not in grouped:
                    grouped[label] = {
                        "label": label,
                        "strategy_returns": [],
                        "benchmark_returns": [],
                        "sort_key": cls._curve_sort_tuple(row),
                    }
                grouped[label]["strategy_returns"].append(cls._to_number(row.get("simulated_return_pct")) or 0.0)
                grouped[label]["benchmark_returns"].append(cls._to_number(row.get("stock_return_pct")) or 0.0)

            timeline = sorted(grouped.values(), key=lambda item: item["sort_key"])
            strategy_equity = 1.0
            benchmark_equity = 1.0
            peak = 1.0
            points: List[Dict[str, Any]] = []
            for point in timeline:
                strategy_return = sum(point["strategy_returns"]) / len(point["strategy_returns"]) if point["strategy_returns"] else 0.0
                benchmark_return = sum(point["benchmark_returns"]) / len(point["benchmark_returns"]) if point["benchmark_returns"] else 0.0
                strategy_equity *= 1 + strategy_return / 100
                benchmark_equity *= 1 + benchmark_return / 100
                if strategy_equity > peak:
                    peak = strategy_equity
                drawdown = ((strategy_equity / peak) - 1) * 100
                points.append(
                    {
                        "label": point["label"],
                        "strategy_return_pct": cls._round((strategy_equity - 1) * 100, 4),
                        "benchmark_return_pct": cls._round((benchmark_equity - 1) * 100, 4),
                        "drawdown_pct": cls._round(drawdown, 4),
                    }
                )
            return points

        strategy_equity = 1.0
        benchmark_equity = 1.0
        peak = 1.0
        points: List[Dict[str, Any]] = []
        for row in completed:
            strategy = cls._to_number(row.get("simulated_return_pct")) or 0.0
            benchmark = cls._to_number(row.get("stock_return_pct")) or 0.0
            strategy_equity *= 1 + strategy / 100
            benchmark_equity *= 1 + benchmark / 100

            if strategy_equity > peak:
                peak = strategy_equity
            drawdown = ((strategy_equity / peak) - 1) * 100

            label = str(row.get("analysis_date") or row.get("evaluated_at") or "")
            points.append(
                {
                    "label": label,
                    "strategy_return_pct": cls._round((strategy_equity - 1) * 100, 4),
                    "benchmark_return_pct": cls._round((benchmark_equity - 1) * 100, 4),
                    "drawdown_pct": cls._round(drawdown, 4),
                }
            )

        return points

    @classmethod
    def _curve_sort_key(cls, item: Dict[str, Any]) -> float:
        """提取曲线排序使用的主时间键。"""
        analysis_date = cls._parse_date(item.get("analysis_date"))
        if analysis_date is not None:
            return datetime.combine(analysis_date, datetime.min.time(), tzinfo=timezone.utc).timestamp()
        evaluated = cls._parse_datetime(item.get("evaluated_at"))
        if evaluated is None:
            return 0.0
        return evaluated.timestamp()

    @classmethod
    def _curve_sort_tuple(cls, item: Dict[str, Any]) -> tuple[float, float, str, int]:
        """构造稳定排序所需的复合键。"""
        primary = cls._curve_sort_key(item)
        evaluated = cls._parse_datetime(item.get("evaluated_at"))
        secondary = evaluated.timestamp() if evaluated is not None else 0.0
        code = str(item.get("code") or "")
        analysis_history_id = int(cls._to_number(item.get("analysis_history_id")) or 0)
        return (primary, secondary, code, analysis_history_id)

    @classmethod
    def _compute_max_drawdown_from_returns(cls, strategy_returns: List[float]) -> Optional[float]:
        """根据收益率序列估算最大回撤。"""
        if not strategy_returns:
            return None
        equity = 1.0
        peak = 1.0
        worst_drawdown = 0.0
        for value in strategy_returns:
            equity *= 1 + value / 100
            if equity > peak:
                peak = equity
            drawdown = ((equity / peak) - 1) * 100
            if drawdown < worst_drawdown:
                worst_drawdown = drawdown
        return cls._round(worst_drawdown, 4)

    @classmethod
    def _compute_ma_at(cls, closes: List[Optional[float]], index: int, window: int) -> Optional[float]:
        """计算指定位置的简单移动平均。"""
        if window <= 0 or index + 1 < window:
            return None
        chunk = closes[index + 1 - window : index + 1]
        if any(item is None for item in chunk):
            return None
        values = [float(item) for item in chunk if item is not None]
        return cls._round(sum(values) / len(values), 4)

    @classmethod
    def _compute_rsi14_at(cls, closes: List[Optional[float]], index: int) -> Optional[float]:
        """按经典 14 日口径计算某一位置的 RSI。"""
        if index < 14:
            return None
        if any(item is None for item in closes[: index + 1]):
            return None

        numeric = [float(item) for item in closes[: index + 1] if item is not None]
        deltas = [numeric[i] - numeric[i - 1] for i in range(1, len(numeric))]

        gain = 0.0
        loss = 0.0
        for delta in deltas[:14]:
            if delta >= 0:
                gain += delta
            else:
                loss += abs(delta)

        avg_gain = gain / 14
        avg_loss = loss / 14

        for delta in deltas[14:]:
            current_gain = delta if delta > 0 else 0.0
            current_loss = abs(delta) if delta < 0 else 0.0
            avg_gain = (avg_gain * 13 + current_gain) / 14
            avg_loss = (avg_loss * 13 + current_loss) / 14

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return cls._round(100 - 100 / (1 + rs), 4)

    @classmethod
    def _is_ma20_cross_up(cls, closes: List[Optional[float]], index: int) -> bool:
        """判断是否在指定位置发生 MA20 上穿。"""
        if index < 20 or index >= len(closes):
            return False
        close_prev = closes[index - 1]
        close_now = closes[index]
        ma20_prev = cls._compute_ma_at(closes, index - 1, 20)
        ma20_now = cls._compute_ma_at(closes, index, 20)
        if close_prev is None or close_now is None or ma20_prev is None or ma20_now is None:
            return False
        return close_prev <= ma20_prev and close_now > ma20_now

    @classmethod
    def _is_ma20_cross_down(cls, closes: List[Optional[float]], index: int) -> bool:
        """判断是否在指定位置发生 MA20 下穿。"""
        if index < 20 or index >= len(closes):
            return False
        close_prev = closes[index - 1]
        close_now = closes[index]
        ma20_prev = cls._compute_ma_at(closes, index - 1, 20)
        ma20_now = cls._compute_ma_at(closes, index, 20)
        if close_prev is None or close_now is None or ma20_prev is None or ma20_now is None:
            return False
        return close_prev >= ma20_prev and close_now < ma20_now

    @classmethod
    def _is_technical_entry_signal(cls, strategy_code: str, closes: List[Optional[float]], index: int) -> bool:
        """判断技术策略在指定位置是否产生入场信号。"""
        if strategy_code == "ma20_trend":
            return cls._is_ma20_cross_up(closes, index)
        if strategy_code == "rsi14_mean_reversion":
            rsi14 = cls._compute_rsi14_at(closes, index)
            return rsi14 is not None and rsi14 < 30
        return False

    @classmethod
    def _is_technical_exit_signal(cls, strategy_code: str, closes: List[Optional[float]], index: int) -> bool:
        """判断技术策略在指定位置是否产生离场信号。"""
        if strategy_code == "ma20_trend":
            return cls._is_ma20_cross_down(closes, index)
        if strategy_code == "rsi14_mean_reversion":
            rsi14 = cls._compute_rsi14_at(closes, index)
            return rsi14 is not None and rsi14 > 70
        return False

    @staticmethod
    def _find_start_index(bars: List[DailyBar], analysis_date: date) -> int:
        """在行情序列中找到不晚于分析日的最后一根 bar。"""
        found = -1
        for idx, bar in enumerate(bars):
            if bar.day <= analysis_date:
                found = idx
            else:
                break
        return found

    def _evaluate_compare_strategy_row(
        self,
        strategy_code: str,
        row: Dict[str, Any],
        stock_return_pct: float,
        *,
        eval_window_days: int,
        bars_by_code: Dict[str, List[DailyBar]],
    ) -> Dict[str, Any]:
        """按指定策略口径重算单条记录的策略收益。"""
        operation_advice = str(row.get("operation_advice") or "")
        position_recommendation_raw = str(row.get("position_recommendation") or "").strip().lower()
        if position_recommendation_raw in {"long", "cash"}:
            agent_position = position_recommendation_raw
        else:
            agent_position = self.infer_position_recommendation(operation_advice)

        code = self._normalize_code(row.get("code"))
        analysis_date = self._parse_date(row.get("analysis_date"))
        stop_loss = self._to_number(row.get("stop_loss"))
        take_profit = self._to_number(row.get("take_profit"))

        if strategy_code == "agent_v1":
            if agent_position != "long":
                return {"position": "cash", "strategy_return_pct": 0.0}

            if not code or analysis_date is None:
                return {"position": "long", "strategy_return_pct": self._to_number(row.get("simulated_return_pct")) or 0.0}

            bars = bars_by_code.get(code) or []
            start_index = self._find_start_index(bars, analysis_date)
            if start_index < 0:
                return {"position": "long", "strategy_return_pct": self._to_number(row.get("simulated_return_pct")) or 0.0}
            start_price = self._to_number(bars[start_index].close)
            if start_price is None:
                return {"position": "long", "strategy_return_pct": self._to_number(row.get("simulated_return_pct")) or 0.0}

            forward_bars = bars[start_index + 1 : start_index + 1 + eval_window_days]
            if len(forward_bars) < eval_window_days:
                return {"position": "long", "strategy_return_pct": self._to_number(row.get("simulated_return_pct")) or 0.0}

            replay = self.replay_engine.simulate_long_trade(
                analysis_date=bars[start_index].day,
                start_price=start_price,
                forward_bars=forward_bars,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            replay_return = self._to_number(replay.get("return_pct"))
            return {
                "position": "long",
                "strategy_return_pct": replay_return if replay_return is not None else (self._to_number(row.get("simulated_return_pct")) or 0.0),
            }

        if not code or analysis_date is None:
            return {"position": "cash", "strategy_return_pct": 0.0}

        bars = bars_by_code.get(code) or []
        start_index = self._find_start_index(bars, analysis_date)
        if start_index < 0:
            return {"position": "cash", "strategy_return_pct": 0.0}

        start_price = self._to_number(bars[start_index].close)
        if start_price is None:
            return {"position": "cash", "strategy_return_pct": 0.0}

        forward_bars = bars[start_index + 1 : start_index + 1 + eval_window_days]
        if len(forward_bars) < eval_window_days:
            return {"position": "cash", "strategy_return_pct": 0.0}

        closes: List[Optional[float]] = [self._to_number(item.close) for item in bars]
        if not self._is_technical_entry_signal(strategy_code, closes, start_index):
            return {"position": "cash", "strategy_return_pct": 0.0}

        exit_global_index = start_index + eval_window_days
        for index in range(start_index + 1, start_index + eval_window_days + 1):
            if index >= len(bars):
                break
            if self._is_technical_exit_signal(strategy_code, closes, index):
                exit_global_index = index
                break
        trade_forward_bars = bars[start_index + 1 : exit_global_index + 1]

        replay = self.replay_engine.simulate_long_trade(
            analysis_date=bars[start_index].day,
            start_price=start_price,
            forward_bars=trade_forward_bars,
            stop_loss=None,
            take_profit=None,
        )
        replay_return = self._to_number(replay.get("return_pct"))
        return {
            "position": "long",
            "strategy_return_pct": replay_return if replay_return is not None else stock_return_pct,
        }

    def _compute_strategy_compare_metrics(
        self,
        strategy_code: str,
        rows: list[Dict[str, Any]],
        neutral_band_pct: float,
        *,
        eval_window_days: int,
        bars_by_code: Dict[str, List[DailyBar]],
    ) -> Dict[str, Any]:
        """计算某个策略模板在一组记录上的对比指标。"""
        completed_rows = [
            row
            for row in rows
            if row.get("eval_status") == "completed" and self._to_number(row.get("stock_return_pct")) is not None
        ]
        completed_rows.sort(key=lambda item: self._curve_sort_tuple(item))

        strategy_returns: List[float] = []
        stock_returns: List[float] = []
        direction_correct_count = 0
        prediction_win_count = 0
        prediction_loss_count = 0
        trade_win_count = 0
        trade_loss_count = 0
        band = abs(neutral_band_pct)

        for row in completed_rows:
            stock_return_pct = float(self._to_number(row.get("stock_return_pct")) or 0.0)
            evaluation = self._evaluate_compare_strategy_row(
                strategy_code,
                row,
                stock_return_pct,
                eval_window_days=eval_window_days,
                bars_by_code=bars_by_code,
            )
            strategy_return_pct = float(evaluation["strategy_return_pct"])
            position = str(evaluation["position"])

            strategy_returns.append(strategy_return_pct)
            stock_returns.append(stock_return_pct)

            if position == "long":
                if stock_return_pct >= band:
                    direction_correct_count += 1
                    prediction_win_count += 1
                elif stock_return_pct <= -band:
                    prediction_loss_count += 1
            else:
                if stock_return_pct <= -band:
                    direction_correct_count += 1
                    prediction_win_count += 1
                elif abs(stock_return_pct) <= band:
                    direction_correct_count += 1
                elif stock_return_pct >= band:
                    prediction_loss_count += 1

            if strategy_return_pct >= band:
                trade_win_count += 1
            elif strategy_return_pct <= -band:
                trade_loss_count += 1

        completed_count = len(completed_rows)
        direction_accuracy_pct = self._round((direction_correct_count / completed_count) * 100, 2) if completed_count else None
        prediction_win_loss_denominator = prediction_win_count + prediction_loss_count
        prediction_win_rate_pct = (
            self._round((prediction_win_count / prediction_win_loss_denominator) * 100, 2)
            if prediction_win_loss_denominator
            else None
        )
        trade_win_loss_denominator = trade_win_count + trade_loss_count
        trade_win_rate_pct = (
            self._round((trade_win_count / trade_win_loss_denominator) * 100, 2)
            if trade_win_loss_denominator
            else None
        )

        avg_simulated_return_pct = self._average(strategy_returns)
        avg_stock_return_pct = self._average(stock_returns)

        return {
            "total_evaluations": len(rows),
            "completed_count": completed_count,
            "direction_accuracy_pct": direction_accuracy_pct,
            "prediction_win_rate_pct": prediction_win_rate_pct,
            "trade_win_rate_pct": trade_win_rate_pct,
            "win_rate_pct": prediction_win_rate_pct,
            "avg_simulated_return_pct": self._round(avg_simulated_return_pct, 4) if avg_simulated_return_pct is not None else None,
            "avg_stock_return_pct": self._round(avg_stock_return_pct, 4) if avg_stock_return_pct is not None else None,
            "max_drawdown_pct": self._compute_max_drawdown_from_returns(strategy_returns),
        }

    @classmethod
    def _parse_date(cls, raw: Any) -> Optional[date]:
        """将输入安全解析为日期。"""
        if raw is None:
            return None
        if isinstance(raw, date) and not isinstance(raw, datetime):
            return raw
        if isinstance(raw, datetime):
            return raw.date()
        text = str(raw).strip()
        if not text:
            return None
        try:
            if len(text) >= 10:
                return date.fromisoformat(text[:10])
        except Exception:
            return None
        return None

    @classmethod
    def _parse_datetime(cls, raw: Any) -> Optional[datetime]:
        """将输入安全解析为 UTC 时间。"""
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw.astimezone(timezone.utc) if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        text = str(raw).strip()
        if not text:
            return None
        try:
            normalized = text.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    @classmethod
    def _safe_json_loads(cls, raw: Any) -> Any:
        """安全解析 JSON 字符串；失败时返回空值。"""
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return None
            try:
                import json

                return json.loads(text)
            except Exception:
                return None
        return None

    @staticmethod
    def _as_dict(raw: Any) -> Dict[str, Any]:
        """将任意值安全转换为字典。"""
        if isinstance(raw, dict):
            return raw
        return {}

    @classmethod
    def _resolve_analysis_date(cls, context_snapshot: Any, created_at: Any) -> date:
        """优先从上下文快照中解析分析日，缺失时退回创建时间。"""
        payload = cls._as_dict(cls._safe_json_loads(context_snapshot))
        enhanced_context = payload.get("enhanced_context")
        if isinstance(enhanced_context, dict):
            day_text = str(enhanced_context.get("date") or "")
            parsed = cls._parse_date(day_text)
            if parsed is not None:
                return parsed

        created_dt = cls._parse_datetime(created_at)
        if created_dt is not None:
            return created_dt.date()

        return datetime.now(timezone.utc).date()

    @classmethod
    def _normalize_code(cls, raw: Any) -> str:
        """规范化股票代码格式。"""
        code = canonical_stock_code(str(raw or ""))
        return normalize_stock_code(code)

    def _fetch_bars_for_code(
        self,
        code: str,
        min_analysis_date: date,
        max_analysis_date: date,
        eval_window_days: int,
    ) -> List[DailyBar]:
        """为给定股票抓取覆盖分析窗口及前瞻区间的行情。"""
        start_date = (min_analysis_date - timedelta(days=40)).isoformat()
        end_date = (max_analysis_date + timedelta(days=max(eval_window_days * 3, 80))).isoformat()
        frame, _ = self.fetcher.get_daily_data(code, start_date=start_date, end_date=end_date, days=800)

        bars: List[DailyBar] = []
        if frame is None or frame.empty:
            return bars

        for _, row in frame.iterrows():
            row_date = self._parse_date(row.get("date"))
            if row_date is None:
                continue
            bars.append(
                DailyBar(
                    day=row_date,
                    high=self._to_number(row.get("high")),
                    low=self._to_number(row.get("low")),
                    close=self._to_number(row.get("close")),
                )
            )

        bars.sort(key=lambda item: item.day)
        return bars

    @staticmethod
    def _find_start_and_forward_bars(
        bars: List[DailyBar],
        analysis_date: date,
        eval_window_days: int,
    ) -> tuple[Optional[date], Optional[float], List[DailyBar]]:
        """找到分析起点以及后续评估窗口的 bar 序列。"""
        start_candidates = [bar for bar in bars if bar.day <= analysis_date]
        if not start_candidates:
            return None, None, []

        start_bar = start_candidates[-1]
        forward = [bar for bar in bars if bar.day > start_bar.day][:eval_window_days]
        return start_bar.day, start_bar.close, forward

    def _build_compare_bars_by_code(self, rows: List[Dict[str, Any]], eval_window_days: int) -> Dict[str, List[DailyBar]]:
        """为策略对比一次性构建各股票的行情缓存。"""
        by_code_dates: Dict[str, List[date]] = {}
        for row in rows:
            code = self._normalize_code(row.get("code"))
            analysis_date = self._parse_date(row.get("analysis_date"))
            if not code or analysis_date is None:
                continue
            by_code_dates.setdefault(code, []).append(analysis_date)

        bars_by_code: Dict[str, List[DailyBar]] = {}
        failures: List[str] = []
        for code, days in by_code_dates.items():
            try:
                bars = self._fetch_bars_for_code(code, min(days), max(days), eval_window_days)
            except Exception as exc:
                failures.append(f"{code}: {str(exc)}")
                continue
            if not bars:
                failures.append(f"{code}: no daily bars available")
                continue
            bars_by_code[code] = bars
        if failures:
            raise ValueError(f"compare_fetch_failed: {'; '.join(failures)}")
        return bars_by_code

    def _normalize_compare_window(self, raw: Any) -> Optional[int]:
        """规范化对比窗口天数，并限制在允许范围内。"""
        value = self._to_number(raw)
        if value is None:
            return None
        window = int(value)
        if abs(value - float(window)) > 1e-9:
            return None
        if not (0 < window <= 120):
            return None
        return window

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行批量回测，并返回逐条评估结果。"""
        eval_window_days = max(1, int(payload.get("eval_window_days") or 10))
        engine_version = str(payload.get("engine_version") or "v1")
        neutral_band_pct = float(payload.get("neutral_band_pct") or 2.0)
        candidates_raw = payload.get("candidates")
        candidates = candidates_raw if isinstance(candidates_raw, list) else []

        candidate_rows: List[Dict[str, Any]] = []
        by_code_dates: Dict[str, List[date]] = {}
        for item in candidates:
            # 先把候选记录规整成统一结构，后面才能按股票批量拉取行情。
            if not isinstance(item, dict):
                continue
            code = self._normalize_code(item.get("code"))
            if not code:
                continue
            analysis_date = self._resolve_analysis_date(item.get("context_snapshot"), item.get("created_at"))
            row = {
                "analysis_history_id": int(item.get("analysis_history_id") or 0),
                "owner_user_id": int(item["owner_user_id"]) if item.get("owner_user_id") is not None else None,
                "code": code,
                "analysis_date": analysis_date,
                "created_at": item.get("created_at"),
                "context_snapshot": item.get("context_snapshot"),
                "operation_advice": item.get("operation_advice"),
                "stop_loss": self._to_number(item.get("stop_loss")),
                "take_profit": self._to_number(item.get("take_profit")),
            }
            candidate_rows.append(row)
            by_code_dates.setdefault(code, []).append(analysis_date)

        bars_by_code: Dict[str, List[DailyBar]] = {}
        fetch_errors = 0
        for code, days in by_code_dates.items():
            try:
                bars_by_code[code] = self._fetch_bars_for_code(code, min(days), max(days), eval_window_days)
            except Exception:
                bars_by_code[code] = []
                fetch_errors += len([row for row in candidate_rows if row["code"] == code])

        items: List[Dict[str, Any]] = []
        completed = 0
        insufficient = 0
        errors = 0

        for candidate in candidate_rows:
            bars = bars_by_code.get(candidate["code"], [])
            start_day, start_price, forward_bars = self._find_start_and_forward_bars(
                bars,
                candidate["analysis_date"],
                eval_window_days,
            )

            if start_day is None or start_price is None:
                evaluation = {
                    "analysis_date": candidate["analysis_date"].isoformat(),
                    "eval_window_days": eval_window_days,
                    "engine_version": engine_version,
                    "eval_status": "insufficient_data",
                    "operation_advice": candidate.get("operation_advice"),
                    "position_recommendation": self.infer_position_recommendation(candidate.get("operation_advice")),
                    "direction_expected": self.infer_direction_expected(candidate.get("operation_advice")),
                }
            else:
                evaluation = self.evaluate_single(
                    operation_advice=candidate.get("operation_advice"),
                    analysis_date=start_day,
                    start_price=start_price,
                    forward_bars=forward_bars,
                    stop_loss=candidate.get("stop_loss"),
                    take_profit=candidate.get("take_profit"),
                    eval_window_days=eval_window_days,
                    neutral_band_pct=neutral_band_pct,
                    engine_version=engine_version,
                    replay_engine=self.replay_engine,
                )

            status = str(evaluation.get("eval_status") or "")
            if status == "completed":
                completed += 1
            elif status == "insufficient_data":
                insufficient += 1
            else:
                errors += 1

            merged = {
                "analysis_history_id": candidate["analysis_history_id"],
                "owner_user_id": candidate.get("owner_user_id"),
                "code": candidate["code"],
                **evaluation,
            }
            items.append(merged)

        errors += fetch_errors

        return {
            "processed": len(candidate_rows),
            "saved": len(items),
            "completed": completed,
            "insufficient": insufficient,
            "errors": errors,
            "items": items,
        }

    def summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """计算回测结果的汇总统计。"""
        rows = payload.get("rows")
        return self.compute_summary(
            rows=rows if isinstance(rows, list) else [],
            scope=str(payload.get("scope") or "overall"),
            code=payload.get("code"),
            eval_window_days=max(1, int(payload.get("eval_window_days") or 10)),
            engine_version=str(payload.get("engine_version") or "v1"),
            neutral_band_pct=abs(float(payload.get("neutral_band_pct") or 2.0)),
        )

    def curves(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """构建顺序曲线与组合曲线。"""
        rows = payload.get("rows")
        scope = str(payload.get("scope") or "overall")
        code = payload.get("code")
        eval_window_days = max(1, int(payload.get("eval_window_days") or 10))
        mode = str(payload.get("equity_mode") or "portfolio").strip().lower()
        if mode not in {"portfolio", "sequential"}:
            mode = "portfolio"
        typed_rows = rows if isinstance(rows, list) else []
        signal_curves = self.build_curves(typed_rows, mode="sequential")
        portfolio_curves = self.build_curves(typed_rows, mode="portfolio")
        return {
            "scope": scope,
            "code": code,
            "eval_window_days": eval_window_days,
            "metric_definition_version": "v2",
            "equity_mode": mode,
            "curves": portfolio_curves if mode == "portfolio" else signal_curves,
            "signal_curves": signal_curves,
            "portfolio_curves": portfolio_curves,
        }

    def distribution(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """构建仓位与结果分布统计。"""
        rows = payload.get("rows")
        scope = str(payload.get("scope") or "overall")
        code = payload.get("code")
        eval_window_days = max(1, int(payload.get("eval_window_days") or 10))
        engine_version = str(payload.get("engine_version") or "v1")

        summary = self.compute_summary(
            rows=rows if isinstance(rows, list) else [],
            scope=scope,
            code=code,
            eval_window_days=eval_window_days,
            engine_version=engine_version,
            neutral_band_pct=abs(float(payload.get("neutral_band_pct") or 2.0)),
        )
        return {
            "scope": scope,
            "code": code,
            "eval_window_days": eval_window_days,
            "metric_definition_version": "v2",
            "distribution": {
                "position_distribution": {
                    "long_count": int(summary.get("long_count") or 0),
                    "cash_count": int(summary.get("cash_count") or 0),
                },
                "outcome_distribution": {
                    "win_count": int(summary.get("win_count") or 0),
                    "loss_count": int(summary.get("loss_count") or 0),
                    "neutral_count": int(summary.get("neutral_count") or 0),
                },
                "long_count": int(summary.get("long_count") or 0),
                "cash_count": int(summary.get("cash_count") or 0),
                "win_count": int(summary.get("win_count") or 0),
                "loss_count": int(summary.get("loss_count") or 0),
                "neutral_count": int(summary.get("neutral_count") or 0),
            },
        }

    def compare(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """比较多种策略模板在不同窗口下的表现。"""
        neutral_band_pct = abs(float(payload.get("neutral_band_pct") or 2.0))

        rows_by_window_raw = payload.get("rows_by_window")
        rows_by_window = rows_by_window_raw if isinstance(rows_by_window_raw, dict) else {}

        windows_raw = payload.get("eval_window_days_list")
        windows: List[int] = []
        if isinstance(windows_raw, list) and windows_raw:
            parsed_windows = {self._normalize_compare_window(item) for item in windows_raw}
            windows = sorted(item for item in parsed_windows if item is not None)
            if not windows:
                raise ValueError("validation_error: eval_window_days_list must contain integers in range 1..120")
        else:
            inferred_windows = {self._normalize_compare_window(key) for key in rows_by_window.keys()}
            windows = sorted(item for item in inferred_windows if item is not None)
            if not windows:
                raise ValueError(
                    "validation_error: eval_window_days_list is required or rows_by_window must contain numeric window keys in range 1..120"
                )

        strategy_codes_raw = payload.get("strategy_codes")
        if isinstance(strategy_codes_raw, list):
            strategy_codes = [str(item).strip() for item in strategy_codes_raw if str(item).strip() in BACKTEST_COMPARE_STRATEGY_CODES]
            strategy_codes = list(dict.fromkeys(strategy_codes))
        else:
            strategy_codes = []
        if not strategy_codes:
            strategy_codes = list(BACKTEST_COMPARE_STRATEGY_CODES)

        items: List[Dict[str, Any]] = []
        for window in windows:
            rows = rows_by_window.get(str(window))
            if rows is None:
                rows = rows_by_window.get(window)
            typed_rows = [item for item in (rows if isinstance(rows, list) else []) if isinstance(item, dict)]
            bars_by_code = self._build_compare_bars_by_code(typed_rows, window)
            for strategy_code in strategy_codes:
                metrics = self._compute_strategy_compare_metrics(
                    strategy_code,
                    typed_rows,
                    neutral_band_pct,
                    eval_window_days=window,
                    bars_by_code=bars_by_code,
                )
                items.append(
                    {
                        "strategy_code": strategy_code,
                        "strategy_name": BACKTEST_COMPARE_STRATEGY_NAMES[strategy_code],
                        "eval_window_days": window,
                        **metrics,
                        "data_source": "api",
                    }
                )

        return {"metric_definition_version": "v2", "items": items}


_backtest_service: Optional[BacktestService] = None


def get_backtest_service() -> BacktestService:
    """返回回测服务单例。"""
    global _backtest_service
    if _backtest_service is None:
        _backtest_service = BacktestService()
    return _backtest_service
