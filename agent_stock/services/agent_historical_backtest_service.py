# -*- coding: utf-8 -*-
"""Historical replay backtest service for Agent pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import math
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from data_provider import DataFetcherManager
from data_provider.base import canonical_stock_code
from agent_stock.agents.contracts import AgentState, RiskAgentOutput, SignalAgentOutput
from agent_stock.agents.risk_agent import RiskAgent
from src.analyzer import GeminiAnalyzer
from src.config import Config, RuntimeLlmConfig, RuntimeStrategyConfig, get_config
from src.stock_analyzer import BuySignal, StockTrendAnalyzer, TrendAnalysisResult

ENGINE_VERSION = "agent_replay_v1"
SIGNAL_PROFILE_VERSION = "agent_signal_profile_v1"
WARMUP_CALENDAR_DAYS = 180
ROLLING_WINDOW_BARS = 60
MAX_LLM_ANCHOR_CALLS = 20


@dataclass(frozen=True)
class HistoricalRunParams:
    code: str
    start_date: date
    end_date: date
    phase: str
    initial_capital: float
    commission_rate: float
    slippage_bps: float
    runtime_strategy: RuntimeStrategyConfig
    runtime_llm: RuntimeLlmConfig | None
    signal_profile_hash: str
    snapshot_version: int


@dataclass(frozen=True)
class ReplayBar:
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    pct_chg: float
    ma5: float
    ma10: float
    ma20: float
    ma60: float
    rsi14: float
    momentum20: float
    vol_ratio5: float


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        number = float(value)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def _to_int(value: Any, default: int = 0) -> int:
    number = _to_float(value, float(default))
    if number is None:
        return default
    return int(number)


def _to_day(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _iso_day(value: date | None) -> str | None:
    return value.isoformat() if isinstance(value, date) else None


class HistoricalDataAgent:
    """Prepare historical rolling data windows without realtime leakage."""

    def __init__(self, fetcher_manager: Optional[DataFetcherManager] = None) -> None:
        self.fetcher = fetcher_manager or DataFetcherManager()

    def load(self, params: HistoricalRunParams) -> tuple[pd.DataFrame, pd.DataFrame]:
        warmup_start = params.start_date - timedelta(days=WARMUP_CALENDAR_DAYS)
        frame, _ = self.fetcher.get_daily_data(
            params.code,
            start_date=warmup_start.isoformat(),
            end_date=params.end_date.isoformat(),
            days=5000,
        )
        if frame is None or frame.empty:
            raise ValueError("insufficient_data: no daily bars in requested range")

        normalized = self._normalize_frame(frame)
        trade_window = normalized[
            (normalized["date"] >= pd.Timestamp(params.start_date))
            & (normalized["date"] <= pd.Timestamp(params.end_date))
        ].copy()
        if trade_window.empty:
            raise ValueError("insufficient_data: no trading bars inside date range")

        trade_end = pd.Timestamp(trade_window["date"].iloc[-1])
        prepared = normalized[normalized["date"] <= trade_end].copy()
        if prepared.empty:
            raise ValueError("insufficient_data: no bars available after warmup preparation")
        return prepared.reset_index(drop=True), trade_window.reset_index(drop=True)

    def rolling_window(self, prepared: pd.DataFrame, trade_day: date) -> pd.DataFrame:
        scoped = prepared[prepared["date"] <= pd.Timestamp(trade_day)].copy()
        return scoped.tail(ROLLING_WINDOW_BARS).reset_index(drop=True)

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for _, row in frame.iterrows():
            day = _to_day(row.get("date"))
            if day is None:
                continue
            close_price = _to_float(row.get("close"))
            if close_price is None or close_price <= 0:
                continue

            open_price = _to_float(row.get("open"), close_price) or close_price
            high_price = _to_float(row.get("high"), max(open_price, close_price)) or max(open_price, close_price)
            low_price = _to_float(row.get("low"), min(open_price, close_price)) or min(open_price, close_price)
            volume = max(_to_float(row.get("volume"), 1.0) or 1.0, 1.0)
            amount = _to_float(row.get("amount"), volume * close_price) or (volume * close_price)

            rows.append(
                {
                    "date": pd.Timestamp(day),
                    "open": float(open_price),
                    "high": float(max(high_price, open_price, close_price)),
                    "low": float(min(low_price, open_price, close_price)),
                    "close": float(close_price),
                    "volume": float(volume),
                    "amount": float(amount),
                }
            )

        if not rows:
            raise ValueError("insufficient_data: no valid OHLC bars after normalization")

        normalized = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        normalized["pct_chg"] = normalized["close"].pct_change().fillna(0.0) * 100.0
        normalized["MA5"] = normalized["close"].rolling(5, min_periods=1).mean()
        normalized["MA10"] = normalized["close"].rolling(10, min_periods=1).mean()
        normalized["MA20"] = normalized["close"].rolling(20, min_periods=1).mean()
        normalized["MA60"] = normalized["close"].rolling(60, min_periods=1).mean()
        normalized["momentum20"] = normalized["close"].pct_change(20).fillna(0.0) * 100.0
        normalized["vol_ratio5"] = (
            normalized["volume"] / normalized["volume"].rolling(5, min_periods=1).mean().replace(0, pd.NA)
        ).fillna(1.0)
        normalized["rsi14"] = HistoricalDataAgent._compute_rsi(normalized["close"], 14)
        return normalized

    @staticmethod
    def _compute_rsi(close_series: pd.Series, period: int) -> pd.Series:
        delta = close_series.diff().fillna(0.0)
        gains = delta.clip(lower=0)
        losses = (-delta.clip(upper=0))
        avg_gain = gains.rolling(period, min_periods=1).mean()
        avg_loss = losses.rolling(period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)


class HistoricalRiskAdapter:
    """Adapter around live RiskAgent formulas for replay."""

    def __init__(self, risk_agent: Optional[RiskAgent] = None) -> None:
        self.risk_agent = risk_agent or RiskAgent()

    def run(
        self,
        *,
        code: str,
        trade_day: date,
        current_price: float,
        signal_output: SignalAgentOutput,
        simulator: "HistoricalExecutionSimulator",
        runtime_strategy: RuntimeStrategyConfig,
    ) -> RiskAgentOutput:
        account_snapshot = simulator.account_snapshot(code, current_price)
        current_position_value = simulator.position_qty * current_price
        return self.risk_agent.run(
            code=code,
            trade_date=trade_day,
            current_price=current_price,
            signal_output=signal_output,
            account_snapshot=account_snapshot,
            current_position_value=current_position_value,
            runtime_strategy=runtime_strategy,
        )


class HistoricalExecutionSimulator:
    """Historical execution simulator with next-open fills and intraday stops."""

    def __init__(self, initial_capital: float, commission_rate: float, slippage_bps: float) -> None:
        self.initial_capital = float(initial_capital)
        self.commission_rate = max(0.0, float(commission_rate))
        self.slippage_rate = max(0.0, float(slippage_bps) / 10000.0)
        self.cash = float(initial_capital)
        self.position_qty = 0
        self.avg_cost = 0.0
        self.pending_action: Dict[str, Any] | None = None
        self.current_entry: Dict[str, Any] | None = None
        self.trades: List[Dict[str, Any]] = []
        self.benchmark_shares = 0
        self.benchmark_cash = float(initial_capital)
        self._benchmark_initialized = False

    def start_day(self, bar: ReplayBar) -> Dict[str, Any]:
        self._ensure_benchmark(bar.close)
        payload = {
            "action": "none",
            "reason": "no_fill",
            "fill_price": None,
            "traded_qty": 0,
            "fee": 0.0,
            "cash_before": round(self.cash, 4),
            "cash_after": round(self.cash, 4),
            "position_before": self.position_qty,
            "position_after": self.position_qty,
        }
        action = self.pending_action
        self.pending_action = None
        if not action:
            return payload

        open_price = max(0.01, float(bar.open))
        if action["action"] == "buy" and self.position_qty <= 0:
            fill_price = open_price * (1.0 + self.slippage_rate)
            estimated_unit_cost = fill_price * (1.0 + self.commission_rate)
            target_qty = max(0, int(action.get("target_qty") or 0))
            affordable = int(self.cash // estimated_unit_cost) if estimated_unit_cost > 0 else 0
            qty = min(target_qty, affordable)
            if qty > 0:
                notional = qty * fill_price
                fee = notional * self.commission_rate
                cash_before = self.cash
                self.cash -= (notional + fee)
                self.position_qty = qty
                self.avg_cost = fill_price
                self.current_entry = {
                    "entry_date": bar.day.isoformat(),
                    "entry_price": round(fill_price, 4),
                    "qty": qty,
                    "entry_fees": fee,
                }
                return {
                    "action": "buy",
                    "reason": action.get("reason") or "signal_entry",
                    "fill_price": round(fill_price, 4),
                    "traded_qty": qty,
                    "fee": round(fee, 6),
                    "cash_before": round(cash_before, 4),
                    "cash_after": round(self.cash, 4),
                    "position_before": 0,
                    "position_after": self.position_qty,
                }
            return {**payload, "reason": "buy_rejected_cash"}

        if action["action"] == "sell" and self.position_qty > 0:
            fill_price = open_price * (1.0 - self.slippage_rate)
            return self._close_position(bar.day, fill_price, str(action.get("reason") or "signal_exit"), payload)

        return payload

    def check_intraday_exit(self, bar: ReplayBar, runtime_strategy: RuntimeStrategyConfig) -> Dict[str, Any] | None:
        if self.position_qty <= 0 or self.avg_cost <= 0:
            return None

        stop_loss = None
        take_profit = None
        if runtime_strategy.stop_loss_pct and float(runtime_strategy.stop_loss_pct) > 0:
            stop_loss = self.avg_cost * (1.0 - float(runtime_strategy.stop_loss_pct) / 100.0)
        if runtime_strategy.take_profit_pct and float(runtime_strategy.take_profit_pct) > 0:
            take_profit = self.avg_cost * (1.0 + float(runtime_strategy.take_profit_pct) / 100.0)

        stop_hit = stop_loss is not None and bar.low <= stop_loss
        take_hit = take_profit is not None and bar.high >= take_profit
        if not stop_hit and not take_hit:
            return None

        if stop_hit:
            return self._close_position(bar.day, float(stop_loss), "stop_loss", None)
        return self._close_position(bar.day, float(take_profit), "take_profit", None)

    def plan_next_open(self, risk_output: RiskAgentOutput, current_price: float, is_last_day: bool) -> Dict[str, Any]:
        if is_last_day:
            self.pending_action = None
            return {
                "pending_action": "none",
                "pending_reason": "last_day_no_new_entry",
                "target_qty": 0,
            }

        if self.position_qty > 0:
            if float(risk_output.target_weight or 0.0) <= 0 or float(risk_output.target_notional or 0.0) <= 0:
                self.pending_action = {"action": "sell", "reason": "signal_exit"}
                return {
                    "pending_action": "sell",
                    "pending_reason": "signal_exit",
                    "target_qty": 0,
                }
            self.pending_action = None
            return {
                "pending_action": "none",
                "pending_reason": "hold_position",
                "target_qty": self.position_qty,
            }

        if float(risk_output.target_notional or 0.0) > 0 and current_price > 0:
            target_qty = int(float(risk_output.target_notional) // current_price)
            if target_qty > 0:
                self.pending_action = {"action": "buy", "reason": "signal_entry", "target_qty": target_qty}
                return {
                    "pending_action": "buy",
                    "pending_reason": "signal_entry",
                    "target_qty": target_qty,
                }

        self.pending_action = None
        return {
            "pending_action": "none",
            "pending_reason": "no_entry_signal",
            "target_qty": 0,
        }

    def equity_point(self, bar: ReplayBar) -> Dict[str, Any]:
        position_value = self.position_qty * bar.close
        equity = self.cash + position_value
        benchmark_equity = self.benchmark_cash + self.benchmark_shares * bar.close
        position_ratio = (position_value / equity) if equity > 0 else 0.0
        return {
            "trade_date": bar.day.isoformat(),
            "equity": round(equity, 4),
            "benchmark_equity": round(benchmark_equity, 4),
            "position_ratio": round(position_ratio * 100.0, 4),
            "cash": round(self.cash, 4),
        }

    def account_snapshot(self, code: str, current_price: float) -> Dict[str, Any]:
        market_value = self.position_qty * current_price
        total_asset = self.cash + market_value
        positions: List[Dict[str, Any]] = []
        if self.position_qty > 0:
            positions.append(
                {
                    "code": code,
                    "quantity": self.position_qty,
                    "available_qty": self.position_qty,
                    "avg_cost": round(self.avg_cost, 4),
                    "market_value": round(market_value, 4),
                }
            )
        return {
            "cash": round(self.cash, 4),
            "total_asset": round(total_asset, 4),
            "total_market_value": round(market_value, 4),
            "positions": positions,
        }

    def _ensure_benchmark(self, first_close: float) -> None:
        if self._benchmark_initialized or first_close <= 0:
            return
        self._benchmark_initialized = True
        self.benchmark_shares = int(self.initial_capital // first_close)
        self.benchmark_cash = self.initial_capital - self.benchmark_shares * first_close

    def _close_position(
        self,
        trade_day: date,
        fill_price: float,
        reason: str,
        base_payload: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        qty = self.position_qty
        if qty <= 0:
            return base_payload or {}

        fill_price = max(0.01, float(fill_price))
        cash_before = self.cash
        gross = qty * fill_price
        fee = gross * self.commission_rate
        self.cash += gross - fee
        entry = self.current_entry or {
            "entry_date": trade_day.isoformat(),
            "entry_price": self.avg_cost,
            "qty": qty,
            "entry_fees": 0.0,
        }
        entry_price = float(entry.get("entry_price") or self.avg_cost or fill_price)
        entry_fees = float(entry.get("entry_fees") or 0.0)
        gross_return_pct = ((fill_price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
        notional = entry_price * qty
        net_pnl = (fill_price - entry_price) * qty - entry_fees - fee
        net_return_pct = (net_pnl / notional) * 100.0 if notional > 0 else 0.0
        self.trades.append(
            {
                "entry_date": str(entry.get("entry_date") or trade_day.isoformat()),
                "exit_date": trade_day.isoformat(),
                "entry_price": round(entry_price, 4),
                "exit_price": round(fill_price, 4),
                "qty": qty,
                "gross_return_pct": round(gross_return_pct, 4),
                "net_return_pct": round(net_return_pct, 4),
                "fees": round(entry_fees + fee, 6),
                "exit_reason": reason,
            }
        )
        self.position_qty = 0
        self.avg_cost = 0.0
        self.current_entry = None
        payload = base_payload or {}
        payload.update(
            {
                "action": "sell",
                "reason": reason,
                "fill_price": round(fill_price, 4),
                "traded_qty": qty,
                "fee": round(fee, 6),
                "cash_before": round(cash_before, 4),
                "cash_after": round(self.cash, 4),
                "position_before": qty,
                "position_after": 0,
            }
        )
        return payload


class HistoricalSignalReplayService:
    """Build deterministic signals and refine sparse anchor days with AI."""

    def __init__(
        self,
        trend_analyzer: Optional[StockTrendAnalyzer] = None,
        ai_analyzer_factory: Optional[Callable[[], Any]] = None,
        runtime_llm: Optional[RuntimeLlmConfig] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.trend_analyzer = trend_analyzer or StockTrendAnalyzer()
        resolved_config = (config or get_config()).clone_for_runtime_llm(runtime_llm)
        self._ai_analyzer_factory = ai_analyzer_factory or (lambda: GeminiAnalyzer(config=resolved_config, runtime_llm=runtime_llm))
        self._ai_analyzer: Any | None = None

    def build_anchor_days(self, trade_frame: pd.DataFrame, fast_snapshots: List[Dict[str, Any]]) -> List[str]:
        anchors: Dict[str, int] = {}
        previous = None
        for index, (_, row) in enumerate(trade_frame.iterrows()):
            day = row["date"].date().isoformat()
            if index == 0:
                anchors[day] = 0
            if index % 5 == 0:
                anchors[day] = min(anchors.get(day, 99), 3)

            fast = fast_snapshots[index]
            current_signature = (
                str(fast["signal_payload"].get("operation_advice") or ""),
                str(fast["factor_payload"].get("ma_bucket") or ""),
                str(fast["factor_payload"].get("rsi_bucket") or ""),
                str(fast["factor_payload"].get("momentum_sign") or ""),
            )
            if previous is not None:
                if current_signature[0] != previous[0]:
                    anchors[day] = min(anchors.get(day, 99), 1)
                if current_signature[1:] != previous[1:]:
                    anchors[day] = min(anchors.get(day, 99), 2)
            previous = current_signature

        ordered = sorted(anchors.items(), key=lambda item: (item[1], item[0]))
        return [day for day, _priority in ordered[:MAX_LLM_ANCHOR_CALLS]]

    def build_fast_snapshot(
        self,
        *,
        code: str,
        trade_day: date,
        window_frame: pd.DataFrame,
        archived_news_payload: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trend_result = None
        if not window_frame.empty:
            trend_result = self.trend_analyzer.analyze(window_frame.copy(), code)
        factor_payload = self._build_factor_payload(window_frame, trend_result)
        advice = self._fast_advice_from_trend(trend_result)
        sentiment = int(factor_payload.get("signal_score") or 50)
        signal_payload = {
            "operation_advice": advice,
            "sentiment_score": sentiment,
            "trend_signal": str(factor_payload.get("buy_signal") or "WAIT"),
            "trend_score": int(factor_payload.get("signal_score") or 0),
            "stop_loss": None,
            "take_profit": None,
            "resolved_stop_loss": None,
            "resolved_take_profit": None,
        }
        return {
            "trade_date": trade_day.isoformat(),
            "decision_source": "fast_rule",
            "llm_used": False,
            "confidence": round(min(max(sentiment / 100.0, 0.0), 1.0), 4),
            "factor_payload": factor_payload,
            "archived_news_payload": archived_news_payload,
            "signal_payload": signal_payload,
            "ai_overlay": {},
        }

    def load_cached_snapshot(self, cached: Dict[str, Any], fallback_fast: Dict[str, Any]) -> Dict[str, Any]:
        factor_payload = _as_dict(cached.get("factor_payload")) or fallback_fast["factor_payload"]
        signal_payload = _as_dict(cached.get("signal_payload")) or fallback_fast["signal_payload"]
        archived_news_payload = _as_list_of_dicts(cached.get("archived_news_payload")) or fallback_fast["archived_news_payload"]
        ai_overlay = _as_dict(cached.get("ai_overlay"))
        return {
            "trade_date": str(cached.get("trade_date") or fallback_fast["trade_date"]),
            "decision_source": str(cached.get("decision_source") or fallback_fast["decision_source"]),
            "llm_used": bool(cached.get("llm_used")),
            "confidence": _to_float(cached.get("confidence"), fallback_fast["confidence"]),
            "factor_payload": factor_payload,
            "archived_news_payload": archived_news_payload,
            "signal_payload": signal_payload,
            "ai_overlay": ai_overlay,
        }

    def refine_anchor_snapshot(
        self,
        *,
        code: str,
        trade_day: date,
        window_frame: pd.DataFrame,
        archived_news_payload: List[Dict[str, Any]],
        fast_snapshot: Dict[str, Any],
        account_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        analyzer = self._get_ai_analyzer()
        context = self._build_ai_context(
            code=code,
            trade_day=trade_day,
            window_frame=window_frame,
            fast_snapshot=fast_snapshot,
            account_snapshot=account_snapshot,
        )
        news_context = self._build_news_context(archived_news_payload)
        ai_result = analyzer.analyze(context, news_context=news_context)
        sniper_points = ai_result.get_sniper_points() if hasattr(ai_result, "get_sniper_points") else {}
        ai_overlay = {
            "operation_advice": getattr(ai_result, "operation_advice", None),
            "sentiment_score": getattr(ai_result, "sentiment_score", None),
            "analysis_summary": getattr(ai_result, "analysis_summary", None),
            "trend_prediction": getattr(ai_result, "trend_prediction", None),
            "stop_loss": _to_float(sniper_points.get("stop_loss")),
            "take_profit": _to_float(sniper_points.get("take_profit")),
        }
        signal_payload = dict(fast_snapshot["signal_payload"])
        if ai_overlay.get("operation_advice"):
            signal_payload["operation_advice"] = str(ai_overlay["operation_advice"])
        if ai_overlay.get("sentiment_score") is not None:
            signal_payload["sentiment_score"] = _to_int(ai_overlay["sentiment_score"], signal_payload["sentiment_score"])
        if ai_overlay.get("stop_loss") is not None:
            signal_payload["stop_loss"] = ai_overlay["stop_loss"]
            signal_payload["resolved_stop_loss"] = ai_overlay["stop_loss"]
        if ai_overlay.get("take_profit") is not None:
            signal_payload["take_profit"] = ai_overlay["take_profit"]
            signal_payload["resolved_take_profit"] = ai_overlay["take_profit"]
        return {
            "trade_date": trade_day.isoformat(),
            "decision_source": "llm_anchor",
            "llm_used": True,
            "confidence": round(min(max(float(signal_payload.get("sentiment_score") or 50) / 100.0, 0.0), 1.0), 4),
            "factor_payload": fast_snapshot["factor_payload"],
            "archived_news_payload": archived_news_payload,
            "signal_payload": signal_payload,
            "ai_overlay": ai_overlay,
        }

    def apply_overlay(self, fast_snapshot: Dict[str, Any], overlay_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        ai_overlay = _as_dict(overlay_snapshot.get("ai_overlay"))
        if not ai_overlay:
            return fast_snapshot
        signal_payload = dict(fast_snapshot["signal_payload"])
        if ai_overlay.get("operation_advice"):
            signal_payload["operation_advice"] = str(ai_overlay["operation_advice"])
        if ai_overlay.get("sentiment_score") is not None:
            signal_payload["sentiment_score"] = _to_int(ai_overlay["sentiment_score"], signal_payload["sentiment_score"])
        if ai_overlay.get("stop_loss") is not None:
            signal_payload["stop_loss"] = _to_float(ai_overlay["stop_loss"])
            signal_payload["resolved_stop_loss"] = _to_float(ai_overlay["stop_loss"])
        if ai_overlay.get("take_profit") is not None:
            signal_payload["take_profit"] = _to_float(ai_overlay["take_profit"])
            signal_payload["resolved_take_profit"] = _to_float(ai_overlay["take_profit"])
        return {
            "trade_date": fast_snapshot["trade_date"],
            "decision_source": "refined",
            "llm_used": False,
            "confidence": overlay_snapshot.get("confidence", fast_snapshot.get("confidence")),
            "factor_payload": fast_snapshot["factor_payload"],
            "archived_news_payload": fast_snapshot["archived_news_payload"],
            "signal_payload": signal_payload,
            "ai_overlay": ai_overlay,
        }

    def to_signal_output(self, code: str, trade_day: date, snapshot: Dict[str, Any]) -> SignalAgentOutput:
        signal_payload = _as_dict(snapshot.get("signal_payload"))
        factor_payload = _as_dict(snapshot.get("factor_payload"))
        ai_overlay = _as_dict(snapshot.get("ai_overlay"))
        return SignalAgentOutput(
            code=code,
            trade_date=trade_day,
            state=AgentState.READY,
            operation_advice=str(signal_payload.get("operation_advice") or "观望"),
            sentiment_score=_to_int(signal_payload.get("sentiment_score"), 50),
            trend_signal=str(signal_payload.get("trend_signal") or factor_payload.get("buy_signal") or "WAIT"),
            trend_score=_to_int(signal_payload.get("trend_score"), _to_int(factor_payload.get("signal_score"), 0)),
            stop_loss=_to_float(signal_payload.get("stop_loss")),
            take_profit=_to_float(signal_payload.get("take_profit")),
            resolved_stop_loss=_to_float(signal_payload.get("resolved_stop_loss")),
            resolved_take_profit=_to_float(signal_payload.get("resolved_take_profit")),
            ai_refreshed=bool(snapshot.get("llm_used")),
            ai_payload=ai_overlay,
            trend_payload=factor_payload,
        )

    @staticmethod
    def _fast_advice_from_trend(trend_result: Optional[TrendAnalysisResult]) -> str:
        if trend_result is None:
            return "观望"
        if trend_result.buy_signal in (BuySignal.STRONG_BUY, BuySignal.BUY):
            return "买入"
        if trend_result.buy_signal == BuySignal.HOLD:
            return "持有"
        if trend_result.buy_signal in (BuySignal.SELL, BuySignal.STRONG_SELL):
            return "卖出"
        return "观望"

    def _get_ai_analyzer(self) -> Any:
        if self._ai_analyzer is None:
            self._ai_analyzer = self._ai_analyzer_factory()
        return self._ai_analyzer

    @staticmethod
    def _build_factor_payload(window_frame: pd.DataFrame, trend_result: Optional[TrendAnalysisResult]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if window_frame.empty:
            return payload
        latest = window_frame.iloc[-1]
        previous = window_frame.iloc[-2] if len(window_frame.index) > 1 else latest
        ma5 = float(latest.get("MA5", latest.get("ma5", latest["close"])))
        ma10 = float(latest.get("MA10", latest.get("ma10", latest["close"])))
        ma20 = float(latest.get("MA20", latest.get("ma20", latest["close"])))
        ma60 = float(latest.get("MA60", latest.get("ma60", latest["close"])))
        rsi14 = float(latest.get("rsi14", 50.0))
        momentum20 = float(latest.get("momentum20", 0.0))
        vol_ratio5 = float(latest.get("vol_ratio5", 1.0))
        ma_bucket = "mixed"
        if ma5 > ma10 > ma20:
            ma_bucket = "bull"
        elif ma5 < ma10 < ma20:
            ma_bucket = "bear"
        rsi_bucket = "neutral"
        if rsi14 >= 70:
            rsi_bucket = "overbought"
        elif rsi14 <= 30:
            rsi_bucket = "oversold"
        momentum_sign = "positive" if momentum20 >= 0 else "negative"
        payload.update(
            {
                "close": round(float(latest["close"]), 4),
                "open": round(float(latest["open"]), 4),
                "high": round(float(latest["high"]), 4),
                "low": round(float(latest["low"]), 4),
                "volume": round(float(latest["volume"]), 4),
                "amount": round(float(latest.get("amount", 0.0)), 4),
                "pct_chg": round(float(latest.get("pct_chg", 0.0)), 4),
                "ma5": round(ma5, 4),
                "ma10": round(ma10, 4),
                "ma20": round(ma20, 4),
                "ma60": round(ma60, 4),
                "rsi14": round(rsi14, 4),
                "momentum20": round(momentum20, 4),
                "vol_ratio5": round(vol_ratio5, 4),
                "recent_close_change_5d": round(float(latest["close"] / max(float(window_frame.iloc[max(0, len(window_frame.index) - 5)]["close"]), 0.01) - 1.0) * 100.0, 4),
                "recent_volume_change_1d": round(float(latest["volume"] / max(float(previous["volume"]), 1.0)), 4),
                "ma_bucket": ma_bucket,
                "rsi_bucket": rsi_bucket,
                "momentum_sign": momentum_sign,
            }
        )
        if trend_result is not None:
            payload.update(
                {
                    "trend_status": trend_result.trend_status.value,
                    "ma_alignment": trend_result.ma_alignment,
                    "trend_strength": round(float(trend_result.trend_strength), 4),
                    "buy_signal": trend_result.buy_signal.value,
                    "signal_score": int(trend_result.signal_score),
                    "signal_reasons": list(trend_result.signal_reasons),
                    "risk_factors": list(trend_result.risk_factors),
                    "bias_ma5": round(float(trend_result.bias_ma5), 4),
                    "bias_ma10": round(float(trend_result.bias_ma10), 4),
                    "bias_ma20": round(float(trend_result.bias_ma20), 4),
                    "volume_status": trend_result.volume_status.value,
                    "volume_trend": trend_result.volume_trend,
                }
            )
        return payload

    @staticmethod
    def _build_ai_context(
        *,
        code: str,
        trade_day: date,
        window_frame: pd.DataFrame,
        fast_snapshot: Dict[str, Any],
        account_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        latest = window_frame.iloc[-1]
        previous = window_frame.iloc[-2] if len(window_frame.index) > 1 else latest
        factor_payload = _as_dict(fast_snapshot.get("factor_payload"))
        ma_bucket = str(factor_payload.get("ma_bucket") or "mixed")
        ma_status = "均线缠绕"
        if ma_bucket == "bull":
            ma_status = "多头排列"
        elif ma_bucket == "bear":
            ma_status = "空头排列"
        runtime_account = dict(account_snapshot)
        runtime_account["position"] = account_snapshot.get("positions", [{}])[0] if account_snapshot.get("positions") else {}
        return {
            "code": code,
            "date": trade_day.isoformat(),
            "today": {
                "open": round(float(latest["open"]), 4),
                "high": round(float(latest["high"]), 4),
                "low": round(float(latest["low"]), 4),
                "close": round(float(latest["close"]), 4),
                "pct_chg": round(float(latest.get("pct_chg", 0.0)), 4),
                "volume": round(float(latest["volume"]), 4),
                "amount": round(float(latest.get("amount", 0.0)), 4),
                "ma5": round(float(latest.get("MA5", latest["close"])), 4),
                "ma10": round(float(latest.get("MA10", latest["close"])), 4),
                "ma20": round(float(latest.get("MA20", latest["close"])), 4),
                "ma60": round(float(latest.get("MA60", latest["close"])), 4),
            },
            "yesterday": {
                "close": round(float(previous["close"]), 4),
                "volume": round(float(previous["volume"]), 4),
            },
            "price_change_ratio": round(float(latest["close"] / max(float(previous["close"]), 0.01) - 1.0) * 100.0, 4),
            "volume_change_ratio": round(float(latest["volume"] / max(float(previous["volume"]), 1.0)), 4),
            "ma_status": ma_status,
            "trend_analysis": factor_payload,
            "runtime_account": runtime_account,
        }

    @staticmethod
    def _build_news_context(archived_news_payload: List[Dict[str, Any]]) -> str | None:
        if not archived_news_payload:
            return None
        parts: List[str] = []
        for item in archived_news_payload[:9]:
            published = str(item.get("published_date") or "")
            title = str(item.get("title") or "")
            snippet = str(item.get("snippet") or "")
            source = str(item.get("source") or "")
            parts.append(f"- [{published}] {title} ({source}) {snippet}".strip())
        return "\n".join(parts)


class AgentHistoricalBacktestService:
    """Stateless historical replay entry point for Backend_stock."""

    def __init__(
        self,
        fetcher_manager: Optional[DataFetcherManager] = None,
        trend_analyzer: Optional[StockTrendAnalyzer] = None,
        risk_agent: Optional[RiskAgent] = None,
        ai_analyzer_factory: Optional[Callable[[], Any]] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.config = config or get_config()
        self.data_agent = HistoricalDataAgent(fetcher_manager=fetcher_manager)
        self._trend_analyzer = trend_analyzer
        self._ai_analyzer_factory = ai_analyzer_factory
        self.risk_adapter = HistoricalRiskAdapter(risk_agent=risk_agent)

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        params = self._parse_payload(payload)
        signal_replay = HistoricalSignalReplayService(
            trend_analyzer=self._trend_analyzer,
            ai_analyzer_factory=self._ai_analyzer_factory,
            runtime_llm=params.runtime_llm,
            config=self.config,
        )
        archived_news_by_date = {
            str(day): _as_list_of_dicts(items)
            for day, items in _as_dict(payload.get("archived_news_by_date")).items()
        }
        cached_snapshots = self._cached_snapshot_map(_as_list_of_dicts(payload.get("cached_snapshots")))

        prepared, trade_window = self.data_agent.load(params)
        fast_snapshots: List[Dict[str, Any]] = []
        trade_days: List[date] = []
        for _, row in trade_window.iterrows():
            trade_day = row["date"].date()
            trade_days.append(trade_day)
            window_frame = self.data_agent.rolling_window(prepared, trade_day)
            fast_snapshots.append(
                signal_replay.build_fast_snapshot(
                    code=params.code,
                    trade_day=trade_day,
                    window_frame=window_frame,
                    archived_news_payload=self._collect_recent_news(archived_news_by_date, trade_day),
                )
            )

        anchor_days = signal_replay.build_anchor_days(trade_window, fast_snapshots)
        pending_anchor_days = [
            day for day in anchor_days
            if not self._is_refined_snapshot(cached_snapshots.get(day))
        ]

        simulator = HistoricalExecutionSimulator(
            initial_capital=params.initial_capital,
            commission_rate=params.commission_rate,
            slippage_bps=params.slippage_bps,
        )

        daily_steps: List[Dict[str, Any]] = []
        signal_snapshots: List[Dict[str, Any]] = []
        snapshot_hit_count = 0
        snapshot_miss_count = 0
        no_news_days = 0
        llm_anchor_count = 0
        divergence_days = 0
        active_overlay_snapshot: Dict[str, Any] | None = None

        for index, trade_day in enumerate(trade_days):
            bar = self._frame_row_to_bar(trade_window.iloc[index])
            open_execution = simulator.start_day(bar)
            intraday_execution = simulator.check_intraday_exit(bar, params.runtime_strategy)
            execution_stage = intraday_execution or open_execution

            archived_news_payload = self._collect_recent_news(archived_news_by_date, trade_day)
            if not archived_news_payload:
                no_news_days += 1
            window_frame = self.data_agent.rolling_window(prepared, trade_day)
            fast_snapshot = fast_snapshots[index]
            snapshot: Dict[str, Any]
            fast_operation_advice = str(fast_snapshot["signal_payload"].get("operation_advice") or "")

            if params.phase == "fast":
                cached = cached_snapshots.get(trade_day.isoformat())
                if cached:
                    snapshot = signal_replay.load_cached_snapshot(cached, fast_snapshot)
                    snapshot_hit_count += 1
                else:
                    snapshot = fast_snapshot
                    snapshot_miss_count += 1
            else:
                cached = cached_snapshots.get(trade_day.isoformat())
                if trade_day.isoformat() in anchor_days:
                    if self._is_refined_snapshot(cached):
                        snapshot = signal_replay.load_cached_snapshot(cached or {}, fast_snapshot)
                        snapshot_hit_count += 1
                    else:
                        snapshot = signal_replay.refine_anchor_snapshot(
                            code=params.code,
                            trade_day=trade_day,
                            window_frame=window_frame,
                            archived_news_payload=archived_news_payload,
                            fast_snapshot=fast_snapshot,
                            account_snapshot=simulator.account_snapshot(params.code, bar.close),
                        )
                        snapshot_miss_count += 1
                    active_overlay_snapshot = snapshot
                    llm_anchor_count += 1
                elif active_overlay_snapshot:
                    snapshot = signal_replay.apply_overlay(fast_snapshot, active_overlay_snapshot)
                else:
                    snapshot = fast_snapshot

                if str(snapshot["signal_payload"].get("operation_advice") or "") != fast_operation_advice:
                    divergence_days += 1

            signal_output = signal_replay.to_signal_output(params.code, trade_day, snapshot)
            risk_output = self.risk_adapter.run(
                code=params.code,
                trade_day=trade_day,
                current_price=bar.close,
                signal_output=signal_output,
                simulator=simulator,
                runtime_strategy=params.runtime_strategy,
            )
            pending_plan = simulator.plan_next_open(
                risk_output,
                current_price=bar.close,
                is_last_day=(index == len(trade_days) - 1),
            )

            stage_execution_payload = {
                **execution_stage,
                **pending_plan,
                "account_snapshot": simulator.account_snapshot(params.code, bar.close),
            }
            stage_signal_payload = signal_output.to_dict()
            stage_risk_payload = risk_output.to_dict()
            stage_data_payload = {
                "trade_date": trade_day.isoformat(),
                "bar": asdict(bar),
                "window_size": len(window_frame.index),
                "archived_news": archived_news_payload,
                "factor_payload": snapshot["factor_payload"],
            }

            daily_steps.append(
                {
                    "trade_date": trade_day.isoformat(),
                    "decision_source": str(snapshot.get("decision_source") or "fast_rule"),
                    "ai_used": bool(snapshot.get("llm_used")),
                    "data_payload": stage_data_payload,
                    "signal_payload": stage_signal_payload,
                    "risk_payload": stage_risk_payload,
                    "execution_payload": stage_execution_payload,
                }
            )
            signal_snapshots.append(snapshot)

        equity = self._with_drawdown(simulator, daily_steps)
        summary = self._build_summary(params.initial_capital, equity, simulator.trades)
        decision_source_breakdown = self._decision_source_breakdown(daily_steps)
        diagnostics = {
            "snapshot_hit_count": snapshot_hit_count,
            "snapshot_miss_count": snapshot_miss_count,
            "llm_anchor_count": llm_anchor_count,
            "no_news_days": no_news_days,
            "fast_refined_divergence_days": divergence_days if params.phase == "refine" else 0,
            "pending_anchor_dates": pending_anchor_days if params.phase == "fast" else [],
            "decision_source_breakdown": decision_source_breakdown,
        }

        return {
            "engine_version": ENGINE_VERSION,
            "code": params.code,
            "phase": params.phase,
            "requested_range": {
                "start_date": params.start_date.isoformat(),
                "end_date": params.end_date.isoformat(),
            },
            "effective_range": {
                "start_date": trade_days[0].isoformat() if trade_days else None,
                "end_date": trade_days[-1].isoformat() if trade_days else None,
            },
            "summary": {
                **summary,
                "llm_anchor_count": llm_anchor_count,
                "snapshot_hit_rate": round((snapshot_hit_count / len(trade_days)) * 100.0, 2) if trade_days else 0.0,
            },
            "diagnostics": diagnostics,
            "daily_steps": daily_steps,
            "trades": simulator.trades,
            "equity": equity,
            "signal_snapshots": signal_snapshots,
            "pending_anchor_dates": pending_anchor_days if params.phase == "fast" else [],
        }

    def _parse_payload(self, payload: Dict[str, Any]) -> HistoricalRunParams:
        code = canonical_stock_code(str(payload.get("code") or ""))
        if not code:
            raise ValueError("code is required")
        start_date = _to_day(payload.get("start_date"))
        end_date = _to_day(payload.get("end_date"))
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required")
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")
        phase = str(payload.get("phase") or "fast").strip().lower()
        if phase not in {"fast", "refine"}:
            raise ValueError("phase must be fast or refine")
        runtime_strategy_payload = _as_dict(payload.get("runtime_strategy"))
        runtime_strategy = RuntimeStrategyConfig(
            position_max_pct=_to_float(runtime_strategy_payload.get("position_max_pct"), 30.0),
            stop_loss_pct=_to_float(runtime_strategy_payload.get("stop_loss_pct"), 8.0),
            take_profit_pct=_to_float(runtime_strategy_payload.get("take_profit_pct"), 15.0),
        )
        runtime_llm_payload = _as_dict(payload.get("runtime_llm"))
        runtime_llm = None
        if runtime_llm_payload:
            provider = str(runtime_llm_payload.get("provider") or "").strip().lower()
            if provider not in {"gemini", "anthropic", "openai", "deepseek", "custom"}:
                raise ValueError("runtime_llm.provider must be one of gemini|anthropic|openai|deepseek|custom")
            base_url = str(runtime_llm_payload.get("base_url") or "").strip()
            model = str(runtime_llm_payload.get("model") or "").strip()
            if not base_url:
                raise ValueError("runtime_llm.base_url is required")
            if not model:
                raise ValueError("runtime_llm.model is required")
            api_token_raw = str(runtime_llm_payload.get("api_token") or "").strip()
            runtime_llm = RuntimeLlmConfig(
                provider=provider,
                base_url=base_url,
                model=model,
                api_token=api_token_raw or None,
                has_token=bool(runtime_llm_payload.get("has_token") or api_token_raw),
            )
        return HistoricalRunParams(
            code=code,
            start_date=start_date,
            end_date=end_date,
            phase=phase,
            initial_capital=max(1.0, _to_float(payload.get("initial_capital"), 100000.0) or 100000.0),
            commission_rate=max(0.0, _to_float(payload.get("commission_rate"), 0.0003) or 0.0003),
            slippage_bps=max(0.0, _to_float(payload.get("slippage_bps"), 2.0) or 2.0),
            runtime_strategy=runtime_strategy,
            runtime_llm=runtime_llm,
            signal_profile_hash=str(payload.get("signal_profile_hash") or SIGNAL_PROFILE_VERSION),
            snapshot_version=max(1, _to_int(payload.get("snapshot_version"), 1)),
        )

    @staticmethod
    def _cached_snapshot_map(cached_rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for row in cached_rows:
            trade_date = str(row.get("trade_date") or "")
            if trade_date:
                result[trade_date] = row
        return result

    @staticmethod
    def _is_refined_snapshot(row: Dict[str, Any] | None) -> bool:
        if not row:
            return False
        source = str(row.get("decision_source") or "")
        return bool(row.get("llm_used")) or source in {"llm_anchor", "refined"}

    @staticmethod
    def _collect_recent_news(
        archived_news_by_date: Dict[str, List[Dict[str, Any]]],
        trade_day: date,
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for offset in range(0, 4):
            day = (trade_day - timedelta(days=offset)).isoformat()
            payload.extend(archived_news_by_date.get(day, []))
        return payload[:9]

    @staticmethod
    def _frame_row_to_bar(row: pd.Series) -> ReplayBar:
        day = row["date"].date()
        return ReplayBar(
            day=day,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            amount=float(row.get("amount", row["close"] * row["volume"])),
            pct_chg=float(row.get("pct_chg", 0.0)),
            ma5=float(row.get("MA5", row["close"])),
            ma10=float(row.get("MA10", row["close"])),
            ma20=float(row.get("MA20", row["close"])),
            ma60=float(row.get("MA60", row["close"])),
            rsi14=float(row.get("rsi14", 50.0)),
            momentum20=float(row.get("momentum20", 0.0)),
            vol_ratio5=float(row.get("vol_ratio5", 1.0)),
        )

    @staticmethod
    def _with_drawdown(simulator: HistoricalExecutionSimulator, daily_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        peak = 0.0
        for row in daily_steps:
            execution_payload = _as_dict(row.get("execution_payload"))
            account_snapshot = _as_dict(execution_payload.get("account_snapshot"))
            equity = _to_float(account_snapshot.get("total_asset"), 0.0) or 0.0
            benchmark = 0.0
            data_payload = _as_dict(row.get("data_payload"))
            bar_payload = _as_dict(data_payload.get("bar"))
            close_price = _to_float(bar_payload.get("close"), 0.0) or 0.0
            benchmark = simulator.benchmark_cash + simulator.benchmark_shares * close_price
            peak = max(peak, equity)
            drawdown = ((peak - equity) / peak) * 100.0 if peak > 0 else 0.0
            points.append(
                {
                    "trade_date": row["trade_date"],
                    "equity": round(equity, 4),
                    "drawdown_pct": round(drawdown, 4),
                    "benchmark_equity": round(benchmark, 4),
                    "position_ratio": 0.0,
                    "cash": round(_to_float(account_snapshot.get("cash"), 0.0) or 0.0, 4),
                }
            )
            total_asset = _to_float(account_snapshot.get("total_asset"), 0.0) or 0.0
            market_value = _to_float(account_snapshot.get("total_market_value"), 0.0) or 0.0
            points[-1]["position_ratio"] = round((market_value / total_asset) * 100.0, 4) if total_asset > 0 else 0.0
        return points

    @staticmethod
    def _build_summary(initial_capital: float, equity: List[Dict[str, Any]], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_equity = float(equity[-1]["equity"]) if equity else float(initial_capital)
        final_benchmark = float(equity[-1]["benchmark_equity"]) if equity else float(initial_capital)
        total_return_pct = ((final_equity / initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0
        benchmark_return_pct = ((final_benchmark / initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0
        max_drawdown_pct = max((float(point.get("drawdown_pct") or 0.0) for point in equity), default=0.0)
        total_trades = len(trades)
        wins = sum(1 for item in trades if (_to_float(item.get("net_return_pct"), 0.0) or 0.0) > 0)
        win_rate_pct = (wins / total_trades) * 100.0 if total_trades > 0 else 0.0
        return {
            "initial_capital": round(initial_capital, 4),
            "final_equity": round(final_equity, 4),
            "total_return_pct": round(total_return_pct, 4),
            "benchmark_return_pct": round(benchmark_return_pct, 4),
            "excess_return_pct": round(total_return_pct - benchmark_return_pct, 4),
            "max_drawdown_pct": round(max_drawdown_pct, 4),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate_pct, 4),
        }

    @staticmethod
    def _decision_source_breakdown(daily_steps: List[Dict[str, Any]]) -> Dict[str, int]:
        payload: Dict[str, int] = {}
        for row in daily_steps:
            key = str(row.get("decision_source") or "unknown")
            payload[key] = payload.get(key, 0) + 1
        return payload


_agent_historical_backtest_service: AgentHistoricalBacktestService | None = None


def get_agent_historical_backtest_service() -> AgentHistoricalBacktestService:
    global _agent_historical_backtest_service
    if _agent_historical_backtest_service is None:
        _agent_historical_backtest_service = AgentHistoricalBacktestService()
    return _agent_historical_backtest_service
