# -*- coding: utf-8 -*-
"""为 Backend_stock 提供统一的固定行情源内部服务。"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from agent_stock.config import ALLOWED_MARKET_SOURCES, Config, get_config
from agent_stock.time_utils import shanghai_now
from data_provider import DataFetcherManager
from data_provider.base import DataSourceUnavailableError, canonical_stock_code, normalize_stock_code
from data_provider.realtime_types import CircuitBreaker, get_realtime_circuit_breaker

MARKET_SOURCE_META: dict[str, dict[str, str]] = {
    "tencent": {
        "label": "腾讯行情",
        "description": "腾讯通道，适合轻量级 A 股实时与历史行情请求。",
    },
    "sina": {
        "label": "新浪行情",
        "description": "新浪通道，适合快速获取单只股票实时与历史行情。",
    },
    "efinance": {
        "label": "EFinance",
        "description": "EFinance 通道，适合较完整的实时与历史行情数据。",
    },
    "eastmoney": {
        "label": "东方财富",
        "description": "东方财富通道，经 Akshare-EM 获取实时与历史行情。",
    },
    "tushare": {
        "label": "Tushare",
        "description": "Tushare 通道，需先配置 TUSHARE_TOKEN。",
    },
}

MARKET_SOURCE_REALTIME_BREAKER_KEY: dict[str, str] = {
    "eastmoney": "akshare_em",
    "sina": "akshare_sina",
    "tencent": "tencent",
    "efinance": "efinance",
    "tushare": "tushare",
}

QUOTE_FALLBACK_ORDER: dict[str, tuple[str, ...]] = {
    "eastmoney": ("tencent", "sina", "efinance", "tushare"),
    "sina": ("tencent", "efinance", "eastmoney", "tushare"),
    "tencent": ("sina", "efinance", "eastmoney", "tushare"),
    "efinance": ("tencent", "sina", "eastmoney", "tushare"),
    "tushare": ("tencent", "sina", "efinance", "eastmoney"),
}

_runtime_market_service: RuntimeMarketService | None = None


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        if pd.isna(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _normalize_window_value(value: Any) -> int | None:
    """将窗口值收窄为整数，避免对非浮点值调用浮点专有方法。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _date_text(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    text = str(value or "").strip()
    return text[:10]


class RuntimeMarketService:
    """为 Backend 提供固定行情源候选列表与股票行情查询。"""

    def __init__(
        self,
        *,
        config: Config | None = None,
        fetcher_manager: DataFetcherManager | None = None,
        realtime_circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self.config = config or get_config()
        self.fetcher_manager = fetcher_manager or DataFetcherManager()
        self.realtime_circuit_breaker = realtime_circuit_breaker or get_realtime_circuit_breaker()

    def get_market_source_options(self) -> dict[str, Any]:
        available_fetchers = set(self.fetcher_manager.available_fetchers)
        options: list[dict[str, Any]] = []

        for code in ALLOWED_MARKET_SOURCES:
            meta = MARKET_SOURCE_META[code]
            available = True
            reason: str | None = None

            if code in {"tencent", "sina", "eastmoney"} and "AkshareFetcher" not in available_fetchers:
                available = False
                reason = "AkshareFetcher 当前不可用"
            elif code == "efinance" and "EfinanceFetcher" not in available_fetchers:
                available = False
                reason = "EfinanceFetcher 当前不可用"
            elif code == "tushare":
                if "TushareFetcher" not in available_fetchers:
                    available = False
                    reason = "TushareFetcher 当前不可用"
                elif not str(self.config.tushare_token or "").strip():
                    available = False
                    reason = "未配置 TUSHARE_TOKEN"

            if available:
                breaker_key = MARKET_SOURCE_REALTIME_BREAKER_KEY.get(code)
                if breaker_key:
                    breaker_state = self.realtime_circuit_breaker.get_state_info(breaker_key)
                    if breaker_state["state"] == CircuitBreaker.OPEN:
                        available = False
                        reason = self._format_circuit_breaker_reason(code, breaker_state)

            option = {
                "code": code,
                "label": meta["label"],
                "description": meta["description"],
                "available": available,
            }
            if reason:
                option["reason"] = reason
            options.append(option)

        return {"options": options}

    def get_quote(self, stock_code: str, market_source: str) -> dict[str, Any]:
        code = self._normalize_stock_code(stock_code)
        requested_source = self._normalize_market_source(market_source)
        quote, effective_source = self._get_quote_with_fallback(code, requested_source)
        stock_name = str(getattr(quote, "name", "") or "").strip() or canonical_stock_code(code)

        payload = {
            "stock_code": code,
            "stock_name": stock_name,
            "current_price": _to_float(getattr(quote, "price", None)),
            "change": _to_float(getattr(quote, "change_amount", None)),
            "change_percent": _to_float(getattr(quote, "change_pct", None)),
            "open": _to_float(getattr(quote, "open_price", None)),
            "high": _to_float(getattr(quote, "high", None)),
            "low": _to_float(getattr(quote, "low", None)),
            "prev_close": _to_float(getattr(quote, "pre_close", None)),
            "volume": _to_int(getattr(quote, "volume", None)),
            "amount": _to_float(getattr(quote, "amount", None)),
            "update_time": self._now_iso(),
            "source": effective_source,
        }
        if effective_source != requested_source:
            payload["requested_source"] = requested_source
            payload["warning"] = self._format_quote_fallback_warning(requested_source, effective_source)
        return payload

    def get_history(self, stock_code: str, days: int, market_source: str) -> dict[str, Any]:
        code = self._normalize_stock_code(stock_code)
        requested_source = self._normalize_market_source(market_source)
        frame, effective_source = self._get_daily_frame_with_fallback(code, requested_source, days=max(days, 30))
        rows = self._frame_to_history_rows(frame)[-days:]
        stock_name = self._resolve_stock_name(code, effective_source)

        payload = {
            "stock_code": code,
            "stock_name": stock_name,
            "period": "daily",
            "data": rows,
            "source": effective_source,
        }
        return self._attach_requested_source_metadata(
            payload,
            requested_source=requested_source,
            effective_source=effective_source,
            warning=self._format_daily_fallback_warning(requested_source, effective_source),
        )

    def get_indicators(self, stock_code: str, days: int, windows: list[int], market_source: str) -> dict[str, Any]:
        code = self._normalize_stock_code(stock_code)
        requested_source = self._normalize_market_source(market_source)
        normalized_windows = self._normalize_windows(windows)
        lookback_days = max(days + max(normalized_windows, default=0), 120)
        frame, effective_source = self._get_daily_frame_with_fallback(code, requested_source, days=lookback_days)
        bars = self._frame_to_indicator_bars(frame)
        items = self._build_indicator_items(bars, normalized_windows)[-days:]

        payload = {
            "stock_code": code,
            "period": "daily",
            "days": days,
            "windows": normalized_windows,
            "items": items,
            "source": effective_source,
        }
        return self._attach_requested_source_metadata(
            payload,
            requested_source=requested_source,
            effective_source=effective_source,
            warning=self._format_daily_fallback_warning(requested_source, effective_source),
        )

    def get_factors(self, stock_code: str, market_source: str, target_date: str | None = None) -> dict[str, Any]:
        code = self._normalize_stock_code(stock_code)
        requested_source = self._normalize_market_source(market_source)
        frame, effective_source = self._get_daily_frame_with_fallback(code, requested_source, days=365)
        bars = self._frame_to_indicator_bars(frame)
        index = self._find_nearest_index_by_date(bars, target_date)
        if index < 0:
            raise ValueError("No available daily bar for the specified date")

        factors = self._compute_factors_at(bars, index)
        payload = {
            "stock_code": code,
            "date": bars[index]["date"],
            "factors": factors,
            "source": effective_source,
        }
        return self._attach_requested_source_metadata(
            payload,
            requested_source=requested_source,
            effective_source=effective_source,
            warning=self._format_daily_fallback_warning(requested_source, effective_source),
        )

    @staticmethod
    def _normalize_market_source(source: str) -> str:
        normalized = str(source or "").strip().lower()
        if normalized not in ALLOWED_MARKET_SOURCES:
            allowed = ", ".join(ALLOWED_MARKET_SOURCES)
            raise DataSourceUnavailableError(f"unsupported market source: {normalized or '<empty>'}. allowed={allowed}")
        return normalized

    @staticmethod
    def _normalize_stock_code(stock_code: str) -> str:
        normalized = normalize_stock_code(stock_code)
        text = canonical_stock_code(normalized)
        if not text:
            raise ValueError("stock_code is required")
        return text

    @staticmethod
    def _normalize_windows(windows: list[int]) -> list[int]:
        cleaned: list[int] = []
        for value in windows:
            normalized = _normalize_window_value(value)
            if normalized is not None:
                cleaned.append(normalized)
        unique = sorted({value for value in cleaned if 0 < value <= 250})
        return unique or [5, 10, 20, 60]

    @staticmethod
    def _now_iso() -> str:
        return shanghai_now().replace(microsecond=0).isoformat()

    def _resolve_stock_name(self, stock_code: str, market_source: str) -> str:
        try:
            quote = self.fetcher_manager.get_realtime_quote(stock_code, fixed_source=market_source)
            if quote is not None:
                name = str(getattr(quote, "name", "") or "").strip()
                if name:
                    return name
        except Exception:
            pass
        return canonical_stock_code(stock_code)

    def _load_daily_frame(self, stock_code: str, *, source: str, days: int) -> pd.DataFrame:
        frame, _resolved_source = self.fetcher_manager.get_daily_data(
            stock_code,
            days=max(days, 30),
            fixed_source=source,
        )
        if frame is None or frame.empty:
            raise DataSourceUnavailableError(f"{source} returned no history data for {stock_code}")
        return frame.sort_values("date", ascending=True).reset_index(drop=True)

    def _frame_to_history_rows(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        previous_close: float | None = None

        for _, row in frame.iterrows():
            close = _to_float(row.get("close"))
            change_percent = _to_float(row.get("pct_chg"))
            if change_percent is None and previous_close and previous_close > 0 and close is not None:
                change_percent = ((close - previous_close) / previous_close) * 100

            rows.append(
                {
                    "date": _date_text(row.get("date")),
                    "open": _to_float(row.get("open")),
                    "high": _to_float(row.get("high")),
                    "low": _to_float(row.get("low")),
                    "close": close,
                    "volume": _to_int(row.get("volume")),
                    "amount": _to_float(row.get("amount")),
                    "change_percent": _round4(change_percent),
                }
            )
            previous_close = close if close is not None else previous_close

        return rows

    def _frame_to_indicator_bars(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        bars: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            date_text = _date_text(row.get("date"))
            if len(date_text) != 10:
                continue
            bars.append(
                {
                    "date": date_text,
                    "open": _to_float(row.get("open")),
                    "high": _to_float(row.get("high")),
                    "low": _to_float(row.get("low")),
                    "close": _to_float(row.get("close")),
                    "volume": _to_float(row.get("volume")),
                }
            )
        return bars

    @staticmethod
    def _average(values: list[float | None]) -> float | None:
        filtered = [value for value in values if value is not None]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    def _compute_moving_average_at(self, bars: list[dict[str, Any]], index: int, window: int) -> float | None:
        if window <= 0 or index + 1 < window:
            return None
        closes = [bar.get("close") for bar in bars[index + 1 - window:index + 1]]
        return _round4(self._average(closes))

    def _compute_rsi14_at(self, bars: list[dict[str, Any]], index: int) -> float | None:
        if index < 14:
            return None

        closes = [bar.get("close") for bar in bars[:index + 1]]
        if any(close is None for close in closes):
            return None

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]  # type: ignore[operator]
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
        return _round4(100 - 100 / (1 + rs))

    @staticmethod
    def _compute_momentum20_at(bars: list[dict[str, Any]], index: int) -> float | None:
        if index < 20:
            return None
        current = bars[index].get("close")
        base = bars[index - 20].get("close")
        if current is None or base is None or base == 0:
            return None
        return _round4(((current / base) - 1) * 100)

    def _compute_vol_ratio5_at(self, bars: list[dict[str, Any]], index: int) -> float | None:
        if index < 4:
            return None
        current_volume = bars[index].get("volume")
        if current_volume is None:
            return None
        volumes = [bar.get("volume") for bar in bars[index - 4:index + 1]]
        avg_volume = self._average(volumes)
        if avg_volume is None or avg_volume == 0:
            return None
        return _round4(current_volume / avg_volume)

    @staticmethod
    def _compute_amplitude_at(bars: list[dict[str, Any]], index: int) -> float | None:
        bar = bars[index]
        open_price = bar.get("open")
        high = bar.get("high")
        low = bar.get("low")
        if open_price is None or high is None or low is None or open_price == 0:
            return None
        return _round4(((high - low) / open_price) * 100)

    def _build_indicator_items(self, bars: list[dict[str, Any]], windows: list[int]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for index, bar in enumerate(bars):
            mas = {
                f"ma{window}": self._compute_moving_average_at(bars, index, window)
                for window in windows
            }
            items.append(
                {
                    "date": bar["date"],
                    "close": bar.get("close"),
                    "mas": mas,
                }
            )
        return items

    @staticmethod
    def _find_nearest_index_by_date(bars: list[dict[str, Any]], target_date: str | None) -> int:
        if not bars:
            return -1
        if not target_date:
            return len(bars) - 1
        normalized = str(target_date).strip()[:10]
        for index in range(len(bars) - 1, -1, -1):
            if bars[index]["date"] <= normalized:
                return index
        return -1

    def _compute_factors_at(self, bars: list[dict[str, Any]], index: int) -> dict[str, Any]:
        return {
            "ma5": self._compute_moving_average_at(bars, index, 5),
            "ma10": self._compute_moving_average_at(bars, index, 10),
            "ma20": self._compute_moving_average_at(bars, index, 20),
            "ma60": self._compute_moving_average_at(bars, index, 60),
            "rsi14": self._compute_rsi14_at(bars, index),
            "momentum20": self._compute_momentum20_at(bars, index),
            "volRatio5": self._compute_vol_ratio5_at(bars, index),
            "amplitude": self._compute_amplitude_at(bars, index),
        }

    @staticmethod
    def _dedupe_sources(sources: list[str]) -> list[str]:
        deduped: list[str] = []
        for source in sources:
            if source in ALLOWED_MARKET_SOURCES and source not in deduped:
                deduped.append(source)
        return deduped

    def _build_candidate_sources(self, requested_source: str) -> list[str]:
        return self._dedupe_sources([requested_source, *QUOTE_FALLBACK_ORDER.get(requested_source, ())])

    def _get_quote_from_source(self, stock_code: str, source: str):
        quote = self.fetcher_manager.get_realtime_quote(stock_code, fixed_source=source)
        if quote is None or not quote.has_basic_data():
            raise DataSourceUnavailableError(f"{source} realtime quote returned no usable data for {stock_code}")
        return quote

    def _get_quote_with_fallback(self, stock_code: str, requested_source: str) -> tuple[Any, str]:
        errors: list[str] = []

        for source in self._build_candidate_sources(requested_source):
            try:
                return self._get_quote_from_source(stock_code, source), source
            except Exception as error:
                errors.append(f"{source}: {str(error).strip() or 'unknown error'}")

        joined = "; ".join(errors) if errors else f"{requested_source} realtime quote returned no usable data for {stock_code}"
        raise DataSourceUnavailableError(joined)

    def _get_daily_frame_with_fallback(self, stock_code: str, requested_source: str, *, days: int) -> tuple[pd.DataFrame, str]:
        errors: list[str] = []

        for source in self._build_candidate_sources(requested_source):
            try:
                return self._load_daily_frame(stock_code, source=source, days=days), source
            except Exception as error:
                errors.append(f"{source}: {str(error).strip() or 'unknown error'}")

        joined = "; ".join(errors) if errors else f"{requested_source} daily data returned no usable rows for {stock_code}"
        raise DataSourceUnavailableError(joined)

    @staticmethod
    def _attach_requested_source_metadata(
        payload: dict[str, Any],
        *,
        requested_source: str,
        effective_source: str,
        warning: str,
    ) -> dict[str, Any]:
        if effective_source != requested_source:
            payload["requested_source"] = requested_source
            payload["warning"] = warning
        return payload

    @staticmethod
    def _market_source_label(source: str) -> str:
        return MARKET_SOURCE_META.get(source, {}).get("label", source)

    def _format_quote_fallback_warning(self, requested_source: str, effective_source: str) -> str:
        requested_label = self._market_source_label(requested_source)
        effective_label = self._market_source_label(effective_source)
        return f"实时行情源 {requested_label} 暂不可用，已自动降级到 {effective_label}"

    def _format_daily_fallback_warning(self, requested_source: str, effective_source: str) -> str:
        requested_label = self._market_source_label(requested_source)
        effective_label = self._market_source_label(effective_source)
        return f"日线行情源 {requested_label} 暂不可用，已自动降级到 {effective_label}"

    def _format_circuit_breaker_reason(self, market_source: str, state: dict[str, Any]) -> str:
        remaining_seconds = max(1, math.ceil(float(state.get("remaining_cooldown_seconds") or 0.0)))
        source_label = self._market_source_label(market_source)
        last_error = str(state.get("last_error") or "").strip()
        if last_error:
            return f"{source_label} 实时行情熔断中，约 {remaining_seconds} 秒后自动恢复；最近错误：{last_error}"
        return f"{source_label} 实时行情熔断中，约 {remaining_seconds} 秒后自动恢复"


def get_runtime_market_service(config: Config | None = None) -> RuntimeMarketService:
    """返回内部市场服务单例。"""
    global _runtime_market_service
    if _runtime_market_service is None:
        _runtime_market_service = RuntimeMarketService(config=config)
    return _runtime_market_service


def reset_runtime_market_service() -> None:
    """重置内部市场服务单例，供测试使用。"""
    global _runtime_market_service
    _runtime_market_service = None
    get_realtime_circuit_breaker().reset()
