# -*- coding: utf-8 -*-
"""为 Backend_stock 提供统一的固定行情源内部服务。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from agent_stock.config import ALLOWED_MARKET_SOURCES, Config, get_config
from data_provider import DataFetcherManager
from data_provider.base import DataSourceUnavailableError, canonical_stock_code, normalize_stock_code

MARKET_SOURCE_META: Dict[str, Dict[str, str]] = {
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

_runtime_market_service: "RuntimeMarketService | None" = None


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
        config: Optional[Config] = None,
        fetcher_manager: Optional[DataFetcherManager] = None,
    ) -> None:
        self.config = config or get_config()
        self.fetcher_manager = fetcher_manager or DataFetcherManager()

    def get_market_source_options(self) -> Dict[str, Any]:
        available_fetchers = set(self.fetcher_manager.available_fetchers)
        options: List[Dict[str, Any]] = []

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

    def get_quote(self, stock_code: str, market_source: str) -> Dict[str, Any]:
        code = self._normalize_stock_code(stock_code)
        source = self._normalize_market_source(market_source)
        quote = self.fetcher_manager.get_realtime_quote(code, fixed_source=source)
        stock_name = str(getattr(quote, "name", "") or "").strip() or canonical_stock_code(code)

        return {
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
            "source": source,
        }

    def get_history(self, stock_code: str, days: int, market_source: str) -> Dict[str, Any]:
        code = self._normalize_stock_code(stock_code)
        source = self._normalize_market_source(market_source)
        frame = self._load_daily_frame(code, source=source, days=max(days, 30))
        rows = self._frame_to_history_rows(frame)[-days:]
        stock_name = self._resolve_stock_name(code, source)

        return {
            "stock_code": code,
            "stock_name": stock_name,
            "period": "daily",
            "data": rows,
            "source": source,
        }

    def get_indicators(self, stock_code: str, days: int, windows: List[int], market_source: str) -> Dict[str, Any]:
        code = self._normalize_stock_code(stock_code)
        source = self._normalize_market_source(market_source)
        normalized_windows = self._normalize_windows(windows)
        lookback_days = max(days + max(normalized_windows, default=0), 120)
        frame = self._load_daily_frame(code, source=source, days=lookback_days)
        bars = self._frame_to_indicator_bars(frame)
        items = self._build_indicator_items(bars, normalized_windows)[-days:]

        return {
            "stock_code": code,
            "period": "daily",
            "days": days,
            "windows": normalized_windows,
            "items": items,
            "source": source,
        }

    def get_factors(self, stock_code: str, market_source: str, target_date: Optional[str] = None) -> Dict[str, Any]:
        code = self._normalize_stock_code(stock_code)
        source = self._normalize_market_source(market_source)
        frame = self._load_daily_frame(code, source=source, days=365)
        bars = self._frame_to_indicator_bars(frame)
        index = self._find_nearest_index_by_date(bars, target_date)
        if index < 0:
            raise ValueError("No available daily bar for the specified date")

        factors = self._compute_factors_at(bars, index)
        return {
            "stock_code": code,
            "date": bars[index]["date"],
            "factors": factors,
            "source": source,
        }

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
    def _normalize_windows(windows: List[int]) -> List[int]:
        cleaned = [
            int(value)
            for value in windows
            if isinstance(value, int) or (isinstance(value, float) and value.is_integer())
        ]
        unique = sorted({value for value in cleaned if 0 < value <= 250})
        return unique or [5, 10, 20, 60]

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=0).isoformat()

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

    def _frame_to_history_rows(self, frame: pd.DataFrame) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
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

    def _frame_to_indicator_bars(self, frame: pd.DataFrame) -> List[Dict[str, Any]]:
        bars: List[Dict[str, Any]] = []
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
    def _average(values: List[float | None]) -> float | None:
        filtered = [value for value in values if value is not None]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    def _compute_moving_average_at(self, bars: List[Dict[str, Any]], index: int, window: int) -> float | None:
        if window <= 0 or index + 1 < window:
            return None
        closes = [bar.get("close") for bar in bars[index + 1 - window:index + 1]]
        return _round4(self._average(closes))

    def _compute_rsi14_at(self, bars: List[Dict[str, Any]], index: int) -> float | None:
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
    def _compute_momentum20_at(bars: List[Dict[str, Any]], index: int) -> float | None:
        if index < 20:
            return None
        current = bars[index].get("close")
        base = bars[index - 20].get("close")
        if current is None or base is None or base == 0:
            return None
        return _round4(((current / base) - 1) * 100)

    def _compute_vol_ratio5_at(self, bars: List[Dict[str, Any]], index: int) -> float | None:
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
    def _compute_amplitude_at(bars: List[Dict[str, Any]], index: int) -> float | None:
        bar = bars[index]
        open_price = bar.get("open")
        high = bar.get("high")
        low = bar.get("low")
        if open_price is None or high is None or low is None or open_price == 0:
            return None
        return _round4(((high - low) / open_price) * 100)

    def _build_indicator_items(self, bars: List[Dict[str, Any]], windows: List[int]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
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
    def _find_nearest_index_by_date(bars: List[Dict[str, Any]], target_date: Optional[str]) -> int:
        if not bars:
            return -1
        if not target_date:
            return len(bars) - 1
        normalized = str(target_date).strip()[:10]
        for index in range(len(bars) - 1, -1, -1):
            if bars[index]["date"] <= normalized:
                return index
        return -1

    def _compute_factors_at(self, bars: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
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


def get_runtime_market_service(config: Optional[Config] = None) -> RuntimeMarketService:
    """返回内部市场服务单例。"""
    global _runtime_market_service
    if _runtime_market_service is None:
        _runtime_market_service = RuntimeMarketService(config=config)
    return _runtime_market_service


def reset_runtime_market_service() -> None:
    """重置内部市场服务单例，供测试使用。"""
    global _runtime_market_service
    _runtime_market_service = None
