# -*- coding: utf-8 -*-
"""`RuntimeMarketService` 的固定行情源回退与可用性测试。"""

from __future__ import annotations

import unittest

import pandas as pd

from agent_stock.config import Config
from agent_stock.services.runtime_market_service import RuntimeMarketService
from data_provider.base import DataSourceUnavailableError
from data_provider.realtime_types import CircuitBreaker, RealtimeSource, UnifiedRealtimeQuote


class _FakeFetcherManager:
    def __init__(
        self,
        quotes: dict[str, UnifiedRealtimeQuote | Exception | None],
        histories: dict[str, pd.DataFrame | Exception | None] | None = None,
        available_fetchers: list[str] | None = None,
    ):
        self.quotes = quotes
        self.histories = histories or {}
        self.available_fetchers = available_fetchers or ["AkshareFetcher", "EfinanceFetcher", "TushareFetcher"]
        self.quote_calls: list[tuple[str, str | None]] = []
        self.daily_calls: list[tuple[str, str | None]] = []

    def get_realtime_quote(self, stock_code: str, fixed_source: str | None = None):
        self.quote_calls.append((stock_code, fixed_source))
        payload = self.quotes.get(str(fixed_source))
        if isinstance(payload, Exception):
            raise payload
        return payload

    def get_daily_data(self, stock_code: str, days: int = 30, fixed_source: str | None = None):
        self.daily_calls.append((stock_code, fixed_source))
        payload = self.histories.get(str(fixed_source))
        if isinstance(payload, Exception):
            raise payload
        if payload is None:
            return None, None
        return payload.copy(), str(fixed_source)


class RuntimeMarketServiceTestCase(unittest.TestCase):
    def _build_history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": "2026-03-18",
                    "open": 1400.0,
                    "high": 1412.0,
                    "low": 1398.0,
                    "close": 1408.0,
                    "volume": 1000000.0,
                    "amount": 1408000000.0,
                    "pct_chg": 0.1,
                },
                {
                    "date": "2026-03-19",
                    "open": 1408.0,
                    "high": 1415.0,
                    "low": 1401.0,
                    "close": 1410.0,
                    "volume": 1100000.0,
                    "amount": 1551000000.0,
                    "pct_chg": 0.142,
                },
                {
                    "date": "2026-03-20",
                    "open": 1410.0,
                    "high": 1418.0,
                    "low": 1402.0,
                    "close": 1406.5,
                    "volume": 1200000.0,
                    "amount": 1687800000.0,
                    "pct_chg": -0.2482,
                },
            ]
        )

    def test_quote_falls_back_when_requested_source_is_temporarily_unavailable(self):
        fetcher_manager = _FakeFetcherManager(
            quotes={
                "eastmoney": DataSourceUnavailableError("akshare_em circuit is open"),
                "tencent": UnifiedRealtimeQuote(
                    code="600519",
                    name="贵州茅台",
                    source=RealtimeSource.TENCENT,
                    price=1406.59,
                    change_pct=-0.11,
                    change_amount=-1.56,
                ),
            },
        )
        service = RuntimeMarketService(config=Config(), fetcher_manager=fetcher_manager)

        payload = service.get_quote("600519", "eastmoney")

        self.assertEqual(payload["stock_code"], "600519")
        self.assertEqual(payload["source"], "tencent")
        self.assertEqual(payload["requested_source"], "eastmoney")
        self.assertIn("东方财富", payload["warning"])
        self.assertIn("腾讯行情", payload["warning"])
        self.assertEqual(fetcher_manager.quote_calls, [("600519", "eastmoney"), ("600519", "tencent")])

    def test_invalid_source_does_not_trigger_quote_fallback(self):
        fetcher_manager = _FakeFetcherManager(quotes={})
        service = RuntimeMarketService(config=Config(), fetcher_manager=fetcher_manager)

        with self.assertRaises(DataSourceUnavailableError):
            service.get_quote("600519", "unknown-source")

        self.assertEqual(fetcher_manager.quote_calls, [])
        self.assertEqual(fetcher_manager.daily_calls, [])

    def test_invalid_stock_code_does_not_trigger_quote_fallback(self):
        fetcher_manager = _FakeFetcherManager(quotes={})
        service = RuntimeMarketService(config=Config(), fetcher_manager=fetcher_manager)

        with self.assertRaises(ValueError):
            service.get_quote("", "eastmoney")

        self.assertEqual(fetcher_manager.quote_calls, [])
        self.assertEqual(fetcher_manager.daily_calls, [])

    def test_history_uses_requested_source_without_warning_when_source_is_available(self):
        frame = self._build_history_frame()
        fetcher_manager = _FakeFetcherManager(
            quotes={
                "eastmoney": UnifiedRealtimeQuote(
                    code="600519",
                    name="贵州茅台",
                    source=RealtimeSource.AKSHARE_EM,
                    price=1406.59,
                ),
            },
            histories={"eastmoney": frame},
        )
        service = RuntimeMarketService(config=Config(), fetcher_manager=fetcher_manager)

        payload = service.get_history("600519", 2, "eastmoney")

        self.assertEqual(payload["source"], "eastmoney")
        self.assertNotIn("requested_source", payload)
        self.assertNotIn("warning", payload)
        self.assertEqual(payload["stock_name"], "贵州茅台")
        self.assertEqual(len(payload["data"]), 2)
        self.assertEqual(fetcher_manager.daily_calls, [("600519", "eastmoney")])
        self.assertEqual(fetcher_manager.quote_calls, [("600519", "eastmoney")])

    def test_history_falls_back_when_requested_source_is_temporarily_unavailable(self):
        frame = self._build_history_frame()
        fetcher_manager = _FakeFetcherManager(
            quotes={
                "tencent": UnifiedRealtimeQuote(
                    code="600519",
                    name="贵州茅台",
                    source=RealtimeSource.TENCENT,
                    price=1406.59,
                ),
            },
            histories={
                "eastmoney": DataSourceUnavailableError("eastmoney daily data failed"),
                "tencent": frame,
            },
        )
        service = RuntimeMarketService(config=Config(), fetcher_manager=fetcher_manager)

        payload = service.get_history("600519", 2, "eastmoney")

        self.assertEqual(payload["source"], "tencent")
        self.assertEqual(payload["requested_source"], "eastmoney")
        self.assertIn("日线行情源 东方财富", payload["warning"])
        self.assertIn("腾讯行情", payload["warning"])
        self.assertEqual(fetcher_manager.daily_calls, [("600519", "eastmoney"), ("600519", "tencent")])
        self.assertEqual(fetcher_manager.quote_calls, [("600519", "tencent")])

    def test_history_raises_when_all_sources_fail(self):
        fetcher_manager = _FakeFetcherManager(
            quotes={},
            histories={
                "eastmoney": DataSourceUnavailableError("eastmoney unavailable"),
                "tencent": DataSourceUnavailableError("tencent unavailable"),
                "sina": DataSourceUnavailableError("sina unavailable"),
                "efinance": DataSourceUnavailableError("efinance unavailable"),
                "tushare": DataSourceUnavailableError("tushare unavailable"),
            },
        )
        service = RuntimeMarketService(config=Config(), fetcher_manager=fetcher_manager)

        with self.assertRaises(DataSourceUnavailableError) as ctx:
            service.get_history("600519", 2, "eastmoney")

        self.assertIn("eastmoney", str(ctx.exception))
        self.assertIn("tushare", str(ctx.exception))
        self.assertEqual(
            fetcher_manager.daily_calls,
            [
                ("600519", "eastmoney"),
                ("600519", "tencent"),
                ("600519", "sina"),
                ("600519", "efinance"),
                ("600519", "tushare"),
            ],
        )

    def test_indicators_use_requested_source_without_warning_when_source_is_available(self):
        fetcher_manager = _FakeFetcherManager(quotes={}, histories={"eastmoney": self._build_history_frame()})
        service = RuntimeMarketService(config=Config(), fetcher_manager=fetcher_manager)

        payload = service.get_indicators("600519", 2, [5, 10], "eastmoney")

        self.assertEqual(payload["source"], "eastmoney")
        self.assertNotIn("requested_source", payload)
        self.assertNotIn("warning", payload)
        self.assertEqual(payload["windows"], [5, 10])
        self.assertEqual(fetcher_manager.daily_calls, [("600519", "eastmoney")])

    def test_indicators_fall_back_when_requested_source_is_temporarily_unavailable(self):
        fetcher_manager = _FakeFetcherManager(
            quotes={},
            histories={
                "eastmoney": DataSourceUnavailableError("eastmoney unavailable"),
                "tencent": self._build_history_frame(),
            },
        )
        service = RuntimeMarketService(config=Config(), fetcher_manager=fetcher_manager)

        payload = service.get_indicators("600519", 2, [5, 10], "eastmoney")

        self.assertEqual(payload["source"], "tencent")
        self.assertEqual(payload["requested_source"], "eastmoney")
        self.assertIn("日线行情源 东方财富", payload["warning"])
        self.assertEqual(fetcher_manager.daily_calls, [("600519", "eastmoney"), ("600519", "tencent")])

    def test_indicators_raise_when_all_sources_fail(self):
        fetcher_manager = _FakeFetcherManager(
            quotes={},
            histories={
                "eastmoney": DataSourceUnavailableError("eastmoney unavailable"),
                "tencent": DataSourceUnavailableError("tencent unavailable"),
                "sina": DataSourceUnavailableError("sina unavailable"),
                "efinance": DataSourceUnavailableError("efinance unavailable"),
                "tushare": DataSourceUnavailableError("tushare unavailable"),
            },
        )
        service = RuntimeMarketService(config=Config(), fetcher_manager=fetcher_manager)

        with self.assertRaises(DataSourceUnavailableError):
            service.get_indicators("600519", 2, [5, 10], "eastmoney")

        self.assertEqual(fetcher_manager.daily_calls[0], ("600519", "eastmoney"))
        self.assertEqual(fetcher_manager.daily_calls[-1], ("600519", "tushare"))

    def test_factors_use_requested_source_without_warning_when_source_is_available(self):
        fetcher_manager = _FakeFetcherManager(quotes={}, histories={"eastmoney": self._build_history_frame()})
        service = RuntimeMarketService(config=Config(), fetcher_manager=fetcher_manager)

        payload = service.get_factors("600519", "eastmoney")

        self.assertEqual(payload["source"], "eastmoney")
        self.assertNotIn("requested_source", payload)
        self.assertNotIn("warning", payload)
        self.assertIn("factors", payload)
        self.assertEqual(fetcher_manager.daily_calls, [("600519", "eastmoney")])

    def test_factors_fall_back_when_requested_source_is_temporarily_unavailable(self):
        fetcher_manager = _FakeFetcherManager(
            quotes={},
            histories={
                "eastmoney": DataSourceUnavailableError("eastmoney unavailable"),
                "tencent": self._build_history_frame(),
            },
        )
        service = RuntimeMarketService(config=Config(), fetcher_manager=fetcher_manager)

        payload = service.get_factors("600519", "eastmoney")

        self.assertEqual(payload["source"], "tencent")
        self.assertEqual(payload["requested_source"], "eastmoney")
        self.assertIn("日线行情源 东方财富", payload["warning"])
        self.assertEqual(fetcher_manager.daily_calls, [("600519", "eastmoney"), ("600519", "tencent")])

    def test_factors_raise_when_all_sources_fail(self):
        fetcher_manager = _FakeFetcherManager(
            quotes={},
            histories={
                "eastmoney": DataSourceUnavailableError("eastmoney unavailable"),
                "tencent": DataSourceUnavailableError("tencent unavailable"),
                "sina": DataSourceUnavailableError("sina unavailable"),
                "efinance": DataSourceUnavailableError("efinance unavailable"),
                "tushare": DataSourceUnavailableError("tushare unavailable"),
            },
        )
        service = RuntimeMarketService(config=Config(), fetcher_manager=fetcher_manager)

        with self.assertRaises(DataSourceUnavailableError):
            service.get_factors("600519", "eastmoney")

        self.assertEqual(fetcher_manager.daily_calls[0], ("600519", "eastmoney"))
        self.assertEqual(fetcher_manager.daily_calls[-1], ("600519", "tushare"))

    def test_market_source_options_reflect_open_circuit_breaker(self):
        circuit_breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=300.0)
        for _ in range(3):
            circuit_breaker.record_failure("akshare_em", "Remote end closed connection without response")

        fetcher_manager = _FakeFetcherManager(quotes={})
        service = RuntimeMarketService(
            config=Config(),
            fetcher_manager=fetcher_manager,
            realtime_circuit_breaker=circuit_breaker,
        )

        options = service.get_market_source_options()["options"]
        eastmoney = next(item for item in options if item["code"] == "eastmoney")
        tencent = next(item for item in options if item["code"] == "tencent")

        self.assertFalse(eastmoney["available"])
        self.assertIn("熔断", eastmoney["reason"])
        self.assertTrue(tencent["available"])


if __name__ == "__main__":
    unittest.main()
