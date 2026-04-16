# -*- coding: utf-8 -*-
"""DataFetcherManager 与 Tushare 健康降级测试。"""

from __future__ import annotations

import unittest

import pandas as pd

from data_provider.base import DataFetcherManager
from data_provider.tushare_fetcher import TushareFetcher


class FakeStockListFetcher:
    def __init__(
        self,
        *,
        name: str,
        priority: int,
        rows: list[dict[str, object]] | None = None,
        stock_list_error: Exception | None = None,
        daily_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.priority = priority
        self.rows = [dict(item) for item in (rows or [])]
        self.stock_list_error = stock_list_error
        self.daily_error = daily_error
        self.stock_list_calls = 0
        self.daily_calls = 0
        self._available = True

    def is_available(self) -> bool:
        return self._available

    def get_stock_list(self):
        self.stock_list_calls += 1
        if self.stock_list_error is not None:
            self._available = False
            raise self.stock_list_error
        return pd.DataFrame(self.rows)

    def get_daily_data(self, *, stock_code: str, start_date=None, end_date=None, days: int = 30):  # noqa: ARG002
        self.daily_calls += 1
        if self.daily_error is not None:
            self._available = False
            raise self.daily_error
        return pd.DataFrame(
            [
                {
                    "date": "2026-04-14",
                    "open": 1.0,
                    "high": 1.2,
                    "low": 0.9,
                    "close": 1.1,
                    "volume": 1000,
                    "amount": 1100,
                    "pct_chg": 1.0,
                }
            ]
        )


class FakeTushareApi:
    def stock_basic(self, **_kwargs):
        raise Exception("您的token不对，请确认。")


class DataFetcherManagerTestCase(unittest.TestCase):
    def test_get_stock_list_falls_back_to_next_fetcher_and_skips_unavailable(self):
        primary = FakeStockListFetcher(
            name="TushareFetcher",
            priority=-1,
            stock_list_error=Exception("您的token不对，请确认。"),
        )
        fallback = FakeStockListFetcher(
            name="BaostockFetcher",
            priority=3,
            rows=[
                {"code": "601988", "name": "中国银行"},
                {"code": "600036", "name": "招商银行"},
            ],
        )
        manager = DataFetcherManager(fetchers=[primary, fallback])

        first = manager.get_stock_list()
        second = manager.get_stock_list()

        assert first is not None
        assert second is not None
        self.assertEqual(first["code"].tolist(), ["601988", "600036"])
        self.assertEqual(second["code"].tolist(), ["601988", "600036"])
        self.assertEqual(primary.stock_list_calls, 1)
        self.assertEqual(fallback.stock_list_calls, 2)

    def test_get_daily_data_skips_fetcher_marked_unavailable_after_auth_failure(self):
        primary = FakeStockListFetcher(
            name="TushareFetcher",
            priority=-1,
            daily_error=Exception("您的token不对，请确认。"),
        )
        fallback = FakeStockListFetcher(
            name="EfinanceFetcher",
            priority=0,
        )
        manager = DataFetcherManager(fetchers=[primary, fallback])

        first_df, first_source = manager.get_daily_data("601988", days=30)
        second_df, second_source = manager.get_daily_data("601988", days=30)

        self.assertEqual(first_source, "EfinanceFetcher")
        self.assertEqual(second_source, "EfinanceFetcher")
        self.assertFalse(first_df.empty)
        self.assertFalse(second_df.empty)
        self.assertEqual(primary.daily_calls, 1)
        self.assertEqual(fallback.daily_calls, 2)


class TushareFetcherHealthTestCase(unittest.TestCase):
    def test_get_stock_list_marks_fetcher_unavailable_after_auth_failure(self):
        fetcher = object.__new__(TushareFetcher)
        fetcher.rate_limit_per_minute = 80
        fetcher._call_count = 0
        fetcher._minute_start = None
        fetcher._api = FakeTushareApi()
        fetcher._auth_verified = True
        fetcher._auth_failed = False
        fetcher._auth_failure_reason = None
        fetcher.priority = -1
        fetcher._stock_name_cache = {}

        result = fetcher.get_stock_list()

        self.assertIsNone(result)
        self.assertFalse(fetcher.is_available())
        self.assertEqual(fetcher.priority, 2)
        self.assertIn("token", str(fetcher._auth_failure_reason or "").lower())
