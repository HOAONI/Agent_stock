# -*- coding: utf-8 -*-
"""DataAgent 单元测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date, timedelta

import pandas as pd

from agent_stock.agents.data_agent import DataAgent
from agent_stock.config import Config
from agent_stock.storage import DatabaseManager


class _UnavailableAnalyzer:
    def is_available(self) -> bool:
        return False


class _FakeQuote:
    def __init__(self, price: float):
        self.price = price

    def to_dict(self) -> dict[str, float | str]:
        return {"price": self.price, "source": "fake"}


class _FakeFetcherManager:
    def get_daily_data(self, stock_code: str, days: int = 60, fixed_source: str | None = None) -> tuple[pd.DataFrame | None, str | None]:
        today = date.today()
        yesterday = today - timedelta(days=1)
        df = pd.DataFrame(
            [
                {
                    "date": yesterday,
                    "open": 9.5,
                    "high": 10.1,
                    "low": 9.4,
                    "close": 10.0,
                    "volume": 1000,
                    "amount": 10000,
                    "pct_chg": 0.5,
                    "ma5": 9.8,
                    "ma10": 9.7,
                    "ma20": 9.6,
                    "volume_ratio": 1.1,
                },
                {
                    "date": today,
                    "open": 10.0,
                    "high": 10.6,
                    "low": 9.9,
                    "close": 10.5,
                    "volume": 1200,
                    "amount": 12600,
                    "pct_chg": 5.0,
                    "ma5": 10.0,
                    "ma10": 9.8,
                    "ma20": 9.7,
                    "volume_ratio": 1.2,
                },
            ]
        )
        return df, fixed_source or "FakeFetcher"

    def get_realtime_quote(self, stock_code: str, fixed_source: str | None = None) -> _FakeQuote:
        return _FakeQuote(price=10.6)


class _FallbackFetcherManager(_FakeFetcherManager):
    def get_daily_data(self, stock_code: str, days: int = 60, fixed_source: str | None = None) -> tuple[pd.DataFrame | None, str | None]:
        if fixed_source == "tencent":
            raise RuntimeError("tencent unavailable")
        return super().get_daily_data(stock_code, days=days, fixed_source=fixed_source)

    def get_realtime_quote(self, stock_code: str, fixed_source: str | None = None) -> _FakeQuote:
        if fixed_source == "tencent":
            raise RuntimeError("tencent quote unavailable")
        return super().get_realtime_quote(stock_code, fixed_source=fixed_source)


class DataAgentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_data_agent.db")
        self._env_backup = {
            "DATABASE_PATH": os.environ.get("DATABASE_PATH"),
            "DATABASE_URL": os.environ.get("DATABASE_URL"),
            "REALTIME_SOURCE_PRIORITY": os.environ.get("REALTIME_SOURCE_PRIORITY"),
            "TUSHARE_TOKEN": os.environ.get("TUSHARE_TOKEN"),
        }

        os.environ["DATABASE_PATH"] = self.db_path
        os.environ["DATABASE_URL"] = ""
        os.environ["REALTIME_SOURCE_PRIORITY"] = "tencent,sina,efinance,eastmoney"
        os.environ["TUSHARE_TOKEN"] = ""
        Config.reset_instance()
        DatabaseManager.reset_instance()

        self.db = DatabaseManager.get_instance()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp_dir.cleanup()

    def test_data_agent_fetches_and_saves(self):
        fetcher = _FakeFetcherManager()
        agent = DataAgent(fetcher_manager=fetcher, db_manager=self.db, analyzer=_UnavailableAnalyzer())

        output = agent.run("600519")

        self.assertEqual(output.code, "600519")
        self.assertEqual(output.state.value, "ready")
        self.assertIn("price", output.realtime_quote)
        self.assertTrue(self.db.has_today_data("600519", date.today()))
        self.assertIsNotNone(output.analysis_context)

    def test_data_agent_falls_back_to_backup_source(self):
        fetcher = _FallbackFetcherManager()
        agent = DataAgent(fetcher_manager=fetcher, db_manager=self.db, analyzer=_UnavailableAnalyzer())

        output = agent.run("600519")

        self.assertEqual(output.state.value, "ready")
        self.assertEqual(output.data_source, "sina")
        self.assertIn("tencent", output.fallback_chain)
        self.assertIn("sina", output.fallback_chain)
        self.assertTrue(any("自动切换" in item for item in output.warnings))


if __name__ == "__main__":
    unittest.main()
