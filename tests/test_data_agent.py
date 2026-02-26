# -*- coding: utf-8 -*-
"""Unit tests for DataAgent."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date, timedelta

import pandas as pd

from src.agents.data_agent import DataAgent
from src.config import Config
from src.storage import DatabaseManager


class _FakeQuote:
    def __init__(self, price: float):
        self.price = price

    def to_dict(self):
        return {"price": self.price, "source": "fake"}


class _FakeFetcherManager:
    def get_daily_data(self, stock_code: str, days: int = 60):
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
        return df, "FakeFetcher"

    def get_realtime_quote(self, stock_code: str):
        return _FakeQuote(price=10.6)


class DataAgentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_data_agent.db")

        os.environ["DATABASE_PATH"] = self.db_path
        Config.reset_instance()
        DatabaseManager.reset_instance()

        self.db = DatabaseManager.get_instance()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_data_agent_fetches_and_saves(self):
        fetcher = _FakeFetcherManager()
        agent = DataAgent(fetcher_manager=fetcher, db_manager=self.db)

        output = agent.run("600519")

        self.assertEqual(output.code, "600519")
        self.assertEqual(output.state.value, "ready")
        self.assertIn("price", output.realtime_quote)
        self.assertTrue(self.db.has_today_data("600519", date.today()))
        self.assertIsNotNone(output.analysis_context)


if __name__ == "__main__":
    unittest.main()
