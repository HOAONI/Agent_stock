# -*- coding: utf-8 -*-
"""Prompt assembly tests for runtime account constraints."""

from __future__ import annotations

import os
import tempfile
import unittest

from src.analyzer import GeminiAnalyzer
from src.config import Config
from src.storage import DatabaseManager


class AnalyzerRuntimeAccountPromptTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "analyzer_runtime_prompt.db")
        os.environ["DATABASE_URL"] = ""
        Config.reset_instance()
        DatabaseManager.reset_instance()
        self.analyzer = GeminiAnalyzer()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_prompt_contains_runtime_account_constraints_when_present(self):
        context = {
            "code": "600519",
            "stock_name": "贵州茅台",
            "date": "2026-03-05",
            "today": {
                "close": 100.0,
                "open": 99.5,
                "high": 101.0,
                "low": 98.5,
                "pct_chg": 1.2,
            },
            "runtime_account": {
                "cash": 6507.96,
                "total_asset": 99971.96,
                "total_market_value": 93464.0,
                "position": {
                    "code": "600519",
                    "quantity": 20000,
                    "available_qty": 100,
                    "market_value": 93464.0,
                },
                "snapshot_at": "2026-03-05T10:00:00",
                "data_source": "upstream",
            },
        }

        prompt = self.analyzer._format_prompt(context, name="贵州茅台", news_context=None)
        self.assertIn("账户资金约束（运行时快照）", prompt)
        self.assertIn("可用现金", prompt)
        self.assertIn("总资产", prompt)
        self.assertIn("持仓市值", prompt)
        self.assertIn("当前标的持仓", prompt)

    def test_prompt_without_runtime_account_keeps_backward_compatibility(self):
        context = {
            "code": "600519",
            "stock_name": "贵州茅台",
            "date": "2026-03-05",
            "today": {"close": 100.0},
        }

        prompt = self.analyzer._format_prompt(context, name="贵州茅台", news_context=None)
        self.assertNotIn("账户资金约束（运行时快照）", prompt)


if __name__ == "__main__":
    unittest.main()
