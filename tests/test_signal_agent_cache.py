# -*- coding: utf-8 -*-
"""Unit tests for SignalAgent daily AI cache behavior."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date, timedelta

from src.agents.contracts import DataAgentOutput
from src.agents.signal_agent import SignalAgent
from src.config import AgentRuntimeConfig, Config, RuntimeLlmConfig
from src.storage import DatabaseManager


class _FakeAIResult:
    def __init__(self, operation_advice: str, sentiment_score: int):
        self.operation_advice = operation_advice
        self.sentiment_score = sentiment_score
        self.trend_prediction = "看多"
        self.analysis_summary = "fake summary"

    def get_sniper_points(self):
        return {"stop_loss": "9.50", "take_profit": "12.00"}


class SignalAgentCacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_signal_agent.db")

        os.environ["DATABASE_PATH"] = self.db_path
        os.environ["DATABASE_URL"] = ""
        os.environ["AGENT_AI_REFRESH_POLICY"] = "daily_once"
        Config.reset_instance()
        DatabaseManager.reset_instance()
        self.db = DatabaseManager.get_instance()

        self.calls = 0

        def _resolver(code: str):
            self.calls += 1
            return _FakeAIResult(operation_advice="买入", sentiment_score=77)

        self.agent = SignalAgent(db_manager=self.db, ai_resolver=_resolver)

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    @staticmethod
    def _build_raw_data():
        today = date.today()
        rows = []
        for idx in range(30):
            day = today - timedelta(days=29 - idx)
            close = 10.0 + idx * 0.1
            rows.append(
                {
                    "date": day.isoformat(),
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000 + idx * 10,
                    "amount": (1000 + idx * 10) * close,
                    "pct_chg": 0.5,
                    "ma5": close - 0.05,
                    "ma10": close - 0.08,
                    "ma20": close - 0.12,
                    "volume_ratio": 1.1,
                }
            )
        return rows

    def test_daily_ai_cache_reused(self):
        context = {"raw_data": self._build_raw_data()}
        data_output = DataAgentOutput(
            code="600519",
            trade_date=date.today(),
            analysis_context=context,
            realtime_quote={"price": 12.5},
        )

        first = self.agent.run(data_output)
        second = self.agent.run(data_output)

        self.assertEqual(self.calls, 1)
        self.assertTrue(first.ai_refreshed)
        self.assertFalse(second.ai_refreshed)
        self.assertEqual(first.operation_advice, "买入")
        self.assertEqual(second.operation_advice, "买入")
        self.assertAlmostEqual(float(second.stop_loss), 9.5)

    def test_runtime_llm_request_still_uses_global_ai_cache(self):
        context = {"raw_data": self._build_raw_data()}
        data_output = DataAgentOutput(
            code="600519",
            trade_date=date.today(),
            analysis_context=context,
            realtime_quote={"price": 12.5},
        )
        runtime_config = AgentRuntimeConfig(
            llm=RuntimeLlmConfig(
                provider="openai",
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
                api_token="runtime-token-abc",
                has_token=True,
            )
        )

        first = self.agent.run(data_output, runtime_config=runtime_config)
        second = self.agent.run(data_output, runtime_config=runtime_config)

        self.assertEqual(self.calls, 1)
        self.assertTrue(first.ai_refreshed)
        self.assertFalse(second.ai_refreshed)


if __name__ == "__main__":
    unittest.main()
