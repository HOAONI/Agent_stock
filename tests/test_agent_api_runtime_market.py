# -*- coding: utf-8 -*-
"""内部市场接口测试。"""

from __future__ import annotations

import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from agent_api.app import create_app
from agent_api.deps import get_runtime_market_service_dep
from agent_stock.config import Config
from agent_stock.services.agent_task_service import reset_agent_task_service
from agent_stock.services.runtime_market_service import reset_runtime_market_service
from agent_stock.storage import DatabaseManager


class _FakeRuntimeMarketService:
    def get_market_source_options(self):
        return {
            "options": [
                {
                    "code": "sina",
                    "label": "新浪行情",
                    "description": "fake",
                    "available": True,
                },
            ],
        }

    def get_quote(self, stock_code: str, market_source: str):
        return {
            "stock_code": stock_code,
            "stock_name": "贵州茅台",
            "current_price": 1500.0,
            "update_time": "2026-03-20T10:00:00+08:00",
            "source": market_source,
        }

    def get_history(self, stock_code: str, days: int, market_source: str):
        return {
            "stock_code": stock_code,
            "stock_name": "贵州茅台",
            "period": "daily",
            "data": [{"date": "2026-03-20", "close": 1500.0}],
            "source": market_source,
        }

    def get_indicators(self, stock_code: str, days: int, windows: list[int], market_source: str):
        return {
            "stock_code": stock_code,
            "period": "daily",
            "days": days,
            "windows": windows,
            "items": [{"date": "2026-03-20", "close": 1500.0, "mas": {"ma5": 1490.0}}],
            "source": market_source,
        }

    def get_factors(self, stock_code: str, market_source: str, target_date: str | None = None):
        return {
            "stock_code": stock_code,
            "date": target_date or "2026-03-20",
            "factors": {"ma5": 1490.0},
            "source": market_source,
        }


class AgentApiRuntimeMarketTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "runtime_market_api.db")
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()
        reset_runtime_market_service()

        self.app = create_app()
        self.app.dependency_overrides[get_runtime_market_service_dep] = lambda: _FakeRuntimeMarketService()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.app.dependency_overrides.clear()
        reset_agent_task_service()
        reset_runtime_market_service()
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_quote_endpoint_forwards_market_source(self):
        response = self.client.get(
            "/internal/v1/stocks/600519/quote",
            params={"market_source": "sina"},
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["stock_code"], "600519")
        self.assertEqual(payload["source"], "sina")

    def test_indicators_endpoint_preserves_windows(self):
        response = self.client.get(
            "/internal/v1/stocks/600519/indicators",
            params={"market_source": "sina", "days": 120, "windows": "5,10,20"},
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["windows"], [5, 10, 20])
        self.assertEqual(payload["source"], "sina")


if __name__ == "__main__":
    unittest.main()
