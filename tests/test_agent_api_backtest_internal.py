# -*- coding: utf-8 -*-
"""Internal backtest API endpoint tests."""

from __future__ import annotations

import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from agent_api.app import create_app
from agent_api.deps import get_backtest_service_dep, get_strategy_backtest_service_dep
from agent_stock.services.agent_task_service import reset_agent_task_service
from agent_stock.storage import DatabaseManager
from src.config import Config


class _FakeBacktestService:
    def run(self, payload):  # noqa: D401
        return {"processed": len(payload.get("candidates", [])), "saved": 0, "completed": 0, "insufficient": 0, "errors": 0, "items": []}

    def summary(self, payload):
        return {
            "scope": payload.get("scope", "overall"),
            "code": payload.get("code"),
            "eval_window_days": payload.get("eval_window_days", 10),
            "engine_version": payload.get("engine_version", "v1"),
            "total_evaluations": 0,
            "completed_count": 0,
            "insufficient_count": 0,
            "long_count": 0,
            "cash_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "neutral_count": 0,
            "direction_accuracy_pct": None,
            "win_rate_pct": None,
            "neutral_rate_pct": None,
            "avg_stock_return_pct": None,
            "avg_simulated_return_pct": None,
            "stop_loss_trigger_rate": None,
            "take_profit_trigger_rate": None,
            "ambiguous_rate": None,
            "avg_days_to_first_hit": None,
            "advice_breakdown": {},
            "diagnostics": {},
        }

    def curves(self, payload):
        return {"scope": payload.get("scope", "overall"), "code": payload.get("code"), "eval_window_days": payload.get("eval_window_days", 10), "curves": []}

    def distribution(self, payload):
        return {
            "scope": payload.get("scope", "overall"),
            "code": payload.get("code"),
            "eval_window_days": payload.get("eval_window_days", 10),
            "distribution": {"long_count": 0, "cash_count": 0, "win_count": 0, "loss_count": 0, "neutral_count": 0},
        }

    def compare(self, payload):
        return {"items": [{"strategy_code": "agent_v1", "strategy_name": "Agent v1", "eval_window_days": payload.get("eval_window_days_list", [10])[0], "total_evaluations": 0, "completed_count": 0}]}


class _FakeStrategyBacktestService:
    def run(self, payload):
        return {
            "engine_version": "backtrader_v1",
            "code": payload.get("code"),
            "requested_range": {"start_date": payload.get("start_date"), "end_date": payload.get("end_date")},
            "effective_range": {"start_date": payload.get("start_date"), "end_date": payload.get("end_date")},
            "items": [
                {
                    "strategy_code": "ma20_trend",
                    "strategy_name": "MA20 Trend",
                    "strategy_version": "v1",
                    "params": {},
                    "metrics": {},
                    "benchmark": {},
                    "trades": [],
                    "equity": [],
                }
            ],
        }


class AgentApiBacktestInternalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_api_backtest_internal.db")
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()

        self.app = create_app()
        self.app.dependency_overrides[get_backtest_service_dep] = lambda: _FakeBacktestService()
        self.app.dependency_overrides[get_strategy_backtest_service_dep] = lambda: _FakeStrategyBacktestService()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.app.dependency_overrides.clear()
        reset_agent_task_service()
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_internal_backtest_requires_auth(self):
        response = self.client.post("/internal/v1/backtest/run", json={"eval_window_days": 10, "candidates": []})
        self.assertEqual(response.status_code, 401)

    def test_internal_backtest_run_envelope(self):
        response = self.client.post(
            "/internal/v1/backtest/run",
            headers={"Authorization": "Bearer test-token"},
            json={"eval_window_days": 10, "candidates": []},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertIn("data", data)
        self.assertEqual(data["data"]["processed"], 0)

    def test_internal_strategy_backtest_run_envelope(self):
        response = self.client.post(
            "/internal/v1/backtest/strategy/run",
            headers={"Authorization": "Bearer test-token"},
            json={
                "code": "600519",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "strategy_codes": ["ma20_trend"],
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["data"]["code"], "600519")
        self.assertEqual(len(data["data"]["items"]), 1)

    def test_internal_backtest_compare_validation_error(self):
        class _FailCompareBacktestService(_FakeBacktestService):
            def compare(self, payload):  # noqa: ARG002
                raise ValueError("compare_fetch_failed: 600519: network down")

        self.app.dependency_overrides[get_backtest_service_dep] = lambda: _FailCompareBacktestService()

        response = self.client.post(
            "/internal/v1/backtest/compare",
            headers={"Authorization": "Bearer test-token"},
            json={"eval_window_days_list": [5], "rows_by_window": {"5": []}},
        )

        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertEqual(detail["error"], "validation_error")
        self.assertIn("compare_fetch_failed", detail["message"])


if __name__ == "__main__":
    unittest.main()
