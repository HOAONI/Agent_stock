# -*- coding: utf-8 -*-
"""Runtime config validation tests for run endpoint."""

from __future__ import annotations

import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from agent_api.app import create_app
from agent_api.deps import get_task_service_dep
from agent_stock.services.agent_task_service import reset_agent_task_service
from agent_stock.storage import DatabaseManager
from src.config import Config


class _NoopTaskService:
    def run_sync(self, *, stock_codes, request_id=None, account_name=None, runtime_config=None):
        return {
            "run_id": "run-validation",
            "mode": "once",
            "trade_date": "2026-02-24",
            "stock_codes": stock_codes,
            "account_name": account_name or "paper-default",
            "status": "completed",
            "data_snapshot": {},
            "signal_snapshot": {},
            "risk_snapshot": {},
            "execution_snapshot": {},
            "account_snapshot": {"cash": 100000.0, "total_asset": 100000.0, "positions": []},
            "started_at": "2026-02-24T10:00:00",
            "ended_at": "2026-02-24T10:00:05",
            "created_at": "2026-02-24T10:00:05",
        }


class AgentRuntimeConfigValidationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "runtime_validation.db")
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()

        self.app = create_app()
        self.app.dependency_overrides[get_task_service_dep] = lambda: _NoopTaskService()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.app.dependency_overrides.clear()
        reset_agent_task_service()
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_invalid_runtime_strategy_returns_422(self):
        payload = {
            "stock_codes": ["600519"],
            "async_mode": False,
            "runtime_config": {
                "strategy": {
                    "position_max_pct": 150,
                    "stop_loss_pct": 8,
                    "take_profit_pct": 15,
                }
            },
        }
        response = self.client.post("/api/v1/runs", headers={"Authorization": "Bearer test-token"}, json=payload)
        self.assertEqual(response.status_code, 422)

    def test_account_conflict_returns_400(self):
        payload = {
            "stock_codes": ["600519"],
            "async_mode": False,
            "account_name": "user-top",
            "runtime_config": {
                "account": {
                    "account_name": "user-runtime",
                    "initial_cash": 100000,
                    "account_display_name": "runtime",
                },
                "llm": {
                    "provider": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini",
                    "has_token": False,
                },
                "strategy": {
                    "position_max_pct": 30,
                    "stop_loss_pct": 8,
                    "take_profit_pct": 15,
                },
            },
        }
        response = self.client.post("/api/v1/runs", headers={"Authorization": "Bearer test-token"}, json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("conflicts", response.text)

    def test_zero_strategy_values_are_accepted(self):
        payload = {
            "stock_codes": ["600519"],
            "async_mode": False,
            "runtime_config": {
                "account": {
                    "account_name": "user-zero",
                    "initial_cash": 100000,
                    "account_display_name": "Zero Strategy",
                },
                "strategy": {
                    "position_max_pct": 0,
                    "stop_loss_pct": 0,
                    "take_profit_pct": 0,
                },
            },
        }
        response = self.client.post("/api/v1/runs", headers={"Authorization": "Bearer test-token"}, json=payload)
        self.assertEqual(response.status_code, 200)

    def test_broker_execution_requires_ticket(self):
        payload = {
            "stock_codes": ["600519"],
            "async_mode": False,
            "runtime_config": {
                "execution": {
                    "mode": "broker",
                    "has_ticket": False,
                    "ticket_id": 123,
                    "broker_account_id": 88,
                }
            },
        }
        response = self.client.post("/api/v1/runs", headers={"Authorization": "Bearer test-token"}, json=payload)
        self.assertEqual(response.status_code, 422)

    def test_broker_execution_with_ticket_is_accepted(self):
        payload = {
            "stock_codes": ["600519"],
            "async_mode": False,
            "runtime_config": {
                "execution": {
                    "mode": "broker",
                    "has_ticket": True,
                    "credential_ticket": "agt_ticket_for_test_only",
                    "ticket_id": 123,
                    "broker_account_id": 88,
                }
            },
        }
        response = self.client.post("/api/v1/runs", headers={"Authorization": "Bearer test-token"}, json=payload)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
