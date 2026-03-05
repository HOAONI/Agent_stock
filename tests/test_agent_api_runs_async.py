# -*- coding: utf-8 -*-
"""Async run endpoint tests."""

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


class _AsyncTaskService:
    def __init__(self):
        self.store = {}
        self.last_runtime_config = None
        self.last_account_name = None

    def submit_task(self, *, stock_codes, request_id=None, account_name=None, runtime_config=None):
        self.last_runtime_config = runtime_config
        self.last_account_name = account_name
        key = request_id or "no-request-id"
        if key not in self.store:
            self.store[key] = {
                "task_id": f"task-{len(self.store) + 1}",
                "status": "pending",
                "request_id": request_id,
                "stock_codes": stock_codes,
                "account_name": account_name or "paper-default",
                "run_id": None,
                "error_message": None,
                "created_at": "2026-02-23T10:00:00",
                "started_at": None,
                "completed_at": None,
                "updated_at": "2026-02-23T10:00:00",
            }
        return self.store[key]


class AgentApiRunsAsyncTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_api_runs_async.db")
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()

        self.fake_service = _AsyncTaskService()
        self.app = create_app()
        self.app.dependency_overrides[get_task_service_dep] = lambda: self.fake_service
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.app.dependency_overrides.clear()
        reset_agent_task_service()
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_async_run_returns_task_payload(self):
        response = self.client.post(
            "/api/v1/runs",
            headers={"Authorization": "Bearer test-token"},
            json={"stock_codes": ["600519"], "async_mode": True, "request_id": "req-1"},
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["task_id"], "task-1")
        self.assertIsNone(self.fake_service.last_runtime_config)

    def test_request_id_idempotency_returns_existing_task(self):
        payload = {"stock_codes": ["600519"], "async_mode": True, "request_id": "req-unique"}
        first = self.client.post("/api/v1/runs", headers={"Authorization": "Bearer test-token"}, json=payload)
        second = self.client.post("/api/v1/runs", headers={"Authorization": "Bearer test-token"}, json=payload)

        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(first.json()["task_id"], second.json()["task_id"])

    def test_async_runtime_config_passthrough(self):
        payload = {
            "stock_codes": ["600519"],
            "async_mode": True,
            "request_id": "req-runtime",
            "runtime_config": {
                "account": {
                    "account_name": "user-async",
                    "initial_cash": 50000,
                    "account_display_name": "Async User",
                },
                "llm": {
                    "provider": "deepseek",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-chat",
                    "has_token": True,
                    "api_token": "runtime-token-xyz",
                },
                "strategy": {
                    "position_max_pct": 20,
                    "stop_loss_pct": 5,
                    "take_profit_pct": 12,
                },
                "execution": {
                    "mode": "paper",
                    "has_ticket": True,
                    "broker_account_id": 88,
                },
                "context": {
                    "summary": {
                        "cash": 90000.0,
                        "total_asset": 120000.0,
                    },
                    "positions": [{"code": "600519", "quantity": 100, "market_value": 30000.0}],
                },
            },
        }
        response = self.client.post("/api/v1/runs", headers={"Authorization": "Bearer test-token"}, json=payload)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.fake_service.last_account_name, "user-async")
        self.assertEqual(self.fake_service.last_runtime_config["llm"]["provider"], "deepseek")
        self.assertEqual(self.fake_service.last_runtime_config["execution"]["mode"], "paper")
        self.assertEqual(self.fake_service.last_runtime_config["context"]["summary"]["cash"], 90000.0)


if __name__ == "__main__":
    unittest.main()
