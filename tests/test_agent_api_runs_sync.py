# -*- coding: utf-8 -*-
"""同步运行端点测试。"""

from __future__ import annotations

import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from agent_api.app import create_app
from agent_api.deps import get_task_service_dep
from agent_stock.services.agent_task_service import reset_agent_task_service
from agent_stock.storage import DatabaseManager
from agent_stock.config import Config


class _SyncTaskService:
    def __init__(self):
        self.last_runtime_config = None
        self.last_account_name = None

    def run_sync(self, *, stock_codes, request_id=None, account_name=None, runtime_config=None):
        self.last_runtime_config = runtime_config
        self.last_account_name = account_name
        return {
            "run_id": "run-sync-1",
            "mode": "once",
            "trade_date": "2026-02-23",
            "stock_codes": stock_codes,
            "account_name": account_name or "paper-default",
            "status": "completed",
            "data_snapshot": {},
            "signal_snapshot": {},
            "risk_snapshot": {},
            "execution_snapshot": {},
            "account_snapshot": {"cash": 100000.0, "total_asset": 100000.0, "positions": []},
            "started_at": "2026-02-23T10:00:00",
            "ended_at": "2026-02-23T10:00:10",
            "created_at": "2026-02-23T10:00:10",
        }


class AgentApiRunsSyncTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_api_runs_sync.db")
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()

        self.fake_service = _SyncTaskService()
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

    def test_sync_run_returns_run_payload(self):
        response = self.client.post(
            "/api/v1/runs",
            headers={"Authorization": "Bearer test-token"},
            json={"stock_codes": ["600519"], "async_mode": False},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["run_id"], "run-sync-1")
        self.assertEqual(data["status"], "completed")
        self.assertIsNone(self.fake_service.last_runtime_config)

    def test_sync_run_with_runtime_config_passthrough(self):
        payload = {
            "stock_codes": ["600519"],
            "async_mode": False,
            "runtime_config": {
                "account": {
                    "account_name": "user-123",
                    "initial_cash": 100000,
                    "account_display_name": "User 123",
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
                "execution": {
                    "mode": "paper",
                    "has_ticket": False,
                },
            },
        }
        response = self.client.post(
            "/api/v1/runs",
            headers={"Authorization": "Bearer test-token"},
            json=payload,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.fake_service.last_account_name, "user-123")
        self.assertIsInstance(self.fake_service.last_runtime_config, dict)
        self.assertEqual(
            self.fake_service.last_runtime_config["strategy"]["position_max_pct"],
            30,
        )
        self.assertEqual(self.fake_service.last_runtime_config["execution"]["mode"], "paper")


if __name__ == "__main__":
    unittest.main()
