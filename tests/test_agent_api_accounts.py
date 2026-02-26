# -*- coding: utf-8 -*-
"""Account endpoint tests."""

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


class _AccountService:
    def get_account_snapshot(self, account_name: str):
        if account_name != "paper-default":
            return {}
        return {
            "account_id": 1,
            "name": "paper-default",
            "cash": 90000.0,
            "initial_cash": 100000.0,
            "total_market_value": 10000.0,
            "total_asset": 100000.0,
            "realized_pnl": 500.0,
            "unrealized_pnl": 300.0,
            "cumulative_fees": 50.0,
            "positions": [
                {
                    "code": "600519",
                    "quantity": 100,
                    "available_qty": 100,
                    "avg_cost": 100.0,
                    "last_price": 103.0,
                    "market_value": 10300.0,
                    "unrealized_pnl": 300.0,
                }
            ],
        }


class AgentApiAccountsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_api_accounts.db")
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()

        self.app = create_app()
        self.app.dependency_overrides[get_task_service_dep] = lambda: _AccountService()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.app.dependency_overrides.clear()
        reset_agent_task_service()
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_account_snapshot_success(self):
        response = self.client.get(
            "/api/v1/accounts/paper-default/snapshot",
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "paper-default")

    def test_account_snapshot_not_found(self):
        response = self.client.get(
            "/api/v1/accounts/unknown/snapshot",
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
