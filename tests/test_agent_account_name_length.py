# -*- coding: utf-8 -*-
"""Account name length compatibility tests for run endpoint."""

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


class _LengthTaskService:
    def __init__(self):
        self.last_account_name = None

    def run_sync(self, *, stock_codes, request_id=None, account_name=None, runtime_config=None):
        self.last_account_name = account_name
        return {
            "run_id": "run-len-1",
            "mode": "once",
            "trade_date": "2026-02-25",
            "stock_codes": stock_codes,
            "account_name": account_name or "paper-default",
            "status": "completed",
            "data_snapshot": {},
            "signal_snapshot": {},
            "risk_snapshot": {},
            "execution_snapshot": {},
            "account_snapshot": {"cash": 100000.0, "total_asset": 100000.0, "positions": []},
            "started_at": "2026-02-25T10:00:00",
            "ended_at": "2026-02-25T10:00:05",
            "created_at": "2026-02-25T10:00:05",
        }


class AgentAccountNameLengthTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "account_name_length.db")
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()

        self.service = _LengthTaskService()
        self.app = create_app()
        self.app.dependency_overrides[get_task_service_dep] = lambda: self.service
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.app.dependency_overrides.clear()
        reset_agent_task_service()
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_top_level_account_name_length_128_is_accepted(self):
        account_name = "u" * 128
        response = self.client.post(
            "/api/v1/runs",
            headers={"Authorization": "Bearer test-token"},
            json={"stock_codes": ["600519"], "async_mode": False, "account_name": account_name},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.last_account_name, account_name)

    def test_runtime_account_name_length_129_is_rejected(self):
        account_name = "u" * 129
        response = self.client.post(
            "/api/v1/runs",
            headers={"Authorization": "Bearer test-token"},
            json={
                "stock_codes": ["600519"],
                "async_mode": False,
                "runtime_config": {
                    "account": {
                        "account_name": account_name,
                        "initial_cash": 100000,
                    }
                },
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_top_level_account_name_length_129_is_rejected(self):
        account_name = "u" * 129
        response = self.client.post(
            "/api/v1/runs",
            headers={"Authorization": "Bearer test-token"},
            json={"stock_codes": ["600519"], "async_mode": False, "account_name": account_name},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
