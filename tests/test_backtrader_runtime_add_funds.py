# -*- coding: utf-8 -*-
"""`BacktraderRuntimeService.add_funds` 支持能力测试。"""

from __future__ import annotations

import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from agent_api.app import create_app
from agent_api.deps import get_backtrader_runtime_service_dep
from agent_stock.services.agent_task_service import reset_agent_task_service
from agent_stock.services.backtrader_runtime_service import BacktraderRuntimeService
from agent_stock.storage import DatabaseManager
from agent_stock.config import Config


class BacktraderRuntimeAddFundsServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "backtrader_add_funds.db")
        os.environ["DATABASE_URL"] = ""
        Config.reset_instance()
        DatabaseManager.reset_instance()
        self.service = BacktraderRuntimeService()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def _base_request(self) -> dict:
        return {
            "user_id": 1,
            "broker_account_id": 3,
            "credentials": {
                "initial_capital": 100000,
            },
            "payload": {},
        }

    def test_add_funds_increases_cash_and_initial_capital(self):
        req = self._base_request()
        req["payload"] = {
            "amount": 5000,
            "note": "manual topup",
        }

        result = self.service.add_funds(req)
        summary = result.get("summary", {})

        self.assertEqual(float(result["amount"]), 5000.0)
        self.assertEqual(float(result["cash_before"]), 100000.0)
        self.assertEqual(float(result["cash_after"]), 105000.0)
        self.assertEqual(float(result["initial_capital_before"]), 100000.0)
        self.assertEqual(float(result["initial_capital_after"]), 105000.0)
        self.assertEqual(float(summary.get("cash", 0.0)), 105000.0)
        self.assertEqual(float(summary.get("initial_capital", 0.0)), 105000.0)
        self.assertEqual(float(summary.get("total_asset", 0.0)), 105000.0)

    def test_add_funds_rejects_non_positive_amount(self):
        req = self._base_request()
        req["payload"] = {"amount": 0}
        with self.assertRaises(ValueError):
            self.service.add_funds(req)


class _FakeBacktraderRuntimeService:
    def add_funds(self, req: dict):
        payload = req.get("payload") or {}
        amount = float(payload.get("amount") or 0)
        if amount <= 0:
            raise ValueError("payload.amount must be > 0")
        return {
            "amount": amount,
            "cash_before": 100000.0,
            "cash_after": 100000.0 + amount,
            "initial_capital_before": 100000.0,
            "initial_capital_after": 100000.0 + amount,
            "summary": {"cash": 100000.0 + amount, "initial_capital": 100000.0 + amount},
        }

    def provision_account(self, req: dict):
        return {}

    def get_account_summary(self, req: dict):
        return {}

    def get_positions(self, req: dict):
        return []

    def get_orders(self, req: dict):
        return []

    def get_trades(self, req: dict):
        return []

    def place_order(self, req: dict):
        return {}

    def cancel_order(self, req: dict):
        return {}


class BacktraderRuntimeAddFundsEndpointTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_api_backtrader_add_funds.db")
        os.environ["DATABASE_URL"] = ""
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()

        self.app = create_app()
        self.app.dependency_overrides[get_backtrader_runtime_service_dep] = lambda: _FakeBacktraderRuntimeService()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.app.dependency_overrides.clear()
        reset_agent_task_service()
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def _request_payload(self, amount: float) -> dict:
        return {
            "user_id": 1,
            "broker_account_id": 3,
            "environment": "simulation",
            "account_uid": "bt-3",
            "credentials": {"initial_capital": 100000},
            "payload": {"amount": amount},
        }

    def test_add_funds_endpoint_success(self):
        response = self.client.post(
            "/internal/v1/backtrader/add-funds",
            headers={"Authorization": "Bearer test-token"},
            json=self._request_payload(1200),
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(float(body["data"]["amount"]), 1200.0)

    def test_add_funds_endpoint_validation_error(self):
        response = self.client.post(
            "/internal/v1/backtrader/add-funds",
            headers={"Authorization": "Bearer test-token"},
            json=self._request_payload(0),
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
