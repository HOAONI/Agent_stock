# -*- coding: utf-8 -*-
"""API auth and health endpoint tests."""

from __future__ import annotations

import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from agent_api.app import create_app
from agent_stock.services.agent_task_service import reset_agent_task_service
from agent_stock.storage import DatabaseManager
from src.config import Config


class AgentApiAuthTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_api_auth.db")
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()

        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        self.client.close()
        reset_agent_task_service()
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_health_endpoint_without_auth(self):
        response = self.client.get("/api/health/live")
        self.assertEqual(response.status_code, 200)

    def test_non_health_endpoint_requires_auth(self):
        response = self.client.post(
            "/api/v1/runs",
            json={"stock_codes": ["600519"], "async_mode": False},
        )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
