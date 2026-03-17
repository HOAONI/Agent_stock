# -*- coding: utf-8 -*-
"""API 鉴权与健康检查端点测试。"""

from __future__ import annotations

import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from agent_api.app import create_app
from agent_stock.services.agent_task_service import reset_agent_task_service
from agent_stock.storage import DatabaseManager
from agent_stock.config import Config


class AgentApiAuthTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_api_auth.db")
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"
        os.environ["OPENAI_API_KEY"] = "deepseek-test-key-123456"
        os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"
        os.environ["OPENAI_MODEL"] = "deepseek-chat"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()

        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        self.client.close()
        reset_agent_task_service()
        DatabaseManager.reset_instance()
        Config.reset_instance()
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("OPENAI_MODEL", None)
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

    def test_runtime_default_endpoint_requires_auth(self):
        response = self.client.get("/internal/v1/runtime/llm-default")
        self.assertEqual(response.status_code, 401)

    def test_runtime_default_endpoint_reports_builtin_llm(self):
        response = self.client.get(
            "/internal/v1/runtime/llm-default",
            headers={"Authorization": "Bearer test-token"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["provider"], "deepseek")
        self.assertEqual(payload["model"], "deepseek-chat")
        self.assertEqual(payload["base_url"], "https://api.deepseek.com/v1")
        self.assertTrue(payload["has_token"])


if __name__ == "__main__":
    unittest.main()
