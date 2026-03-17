# -*- coding: utf-8 -*-
"""任务轮询端点测试。"""

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


class _TaskPollingService:
    def get_task(self, task_id: str):
        if task_id == "task-ok":
            return {
                "task_id": "task-ok",
                "status": "completed",
                "request_id": "req-1",
                "stock_codes": ["600519"],
                "account_name": "paper-default",
                "run_id": "run-1",
                "error_message": None,
                "created_at": "2026-02-23T10:00:00",
                "started_at": "2026-02-23T10:00:01",
                "completed_at": "2026-02-23T10:00:10",
                "updated_at": "2026-02-23T10:00:10",
            }
        return {}


class AgentApiTasksPollingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_api_tasks_polling.db")
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()

        self.app = create_app()
        self.app.dependency_overrides[get_task_service_dep] = lambda: _TaskPollingService()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.app.dependency_overrides.clear()
        reset_agent_task_service()
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_task_polling_success(self):
        response = self.client.get("/api/v1/tasks/task-ok", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], "run-1")

    def test_task_not_found(self):
        response = self.client.get("/api/v1/tasks/task-missing", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
