# -*- coding: utf-8 -*-
"""服务重启场景下的任务恢复测试。"""

from __future__ import annotations

import os
import tempfile
import unittest

from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.services.agent_task_service import get_agent_task_service, reset_agent_task_service
from agent_stock.storage import DatabaseManager
from agent_stock.config import Config


class AgentTaskRecoveryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "task_recovery.db")
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()

    def tearDown(self) -> None:
        reset_agent_task_service()
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_pending_tasks_are_marked_failed_after_restart(self):
        db = DatabaseManager.get_instance()
        repo = ExecutionRepository(db)
        repo.create_agent_task(
            task_id="task-restart-1",
            request_id="req-restart-1",
            stock_codes=["600519"],
            account_name="paper-default",
            status="pending",
        )

        # 通过重置单例并重新初始化来模拟进程重启。
        reset_agent_task_service()
        DatabaseManager.reset_instance()

        service = get_agent_task_service()
        task_payload = service.get_task("task-restart-1")

        self.assertEqual(task_payload.get("status"), "failed")
        self.assertEqual(task_payload.get("error_message"), "service_restarted")


if __name__ == "__main__":
    unittest.main()
