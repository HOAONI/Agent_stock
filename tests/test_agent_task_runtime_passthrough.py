# -*- coding: utf-8 -*-
"""Task service runtime_config passthrough tests."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import date, datetime

from agent_stock.services.agent_task_service import AgentTaskService
from agent_stock.storage import DatabaseManager
from src.config import Config


class _FakeRunResult:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.mode = "once"
        self.trade_date = date(2026, 2, 24)
        self.started_at = datetime(2026, 2, 24, 10, 0, 0)
        self.ended_at = datetime(2026, 2, 24, 10, 0, 5)

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "trade_date": self.trade_date.isoformat(),
            "stock_codes": ["600519"],
            "account_name": "paper-default",
            "status": "completed",
            "data_snapshot": {},
            "signal_snapshot": {},
            "risk_snapshot": {},
            "execution_snapshot": {},
            "account_snapshot": {},
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "created_at": self.ended_at.isoformat(),
        }


class _CaptureAgentService:
    def __init__(self):
        self.calls = []

    def run_once(
        self,
        stock_codes,
        *,
        account_name=None,
        request_id=None,
        notify_enabled=None,
        write_reports=None,
        runtime_config=None,
    ):
        self.calls.append(
            {
                "stock_codes": list(stock_codes),
                "account_name": account_name,
                "request_id": request_id,
                "runtime_config": runtime_config,
            }
        )
        run_id = f"run-{len(self.calls)}"
        return _FakeRunResult(run_id=run_id)


class AgentTaskRuntimePassthroughTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "task_runtime_passthrough.db")
        os.environ["DATABASE_URL"] = ""
        os.environ["AGENT_SERVICE_MODE"] = "false"
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        Config.reset_instance()
        DatabaseManager.reset_instance()

        self.capture_service = _CaptureAgentService()
        self.task_service = AgentTaskService(agent_service=self.capture_service)

    def tearDown(self) -> None:
        self.task_service._executor.shutdown(wait=False, cancel_futures=True)
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_run_sync_runtime_config_passthrough(self):
        runtime_config = {
            "account": {
                "account_name": "user-sync",
                "initial_cash": 120000,
                "account_display_name": "sync-user",
            },
            "strategy": {
                "position_max_pct": 35,
                "stop_loss_pct": 7,
                "take_profit_pct": 12,
            },
            "execution": {
                "mode": "paper",
                "has_ticket": False,
            },
        }
        self.task_service.run_sync(
            stock_codes=["600519"],
            request_id="req-sync-runtime",
            runtime_config=runtime_config,
        )
        self.assertEqual(len(self.capture_service.calls), 1)
        self.assertEqual(self.capture_service.calls[0]["account_name"], "user-sync")
        self.assertEqual(self.capture_service.calls[0]["request_id"], "req-sync-runtime")
        self.assertEqual(self.capture_service.calls[0]["runtime_config"]["strategy"]["position_max_pct"], 35)
        self.assertEqual(self.capture_service.calls[0]["runtime_config"]["execution"]["mode"], "paper")

    def test_async_worker_runtime_config_passthrough(self):
        runtime_config = {
            "account": {
                "account_name": "user-async",
                "initial_cash": 80000,
                "account_display_name": "async-user",
            },
            "llm": {
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "api_token": "runtime-secret-token",
                "has_token": True,
            },
            "strategy": {
                "position_max_pct": 25,
                "stop_loss_pct": 6,
                "take_profit_pct": 10,
            },
            "execution": {
                "mode": "broker",
                "has_ticket": True,
                "credential_ticket": "agt-worker-ticket",
                "ticket_id": 11,
                "broker_account_id": 22,
            },
        }
        task = self.task_service.submit_task(
            stock_codes=["600519"],
            request_id="req-async-runtime",
            runtime_config=runtime_config,
        )

        deadline = time.time() + 3
        latest = task
        while time.time() < deadline:
            latest = self.task_service.get_task(task["task_id"])
            if latest.get("status") in {"completed", "failed"}:
                break
            time.sleep(0.05)

        self.assertEqual(latest.get("status"), "completed")
        self.assertGreaterEqual(len(self.capture_service.calls), 1)
        self.assertEqual(self.capture_service.calls[-1]["account_name"], "user-async")
        self.assertEqual(self.capture_service.calls[-1]["request_id"], "req-async-runtime")
        self.assertEqual(self.capture_service.calls[-1]["runtime_config"]["llm"]["model"], "gpt-4o-mini")
        self.assertEqual(self.capture_service.calls[-1]["runtime_config"]["execution"]["mode"], "broker")


if __name__ == "__main__":
    unittest.main()
