# -*- coding: utf-8 -*-
"""并发请求下的运行时配置隔离测试。"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

from agent_stock.agents.contracts import AgentRunResult
from agent_stock.services.agent_service import AgentService
from agent_stock.storage import DatabaseManager
from agent_stock.config import Config, get_config


class _CaptureRepo:
    def __init__(self):
        self.saved = []

    def save_agent_run(self, **kwargs):
        self.saved.append(kwargs)


class _CaptureOrchestrator:
    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()
        self._counter = 0

    def run_once(
        self,
        stock_codes,
        *,
        account_name=None,
        request_id=None,
        initial_cash_override=None,
        runtime_config=None,
    ):
        with self._lock:
            self._counter += 1
            run_id = f"run-iso-{self._counter}"
            self.calls.append(
                {
                    "stock_codes": list(stock_codes),
                    "account_name": account_name,
                    "request_id": request_id,
                    "initial_cash_override": initial_cash_override,
                    "runtime_config": runtime_config,
                }
            )
        now = datetime(2026, 2, 24, 10, 0, 0)
        return AgentRunResult(
            run_id=run_id,
            mode="once",
            started_at=now,
            ended_at=now,
            trade_date=date(2026, 2, 24),
            results=[],
            account_snapshot={"cash": 100000.0, "total_market_value": 0.0, "total_asset": 100000.0, "positions": []},
        )


class AgentRuntimeIsolationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "runtime_isolation.db")
        os.environ["DATABASE_URL"] = ""
        os.environ["AGENT_SERVICE_MODE"] = "false"
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["OPENAI_API_KEY"] = "env-openai-token-1234567890"
        os.environ["OPENAI_MODEL"] = "gpt-env-default"
        Config.reset_instance()
        DatabaseManager.reset_instance()

        self.config = get_config()
        self.capture_repo = _CaptureRepo()
        self.capture_orchestrator = _CaptureOrchestrator()
        self.service = AgentService(
            config=self.config,
            orchestrator=self.capture_orchestrator,
            execution_repo=self.capture_repo,
        )

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_concurrent_runtime_configs_are_isolated(self):
        payload_a = {
            "account": {
                "account_name": "user-a",
                "initial_cash": 100000,
                "account_display_name": "User A",
            },
            "llm": {
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "model": "model-a",
                "api_token": "token-a-123456",
                "has_token": True,
            },
            "strategy": {
                "position_max_pct": 30,
                "stop_loss_pct": 8,
                "take_profit_pct": 15,
            },
            "execution": {
                "mode": "paper",
                "has_ticket": True,
                "broker_account_id": 501,
            },
        }
        payload_b = {
            "account": {
                "account_name": "user-b",
                "initial_cash": 200000,
                "account_display_name": "User B",
            },
            "llm": {
                "provider": "custom",
                "base_url": "https://llm.example.com/v1",
                "model": "model-b",
                "api_token": "token-b-654321",
                "has_token": True,
            },
            "strategy": {
                "position_max_pct": 20,
                "stop_loss_pct": 6,
                "take_profit_pct": 10,
            },
            "execution": {
                "mode": "paper",
                "has_ticket": False,
                "broker_account_id": 502,
            },
        }

        def _run(payload):
            self.service.run_once(["600519"], runtime_config=payload, write_reports=False)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(_run, payload_a)
            future_b = executor.submit(_run, payload_b)
            future_a.result()
            future_b.result()

        self.assertEqual(len(self.capture_orchestrator.calls), 2)
        seen_accounts = {item["account_name"] for item in self.capture_orchestrator.calls}
        self.assertEqual(seen_accounts, {"user-a", "user-b"})

        llm_models = {
            item["runtime_config"].llm.model
            for item in self.capture_orchestrator.calls
            if item["runtime_config"] and item["runtime_config"].llm
        }
        self.assertEqual(llm_models, {"model-a", "model-b"})

        execution_modes = {
            item["runtime_config"].execution.mode
            for item in self.capture_orchestrator.calls
            if item["runtime_config"] and item["runtime_config"].execution
        }
        self.assertEqual(execution_modes, {"paper"})

        # 确保请求级覆盖后，全局单例配置保持不变。
        self.assertEqual(self.service.config.openai_model, "gpt-env-default")
        self.assertEqual(self.service.config.openai_api_key, "env-openai-token-1234567890")


if __name__ == "__main__":
    unittest.main()
