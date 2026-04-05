# -*- coding: utf-8 -*-
"""实时循环周期控制的集成式测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

from agent_stock.agents.contracts import AgentRunResult
from agent_stock.agents.orchestrator import AgentOrchestrator
from agent_stock.config import Config
from agent_stock.storage import DatabaseManager


class _AlwaysOpenGuard:
    timezone = ZoneInfo("Asia/Shanghai")

    @staticmethod
    def is_market_open(now=None):
        return True


class _MinimalRepo:
    def get_or_create_account(self, name, initial_cash):
        class _A:
            id = 1

        return _A()

    def get_account_snapshot(self, name):
        return {"cash": 100000.0, "total_market_value": 0.0, "total_asset": 100000.0, "positions": []}


class _TestOrchestrator(AgentOrchestrator):
    def run_cycle(
        self,
        stock_codes,
        *,
        mode: str,
        request_id=None,
        initial_cash_override=None,
        runtime_config=None,
        account_name=None,
        planning_context=None,
        stage_observer=None,
    ):
        now = self._now()
        return AgentRunResult(
            run_id="run-1",
            mode=mode,
            started_at=now,
            ended_at=now,
            trade_date=date(2026, 2, 23),
            results=[],
            account_snapshot={"cash": 100000.0, "total_market_value": 0.0, "total_asset": 100000.0, "positions": []},
        )


class AgentRealtimeLoopTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "realtime_loop.db")
        os.environ["DATABASE_URL"] = ""
        Config.reset_instance()
        DatabaseManager.reset_instance()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_realtime_max_cycles(self):
        now = datetime(2026, 2, 23, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        orchestrator = _TestOrchestrator(
            execution_repo=_MinimalRepo(),
            market_guard=_AlwaysOpenGuard(),
            now_provider=lambda: now,
            sleep_func=lambda _: None,
        )

        results = orchestrator.run_realtime(["600519"], interval_minutes=5, max_cycles=1, heartbeat_sleep=0)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].mode, "realtime")


if __name__ == "__main__":
    unittest.main()
