# -*- coding: utf-8 -*-
"""单次编排周期的集成式测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

from agent_stock.agents.contracts import DataAgentOutput, ExecutionAgentOutput, RiskAgentOutput, SignalAgentOutput
from agent_stock.agents.orchestrator import AgentOrchestrator, MarketSessionGuard
from agent_stock.config import Config
from agent_stock.storage import DatabaseManager


class _DummyRepo:
    def __init__(self):
        self._account_id = 1

    def get_or_create_account(self, name, initial_cash):
        class _A:
            id = 1

        return _A()

    def get_account_snapshot(self, name):
        return {
            "account_id": self._account_id,
            "cash": 100000.0,
            "total_market_value": 0.0,
            "total_asset": 100000.0,
            "positions": [],
        }


class _FakeDataAgent:
    def run(self, code: str, *, runtime_config=None) -> DataAgentOutput:
        return DataAgentOutput(
            code=code,
            trade_date=date(2026, 2, 23),
            analysis_context={"today": {"close": 10.0}},
            realtime_quote={"price": 10.0},
        )


class _FakeSignalAgent:
    def run(self, data_output: DataAgentOutput, *, runtime_config=None) -> SignalAgentOutput:
        return SignalAgentOutput(
            code=data_output.code,
            trade_date=data_output.trade_date,
            operation_advice="观望",
            sentiment_score=50,
        )


class _FakeRiskAgent:
    def run(self, **kwargs) -> RiskAgentOutput:
        return RiskAgentOutput(
            code=kwargs["code"],
            trade_date=kwargs["trade_date"],
            target_weight=0.0,
            target_notional=0.0,
            current_price=kwargs["current_price"],
        )


class _FakeExecutionAgent:
    def run(self, **kwargs) -> ExecutionAgentOutput:
        return ExecutionAgentOutput(
            code=kwargs["code"],
            trade_date=kwargs["trade_date"],
            action="none",
            reason="target_matched",
            account_snapshot={
                "cash": 100000.0,
                "total_market_value": 0.0,
                "total_asset": 100000.0,
                "positions": [],
            },
        )


class AgentOrchestratorOnceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "orchestrator_once.db")
        os.environ["DATABASE_URL"] = ""
        Config.reset_instance()
        DatabaseManager.reset_instance()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_run_once(self):
        tz = ZoneInfo("Asia/Shanghai")
        now = datetime(2026, 2, 23, 10, 0, tzinfo=tz)
        orchestrator = AgentOrchestrator(
            data_agent=_FakeDataAgent(),
            signal_agent=_FakeSignalAgent(),
            risk_agent=_FakeRiskAgent(),
            execution_agent=_FakeExecutionAgent(),
            execution_repo=_DummyRepo(),
            market_guard=MarketSessionGuard("Asia/Shanghai", "09:30-11:30,13:00-15:00"),
            now_provider=lambda: now,
            sleep_func=lambda _: None,
        )

        result = orchestrator.run_once(["600519", "000001"])

        self.assertEqual(result.mode, "once")
        self.assertEqual(len(result.results), 2)
        self.assertEqual(result.results[0].code, "600519")
        self.assertEqual(result.results[1].code, "000001")

    def test_run_once_rejects_empty_stock_codes(self):
        tz = ZoneInfo("Asia/Shanghai")
        now = datetime(2026, 2, 23, 10, 0, tzinfo=tz)
        orchestrator = AgentOrchestrator(
            data_agent=_FakeDataAgent(),
            signal_agent=_FakeSignalAgent(),
            risk_agent=_FakeRiskAgent(),
            execution_agent=_FakeExecutionAgent(),
            execution_repo=_DummyRepo(),
            market_guard=MarketSessionGuard("Asia/Shanghai", "09:30-11:30,13:00-15:00"),
            now_provider=lambda: now,
            sleep_func=lambda _: None,
        )

        with self.assertRaisesRegex(ValueError, "stock_codes must not be empty"):
            orchestrator.run_once([])


if __name__ == "__main__":
    unittest.main()
