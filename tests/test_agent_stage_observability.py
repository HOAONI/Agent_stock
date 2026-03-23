# -*- coding: utf-8 -*-
"""分阶段耗时、输入与输出快照的可观测性回归测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

from agent_stock.agents.contracts import DataAgentOutput, ExecutionAgentOutput, RiskAgentOutput, SignalAgentOutput
from agent_stock.agents.orchestrator import AgentOrchestrator
from agent_stock.config import Config
from agent_stock.storage import DatabaseManager


class _DummyRepo:
    def get_or_create_account(self, name, initial_cash):
        class _A:
            id = 1

        return _A()

    def get_account_snapshot(self, name):
        return {
            "account_id": 1,
            "name": name,
            "cash": 100000.0,
            "total_market_value": 0.0,
            "total_asset": 100000.0,
            "positions": [],
        }


class _FakeDataAgent:
    def run(self, code: str, *, runtime_config=None) -> DataAgentOutput:
        return DataAgentOutput(
            code=code,
            trade_date=date(2026, 2, 25),
            analysis_context={"today": {"close": 10.0}, "raw_data": [{"date": "2026-02-24", "close": 9.9}]},
            realtime_quote={"price": 10.0},
            data_source="fake",
        )


class _FakeSignalAgent:
    def run(self, data_output: DataAgentOutput, *, runtime_config=None) -> SignalAgentOutput:
        return SignalAgentOutput(
            code=data_output.code,
            trade_date=data_output.trade_date,
            operation_advice="买入",
            sentiment_score=72,
            trend_signal="BUY",
            trend_score=75,
            stop_loss=9.2,
            take_profit=11.8,
            resolved_stop_loss=9.2,
            resolved_take_profit=11.8,
        )


class _FakeRiskAgent:
    def run(self, **kwargs) -> RiskAgentOutput:
        return RiskAgentOutput(
            code=kwargs["code"],
            trade_date=kwargs["trade_date"],
            target_weight=0.3,
            target_notional=30000.0,
            current_price=kwargs["current_price"],
            stop_loss=9.2,
            take_profit=11.8,
            effective_stop_loss=9.2,
            effective_take_profit=11.8,
            position_cap_pct=30.0,
            strategy_applied=True,
        )


class _FakeExecutionAgent:
    def run(self, **kwargs) -> ExecutionAgentOutput:
        return ExecutionAgentOutput(
            code=kwargs["code"],
            trade_date=kwargs["trade_date"],
            action="buy",
            reason="executed",
            traded_qty=100,
            fill_price=10.0,
            cash_before=100000.0,
            cash_after=99000.0,
            position_before=0,
            position_after=100,
            account_snapshot={"cash": 99000.0, "total_market_value": 1000.0, "total_asset": 100000.0, "positions": []},
        )


class AgentStageObservabilityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "stage_observability.db")
        os.environ["DATABASE_URL"] = ""
        Config.reset_instance()
        DatabaseManager.reset_instance()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_stage_duration_input_output_and_risk_effective_fields(self):
        now = datetime(2026, 2, 25, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
        orchestrator = AgentOrchestrator(
            data_agent=_FakeDataAgent(),
            signal_agent=_FakeSignalAgent(),
            risk_agent=_FakeRiskAgent(),
            execution_agent=_FakeExecutionAgent(),
            execution_repo=_DummyRepo(),
            now_provider=lambda: now,
            sleep_func=lambda _: None,
        )

        result = orchestrator.run_once(["600519"], account_name="user-stage")
        stock = result.results[0]

        self.assertIsNotNone(stock.data.duration_ms)
        self.assertIsNotNone(stock.signal.duration_ms)
        self.assertIsNotNone(stock.risk.duration_ms)
        self.assertIsNotNone(stock.execution.duration_ms)

        self.assertIsInstance(stock.data.input, dict)
        self.assertIsInstance(stock.signal.input, dict)
        self.assertIsInstance(stock.risk.input, dict)
        self.assertIsInstance(stock.execution.input, dict)

        self.assertIsInstance(stock.data.output, dict)
        self.assertIsInstance(stock.signal.output, dict)
        self.assertIsInstance(stock.risk.output, dict)
        self.assertIsInstance(stock.execution.output, dict)

        risk_payload = stock.risk.to_dict()
        self.assertIn("effective_stop_loss", risk_payload)
        self.assertIn("effective_take_profit", risk_payload)
        self.assertIn("position_cap_pct", risk_payload)
        self.assertIn("strategy_applied", risk_payload)

        signal_payload = stock.signal.to_dict()
        self.assertIn("resolved_stop_loss", signal_payload)
        self.assertIn("resolved_take_profit", signal_payload)

        execution_payload = stock.execution.to_dict()
        self.assertIn("execution_mode", execution_payload)
        self.assertIn("backend_task_id", execution_payload)
        self.assertIn("broker_requested", execution_payload)
        self.assertIn("executed_via", execution_payload)
        self.assertIn("broker_ticket_id", execution_payload)
        self.assertIn("fallback_reason", execution_payload)


if __name__ == "__main__":
    unittest.main()
