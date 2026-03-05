# -*- coding: utf-8 -*-
"""ExecutionAgent simulation-only runtime tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date

from src.agents.contracts import RiskAgentOutput
from src.agents.execution_agent import ExecutionAgent
from src.config import Config, RuntimeExecutionConfig
from src.repositories.execution_repo import ExecutionRepository
from src.storage import DatabaseManager


class ExecutionAgentSimulationOnlyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "execution_agent_simulation_only.db")

        os.environ["DATABASE_PATH"] = self.db_path
        os.environ["DATABASE_URL"] = ""
        os.environ["AGENT_ACCOUNT_NAME"] = "paper-test"
        os.environ["AGENT_INITIAL_CASH"] = "100000"
        os.environ["AGENT_MIN_TRADE_LOT"] = "100"
        os.environ["AGENT_SLIPPAGE_BPS"] = "5"
        os.environ["AGENT_FEE_RATE"] = "0.0003"
        os.environ["AGENT_SELL_TAX_RATE"] = "0.001"
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "agent-token-test"
        Config.reset_instance()
        DatabaseManager.reset_instance()
        self.db = DatabaseManager.get_instance()
        self.repo = ExecutionRepository(self.db)

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_runtime_execution_context_remains_paper_only(self):
        agent = ExecutionAgent(db_manager=self.db, execution_repo=self.repo)
        trade_date = date.today()
        risk = RiskAgentOutput(
            code="600519",
            trade_date=trade_date,
            target_notional=15000.0,
            target_weight=0.15,
            current_price=10.0,
        )

        output = agent.run(
            run_id="run-paper-mode",
            code="600519",
            trade_date=trade_date,
            current_price=10.0,
            risk_output=risk,
            runtime_execution=RuntimeExecutionConfig(
                mode="paper",
                has_ticket=True,
                broker_account_id=88,
            ),
            backend_task_id="backend-task-1",
        )

        self.assertEqual(output.execution_mode, "paper")
        self.assertFalse(output.broker_requested)
        self.assertEqual(output.executed_via, "paper")
        self.assertIsNone(output.broker_ticket_id)
        self.assertIsNone(output.fallback_reason)
        self.assertEqual(output.backend_task_id, "backend-task-1")

    def test_default_runtime_execution_is_paper(self):
        agent = ExecutionAgent(db_manager=self.db, execution_repo=self.repo)
        trade_date = date.today()
        risk = RiskAgentOutput(
            code="600519",
            trade_date=trade_date,
            target_notional=12000.0,
            target_weight=0.12,
            current_price=10.0,
        )

        output = agent.run(
            run_id="run-no-runtime",
            code="600519",
            trade_date=trade_date,
            current_price=10.0,
            risk_output=risk,
            runtime_execution=None,
            backend_task_id="backend-task-2",
        )

        self.assertEqual(output.execution_mode, "paper")
        self.assertFalse(output.broker_requested)
        self.assertEqual(output.executed_via, "paper")
        self.assertIsNone(output.broker_ticket_id)
        self.assertIsNone(output.fallback_reason)

    def test_skip_path_keeps_compatibility_fields_fixed(self):
        agent = ExecutionAgent(db_manager=self.db, execution_repo=self.repo)
        trade_date = date.today()
        risk = RiskAgentOutput(
            code="600519",
            trade_date=trade_date,
            target_notional=0.0,
            target_weight=0.0,
            current_price=10.0,
        )

        output = agent.run(
            run_id="run-skip",
            code="600519",
            trade_date=trade_date,
            current_price=10.0,
            risk_output=risk,
            runtime_execution=RuntimeExecutionConfig(mode="paper", has_ticket=False, broker_account_id=88),
            backend_task_id="backend-task-3",
        )

        self.assertEqual(output.state.value, "skipped")
        self.assertEqual(output.execution_mode, "paper")
        self.assertFalse(output.broker_requested)
        self.assertEqual(output.executed_via, "paper")
        self.assertIsNone(output.broker_ticket_id)
        self.assertIsNone(output.fallback_reason)


if __name__ == "__main__":
    unittest.main()
