# -*- coding: utf-8 -*-
"""Unit tests for ExecutionAgent intent projection with runtime snapshots."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date

from src.agents.contracts import RiskAgentOutput
from src.agents.execution_agent import ExecutionAgent
from src.config import Config
from src.repositories.execution_repo import ExecutionRepository
from src.storage import DatabaseManager


class ExecutionAgentAccountingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_execution_agent.db")

        os.environ["DATABASE_PATH"] = self.db_path
        os.environ["AGENT_ACCOUNT_NAME"] = "paper-test"
        os.environ["AGENT_INITIAL_CASH"] = "100000"
        os.environ["AGENT_MIN_TRADE_LOT"] = "100"
        os.environ["AGENT_SLIPPAGE_BPS"] = "5"
        os.environ["AGENT_FEE_RATE"] = "0.0003"
        os.environ["AGENT_SELL_TAX_RATE"] = "0.001"
        Config.reset_instance()
        DatabaseManager.reset_instance()

        self.db = DatabaseManager.get_instance()
        self.repo = ExecutionRepository(self.db)
        self.agent = ExecutionAgent(db_manager=self.db, execution_repo=self.repo)

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_buy_hold_sell_with_runtime_snapshot(self):
        trade_date = date.today()
        runtime_snapshot = {
            "name": "paper-test",
            "cash": 100000.0,
            "initial_cash": 100000.0,
            "total_market_value": 0.0,
            "total_asset": 100000.0,
            "positions": [],
        }

        buy_risk = RiskAgentOutput(
            code="600519",
            trade_date=trade_date,
            target_notional=30000.0,
            target_weight=0.3,
            current_price=10.0,
        )
        buy_out = self.agent.run(
            run_id="run-buy",
            code="600519",
            trade_date=trade_date,
            current_price=10.0,
            risk_output=buy_risk,
            account_snapshot=runtime_snapshot,
        )
        self.assertEqual(buy_out.action, "buy")
        self.assertGreater(buy_out.traded_qty, 0)
        runtime_snapshot = buy_out.account_snapshot

        # Keep same target and pass previous snapshot: should skip.
        hold_notional = float(buy_out.target_qty) * 10.0 * (1.0 + 5 / 10000.0)
        hold_risk = RiskAgentOutput(
            code="600519",
            trade_date=trade_date,
            target_notional=hold_notional,
            target_weight=0.0,
            current_price=10.0,
        )
        hold_out = self.agent.run(
            run_id="run-hold",
            code="600519",
            trade_date=trade_date,
            current_price=10.0,
            risk_output=hold_risk,
            account_snapshot=runtime_snapshot,
        )
        self.assertEqual(hold_out.action, "none")
        runtime_snapshot = hold_out.account_snapshot

        # Sell intent uses runtime snapshot and executes immediately in light-state mode.
        flat_risk = RiskAgentOutput(
            code="600519",
            trade_date=trade_date,
            target_notional=0.0,
            target_weight=0.0,
            current_price=10.0,
        )
        sell_out = self.agent.run(
            run_id="run-sell",
            code="600519",
            trade_date=trade_date,
            current_price=10.0,
            risk_output=flat_risk,
            account_snapshot=runtime_snapshot,
        )
        self.assertEqual(sell_out.action, "sell")
        self.assertGreater(sell_out.traded_qty, 0)
        self.assertEqual(sell_out.position_after, 0)


if __name__ == "__main__":
    unittest.main()
