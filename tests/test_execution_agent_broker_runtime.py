# -*- coding: utf-8 -*-
"""ExecutionAgent 券商运行时测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date

from agent_stock.agents.contracts import RiskAgentOutput
from agent_stock.agents.execution_agent import ExecutionAgent
from agent_stock.config import Config, RuntimeExecutionConfig
from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.storage import DatabaseManager


class _FakeBacktraderRuntimeService:
    def __init__(self) -> None:
        self.cash = 100000.0
        self.initial_capital = 100000.0
        self.positions = []
        self.last_order = None
        self.reject_with = None

    def get_account_summary(self, req):
        return {
            "broker_account_id": req["broker_account_id"],
            "cash": self.cash,
            "market_value": sum(float(item["market_value"]) for item in self.positions),
            "total_asset": self.cash + sum(float(item["market_value"]) for item in self.positions),
            "initial_capital": self.initial_capital,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "cumulative_fees": 0.0,
            "snapshot_at": "2026-03-07T10:00:00",
        }

    def get_positions(self, _req):
        return list(self.positions)

    def place_order(self, req):
        self.last_order = req
        if self.reject_with:
            return {
                "order_id": None,
                "trade_id": None,
                "status": "rejected",
                "provider_status": "rejected",
                "provider_order_id": None,
                "filled_quantity": 0,
                "filled_price": None,
                "fee": 0.0,
                "tax": 0.0,
                "message": self.reject_with,
            }

        payload = req["payload"]
        qty = int(payload["quantity"])
        price = float(payload["price"])
        fee = round(price * qty * 0.0003, 4)
        self.cash = round(self.cash - price * qty - fee, 4)
        self.positions = [
            {
                "stock_code": payload["stock_code"],
                "quantity": qty,
                "available_qty": qty,
                "avg_cost": price,
                "last_price": price,
                "market_value": round(price * qty, 4),
                "unrealized_pnl": 0.0,
            }
        ]
        return {
            "order_id": 11,
            "trade_id": 22,
            "status": "filled",
            "provider_status": "filled",
            "provider_order_id": "bt-order-11",
            "filled_quantity": qty,
            "filled_price": price,
            "fee": fee,
            "tax": 0.0,
            "cash_before": 100000.0,
            "cash_after": self.cash,
            "position_before": 0,
            "position_after": qty,
            "message": "ok",
        }


class ExecutionAgentBrokerRuntimeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "execution_agent_broker_runtime.db")

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

    def test_runtime_execution_context_remains_paper(self):
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

    def test_broker_mode_executes_via_internal_runtime(self):
        runtime_service = _FakeBacktraderRuntimeService()
        agent = ExecutionAgent(
            db_manager=self.db,
            execution_repo=self.repo,
            runtime_service=runtime_service,
        )
        trade_date = date.today()
        risk = RiskAgentOutput(
            code="600519",
            trade_date=trade_date,
            target_notional=15000.0,
            target_weight=0.15,
            current_price=10.0,
        )

        output = agent.run(
            run_id="run-broker-mode",
            code="600519",
            trade_date=trade_date,
            current_price=10.0,
            risk_output=risk,
            runtime_execution=RuntimeExecutionConfig(
                mode="broker",
                has_ticket=False,
                broker_account_id=88,
            ),
            backend_task_id="backend-task-2",
        )

        self.assertEqual(output.execution_mode, "broker")
        self.assertTrue(output.broker_requested)
        self.assertEqual(output.executed_via, "backtrader_internal")
        self.assertEqual(output.broker_ticket_id, "bt-order-11")
        self.assertEqual(output.order_id, 11)
        self.assertEqual(output.trade_id, 22)
        self.assertGreater(output.traded_qty, 0)
        self.assertEqual(output.reason, "broker_executed")
        self.assertEqual(runtime_service.last_order["broker_account_id"], 88)
        self.assertEqual(runtime_service.last_order["payload"]["direction"], "buy")

    def test_broker_mode_rejects_when_account_id_missing(self):
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
            run_id="run-broker-missing-account",
            code="600519",
            trade_date=trade_date,
            current_price=10.0,
            risk_output=risk,
            runtime_execution=RuntimeExecutionConfig(mode="broker", has_ticket=False, broker_account_id=None),
            backend_task_id="backend-task-3",
        )

        self.assertEqual(output.state.value, "failed")
        self.assertEqual(output.execution_mode, "broker")
        self.assertEqual(output.reason, "invalid_broker_account")

    def test_broker_mode_surfaces_runtime_rejection(self):
        runtime_service = _FakeBacktraderRuntimeService()
        runtime_service.reject_with = "可用资金不足"
        agent = ExecutionAgent(
            db_manager=self.db,
            execution_repo=self.repo,
            runtime_service=runtime_service,
        )
        trade_date = date.today()
        risk = RiskAgentOutput(
            code="600519",
            trade_date=trade_date,
            target_notional=15000.0,
            target_weight=0.15,
            current_price=10.0,
        )

        output = agent.run(
            run_id="run-broker-reject",
            code="600519",
            trade_date=trade_date,
            current_price=10.0,
            risk_output=risk,
            runtime_execution=RuntimeExecutionConfig(mode="broker", has_ticket=False, broker_account_id=88),
            backend_task_id="backend-task-4",
        )

        self.assertEqual(output.state.value, "failed")
        self.assertEqual(output.execution_mode, "broker")
        self.assertTrue(output.broker_requested)
        self.assertEqual(output.reason, "broker_rejected")
        self.assertEqual(output.error_message, "可用资金不足")


if __name__ == "__main__":
    unittest.main()
