# -*- coding: utf-8 -*-
"""ExecutionAgent broker runtime fallback tests."""

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

from agent_stock.integrations.backend_bridge_client import BackendBridgeError


class _BridgeSuccess:
    def __init__(self):
        self.exchange_calls = []
        self.event_calls = []

    def exchange_credential_ticket(self, ticket: str):
        self.exchange_calls.append(ticket)
        return {
            "ticket_id": 321,
            "user_id": 9,
            "broker_account": {"id": 88},
            "task_id": "backend-task-1",
        }

    def post_execution_event(self, **kwargs):
        self.event_calls.append(kwargs)
        return {"ok": True}


class _BridgeExchangeFail:
    def __init__(self):
        self.exchange_calls = []
        self.event_calls = []

    def exchange_credential_ticket(self, ticket: str):
        self.exchange_calls.append(ticket)
        raise BackendBridgeError("ticket=agt_secret_failure", status_code=500, error_code="internal_error")

    def post_execution_event(self, **kwargs):
        self.event_calls.append(kwargs)
        return {"ok": True}


class _BridgeNoop:
    def __init__(self):
        self.exchange_calls = []

    def exchange_credential_ticket(self, ticket: str):
        self.exchange_calls.append(ticket)
        return {"ticket_id": 999}

    def post_execution_event(self, **kwargs):
        return {"ok": True}


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

    def test_broker_mode_fallback_to_paper_and_post_event(self):
        bridge = _BridgeSuccess()
        agent = ExecutionAgent(db_manager=self.db, execution_repo=self.repo, bridge_client=bridge)
        trade_date = date.today()
        risk = RiskAgentOutput(
            code="600519",
            trade_date=trade_date,
            target_notional=15000.0,
            target_weight=0.15,
            current_price=10.0,
        )

        output = agent.run(
            run_id="run-broker-fallback",
            code="600519",
            trade_date=trade_date,
            current_price=10.0,
            risk_output=risk,
            runtime_execution=RuntimeExecutionConfig(
                mode="broker",
                has_ticket=True,
                credential_ticket="agt_secret_ticket",
                ticket_id=123,
                broker_account_id=88,
            ),
            backend_task_id="backend-task-1",
        )

        self.assertEqual(output.execution_mode, "broker")
        self.assertTrue(output.broker_requested)
        self.assertEqual(output.executed_via, "paper")
        self.assertEqual(output.fallback_reason, "broker_contract_missing")
        self.assertEqual(output.broker_ticket_id, 321)
        self.assertEqual(len(bridge.exchange_calls), 1)
        self.assertEqual(len(bridge.event_calls), 1)
        self.assertEqual(bridge.event_calls[0]["status"], "failed")
        self.assertEqual(bridge.event_calls[0]["error_code"], "broker_contract_missing")
        self.assertNotIn("agt_secret_ticket", str(bridge.event_calls[0]))
        self.assertNotIn("agt_secret_ticket", str(output.to_dict()))

    def test_exchange_failure_fallbacks_to_paper_without_event(self):
        bridge = _BridgeExchangeFail()
        agent = ExecutionAgent(db_manager=self.db, execution_repo=self.repo, bridge_client=bridge)
        trade_date = date.today()
        risk = RiskAgentOutput(
            code="600519",
            trade_date=trade_date,
            target_notional=12000.0,
            target_weight=0.12,
            current_price=10.0,
        )

        output = agent.run(
            run_id="run-broker-exchange-fail",
            code="600519",
            trade_date=trade_date,
            current_price=10.0,
            risk_output=risk,
            runtime_execution=RuntimeExecutionConfig(
                mode="broker",
                has_ticket=True,
                credential_ticket="agt_secret_failure",
                ticket_id=555,
                broker_account_id=88,
            ),
            backend_task_id="backend-task-2",
        )

        self.assertEqual(output.execution_mode, "broker")
        self.assertTrue(output.broker_requested)
        self.assertEqual(output.executed_via, "paper")
        self.assertEqual(output.fallback_reason, "credential_exchange_failed")
        self.assertEqual(output.broker_ticket_id, 555)
        self.assertEqual(len(bridge.exchange_calls), 1)
        self.assertEqual(len(bridge.event_calls), 0)
        self.assertNotIn("agt_secret_failure", str(output.to_dict()))

    def test_missing_ticket_in_broker_mode_fallback(self):
        bridge = _BridgeNoop()
        agent = ExecutionAgent(db_manager=self.db, execution_repo=self.repo, bridge_client=bridge)
        trade_date = date.today()
        risk = RiskAgentOutput(
            code="600519",
            trade_date=trade_date,
            target_notional=8000.0,
            target_weight=0.08,
            current_price=10.0,
        )

        output = agent.run(
            run_id="run-broker-missing-ticket",
            code="600519",
            trade_date=trade_date,
            current_price=10.0,
            risk_output=risk,
            runtime_execution=RuntimeExecutionConfig(mode="broker", has_ticket=False, credential_ticket=None),
            backend_task_id="backend-task-3",
        )

        self.assertEqual(output.execution_mode, "broker")
        self.assertTrue(output.broker_requested)
        self.assertEqual(output.executed_via, "paper")
        self.assertEqual(output.fallback_reason, "missing_credential_ticket")
        self.assertEqual(len(bridge.exchange_calls), 0)


if __name__ == "__main__":
    unittest.main()
