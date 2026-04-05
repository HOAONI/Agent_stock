# -*- coding: utf-8 -*-
"""基于运行时快照的 ExecutionAgent 意图映射单元测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date
from types import SimpleNamespace

from agent_stock.agents.contracts import DataAgentOutput, RiskAgentOutput
from agent_stock.agents.execution_agent import ExecutionAgent
from agent_stock.config import Config
from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.storage import DatabaseManager


class _UnavailableAnalyzer:
    def is_available(self) -> bool:
        return False


class ExecutionAgentAccountingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_execution_agent.db")
        self._env_backup = {
            "DATABASE_PATH": os.environ.get("DATABASE_PATH"),
            "DATABASE_URL": os.environ.get("DATABASE_URL"),
            "AGENT_ACCOUNT_NAME": os.environ.get("AGENT_ACCOUNT_NAME"),
            "AGENT_INITIAL_CASH": os.environ.get("AGENT_INITIAL_CASH"),
            "AGENT_MIN_TRADE_LOT": os.environ.get("AGENT_MIN_TRADE_LOT"),
            "AGENT_SLIPPAGE_BPS": os.environ.get("AGENT_SLIPPAGE_BPS"),
            "AGENT_FEE_RATE": os.environ.get("AGENT_FEE_RATE"),
            "AGENT_SELL_TAX_RATE": os.environ.get("AGENT_SELL_TAX_RATE"),
            "TUSHARE_TOKEN": os.environ.get("TUSHARE_TOKEN"),
        }

        os.environ["DATABASE_PATH"] = self.db_path
        os.environ["DATABASE_URL"] = ""
        os.environ["AGENT_ACCOUNT_NAME"] = "paper-test"
        os.environ["AGENT_INITIAL_CASH"] = "100000"
        os.environ["AGENT_MIN_TRADE_LOT"] = "100"
        os.environ["AGENT_SLIPPAGE_BPS"] = "5"
        os.environ["AGENT_FEE_RATE"] = "0.0003"
        os.environ["AGENT_SELL_TAX_RATE"] = "0.001"
        os.environ["TUSHARE_TOKEN"] = ""
        Config.reset_instance()
        DatabaseManager.reset_instance()

        self.db = DatabaseManager.get_instance()
        self.repo = ExecutionRepository(self.db)
        self.agent = ExecutionAgent(db_manager=self.db, execution_repo=self.repo, analyzer=_UnavailableAnalyzer())

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
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

        # 目标不变且传入上一份快照时，应跳过执行。
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

        # 卖出意图会使用运行时快照，并在轻量状态模式下立即执行。
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

    def test_execution_agent_reduces_order_when_price_is_highly_volatile(self):
        trade_date = date.today()
        runtime_snapshot = {
            "name": "paper-test",
            "cash": 100000.0,
            "initial_cash": 100000.0,
            "total_market_value": 0.0,
            "total_asset": 100000.0,
            "positions": [],
        }
        risk = RiskAgentOutput(
            code="300750",
            trade_date=trade_date,
            target_notional=60000.0,
            target_weight=0.6,
            current_price=200.0,
        )
        data_output = DataAgentOutput(
            code="300750",
            trade_date=trade_date,
            analysis_context={
                "today": {
                    "open": 196.0,
                    "high": 204.0,
                    "low": 194.0,
                    "close": 198.0,
                    "pct_chg": 4.7,
                },
                "yesterday": {"close": 191.0},
            },
            realtime_quote={"price": 200.0, "change_pct": 4.7},
        )

        output = self.agent.run(
            run_id="run-volatility",
            code="300750",
            trade_date=trade_date,
            current_price=200.0,
            risk_output=risk,
            account_snapshot=runtime_snapshot,
            signal_output=SimpleNamespace(operation_advice="买入"),
            data_output=data_output,
        )

        self.assertEqual(output.action, "buy")
        self.assertTrue(output.adjustment_applied)
        self.assertTrue(output.risk_reduction_only)
        self.assertIsNotNone(output.original_order)
        self.assertIsNotNone(output.final_order)
        self.assertLess(int(output.final_order["quantity"]), int(output.original_order["quantity"]))


if __name__ == "__main__":
    unittest.main()
