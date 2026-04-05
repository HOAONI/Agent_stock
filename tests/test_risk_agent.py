# -*- coding: utf-8 -*-
"""RiskAgent 单元测试。"""

from __future__ import annotations

import os
import unittest
from datetime import date

from agent_stock.agents.contracts import SignalAgentOutput
from agent_stock.agents.risk_agent import RiskAgent
from agent_stock.config import Config, RuntimeStrategyConfig


class _UnavailableAnalyzer:
    def is_available(self) -> bool:
        return False


class RiskAgentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["AGENT_WEIGHT_STRONG_BUY"] = "0.5"
        os.environ["AGENT_WEIGHT_BUY"] = "0.3"
        os.environ["AGENT_WEIGHT_HOLD"] = "0.2"
        os.environ["AGENT_WEIGHT_WAIT"] = "0.0"
        os.environ["AGENT_WEIGHT_SELL"] = "0.0"
        os.environ["AGENT_MAX_SINGLE_POSITION_PCT"] = "0.5"
        os.environ["AGENT_MAX_TOTAL_EXPOSURE_PCT"] = "0.9"
        Config.reset_instance()
        self.agent = RiskAgent(analyzer=_UnavailableAnalyzer())

    def tearDown(self) -> None:
        Config.reset_instance()

    def test_buy_weight_mapping(self):
        signal = SignalAgentOutput(code="600519", trade_date=date.today(), operation_advice="买入", sentiment_score=70)
        output = self.agent.run(
            code="600519",
            trade_date=date.today(),
            current_price=10.0,
            signal_output=signal,
            account_snapshot={"cash": 100000.0, "total_asset": 100000.0, "total_market_value": 0.0},
            current_position_value=0.0,
        )
        self.assertAlmostEqual(output.target_weight, 0.3, places=4)
        self.assertAlmostEqual(output.target_notional, 30000.0, places=2)

    def test_stop_loss_force_flat(self):
        signal = SignalAgentOutput(
            code="600519",
            trade_date=date.today(),
            operation_advice="买入",
            sentiment_score=70,
            stop_loss=9.5,
        )
        output = self.agent.run(
            code="600519",
            trade_date=date.today(),
            current_price=9.4,
            signal_output=signal,
            account_snapshot={"cash": 100000.0, "total_asset": 100000.0, "total_market_value": 50000.0},
            current_position_value=20000.0,
        )
        self.assertEqual(output.target_weight, 0.0)
        self.assertIn("stop_loss_triggered", output.risk_flags)

    def test_total_exposure_clamp(self):
        signal = SignalAgentOutput(code="600519", trade_date=date.today(), operation_advice="强烈买入", sentiment_score=90)
        output = self.agent.run(
            code="600519",
            trade_date=date.today(),
            current_price=10.0,
            signal_output=signal,
            account_snapshot={"cash": 100000.0, "total_asset": 100000.0, "total_market_value": 88000.0},
            current_position_value=10000.0,
        )
        self.assertLessEqual(output.target_notional, 12000.0)
        self.assertIn("total_exposure_clamped", output.risk_flags)

    def test_runtime_position_max_pct_override(self):
        signal = SignalAgentOutput(code="600519", trade_date=date.today(), operation_advice="强烈买入", sentiment_score=90)
        output = self.agent.run(
            code="600519",
            trade_date=date.today(),
            current_price=10.0,
            signal_output=signal,
            account_snapshot={"cash": 100000.0, "total_asset": 100000.0, "total_market_value": 0.0, "positions": []},
            current_position_value=0.0,
            runtime_strategy=RuntimeStrategyConfig(position_max_pct=20, stop_loss_pct=8, take_profit_pct=15),
        )
        self.assertAlmostEqual(output.target_weight, 0.2, places=4)
        self.assertAlmostEqual(output.target_notional, 20000.0, places=2)

    def test_runtime_take_profit_triggered_by_avg_cost(self):
        signal = SignalAgentOutput(code="600519", trade_date=date.today(), operation_advice="持有", sentiment_score=60)
        output = self.agent.run(
            code="600519",
            trade_date=date.today(),
            current_price=11.5,
            signal_output=signal,
            account_snapshot={
                "cash": 50000.0,
                "total_asset": 100000.0,
                "total_market_value": 50000.0,
                "positions": [{"code": "600519", "quantity": 1000, "avg_cost": 10.0, "market_value": 11500.0}],
            },
            current_position_value=11500.0,
            runtime_strategy=RuntimeStrategyConfig(position_max_pct=30, stop_loss_pct=8, take_profit_pct=12),
        )
        self.assertEqual(output.target_weight, 0.0)
        self.assertIn("runtime_take_profit_triggered", output.risk_flags)

    def test_runtime_position_max_pct_zero_blocks_opening(self):
        signal = SignalAgentOutput(code="600519", trade_date=date.today(), operation_advice="买入", sentiment_score=70)
        output = self.agent.run(
            code="600519",
            trade_date=date.today(),
            current_price=10.0,
            signal_output=signal,
            account_snapshot={"cash": 100000.0, "total_asset": 100000.0, "total_market_value": 0.0, "positions": []},
            current_position_value=0.0,
            runtime_strategy=RuntimeStrategyConfig(position_max_pct=0, stop_loss_pct=8, take_profit_pct=15),
        )
        self.assertAlmostEqual(output.target_weight, 0.0, places=4)
        self.assertAlmostEqual(output.target_notional, 0.0, places=2)
        self.assertAlmostEqual(float(output.position_cap_pct or 0.0), 0.0, places=4)

    def test_runtime_zero_threshold_disables_signal_trigger(self):
        signal = SignalAgentOutput(
            code="600519",
            trade_date=date.today(),
            operation_advice="买入",
            sentiment_score=70,
            stop_loss=9.5,
            take_profit=10.5,
        )
        output = self.agent.run(
            code="600519",
            trade_date=date.today(),
            current_price=9.4,
            signal_output=signal,
            account_snapshot={
                "cash": 50000.0,
                "total_asset": 100000.0,
                "total_market_value": 50000.0,
                "positions": [{"code": "600519", "quantity": 1000, "avg_cost": 10.0, "market_value": 9400.0}],
            },
            current_position_value=9400.0,
            runtime_strategy=RuntimeStrategyConfig(position_max_pct=30, stop_loss_pct=0, take_profit_pct=0),
        )
        self.assertNotIn("stop_loss_triggered", output.risk_flags)
        self.assertNotIn("take_profit_triggered", output.risk_flags)
        self.assertIsNone(output.effective_stop_loss)
        self.assertIsNone(output.effective_take_profit)


if __name__ == "__main__":
    unittest.main()
