# -*- coding: utf-8 -*-
"""本地 Agent 报表渲染单元测试。"""

from __future__ import annotations

import unittest

from agent_stock.reporting import generate_agent_execution_report


class AgentReportingTestCase(unittest.TestCase):
    def test_generate_agent_execution_report(self):
        payload = {
            "run_id": "run-123",
            "mode": "once",
            "trade_date": "2026-02-23",
            "account_snapshot": {
                "cash": 90000.0,
                "total_market_value": 10000.0,
                "total_asset": 100000.0,
            },
            "results": [
                {
                    "code": "600519",
                    "signal": {"operation_advice": "买入"},
                    "risk": {"target_weight": 0.3},
                    "execution": {
                        "action": "buy",
                        "traded_qty": 100,
                        "fill_price": 10.05,
                        "cash_after": 90000.0,
                        "position_after": 100,
                    },
                }
            ],
        }

        report = generate_agent_execution_report(payload)

        self.assertIn("Multi-Agent Execution Report", report)
        self.assertIn("run-123", report)
        self.assertIn("600519", report)
        self.assertIn("| buy |", report)


if __name__ == "__main__":
    unittest.main()
