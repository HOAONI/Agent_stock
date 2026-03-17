# -*- coding: utf-8 -*-
"""`MarketSessionGuard` 单元测试。"""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from agent_stock.agents.orchestrator import MarketSessionGuard


class MarketSessionGuardTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tz = ZoneInfo("Asia/Shanghai")
        self.guard = MarketSessionGuard("Asia/Shanghai", "09:30-11:30,13:00-15:00")

    def test_market_open_in_morning_session(self):
        dt = datetime(2026, 2, 23, 10, 0, tzinfo=self.tz)  # 周一
        self.assertTrue(self.guard.is_market_open(dt))

    def test_market_closed_at_noon_break(self):
        dt = datetime(2026, 2, 23, 12, 0, tzinfo=self.tz)
        self.assertFalse(self.guard.is_market_open(dt))

    def test_market_closed_on_weekend(self):
        dt = datetime(2026, 2, 22, 10, 0, tzinfo=self.tz)  # 周日
        self.assertFalse(self.guard.is_market_open(dt))


if __name__ == "__main__":
    unittest.main()
