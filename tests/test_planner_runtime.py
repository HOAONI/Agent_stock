# -*- coding: utf-8 -*-
"""Planner runtime condition compiler tests."""

from __future__ import annotations

import unittest

from agent_stock.agents.planner_runtime import build_request_intent, compile_message_conditions


class PlannerRuntimeConditionTestCase(unittest.TestCase):
    def test_build_request_intent_prefers_structured_constraints_without_regex_fallback(self):
        intent = build_request_intent(
            stock_codes=["600519"],
            planning_context={
                "message": "分析 600519，如果风险低就买100股",
                "constraints": [],
            },
        )

        self.assertEqual(intent.conditions, [])
        self.assertEqual(intent.unsupported_conditions, [])
        self.assertEqual(intent.user_message, "分析 600519，如果风险低就买100股")

    def test_compile_supported_trade_conditions(self):
        supported, unsupported = compile_message_conditions(
            "分析 600519，风险低的话低于10元帮我买100股，并且至少保留5000现金，单票仓位不超过30%"
        )

        self.assertEqual(len(unsupported), 0)
        condition_types = {item.type for item in supported}
        self.assertIn("risk_gate", condition_types)
        self.assertIn("price_gate", condition_types)
        self.assertIn("min_remaining_cash", condition_types)
        self.assertIn("max_single_position_pct", condition_types)

    def test_compile_unsupported_free_form_condition(self):
        supported, unsupported = compile_message_conditions("分析 600519，如果夏普大于1.2就买")

        self.assertEqual(len(supported), 0)
        self.assertEqual(len(unsupported), 1)
        self.assertEqual(unsupported[0].type, "unsupported_condition")


if __name__ == "__main__":
    unittest.main()
