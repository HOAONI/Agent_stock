# -*- coding: utf-8 -*-
"""回测自然语言解读服务测试。"""

from __future__ import annotations

import unittest

from agent_stock.services.backtest_interpretation_service import BacktestInterpretationService


class _UnavailableAnalyzer:
    def is_available(self) -> bool:
        return False


class _FixedTextAnalyzer:
    def __init__(self, text: str) -> None:
        self.text = text

    def is_available(self) -> bool:
        return True

    def generate_text(self, prompt, *, temperature=None, max_output_tokens=4096):  # noqa: ARG002
        return self.text


def _sample_payload() -> dict:
    return {
        "items": [
            {
                "item_key": "strategy-1",
                "item_type": "strategy",
                "label": "Fast MA",
                "code": "600519",
                "requested_range": {"start_date": "2024-01-01", "end_date": "2024-12-31"},
                "effective_range": {"start_date": "2024-01-02", "end_date": "2024-12-31"},
                "metrics": {
                    "total_return_pct": 12.3,
                    "max_drawdown_pct": -8.1,
                    "sharpe_ratio": 1.25,
                    "win_rate_pct": 58.2,
                    "total_trades": 11,
                },
            },
            {
                "item_key": "strategy-2",
                "item_type": "strategy",
                "label": "RSI Swing",
                "code": "600519",
                "requested_range": {"start_date": "2024-01-01", "end_date": "2024-12-31"},
                "effective_range": {"start_date": "2024-01-02", "end_date": "2024-12-31"},
                "metrics": {
                    "total_return_pct": -2.4,
                    "max_drawdown_pct": -18.0,
                    "total_trades": 0,
                },
            },
        ]
    }


class BacktestInterpretationServiceTestCase(unittest.TestCase):
    def test_returns_unavailable_when_no_provider_is_configured(self):
        service = BacktestInterpretationService(
            analyzer_factory=lambda runtime_llm: _UnavailableAnalyzer(),  # noqa: ARG005
        )

        result = service.interpret(_sample_payload())

        self.assertEqual(
            result["items"],
            [
                {
                    "item_key": "strategy-1",
                    "status": "unavailable",
                    "verdict": None,
                    "summary": "AI 解读暂不可用，请先检查当前运行环境里的模型配置。",
                    "error_message": None,
                },
                {
                    "item_key": "strategy-2",
                    "status": "unavailable",
                    "verdict": None,
                    "summary": "AI 解读暂不可用，请先检查当前运行环境里的模型配置。",
                    "error_message": None,
                },
            ],
        )

    def test_parses_batch_response_and_marks_missing_items_failed(self):
        service = BacktestInterpretationService(
            analyzer_factory=lambda runtime_llm: _FixedTextAnalyzer(  # noqa: ARG005
                """{"items":[{"item_key":"strategy-1","verdict":"表现中等","summary":"该策略在样本区间内取得 12.3% 收益，最大回撤约 8.1%，整体回撤可控。夏普和胜率都不差，说明收益质量尚可，但仍然需要结合更多年份验证稳定性。"}]}"""
            ),
        )

        result = service.interpret(_sample_payload())

        self.assertEqual(result["items"][0]["status"], "ready")
        self.assertEqual(result["items"][0]["verdict"], "表现中等")
        self.assertIn("12.3%", result["items"][0]["summary"])

        self.assertEqual(result["items"][1]["status"], "failed")
        self.assertEqual(result["items"][1]["verdict"], None)
        self.assertEqual(result["items"][1]["error_message"], "missing_item_in_model_response")

    def test_invalid_json_response_falls_back_to_failed_items(self):
        service = BacktestInterpretationService(
            analyzer_factory=lambda runtime_llm: _FixedTextAnalyzer("not-json"),  # noqa: ARG005
        )

        result = service.interpret(_sample_payload())

        self.assertTrue(all(item["status"] == "failed" for item in result["items"]))
        self.assertTrue(all(item["summary"] == "AI 解读生成失败，请稍后重试。" for item in result["items"]))


if __name__ == "__main__":
    unittest.main()
