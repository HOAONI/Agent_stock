# -*- coding: utf-8 -*-
"""回测自然语言解读服务测试。"""

from __future__ import annotations

import unittest

from agent_stock.analyzer import GeminiAnalyzer, LlmRequestTimeoutError
from agent_stock.config import Config
from agent_stock.services.backtest_interpretation_service import (
    BACKTEST_INTERPRETATION_SYSTEM_PROMPT,
    BacktestInterpretationService,
)


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


class _RecordingAnalyzer(_FixedTextAnalyzer):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.calls: list[dict[str, object]] = []

    def generate_text(self, prompt, *, temperature=None, max_output_tokens=4096):
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        return self.text


class _TimeoutAnalyzer:
    def is_available(self) -> bool:
        return True

    def generate_text(self, prompt, *, temperature=None, max_output_tokens=4096):  # noqa: ARG002
        raise LlmRequestTimeoutError(provider="OpenAI-compatible", timeout_ms=120000)


class _FakeOpenAiCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Message:
            content = '{"items":[{"item_key":"strategy-1","verdict":"表现中等","summary":"样本期内收益与回撤匹配度尚可。"}]}'

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()


class _FakeOpenAiClient:
    def __init__(self, completions: _FakeOpenAiCompletions) -> None:
        self.chat = type("_Chat", (), {"completions": completions})()


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

    def test_scales_generation_budget_with_item_count(self):
        analyzer = _RecordingAnalyzer(
            """{"items":[{"item_key":"strategy-1","verdict":"表现中等","summary":"策略一表现稳定。"},{"item_key":"strategy-2","verdict":"样本不足","summary":"策略二样本不足。"}]}"""
        )
        service = BacktestInterpretationService(
            analyzer_factory=lambda runtime_llm: analyzer,  # noqa: ARG005
        )

        result = service.interpret(_sample_payload())

        self.assertEqual(len(analyzer.calls), 1)
        self.assertEqual(analyzer.calls[0]["temperature"], 0.2)
        self.assertEqual(analyzer.calls[0]["max_output_tokens"], 760)
        self.assertEqual(result["items"][0]["status"], "ready")
        self.assertEqual(result["items"][1]["status"], "ready")

    def test_timeout_errors_fall_back_to_failed_items(self):
        service = BacktestInterpretationService(
            analyzer_factory=lambda runtime_llm: _TimeoutAnalyzer(),  # noqa: ARG005
        )

        result = service.interpret(_sample_payload())

        self.assertTrue(all(item["status"] == "failed" for item in result["items"]))
        self.assertTrue(all(item["summary"] == "AI 解读生成失败，请稍后重试。" for item in result["items"]))
        self.assertTrue(
            all("[llm_request_timeout] OpenAI-compatible request timed out after 120000ms" in str(item["error_message"]) for item in result["items"])
        )

    def test_dedicated_backtest_prompt_is_sent_to_openai_client(self):
        completions = _FakeOpenAiCompletions()
        analyzer = GeminiAnalyzer(config=Config(), system_prompt=BACKTEST_INTERPRETATION_SYSTEM_PROMPT)
        analyzer._model = None
        analyzer._anthropic_client = None
        analyzer._openai_client = _FakeOpenAiClient(completions)
        analyzer._use_openai = True
        analyzer._current_model_name = "fake-model"

        text = analyzer.generate_text("回测输入", temperature=0.2, max_output_tokens=512)

        self.assertIn('"items"', text)
        self.assertEqual(len(completions.calls), 1)
        messages = completions.calls[0]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], BACKTEST_INTERPRETATION_SYSTEM_PROMPT)
        self.assertNotIn("决策仪表盘", str(messages[0]["content"]))
        self.assertEqual(messages[1]["content"], "回测输入")


if __name__ == "__main__":
    unittest.main()
