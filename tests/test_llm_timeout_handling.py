# -*- coding: utf-8 -*-
"""LLM 超时传播回归测试。"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import date

from agent_stock.agents.contracts import AgentState, DataAgentOutput
from agent_stock.agents.signal_agent import SignalAgent
from agent_stock.services.agent_task_service import AgentTaskService
from agent_stock.storage import DatabaseManager
from agent_stock.analyzer import GeminiAnalyzer, LlmRequestTimeoutError
from agent_stock.config import Config


class _TimeoutAgentService:
    def run_once(
        self,
        stock_codes,
        *,
        account_name=None,
        request_id=None,
        write_reports=None,
        runtime_config=None,
    ):
        raise LlmRequestTimeoutError(provider="DeepSeek", timeout_ms=120000)


class LlmTimeoutHandlingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "llm_timeout.db")
        os.environ["DATABASE_URL"] = ""
        os.environ["AGENT_SERVICE_MODE"] = "false"
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["GEMINI_REQUEST_DELAY"] = "0"
        Config.reset_instance()
        DatabaseManager.reset_instance()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_analyzer_reraises_llm_timeout(self) -> None:
        analyzer = GeminiAnalyzer(config=Config.get_instance())
        analyzer.is_available = lambda: True  # type: ignore[method-assign]
        analyzer._format_prompt = lambda *_args, **_kwargs: "timeout-test-prompt"  # type: ignore[method-assign]
        analyzer._call_api_with_retry = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
            LlmRequestTimeoutError(provider="DeepSeek", timeout_ms=120000)
        )

        with self.assertRaises(LlmRequestTimeoutError):
            analyzer.analyze({"code": "600122", "stock_name": "Timeout Test"})

    def test_signal_agent_run_propagates_llm_timeout(self) -> None:
        def _resolver(_code: str):
            raise LlmRequestTimeoutError(provider="DeepSeek", timeout_ms=120000)

        agent = SignalAgent(ai_resolver=_resolver)
        payload = DataAgentOutput(
            code="600122",
            trade_date=date(2026, 3, 8),
            state=AgentState.READY,
            analysis_context={},
            realtime_quote={},
        )

        with self.assertRaises(LlmRequestTimeoutError):
            agent.run(payload)

    def test_async_task_marks_failed_on_llm_timeout(self) -> None:
        service = AgentTaskService(agent_service=_TimeoutAgentService())
        try:
            task = service.submit_task(stock_codes=["600122"], request_id="timeout-task-1")

            deadline = time.time() + 3
            latest = task
            while time.time() < deadline:
                latest = service.get_task(task["task_id"])
                if latest.get("status") in {"completed", "failed"}:
                    break
                time.sleep(0.05)

            self.assertEqual(latest.get("status"), "failed")
            self.assertIn("[llm_request_timeout]", str(latest.get("error_message") or ""))
        finally:
            service._executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
