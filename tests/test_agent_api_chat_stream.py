# -*- coding: utf-8 -*-
"""Agent 问股流式接口测试。"""

from __future__ import annotations

import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from agent_api.app import create_app
from agent_api.deps import get_agent_chat_service_dep
from agent_stock.config import Config
from agent_stock.services.agent_chat_service import reset_agent_chat_service
from agent_stock.services.agent_task_service import reset_agent_task_service
from agent_stock.storage import DatabaseManager


def _extract_event_names(payload: str) -> list[str]:
    names: list[str] = []
    for block in payload.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("event:"):
                names.append(line.split(":", 1)[1].strip())
                break
    return names


class _StreamingChatService:
    async def handle_chat(self, payload: dict, event_handler=None):
        if event_handler is not None:
            await event_handler("thinking", {"message": "正在理解你的问题"})
            await event_handler("tool_start", {"tool": "run_multi_stock_analysis", "summary": "开始串行分析 300750"})
            await event_handler("tool_done", {"summary": "已完成 1 只股票分析，生成 1 笔候选订单"})
            await event_handler("tool_start", {"tool": "batch_execute_candidate_orders", "summary": "根据分析结果自动执行 1 笔组合候选单"})
            await event_handler("tool_done", {"summary": "已完成候选订单提交，成功 1 笔，失败 0 笔"})
        return {
            "session_id": payload.get("session_id") or "session-stream-1",
            "content": "分析并执行完成",
            "structured_result": {"intent": "analysis_then_execute"},
            "candidate_orders": [{"code": "300750", "action": "buy", "quantity": 100, "price": 198.52}],
            "execution_result": {"mode": "batch", "executed_count": 1, "status": "filled"},
            "status": "simulation_order_filled",
        }


class _FailingStreamingChatService:
    async def handle_chat(self, _payload: dict, event_handler=None):
        if event_handler is not None:
            await event_handler("thinking", {"message": "正在理解你的问题"})
        raise RuntimeError("LLM timeout")


class AgentApiChatStreamTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_api_chat_stream.db")
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        os.environ["AGENT_SERVICE_MODE"] = "false"

        Config.reset_instance()
        DatabaseManager.reset_instance()
        reset_agent_task_service()
        reset_agent_chat_service()

        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.app.dependency_overrides.clear()
        reset_agent_task_service()
        reset_agent_chat_service()
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_chat_stream_emits_analysis_and_batch_execute_done_in_order(self):
        self.app.dependency_overrides[get_agent_chat_service_dep] = lambda: _StreamingChatService()

        with self.client.stream(
            "POST",
            "/internal/v1/chat/stream",
            headers={"Authorization": "Bearer test-token"},
            json={"owner_user_id": 1, "message": "帮我分析一下今天的 300750 行情，并根据结果决定是否去下单"},
        ) as response:
            self.assertEqual(response.status_code, 200)
            payload = "".join(response.iter_text())

        self.assertEqual(
            _extract_event_names(payload),
            ["thinking", "tool_start", "tool_done", "tool_start", "tool_done", "done"],
        )
        self.assertIn('"tool": "run_multi_stock_analysis"', payload)
        self.assertIn('"tool": "batch_execute_candidate_orders"', payload)
        self.assertIn('"session_id": "session-stream-1"', payload)

    def test_chat_stream_emits_error_as_terminal_event(self):
        self.app.dependency_overrides[get_agent_chat_service_dep] = lambda: _FailingStreamingChatService()

        with self.client.stream(
            "POST",
            "/internal/v1/chat/stream",
            headers={"Authorization": "Bearer test-token"},
            json={"owner_user_id": 1, "message": "分析 600519"},
        ) as response:
            self.assertEqual(response.status_code, 200)
            payload = "".join(response.iter_text())

        self.assertEqual(_extract_event_names(payload), ["thinking", "error"])
        self.assertIn('"message": "Agent 问股服务异常，请稍后重试。"', payload)


if __name__ == "__main__":
    unittest.main()
