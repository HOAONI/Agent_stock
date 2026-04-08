# -*- coding: utf-8 -*-
"""Agent 问股流式接口测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
import asyncio

from fastapi.testclient import TestClient

from agent_api.app import create_app
from agent_api.deps import get_agent_chat_service_dep
from agent_api.v1.endpoints.chat_internal import get_chat_monitor_stream
from agent_stock.config import Config
from agent_stock.services.agent_chat_service import AgentChatHandledError, reset_agent_chat_service
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
            await event_handler("message_start", {"role": "assistant", "format": "markdown"})
            await event_handler("message_delta", {"delta": "分析并"})
            await event_handler("message_delta", {"delta": "执行完成"})
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


class _HandledFailureStreamingChatService:
    async def handle_chat(self, payload: dict, event_handler=None):
        if event_handler is not None:
            await event_handler("thinking", {"message": "正在理解你的问题"})
        final_payload = {
            "session_id": payload.get("session_id") or "session-stream-timeout-1",
            "content": "本轮 Agent 问股执行失败：[llm_request_timeout] OpenAI-compatible request timed out after 120000ms",
            "structured_result": {"intent": "error", "message": "[llm_request_timeout] OpenAI-compatible request timed out after 120000ms"},
            "candidate_orders": [],
            "execution_result": None,
            "status": "blocked",
        }
        raise AgentChatHandledError("timeout", final_payload)


class _BlockedStreamingChatService:
    async def handle_chat(self, payload: dict, event_handler=None):
        if event_handler is not None:
            await event_handler("thinking", {"message": "正在理解你的问题"})
            await event_handler(
                "warning",
                {
                    "stage": "execution",
                    "message": "当前非交易时段，本轮未执行模拟盘订单，候选单已保留。下次可执行时间：2026-04-06T09:30:00+08:00。",
                    "stock_code": "600519",
                },
            )
        return {
            "session_id": payload.get("session_id") or "session-stream-blocked-1",
            "content": "当前处于非交易时段，本轮未执行模拟盘订单，候选单已保留，请在 2026-04-06T09:30:00+08:00 后再次确认。",
            "structured_result": {"intent": "order_followup_single"},
            "candidate_orders": [{"code": "600519", "action": "buy", "quantity": 100, "price": 1680.0}],
            "execution_result": {
                "status": "blocked",
                "reason": "outside_trading_session",
                "message": "当前处于非交易时段，本轮未执行模拟盘订单，候选单已保留，请在 2026-04-06T09:30:00+08:00 后再次确认。",
                "session_guard": {
                    "timezone": "Asia/Shanghai",
                    "sessions": ["09:30-11:30", "13:00-15:00"],
                    "next_open_at": "2026-04-06T09:30:00+08:00",
                },
            },
            "status": "blocked",
        }


class _MonitorChatService:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.queue.put_nowait(
            {
                "session": {
                    "session_id": "monitor-session-live-1",
                    "title": "最近一轮 Agent 协作",
                    "live_status": "running",
                },
                "agent_cards": [
                    {"code": "data", "title": "数据 Agent", "status": "running", "total_calls": 9},
                ],
                "execution_chain": [
                    {"node_id": "monitor-session-live-1:300750:data:1", "stock_code": "300750", "stage": "data", "visit": 1},
                ],
                "stock_details": [],
            }
        )

    def get_monitor_snapshot(self, owner_user_id: int) -> dict:
        return {
            "session": {
                "session_id": f"monitor-session-{owner_user_id}",
                "title": "最近一轮 Agent 协作",
                "live_status": "completed",
            },
            "agent_cards": [
                {"code": "data", "title": "数据 Agent", "status": "completed", "total_calls": 8},
                {"code": "signal", "title": "信号 Agent", "status": "completed", "total_calls": 5},
            ],
            "execution_chain": [],
            "stock_details": [],
        }

    def subscribe_monitor(self, _owner_user_id: int) -> asyncio.Queue[dict]:
        return self.queue

    def unsubscribe_monitor(self, _owner_user_id: int, _queue: asyncio.Queue[dict]) -> None:
        return None


class AgentApiChatStreamTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_api_chat_stream.db")
        os.environ["DATABASE_URL"] = ""
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
            ["thinking", "tool_start", "tool_done", "tool_start", "tool_done", "message_start", "message_delta", "message_delta", "done"],
        )
        self.assertIn('"tool": "run_multi_stock_analysis"', payload)
        self.assertIn('"tool": "batch_execute_candidate_orders"', payload)
        self.assertIn('"delta": "分析并"', payload)
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

    def test_chat_stream_returns_persisted_failure_as_done(self):
        self.app.dependency_overrides[get_agent_chat_service_dep] = lambda: _HandledFailureStreamingChatService()

        with self.client.stream(
            "POST",
            "/internal/v1/chat/stream",
            headers={"Authorization": "Bearer test-token"},
            json={"owner_user_id": 1, "message": "分析 600519"},
        ) as response:
            self.assertEqual(response.status_code, 200)
            payload = "".join(response.iter_text())

        self.assertEqual(_extract_event_names(payload), ["thinking", "done"])
        self.assertIn('"status": "blocked"', payload)
        self.assertIn('"session_id": "session-stream-timeout-1"', payload)

    def test_chat_stream_emits_warning_before_blocked_done(self):
        self.app.dependency_overrides[get_agent_chat_service_dep] = lambda: _BlockedStreamingChatService()

        with self.client.stream(
            "POST",
            "/internal/v1/chat/stream",
            headers={"Authorization": "Bearer test-token"},
            json={"owner_user_id": 1, "message": "去下单吧"},
        ) as response:
            self.assertEqual(response.status_code, 200)
            payload = "".join(response.iter_text())

        self.assertEqual(_extract_event_names(payload), ["thinking", "warning", "done"])
        self.assertIn('"stage": "execution"', payload)
        self.assertIn('"status": "blocked"', payload)
        self.assertIn('"session_id": "session-stream-blocked-1"', payload)

    def test_chat_monitor_returns_latest_snapshot(self):
        self.app.dependency_overrides[get_agent_chat_service_dep] = lambda: _MonitorChatService()

        response = self.client.get(
            "/internal/v1/chat/monitor",
            headers={"Authorization": "Bearer test-token"},
            params={"owner_user_id": 1},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["session"]["session_id"], "monitor-session-1")
        self.assertEqual(payload["agent_cards"][0]["code"], "data")
        self.assertEqual(payload["agent_cards"][0]["total_calls"], 8)

    def test_chat_monitor_stream_emits_connected_and_snapshot(self):
        service = _MonitorChatService()
        response = asyncio.run(get_chat_monitor_stream(owner_user_id=1, chat_service=service))

        self.assertEqual(response.media_type, "text/event-stream")
        self.assertEqual(response.headers["Cache-Control"], "no-cache")
        self.assertEqual(response.headers["Connection"], "keep-alive")

        async def collect_chunks() -> str:
            chunks: list[str] = []
            iterator = response.body_iterator
            try:
                for _ in range(3):
                    chunk = await iterator.__anext__()
                    chunks.append(chunk.decode("utf-8"))
            finally:
                if hasattr(iterator, "aclose"):
                    await iterator.aclose()
            return "".join(chunks)

        payload = asyncio.run(collect_chunks())
        self.assertIn("event: connected", payload)
        self.assertIn("event: snapshot", payload)
        self.assertIn('"session_id": "monitor-session-1"', payload)
        self.assertIn('"session_id": "monitor-session-live-1"', payload)


if __name__ == "__main__":
    unittest.main()
