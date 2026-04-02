# -*- coding: utf-8 -*-
"""Agent 问股聊天服务测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date, datetime

from agent_stock.agents.contracts import (
    AgentRunResult,
    AgentState,
    DataAgentOutput,
    ExecutionAgentOutput,
    RiskAgentOutput,
    SignalAgentOutput,
    StockAgentResult,
)
from agent_stock.config import Config
from agent_stock.repositories.chat_repo import AgentChatRepository
from agent_stock.services.agent_chat_service import AgentChatService
from agent_stock.storage import DatabaseManager

STOCK_NAMES = {
    "600519": "贵州茅台",
    "000001": "平安银行",
    "300750": "宁德时代",
}

STOCK_PRICES = {
    "600519": 1680.0,
    "000001": 12.36,
    "300750": 198.52,
}

STOCK_SCORES = {
    "600519": 78,
    "000001": 66,
    "300750": 92,
}


def build_runtime_config() -> dict:
    return {
        "account": {"account_name": "u1", "initial_cash": 100000},
        "strategy": {
            "position_max_pct": 30,
            "stop_loss_pct": 8,
            "take_profit_pct": 15,
        },
    }


class FakeAnalyzer:
    def is_available(self) -> bool:
        return False


class FakeStructuredAnalyzer:
    def is_available(self) -> bool:
        return True

    def generate_text(self, *_args, **_kwargs) -> str:
        return """```json
{
  "stock_name": "贵州茅台",
  "sentiment_score": 78,
  "operation_advice": "买入"
}
```"""


class FakePlannerNoiseAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def generate_text(self, *_args, **_kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            return "this-is-not-json"
        return "## 自主决策结论\n我已经根据分析结果完成判断并提交模拟盘候选单。"


class FakeBackendClient:
    def __init__(self) -> None:
        self.placed_orders: list[dict] = []

    async def get_runtime_account_context(self, *, owner_user_id: int, refresh: bool = True):
        return {
            "simulation_account": {
                "is_bound": True,
                "is_verified": True,
                "broker_account_id": 7,
            },
            "runtime_context": {
                "broker_account_id": 7,
                "provider_code": "backtrader_local",
                "provider_name": "Backtrader Local Sim",
                "account_uid": "bt-u1",
                "account_display_name": "u1",
                "snapshot_at": "2026-04-02T09:30:00",
                "data_source": "cache",
                "summary": {
                    "cash": 100000,
                    "initial_capital": 100000,
                    "total_market_value": 0,
                    "total_asset": 100000,
                },
                "positions": [],
            },
        }

    async def get_analysis_history(self, *, owner_user_id: int, stock_codes: list[str] | None = None, limit: int = 5):
        return {"total": 0, "items": []}

    async def get_backtest_summary(self, *, owner_user_id: int, stock_codes: list[str] | None = None, limit: int = 6):
        return {"total": 0, "items": []}

    async def place_simulated_order(self, *, owner_user_id: int, session_id: str, candidate_order: dict):
        self.placed_orders.append(dict(candidate_order))
        return {
            "status": "filled",
            "candidate_order": candidate_order,
            "order": {
                "provider_status": "filled",
                "stock_code": candidate_order.get("code"),
                "direction": candidate_order.get("action"),
                "quantity": candidate_order.get("quantity"),
            },
        }


class FakeAgentService:
    def __init__(
        self,
        *,
        no_candidate_codes: set[str] | None = None,
        sentiment_scores: dict[str, int] | None = None,
    ) -> None:
        self.no_candidate_codes = no_candidate_codes or set()
        self.sentiment_scores = sentiment_scores or STOCK_SCORES

    def run_once(self, stock_codes, **_kwargs):
        today = date(2026, 4, 2)
        results: list[StockAgentResult] = []
        positions: list[dict] = []
        for index, code in enumerate(stock_codes):
            price = STOCK_PRICES.get(code, 10.0 + index)
            name = STOCK_NAMES.get(code, code)
            sentiment = int(self.sentiment_scores.get(code, 60 - index))
            has_candidate = code not in self.no_candidate_codes
            action = "buy" if has_candidate else "hold"
            traded_qty = 100 * (index + 1) if has_candidate else 0
            target_weight = 0.2 + 0.1 * index if has_candidate else 0.0
            result = StockAgentResult(
                code=code,
                data=DataAgentOutput(
                    code=code,
                    trade_date=today,
                    state=AgentState.READY,
                    analysis_context={"today": {"close": price, "pct_chg": 1.23 + index}, "stock_name": name},
                    realtime_quote={"name": name, "price": price, "change_pct": 1.23 + index},
                ),
                signal=SignalAgentOutput(
                    code=code,
                    trade_date=today,
                    state=AgentState.READY,
                    operation_advice="买入" if has_candidate else "观望",
                    sentiment_score=sentiment,
                    trend_signal="BUY" if has_candidate else "HOLD",
                ),
                risk=RiskAgentOutput(
                    code=code,
                    trade_date=today,
                    state=AgentState.READY,
                    target_weight=target_weight,
                    target_notional=30000 + index * 5000 if has_candidate else 0,
                    current_price=price,
                    risk_flags=["position_cap_ok"] if has_candidate else ["wait_for_signal"],
                ),
                execution=ExecutionAgentOutput(
                    code=code,
                    trade_date=today,
                    state=AgentState.READY,
                    action=action,
                    reason="intent_generated" if has_candidate else "wait_signal",
                    traded_qty=traded_qty,
                    target_qty=traded_qty,
                    fill_price=round(price * 1.0005, 2) if has_candidate else None,
                ),
            )
            results.append(result)
            if has_candidate:
                positions.append({"code": code, "quantity": traded_qty})

        return AgentRunResult(
            run_id="run-chat-1",
            mode="once",
            started_at=datetime(2026, 4, 2, 9, 30, 0),
            ended_at=datetime(2026, 4, 2, 9, 30, 2),
            trade_date=today,
            results=results,
            account_snapshot={"cash": 70000, "positions": positions},
        )


class AgentChatServiceTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_chat_service.db")
        os.environ["DATABASE_URL"] = ""
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        Config.reset_instance()
        DatabaseManager.reset_instance()
        self.db = DatabaseManager.get_instance()
        self.repo = AgentChatRepository(self.db)
        self.backend_client = FakeBackendClient()
        self.agent_service = FakeAgentService()
        self.service = AgentChatService(
            config=Config.get_instance(),
            db_manager=self.db,
            chat_repo=self.repo,
            agent_service=self.agent_service,
            backend_client=self.backend_client,
            analyzer=FakeAnalyzer(),
        )

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        os.environ.pop("DATABASE_URL", None)
        self.temp_dir.cleanup()

    async def test_analysis_chat_creates_candidate_order(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(len(payload["candidate_orders"]), 1)
        self.assertEqual(payload["candidate_orders"][0]["code"], "600519")

        sessions = self.service.list_sessions(1, limit=10)
        self.assertEqual(sessions["total"], 1)
        detail = self.service.get_session_detail(1, payload["session_id"])
        assert detail is not None
        self.assertEqual(len(detail["messages"]), 2)

    async def test_generic_viewing_phrase_with_code_still_triggers_analysis(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我看下 300750",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "analysis")
        self.assertEqual(payload["candidate_orders"][0]["code"], "300750")

    async def test_account_request_with_generic_viewing_phrase_uses_account_intent(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我看下账户情况",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "account")
        self.assertIn("当前模拟盘现金", payload["content"])

    async def test_history_request_with_generic_viewing_phrase_uses_history_intent(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我看下最近分析记录",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "history")

    async def test_follow_up_order_executes_single_candidate(self):
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "runtime_config": build_runtime_config(),
            }
        )

        second = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "去下单吧",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(second["status"], "simulation_order_filled")
        execution = second["execution_result"]
        assert execution is not None
        self.assertEqual(execution["candidate_order"]["code"], "600519")
        self.assertEqual(execution["status"], "filled")
        self.assertEqual(len(self.backend_client.placed_orders), 1)

    async def test_composite_request_analyzes_then_executes_single_candidate(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 300750 行情，并根据结果决定是否去下单",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "simulation_order_filled")
        self.assertEqual(payload["structured_result"]["intent"], "analysis_then_execute")
        self.assertEqual(payload["candidate_orders"][0]["code"], "300750")
        self.assertEqual(payload["execution_result"]["mode"], "batch")
        self.assertEqual(payload["execution_result"]["executed_count"], 1)
        self.assertTrue(payload["structured_result"]["autonomous_execution"]["executed"])
        self.assertEqual(len(self.backend_client.placed_orders), 1)

    async def test_multi_stock_composite_request_executes_all_candidate_orders(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 和 000001，并根据结果决定是否去下单",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "simulation_order_filled")
        self.assertEqual(payload["structured_result"]["intent"], "analysis_then_execute")
        self.assertEqual(len(payload["candidate_orders"]), 2)
        self.assertEqual(payload["execution_result"]["mode"], "batch")
        self.assertEqual(payload["execution_result"]["executed_count"], 2)
        self.assertEqual(len(self.backend_client.placed_orders), 2)

    async def test_no_candidate_orders_stays_analysis_only_for_autonomous_request(self):
        service = AgentChatService(
            config=Config.get_instance(),
            db_manager=self.db,
            chat_repo=self.repo,
            agent_service=FakeAgentService(no_candidate_codes={"300750"}),
            backend_client=self.backend_client,
            analyzer=FakeAnalyzer(),
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 300750 行情，并根据结果决定是否去下单",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["candidate_orders"], [])
        self.assertIsNone(payload["execution_result"])
        self.assertEqual(payload["structured_result"]["intent"], "analysis_then_execute")
        self.assertEqual(payload["structured_result"]["autonomous_execution"]["reason"], "no_candidate_orders")
        self.assertIn("暂时不需要下模拟盘单", payload["content"])

    async def test_follow_up_order_executes_all_candidates_from_latest_round(self):
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 和 000001",
                "runtime_config": build_runtime_config(),
            }
        )

        second = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "去下单吧",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(second["status"], "simulation_order_filled")
        self.assertEqual(second["structured_result"]["intent"], "order_followup_all")
        self.assertEqual(second["execution_result"]["mode"], "batch")
        self.assertEqual(second["execution_result"]["executed_count"], 2)
        self.assertEqual(len(self.backend_client.placed_orders), 2)

    async def test_follow_up_specific_code_executes_single_candidate(self):
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 和 300750",
                "runtime_config": build_runtime_config(),
            }
        )

        second = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "下 300750 的单",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(second["status"], "simulation_order_filled")
        self.assertEqual(second["structured_result"]["intent"], "order_followup_single")
        self.assertEqual(second["execution_result"]["candidate_order"]["code"], "300750")
        self.assertEqual(len(self.backend_client.placed_orders), 1)

    async def test_follow_up_best_order_picks_highest_ranked_candidate(self):
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 和 300750",
                "runtime_config": build_runtime_config(),
            }
        )

        second = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "下最看好的那笔",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(second["status"], "simulation_order_filled")
        self.assertEqual(second["execution_result"]["candidate_order"]["code"], "300750")

    async def test_invalid_planner_output_falls_back_to_rule_plan(self):
        analyzer = FakePlannerNoiseAnalyzer()
        service = AgentChatService(
            config=Config.get_instance(),
            db_manager=self.db,
            chat_repo=self.repo,
            agent_service=self.agent_service,
            backend_client=self.backend_client,
            analyzer=analyzer,
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 300750 行情，并根据结果决定是否去下单",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "simulation_order_filled")
        self.assertEqual(payload["structured_result"]["intent"], "analysis_then_execute")
        self.assertGreaterEqual(analyzer.calls, 2)
        self.assertIn("自主决策结论", payload["content"])

    async def test_structured_llm_output_falls_back_to_natural_language_template(self):
        service = AgentChatService(
            config=Config.get_instance(),
            db_manager=self.db,
            chat_repo=self.repo,
            agent_service=self.agent_service,
            backend_client=self.backend_client,
            analyzer=FakeStructuredAnalyzer(),
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertNotIn("```", payload["content"])
        self.assertNotIn('"stock_name"', payload["content"])
        self.assertIn("综合判断", payload["content"])
        self.assertIn("600519", payload["content"])

    async def test_composite_request_emits_analysis_and_batch_execute_events(self):
        events: list[tuple[str, str | None]] = []

        async def handler(event_name: str, payload: dict):
            events.append((event_name, payload.get("tool")))

        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 300750 行情，并根据结果决定是否去下单",
                "runtime_config": build_runtime_config(),
            },
            event_handler=handler,
        )

        self.assertEqual(payload["status"], "simulation_order_filled")
        self.assertEqual(
            [item[0] for item in events],
            ["thinking", "tool_start", "tool_done", "tool_start", "tool_done", "tool_start", "tool_done"],
        )
        self.assertEqual(events[1][1], "get_runtime_account_context")
        self.assertEqual(events[3][1], "run_multi_stock_analysis")
        self.assertEqual(events[5][1], "batch_execute_candidate_orders")


if __name__ == "__main__":
    unittest.main()
