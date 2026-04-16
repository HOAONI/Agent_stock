# -*- coding: utf-8 -*-
"""Agent 问股监控服务测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any

from agent_stock.repositories.chat_repo import AgentChatRepository
from agent_stock.services.agent_chat_monitor_service import AgentChatMonitorService
from agent_stock.storage import DatabaseManager


def build_trace(
    *,
    stage: str,
    visit: int,
    summary: str,
    stock_code: str = "600519",
    state: str = "ready",
    duration_ms: int = 120,
) -> dict[str, Any]:
    return {
        "stock_code": stock_code,
        "stage": stage,
        "visit": visit,
        "state": state,
        "summary": summary,
        "duration_ms": duration_ms,
        "confidence": 0.82,
        "warnings": [],
        "input": {"stage": stage, "visit": visit},
        "output": {"status": state, "summary": summary},
        "error_message": None if state != "failed" else "stage_failed",
        "decision": {"summary": summary},
        "observations": [{"stage": stage, "visit": visit}],
        "fallback_chain": [],
        "next_action": "done",
        "llm_used": stage in {"signal", "risk", "execution"},
        "started_at": f"2026-04-06T09:3{visit}:00",
        "finished_at": f"2026-04-06T09:3{visit}:01",
    }


def build_stock_item(stock_code: str, stock_name: str, stage_summaries: dict[str, str]) -> dict[str, Any]:
    raw = {}
    for stage, summary in stage_summaries.items():
        raw[stage] = {
            "state": "ready",
            "duration_ms": 100,
            "input": {"stage": stage},
            "output": {"summary": summary},
            "decision": {"summary": summary},
        }
    return {
        "code": stock_code,
        "name": stock_name,
        "planner_trace": [{"stock_code": stock_code, "kind": "planner_step", "tool": "fetch_market_data"}],
        "condition_evaluations": [{"stock_code": stock_code, "passed": True, "reason": "ok"}],
        "candidate_order": {"code": stock_code, "action": "buy", "quantity": 100},
        "execution_result": {"status": "filled"},
        "raw": raw,
    }


def build_meta(
    *,
    stage_traces: list[dict[str, Any]],
    stock_items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "session_id": "ignored-in-meta",
        "content": "分析完成",
        "structured_result": {
            "intent": "analysis",
            "analysis": {
                "controller_plan": {
                    "goal": "分析股票并生成保守执行计划",
                    "stage_priority": ["data", "signal", "risk", "execution"],
                    "policy_snapshot": {"runtime_execution_mode": "paper"},
                },
                "stage_traces": stage_traces,
                "planner_trace": [{"stock_code": "600519", "kind": "planner_step", "tool": "fetch_market_data"}],
                "condition_evaluations": [{"stock_code": "600519", "passed": True, "reason": "ok"}],
                "stocks": stock_items,
            },
        },
        "candidate_orders": [],
        "execution_result": None,
        "status": "analysis_only",
    }


class AgentChatMonitorServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_chat_monitor.db")
        os.environ["DATABASE_URL"] = ""
        DatabaseManager.reset_instance()
        self.db = DatabaseManager.get_instance()
        self.repo = AgentChatRepository(self.db)
        self.service = AgentChatMonitorService(self.repo)

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        self.temp_dir.cleanup()

    def _persist_analysis_message(
        self,
        *,
        session_id: str,
        user_message: str,
        traces: list[dict[str, Any]],
        stock_code: str = "600519",
        stock_name: str = "贵州茅台",
    ) -> None:
        self.repo.ensure_session(owner_user_id=1, session_id=session_id, title=f"{stock_code} 问股", context={"stock_code": stock_code})
        self.repo.add_message(owner_user_id=1, session_id=session_id, role="user", content=user_message)
        self.repo.add_message(
            owner_user_id=1,
            session_id=session_id,
            role="assistant",
            content="分析完成",
            meta=build_meta(
                stage_traces=traces,
                stock_items=[build_stock_item(stock_code, stock_name, {trace["stage"]: trace["summary"] for trace in traces})],
            ),
        )

    def test_builds_persisted_snapshot_and_historical_counts(self) -> None:
        self._persist_analysis_message(
            session_id="session-history-1",
            user_message="分析 600519",
            traces=[
                build_trace(stage="data", visit=1, summary="数据准备完成"),
                build_trace(stage="signal", visit=1, summary="信号生成完成"),
            ],
        )
        self._persist_analysis_message(
            session_id="session-history-2",
            user_message="分析 300750",
            traces=[
                build_trace(stage="data", visit=1, summary="宁德时代数据完成", stock_code="300750"),
                build_trace(stage="signal", visit=1, summary="宁德时代信号完成", stock_code="300750"),
                build_trace(stage="risk", visit=1, summary="宁德时代风控完成", stock_code="300750"),
                build_trace(stage="execution", visit=1, summary="宁德时代执行完成", stock_code="300750"),
            ],
            stock_code="300750",
            stock_name="宁德时代",
        )

        snapshot = self.service.get_snapshot(1)

        self.assertEqual(snapshot["session"]["session_id"], "session-history-2")
        self.assertEqual(snapshot["session"]["user_message"], "分析 300750")
        card_by_code = {item["code"]: item for item in snapshot["agent_cards"]}
        self.assertEqual(card_by_code["data"]["total_calls"], 2)
        self.assertEqual(card_by_code["signal"]["total_calls"], 2)
        self.assertEqual(card_by_code["risk"]["total_calls"], 1)
        self.assertEqual(card_by_code["execution"]["total_calls"], 1)
        self.assertEqual(snapshot["execution_chain"][-1]["summary"], "宁德时代执行完成")

    def test_live_snapshot_transitions_without_double_counting_after_finalize(self) -> None:
        self._persist_analysis_message(
            session_id="session-baseline-1",
            user_message="分析 600519",
            traces=[
                build_trace(stage="data", visit=1, summary="历史数据完成"),
                build_trace(stage="signal", visit=1, summary="历史信号完成"),
            ],
        )

        self.repo.ensure_session(owner_user_id=1, session_id="session-live-1", title="300750 问股", context={"stock_code": "300750"})
        self.repo.add_message(owner_user_id=1, session_id="session-live-1", role="user", content="分析 300750")

        self.service.start_run(owner_user_id=1, session_id="session-live-1", title="300750 问股", user_message="分析 300750")
        self.service.record_event(
            owner_user_id=1,
            session_id="session-live-1",
            event_name="thinking",
            payload={"message": "正在理解你的问题"},
        )
        pre_stage_snapshot = self.service.get_snapshot(1)
        self.assertEqual(pre_stage_snapshot["session"]["live_status"], "running")

        self.service.record_event(
            owner_user_id=1,
            session_id="session-live-1",
            event_name="stage_start",
            payload={"stock_code": "300750", "stage": "data", "visit": 1, "started_at": "2026-04-06T09:30:00", "summary": "数据 Agent 运行中"},
        )
        running_snapshot = self.service.get_snapshot(1)
        data_card = next(item for item in running_snapshot["agent_cards"] if item["code"] == "data")
        self.assertEqual(data_card["status"], "running")
        self.assertEqual(data_card["total_calls"], 2)

        self.service.record_event(
            owner_user_id=1,
            session_id="session-live-1",
            event_name="stage_update",
            payload=build_trace(stage="data", visit=1, summary="实时数据完成", stock_code="300750"),
        )
        completed_snapshot = self.service.get_snapshot(1)
        data_card_after_update = next(item for item in completed_snapshot["agent_cards"] if item["code"] == "data")
        self.assertEqual(data_card_after_update["status"], "completed")
        self.assertEqual(data_card_after_update["total_calls"], 2)

        self.service.record_event(
            owner_user_id=1,
            session_id="session-live-1",
            event_name="error",
            payload={"message": "signal timeout"},
        )
        error_snapshot = self.service.get_snapshot(1)
        self.assertEqual(error_snapshot["session"]["live_status"], "error")

        self.repo.add_message(
            owner_user_id=1,
            session_id="session-live-1",
            role="assistant",
            content="分析完成",
            meta=build_meta(
                stage_traces=[
                    build_trace(stage="data", visit=1, summary="实时数据完成", stock_code="300750"),
                    build_trace(stage="signal", visit=1, summary="实时信号完成", stock_code="300750"),
                ],
                stock_items=[build_stock_item("300750", "宁德时代", {"data": "实时数据完成", "signal": "实时信号完成"})],
            ),
        )

        final_snapshot = self.service.finalize_run(1, "session-live-1")
        data_card_after_finalize = next(item for item in final_snapshot["agent_cards"] if item["code"] == "data")
        signal_card_after_finalize = next(item for item in final_snapshot["agent_cards"] if item["code"] == "signal")
        self.assertEqual(data_card_after_finalize["total_calls"], 2)
        self.assertEqual(signal_card_after_finalize["total_calls"], 2)

    def test_preserves_multiple_visits_for_same_stage(self) -> None:
        self._persist_analysis_message(
            session_id="session-replan-1",
            user_message="分析 600519",
            traces=[
                build_trace(stage="data", visit=1, summary="数据准备完成"),
                build_trace(stage="signal", visit=1, summary="第一次信号需要复核"),
                build_trace(stage="signal", visit=2, summary="第二次信号复核完成"),
                build_trace(stage="risk", visit=1, summary="风控完成"),
            ],
        )

        snapshot = self.service.get_snapshot(1)
        stock_detail = snapshot["stock_details"][0]
        signal_visits = [item for item in stock_detail["stage_visits"] if item["stage"] == "signal"]

        self.assertEqual(len(signal_visits), 2)
        self.assertEqual(signal_visits[0]["visit"], 1)
        self.assertEqual(signal_visits[1]["visit"], 2)
        self.assertEqual(signal_visits[1]["detail"]["summary"], "第二次信号复核完成")

    def test_start_run_marks_previous_live_run_as_superseded(self) -> None:
        first = self.service.start_run(
            owner_user_id=1,
            session_id="session-live-a",
            title="第一次问股",
            user_message="分析一下寒武纪",
        )
        second = self.service.start_run(
            owner_user_id=1,
            session_id="session-live-b",
            title="第二次问股",
            user_message="再分析一下寒武纪",
        )

        self.assertIsNone(first)
        assert second is not None
        self.assertEqual(second["session_id"], "session-live-a")
        self.assertEqual(second["user_message"], "分析一下寒武纪")

        snapshot = self.service.get_snapshot(1)
        self.assertEqual(snapshot["session"]["session_id"], "session-live-b")
        self.assertEqual(snapshot["session"]["live_status"], "running")
        self.assertEqual(snapshot["session"]["superseded_run"]["session_id"], "session-live-a")
        self.assertEqual(snapshot["session"]["superseded_run"]["interrupted_reason"], "superseded_by_new_run")


if __name__ == "__main__":
    unittest.main()
