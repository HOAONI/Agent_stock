# -*- coding: utf-8 -*-
"""显式主控 Agent 的动态路径测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date
from types import SimpleNamespace

from agent_stock.agents.controller_agent import ControllerAgent, ControllerContext
from agent_stock.agents.contracts import AgentState, DataAgentOutput, ExecutionAgentOutput, RiskAgentOutput, SignalAgentOutput
from agent_stock.config import Config
from agent_stock.storage import DatabaseManager


class _UnavailableAnalyzer:
    def is_available(self) -> bool:
        return False


class _SequencedDataAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, code: str, *, runtime_config=None) -> DataAgentOutput:
        self.calls += 1
        return DataAgentOutput(
            code=code,
            trade_date=date(2026, 4, 3),
            state=AgentState.READY,
            analysis_context={"today": {"close": 10.0}, "raw_data": [{"date": "2026-04-02", "close": 9.8}]},
            realtime_quote={"price": 10.0},
            data_source="sina" if self.calls > 1 else "tencent",
            decision={"summary": f"data pass {self.calls}"},
            next_action="signal",
        )


class _SequencedSignalAgent:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, data_output: DataAgentOutput, *, runtime_config=None) -> SignalAgentOutput:
        self.calls += 1
        if self.calls == 1:
            return SignalAgentOutput(
                code=data_output.code,
                trade_date=data_output.trade_date,
                operation_advice="观望",
                sentiment_score=48,
                decision={"summary": "信号不完整，申请补数据"},
                warnings=["need_more_data"],
                next_action="data",
            )
        return SignalAgentOutput(
            code=data_output.code,
            trade_date=data_output.trade_date,
            operation_advice="买入",
            sentiment_score=72,
            decision={"summary": "信号准备完成"},
            next_action="risk",
        )


class _SequencedRiskAgent:
    def run(self, **kwargs) -> RiskAgentOutput:
        return RiskAgentOutput(
            code=kwargs["code"],
            trade_date=kwargs["trade_date"],
            target_weight=0.0,
            target_notional=0.0,
            current_price=kwargs["current_price"],
            risk_flags=["position_full"],
            warnings=["position_full"],
            decision={"summary": "仓位已满，跳过执行"},
            next_action="skip_execution",
        )


class _UnusedExecutionAgent:
    def run(self, **kwargs) -> ExecutionAgentOutput:  # pragma: no cover - should not be reached
        return ExecutionAgentOutput(
            code=kwargs["code"],
            trade_date=kwargs["trade_date"],
            action="buy",
            reason="should_not_run",
        )


class _PreparedExecutionAgent:
    def prepare_order(self, **kwargs) -> ExecutionAgentOutput:
        return ExecutionAgentOutput(
            code=kwargs["code"],
            trade_date=kwargs["trade_date"],
            state=AgentState.READY,
            action="buy",
            reason="intent_generated",
            traded_qty=200,
            target_qty=200,
            cash_before=100000.0,
            cash_after=98000.0,
            position_before=0,
            position_after=200,
            account_snapshot={
                "name": "paper-test",
                "cash": 98000.0,
                "total_market_value": 2000.0,
                "total_asset": 100000.0,
                "positions": [{"code": kwargs["code"], "quantity": 200, "market_value": 2000.0}],
            },
            proposed_order={
                "code": kwargs["code"],
                "action": "buy",
                "quantity": 200,
                "target_qty": 200,
                "price": 10.0,
            },
            proposal_state="proposed",
            proposal_reason="intent_generated",
        )


class ControllerAgentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "controller_agent.db")
        os.environ["DATABASE_URL"] = ""
        Config.reset_instance()
        DatabaseManager.reset_instance()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        self.temp_dir.cleanup()

    def test_controller_agent_can_revisit_data_then_skip_execution(self):
        controller = ControllerAgent(config=Config.get_instance(), analyzer=_UnavailableAnalyzer())
        data_agent = _SequencedDataAgent()
        signal_agent = _SequencedSignalAgent()
        risk_agent = _SequencedRiskAgent()
        execution_agent = _UnusedExecutionAgent()
        stage_events: list[dict] = []

        context = ControllerContext(
            stock_codes=["600519"],
            account_name="paper-test",
            initial_cash=100000.0,
            planning_context={"message": "分析 600519 并在必要时回补数据"},
        )
        plan = controller.build_plan(context=context)

        result, _snapshot, traces, warnings = controller.run_stock(
            code="600519",
            trade_date=date(2026, 4, 3),
            current_account_snapshot={
                "name": "paper-test",
                "cash": 100000.0,
                "total_market_value": 90000.0,
                "total_asset": 100000.0,
                "positions": [{"code": "000001", "quantity": 1000, "market_value": 90000.0}],
            },
            context=context,
            controller_plan=plan,
            data_agent=data_agent,
            signal_agent=signal_agent,
            risk_agent=risk_agent,
            execution_agent=execution_agent,
            stage_observer=stage_events.append,
        )

        self.assertEqual([item["stage"] for item in traces], ["data", "signal", "data", "signal", "risk"])
        self.assertEqual(result.execution.state.value, "skipped")
        self.assertTrue(any(item["event"] == "stage_update" for item in stage_events))
        self.assertTrue(any(item["event"] == "warning" for item in stage_events))
        self.assertTrue(any(item["message"] == "position_full" for item in warnings))

    def test_controller_agent_submits_paper_order_when_risk_low_condition_passes(self):
        controller = ControllerAgent(config=Config.get_instance(), analyzer=_UnavailableAnalyzer())
        submitted_orders: list[dict] = []

        def submitter(candidate_order: dict) -> dict:
            submitted_orders.append(dict(candidate_order))
            return {
                "status": "filled",
                "order": {
                    "provider_status": "filled",
                    "stock_code": candidate_order.get("code"),
                    "direction": candidate_order.get("action"),
                    "quantity": candidate_order.get("quantity"),
                },
            }

        class _LowRiskAgent:
            def run(self, **kwargs) -> RiskAgentOutput:
                return RiskAgentOutput(
                    code=kwargs["code"],
                    trade_date=kwargs["trade_date"],
                    state=AgentState.READY,
                    target_weight=0.2,
                    target_notional=2000.0,
                    current_price=kwargs["current_price"],
                    next_action="execution",
                    risk_level="low",
                    execution_allowed=True,
                    hard_blocks=[],
                    soft_flags=[],
                    status="ready",
                )

        context = ControllerContext(
            stock_codes=["600519"],
            account_name="paper-test",
            initial_cash=100000.0,
            runtime_config=SimpleNamespace(execution=SimpleNamespace(mode="paper"), strategy=None, llm=None),
            planning_context={
                "message": "分析 600519，风险低的话帮我买100股",
                "autonomous_execution_authorized": True,
                "requested_order_side": "buy",
                "requested_quantity": 100,
                "conditions": [{"type": "risk_gate", "value": "risk_low", "label": "risk_low"}],
            },
            paper_order_submitter=submitter,
        )
        plan = controller.build_plan(context=context)

        result, _snapshot, _traces, _warnings = controller.run_stock(
            code="600519",
            trade_date=date(2026, 4, 3),
            current_account_snapshot={
                "name": "paper-test",
                "cash": 100000.0,
                "total_market_value": 0.0,
                "total_asset": 100000.0,
                "positions": [],
            },
            context=context,
            controller_plan=plan,
            data_agent=_SequencedDataAgent(),
            signal_agent=_SequencedSignalAgent(),
            risk_agent=_LowRiskAgent(),
            execution_agent=_PreparedExecutionAgent(),
        )

        self.assertEqual(result.termination_reason, "execution_completed")
        self.assertIsNotNone(result.execution_result)
        assert result.execution_result is not None
        self.assertEqual(result.execution_result["status"], "filled")
        self.assertEqual(submitted_orders[0]["quantity"], 100)
        self.assertTrue(all(item["passed"] for item in result.condition_evaluations))

    def test_controller_agent_never_auto_submits_in_broker_mode(self):
        controller = ControllerAgent(config=Config.get_instance(), analyzer=_UnavailableAnalyzer())
        submitted_orders: list[dict] = []

        def submitter(candidate_order: dict) -> dict:
            submitted_orders.append(dict(candidate_order))
            return {"status": "filled"}

        class _LowRiskAgent:
            def run(self, **kwargs) -> RiskAgentOutput:
                return RiskAgentOutput(
                    code=kwargs["code"],
                    trade_date=kwargs["trade_date"],
                    state=AgentState.READY,
                    target_weight=0.2,
                    target_notional=2000.0,
                    current_price=kwargs["current_price"],
                    next_action="execution",
                    risk_level="low",
                    execution_allowed=True,
                    hard_blocks=[],
                    soft_flags=[],
                    status="ready",
                )

        context = ControllerContext(
            stock_codes=["600519"],
            account_name="paper-test",
            initial_cash=100000.0,
            runtime_config=SimpleNamespace(execution=SimpleNamespace(mode="broker"), strategy=None, llm=None),
            planning_context={
                "message": "分析 600519，风险低的话帮我买100股",
                "autonomous_execution_authorized": True,
                "requested_order_side": "buy",
                "requested_quantity": 100,
                "conditions": [{"type": "risk_gate", "value": "risk_low", "label": "risk_low"}],
            },
            paper_order_submitter=submitter,
        )
        plan = controller.build_plan(context=context)

        result, _snapshot, _traces, _warnings = controller.run_stock(
            code="600519",
            trade_date=date(2026, 4, 3),
            current_account_snapshot={
                "name": "paper-test",
                "cash": 100000.0,
                "total_market_value": 0.0,
                "total_asset": 100000.0,
                "positions": [],
            },
            context=context,
            controller_plan=plan,
            data_agent=_SequencedDataAgent(),
            signal_agent=_SequencedSignalAgent(),
            risk_agent=_LowRiskAgent(),
            execution_agent=_PreparedExecutionAgent(),
        )

        self.assertIn(result.termination_reason, {"paper_submit_disabled", "planner_finished"})
        self.assertEqual(submitted_orders, [])
        self.assertIsNone(result.execution_result)

    def test_controller_agent_blocks_execution_when_system_state_marks_execution_unavailable(self):
        controller = ControllerAgent(config=Config.get_instance(), analyzer=_UnavailableAnalyzer())
        submitted_orders: list[dict] = []

        def submitter(candidate_order: dict) -> dict:
            submitted_orders.append(dict(candidate_order))
            return {"status": "filled"}

        class _LowRiskAgent:
            def run(self, **kwargs) -> RiskAgentOutput:
                return RiskAgentOutput(
                    code=kwargs["code"],
                    trade_date=kwargs["trade_date"],
                    state=AgentState.READY,
                    target_weight=0.2,
                    target_notional=2000.0,
                    current_price=kwargs["current_price"],
                    next_action="execution",
                    risk_level="low",
                    execution_allowed=True,
                    hard_blocks=[],
                    soft_flags=[],
                    status="ready",
                )

        context = ControllerContext(
            stock_codes=["600519"],
            account_name="paper-test",
            initial_cash=100000.0,
            runtime_config=SimpleNamespace(execution=SimpleNamespace(mode="paper"), strategy=None, llm=None),
            planning_context={
                "message": "分析 600519，如果风险低就买100股",
                "autonomous_execution_authorized": True,
                "requested_order_side": "buy",
                "requested_quantity": 100,
                "constraints": [{"type": "risk_gate", "value": "risk_low", "label": "risk_low"}],
                "loaded_context": {
                    "system_state": {
                        "execution": {"available": False, "busy": False, "degraded_reason": "agent_down"},
                    },
                },
            },
            paper_order_submitter=submitter,
        )
        plan = controller.build_plan(context=context)

        result, _snapshot, _traces, _warnings = controller.run_stock(
            code="600519",
            trade_date=date(2026, 4, 3),
            current_account_snapshot={
                "name": "paper-test",
                "cash": 100000.0,
                "total_market_value": 0.0,
                "total_asset": 100000.0,
                "positions": [],
            },
            context=context,
            controller_plan=plan,
            data_agent=_SequencedDataAgent(),
            signal_agent=_SequencedSignalAgent(),
            risk_agent=_LowRiskAgent(),
            execution_agent=_PreparedExecutionAgent(),
        )

        self.assertEqual(submitted_orders, [])
        self.assertIsNone(result.execution_result)
        self.assertNotEqual(result.termination_reason, "execution_completed")


if __name__ == "__main__":
    unittest.main()
