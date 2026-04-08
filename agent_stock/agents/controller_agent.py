# -*- coding: utf-8 -*-
"""Agent planner facade with a constrained tool-calling kernel."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any
import copy

from agent_stock.agents.agentic_decision import clamp_confidence, extract_json_object
from agent_stock.agents.contracts import (
    AgentState,
    DataAgentOutput,
    ExecutionAgentOutput,
    RiskAgentOutput,
    SignalAgentOutput,
    StockAgentResult,
)
from agent_stock.agents.planner_runtime import (
    PlannerAction,
    RequestIntent,
    build_request_intent,
    evaluate_conditions,
)
from agent_stock.analyzer import get_analyzer
from agent_stock.config import Config, get_config, redact_sensitive_text
from agent_stock.time_utils import local_now

logger = logging.getLogger(__name__)

StageObserver = Callable[[dict[str, Any]], None]
PaperOrderSubmitter = Callable[[dict[str, Any]], dict[str, Any]]

DEFAULT_TOOL_REGISTRY = [
    "load_account_context",
    "load_history",
    "load_backtest",
    "fetch_market_data",
    "derive_signal",
    "evaluate_risk",
    "prepare_order",
    "submit_paper_order",
]

STAGE_TO_TOOL = {
    "data": "fetch_market_data",
    "signal": "derive_signal",
    "risk": "evaluate_risk",
    "execution": "prepare_order",
}
TOOL_TO_STAGE = {
    "fetch_market_data": "data",
    "derive_signal": "signal",
    "evaluate_risk": "risk",
    "prepare_order": "execution",
}
STAGE_NEXT_TOOL = {
    "fetch_market_data": "derive_signal",
    "derive_signal": "evaluate_risk",
    "evaluate_risk": "prepare_order",
    "prepare_order": "finish",
    "submit_paper_order": "finish",
}
STAGE_LABELS = {
    "data": "数据 Agent",
    "signal": "信号 Agent",
    "risk": "风控 Agent",
    "execution": "执行 Agent",
}


@dataclass(frozen=True)
class ControllerContext:
    """Planner runtime context shared across one orchestrator cycle."""

    stock_codes: list[str]
    account_name: str
    initial_cash: float
    request_id: str | None = None
    runtime_config: Any | None = None
    planning_context: dict[str, Any] | None = None
    paper_order_submitter: PaperOrderSubmitter | None = None


class ControllerAgent:
    """Explicit planner facade that decides which constrained tool runs next."""

    def __init__(self, config: Config | None = None, analyzer=None) -> None:
        self.config = config or get_config()
        self.analyzer = analyzer or get_analyzer()

    @staticmethod
    def _extract_loaded_context(planning_context: dict[str, Any] | None) -> dict[str, Any]:
        context = planning_context if isinstance(planning_context, dict) else {}
        loaded_context = context.get("loaded_context")
        return dict(loaded_context) if isinstance(loaded_context, dict) else {}

    def _extract_system_state(self, planning_context: dict[str, Any] | None) -> dict[str, Any]:
        loaded_context = self._extract_loaded_context(planning_context)
        system_state = loaded_context.get("system_state")
        return dict(system_state) if isinstance(system_state, dict) else {}

    def _extract_account_state(self, planning_context: dict[str, Any] | None) -> dict[str, Any]:
        loaded_context = self._extract_loaded_context(planning_context)
        account_state = loaded_context.get("account_state")
        return dict(account_state) if isinstance(account_state, dict) else {}

    def _extract_effective_user_preferences(self, planning_context: dict[str, Any] | None) -> dict[str, Any]:
        loaded_context = self._extract_loaded_context(planning_context)
        preferences = loaded_context.get("effective_user_preferences")
        return dict(preferences) if isinstance(preferences, dict) else {}

    def _extract_stage_memory(self, planning_context: dict[str, Any] | None) -> dict[str, Any]:
        context = planning_context if isinstance(planning_context, dict) else {}
        stage_memory = context.get("stage_memory")
        if isinstance(stage_memory, dict):
            return dict(stage_memory)
        loaded_context = self._extract_loaded_context(planning_context)
        loaded_stage_memory = loaded_context.get("stage_memory")
        return dict(loaded_stage_memory) if isinstance(loaded_stage_memory, dict) else {}

    @staticmethod
    def _stage_state_for_tool(tool_name: str, system_state: dict[str, Any]) -> dict[str, Any]:
        stage_name = "execution" if tool_name == "submit_paper_order" else TOOL_TO_STAGE.get(tool_name)
        stage_state = system_state.get(stage_name) if stage_name and isinstance(system_state.get(stage_name), dict) else {}
        return dict(stage_state) if isinstance(stage_state, dict) else {}

    def _tool_allowed_by_system_state(self, tool_name: str, *, system_state: dict[str, Any]) -> bool:
        if not isinstance(system_state, dict) or not system_state:
            return True
        stage_state = self._stage_state_for_tool(tool_name, system_state)
        if not stage_state:
            return True
        if not bool(stage_state.get("available", True)):
            return False
        if bool(stage_state.get("busy")):
            return False
        return True

    def _filter_tools_by_system_state(self, tools: list[str], *, system_state: dict[str, Any]) -> list[str]:
        return [tool for tool in tools if self._tool_allowed_by_system_state(tool, system_state=system_state)]

    @staticmethod
    def _summarize_system_state(system_state: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for stage_name in ("data", "signal", "risk", "execution"):
            stage_state = system_state.get(stage_name) if isinstance(system_state.get(stage_name), dict) else {}
            if not stage_state:
                continue
            summary[stage_name] = {
                "available": bool(stage_state.get("available", True)),
                "busy": bool(stage_state.get("busy", False)),
                "degraded_reason": stage_state.get("degraded_reason"),
            }
        return summary

    @classmethod
    def _blocked_system_stages(cls, system_state: dict[str, Any]) -> list[str]:
        blocked: list[str] = []
        for stage_name, stage_state in cls._summarize_system_state(system_state).items():
            if not bool(stage_state.get("available", True)) or bool(stage_state.get("busy")):
                blocked.append(stage_name)
        return blocked

    def build_plan(self, *, context: ControllerContext) -> dict[str, Any]:
        """Generate a real execution plan plus compatibility summary fields."""
        intent = build_request_intent(
            stock_codes=list(context.stock_codes or []),
            planning_context=context.planning_context,
        )
        runtime_mode = self._resolve_runtime_execution_mode(context.runtime_config)
        system_state = self._extract_system_state(context.planning_context)
        account_state = self._extract_account_state(context.planning_context)
        effective_preferences = self._extract_effective_user_preferences(context.planning_context)
        stage_memory = self._extract_stage_memory(context.planning_context)
        tool_registry = [
            tool
            for tool in DEFAULT_TOOL_REGISTRY
            if tool not in {"load_account_context", "load_history", "load_backtest"}
        ]
        tool_registry = self._filter_tools_by_system_state(tool_registry, system_state=system_state)
        policy_snapshot = self._build_policy_snapshot(
            runtime_mode=runtime_mode,
            autonomous_execution_authorized=intent.autonomous_execution_authorized,
            submitter_available=context.paper_order_submitter is not None,
        )
        policy_snapshot["system_state"] = self._summarize_system_state(system_state)
        policy_snapshot["blocked_stages"] = self._blocked_system_stages(system_state)
        policy_snapshot["autonomous_execution_allowed"] = bool(policy_snapshot.get("autonomous_execution_allowed")) and self._tool_allowed_by_system_state(
            "submit_paper_order",
            system_state=system_state,
        )
        default_plan = {
            "goal": intent.goal,
            "stock_codes": list(intent.stock_codes),
            "include_runtime_context": intent.include_runtime_context,
            "stage_priority": ["data", "signal", "risk", "execution"],
            "autonomous_execution_authorized": intent.autonomous_execution_authorized,
            "termination_conditions": [
                "planner_finished",
                "stage_abort",
                "max_transitions_reached",
                "execution_completed",
                "policy_blocked",
            ],
            "max_transitions_per_stock": 10,
            "max_data_retries": 1,
            "max_signal_retries": 1,
            "max_execution_retries": 1,
            "tool_registry": tool_registry,
            "request_intent": intent.to_dict(),
            "policy_snapshot": policy_snapshot,
        }

        if not getattr(self.analyzer, "is_available", lambda: False)() or not intent.user_message:
            return default_plan

        prompt = (
            "你是股票多Agent主控规划器，只输出严格 JSON，不要输出解释、Markdown 或代码块。\n"
            "输出字段：goal, stage_priority, include_runtime_context, autonomous_execution_authorized。\n"
            "规则：1. stage_priority 只能用于摘要展示，不影响真实工具调度；"
            "2. stage_priority 只能由 data/signal/risk/execution 组成；3. 仅输出 JSON。\n\n"
            f"用户目标：{intent.user_message}\n"
            f"股票范围：{intent.stock_codes}\n"
            f"请求意图：{intent.to_dict()}\n"
            f"系统状态：{system_state}\n"
            f"账户状态：{account_state}\n"
            f"有效偏好：{effective_preferences}\n"
            f"上轮阶段结果：{stage_memory}\n"
            f"默认计划：{default_plan}\n"
        )
        try:
            payload = extract_json_object(self.analyzer.generate_text(prompt, temperature=0.0, max_output_tokens=400))
        except Exception as exc:
            logger.warning("controller plan fallback: %s", redact_sensitive_text(str(exc)))
            return default_plan

        raw_stage_priority = payload.get("stage_priority")
        stage_priority = ["data", "signal", "risk", "execution"]
        if isinstance(raw_stage_priority, list):
            normalized = []
            for item in raw_stage_priority:
                text = str(item or "").strip().lower()
                if text in stage_priority and text not in normalized:
                    normalized.append(text)
            if normalized:
                stage_priority = normalized

        return {
            **default_plan,
            "goal": str(payload.get("goal") or default_plan["goal"]).strip() or default_plan["goal"],
            "stage_priority": stage_priority,
            "include_runtime_context": bool(payload.get("include_runtime_context", default_plan["include_runtime_context"])),
            "autonomous_execution_authorized": bool(
                payload.get("autonomous_execution_authorized", default_plan["autonomous_execution_authorized"])
            ),
        }

    def run_stock(
        self,
        *,
        code: str,
        trade_date: date,
        current_account_snapshot: dict[str, Any],
        context: ControllerContext,
        controller_plan: dict[str, Any],
        data_agent: Any,
        signal_agent: Any,
        risk_agent: Any,
        execution_agent: Any,
        stage_observer: StageObserver | None = None,
    ) -> tuple[StockAgentResult, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        """Run the constrained tool-calling planner for one stock."""
        request_intent = build_request_intent(
            stock_codes=[code],
            planning_context=context.planning_context,
        )
        max_transitions = int(controller_plan.get("max_transitions_per_stock") or 10)
        max_data_retries = int(controller_plan.get("max_data_retries") or 1)
        max_signal_retries = int(controller_plan.get("max_signal_retries") or 1)
        max_execution_retries = int(controller_plan.get("max_execution_retries") or 1)
        policy_snapshot = dict(controller_plan.get("policy_snapshot") or {})
        system_state = self._extract_system_state(context.planning_context)
        account_state = self._extract_account_state(context.planning_context)
        effective_preferences = self._extract_effective_user_preferences(context.planning_context)
        stage_memory = self._extract_stage_memory(context.planning_context)

        data_out: DataAgentOutput | None = None
        signal_out: SignalAgentOutput | None = None
        risk_out: RiskAgentOutput | None = None
        execution_out: ExecutionAgentOutput | None = None
        working_snapshot = dict(current_account_snapshot or {})
        stage_traces: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        planner_trace: list[dict[str, Any]] = []
        condition_evaluations: list[dict[str, Any]] = []
        stage_visits = {"data": 0, "signal": 0, "risk": 0, "execution": 0}
        tool_attempts = {
            "fetch_market_data": 0,
            "derive_signal": 0,
            "evaluate_risk": 0,
            "prepare_order": 0,
            "submit_paper_order": 0,
        }
        replan_count = 0
        termination_reason = "max_transitions_reached"
        stock_execution_result: dict[str, Any] | None = None

        for step in range(1, max_transitions + 1):
            available_tools = self._available_tools(
                data_out=data_out,
                signal_out=signal_out,
                risk_out=risk_out,
                execution_out=execution_out,
                request_intent=request_intent,
                tool_attempts=tool_attempts,
                max_data_retries=max_data_retries,
                max_signal_retries=max_signal_retries,
                max_execution_retries=max_execution_retries,
                policy_snapshot=policy_snapshot,
                system_state=system_state,
            )
            action = self._select_planner_action(
                stock_code=code,
                request_intent=request_intent,
                available_tools=available_tools,
                default_action=self._default_action(available_tools=available_tools),
                data_out=data_out,
                signal_out=signal_out,
                risk_out=risk_out,
                execution_out=execution_out,
                system_state=system_state,
                account_state=account_state,
                effective_preferences=effective_preferences,
                stage_memory=stage_memory,
            )
            planner_entry = {
                "stock_code": code,
                "step": step,
                **action.to_dict(),
                "available_tools": list(available_tools),
            }
            planner_trace.append(planner_entry)
            self._emit_planner_event(stage_observer, "planner_step", planner_entry)

            if action.kind in {"finish", "clarify"} or not action.tool:
                termination_reason = action.reason or "planner_finished"
                break

            if action.tool == "fetch_market_data":
                if signal_out is not None:
                    signal_out = None
                    risk_out = None
                    execution_out = None
                tool_attempts["fetch_market_data"] += 1
                stage_visits["data"] += 1
                data_started_at = local_now().isoformat()
                self._emit_planner_event(
                    stage_observer,
                    "stage_start",
                    {
                        "stock_code": code,
                        "stage": "data",
                        "visit": stage_visits["data"],
                        "summary": f"{STAGE_LABELS['data']} 运行中",
                        "started_at": data_started_at,
                    },
                )
                data_out = self._run_data_tool(
                    code=code,
                    data_agent=data_agent,
                    runtime_config=context.runtime_config,
                    started_at=data_started_at,
                )
                trace = self._build_stage_trace(code=code, stage="data", output=data_out, visit=stage_visits["data"])
                stage_traces.append(trace)
                warnings.extend(self._collect_trace_warnings(trace))
                self._emit_stage_trace(stage_observer, trace)
                replan_count += self._emit_replan_if_needed(
                    stage_observer=stage_observer,
                    planner_trace=planner_trace,
                    stock_code=code,
                    from_tool=action.tool,
                    next_stage=data_out.next_action,
                    reason=str(data_out.status or data_out.error_message or ""),
                )
                continue

            if action.tool == "derive_signal":
                if data_out is None:
                    termination_reason = "missing_data_for_signal"
                    break
                if risk_out is not None:
                    risk_out = None
                    execution_out = None
                tool_attempts["derive_signal"] += 1
                stage_visits["signal"] += 1
                signal_started_at = local_now().isoformat()
                self._emit_planner_event(
                    stage_observer,
                    "stage_start",
                    {
                        "stock_code": code,
                        "stage": "signal",
                        "visit": stage_visits["signal"],
                        "summary": f"{STAGE_LABELS['signal']} 运行中",
                        "started_at": signal_started_at,
                    },
                )
                signal_out = self._run_signal_tool(
                    data_out=data_out,
                    trade_date=trade_date,
                    signal_agent=signal_agent,
                    runtime_config=context.runtime_config,
                    started_at=signal_started_at,
                )
                trace = self._build_stage_trace(code=code, stage="signal", output=signal_out, visit=stage_visits["signal"])
                stage_traces.append(trace)
                warnings.extend(self._collect_trace_warnings(trace))
                self._emit_stage_trace(stage_observer, trace)
                replan_count += self._emit_replan_if_needed(
                    stage_observer=stage_observer,
                    planner_trace=planner_trace,
                    stock_code=code,
                    from_tool=action.tool,
                    next_stage=signal_out.next_action,
                    reason=str(signal_out.review_reason or signal_out.status or ""),
                )
                continue

            if action.tool == "evaluate_risk":
                if data_out is None or signal_out is None:
                    termination_reason = "missing_signal_for_risk"
                    break
                tool_attempts["evaluate_risk"] += 1
                stage_visits["risk"] += 1
                risk_started_at = local_now().isoformat()
                self._emit_planner_event(
                    stage_observer,
                    "stage_start",
                    {
                        "stock_code": code,
                        "stage": "risk",
                        "visit": stage_visits["risk"],
                        "summary": f"{STAGE_LABELS['risk']} 运行中",
                        "started_at": risk_started_at,
                    },
                )
                risk_out = self._run_risk_tool(
                    code=code,
                    trade_date=trade_date,
                    signal_out=signal_out,
                    data_out=data_out,
                    working_snapshot=working_snapshot,
                    risk_agent=risk_agent,
                    runtime_config=context.runtime_config,
                    started_at=risk_started_at,
                )
                trace = self._build_stage_trace(code=code, stage="risk", output=risk_out, visit=stage_visits["risk"])
                stage_traces.append(trace)
                warnings.extend(self._collect_trace_warnings(trace))
                self._emit_stage_trace(stage_observer, trace)
                replan_count += self._emit_replan_if_needed(
                    stage_observer=stage_observer,
                    planner_trace=planner_trace,
                    stock_code=code,
                    from_tool=action.tool,
                    next_stage=risk_out.next_action,
                    reason=str(risk_out.review_reason or risk_out.status or ""),
                )
                continue

            if action.tool == "prepare_order":
                if data_out is None or signal_out is None or risk_out is None:
                    termination_reason = "missing_risk_for_execution"
                    break
                tool_attempts["prepare_order"] += 1
                stage_visits["execution"] += 1
                execution_started_at = local_now().isoformat()
                self._emit_planner_event(
                    stage_observer,
                    "stage_start",
                    {
                        "stock_code": code,
                        "stage": "execution",
                        "visit": stage_visits["execution"],
                        "summary": f"{STAGE_LABELS['execution']} 运行中",
                        "started_at": execution_started_at,
                    },
                )
                execution_out = self._run_prepare_order_tool(
                    code=code,
                    trade_date=trade_date,
                    data_out=data_out,
                    signal_out=signal_out,
                    risk_out=risk_out,
                    working_snapshot=working_snapshot,
                    context=context,
                    execution_agent=execution_agent,
                    started_at=execution_started_at,
                )
                execution_out = self._apply_requested_order_constraints(
                    execution_out=execution_out,
                    request_intent=request_intent,
                )
                if execution_out.account_snapshot:
                    working_snapshot = dict(execution_out.account_snapshot)
                trace = self._build_stage_trace(code=code, stage="execution", output=execution_out, visit=stage_visits["execution"])
                stage_traces.append(trace)
                warnings.extend(self._collect_trace_warnings(trace))
                self._emit_stage_trace(stage_observer, trace)
                current_price = self._resolve_current_price(data_out)
                condition_evaluations = self._evaluate_execution_policy(
                    request_intent=request_intent,
                    stock_code=code,
                    current_price=current_price,
                    risk_out=risk_out,
                    execution_out=execution_out,
                    account_snapshot=working_snapshot,
                    stage_observer=stage_observer,
                )
                if self._evaluations_block_execution(condition_evaluations):
                    reason = self._first_blocking_reason(condition_evaluations)
                    termination_reason = reason or "policy_blocked"
                    self._emit_policy_block(
                        stage_observer=stage_observer,
                        stock_code=code,
                        reason=termination_reason,
                    )
                    break
                if not request_intent.autonomous_execution_authorized:
                    termination_reason = "execution_prepared"
                    break
                if not self._can_submit_paper_order(
                    policy_snapshot=policy_snapshot,
                    request_intent=request_intent,
                    execution_out=execution_out,
                    submitter=context.paper_order_submitter,
                ):
                    termination_reason = "paper_submit_disabled"
                    self._emit_policy_block(
                        stage_observer=stage_observer,
                        stock_code=code,
                        reason=termination_reason,
                    )
                    break
                continue

            if action.tool == "submit_paper_order":
                tool_attempts["submit_paper_order"] += 1
                stock_execution_result = self._submit_paper_order(
                    stock_code=code,
                    execution_out=execution_out,
                    submitter=context.paper_order_submitter,
                )
                if execution_out is not None:
                    execution_out.paper_submit_result = dict(stock_execution_result)
                    if hasattr(execution_agent, "_attach_proposal_metadata"):
                        execution_out = execution_agent._attach_proposal_metadata(  # type: ignore[attr-defined]
                            execution_out,
                            current_price=self._resolve_current_price(data_out) if data_out is not None else 0.0,
                        )
                    else:
                        execution_out.proposal_state = "executed"
                        execution_out.status = "executed"
                        execution_out.execution_allowed = True
                termination_reason = "execution_completed"
                break

        if data_out is None:
            data_out = DataAgentOutput(
                code=code,
                trade_date=trade_date,
                state=AgentState.FAILED,
                error_message="controller_missing_data",
                next_action="abort",
                status="failed",
                retryable=False,
                partial_ok=False,
                suggested_next="abort",
            )
        if signal_out is None:
            signal_out = SignalAgentOutput(
                code=code,
                trade_date=trade_date,
                state=AgentState.SKIPPED,
                next_action="abort",
                status="blocked",
                suggested_next="abort",
            )
        if risk_out is None:
            risk_out = RiskAgentOutput(
                code=code,
                trade_date=trade_date,
                state=AgentState.SKIPPED,
                current_price=self._resolve_current_price(data_out),
                next_action="abort",
                status="blocked",
                risk_level="high",
                execution_allowed=False,
                hard_blocks=["missing_risk"],
                suggested_next="abort",
            )
        if execution_out is None:
            execution_out = ExecutionAgentOutput(
                code=code,
                trade_date=trade_date,
                state=AgentState.SKIPPED,
                action="none",
                reason="controller_no_execution",
                account_snapshot=working_snapshot,
                next_action="done",
                status="blocked",
                execution_allowed=False,
            )

        result = StockAgentResult(
            code=code,
            data=data_out,
            signal=signal_out,
            risk=risk_out,
            execution=execution_out,
            planner_trace=planner_trace,
            condition_evaluations=condition_evaluations,
            termination_reason=termination_reason,
            replan_count=replan_count,
            policy_snapshot=policy_snapshot,
            execution_result=stock_execution_result,
        )
        return result, working_snapshot, stage_traces, warnings

    def _run_data_tool(
        self,
        *,
        code: str,
        data_agent: Any,
        runtime_config: Any,
        started_at: str | None = None,
    ) -> DataAgentOutput:
        stage_started_at = started_at or local_now().isoformat()
        started = time.perf_counter()
        data_out = data_agent.run(code, runtime_config=runtime_config)
        data_out.started_at = stage_started_at
        data_out.finished_at = local_now().isoformat()
        data_out.duration_ms = int((time.perf_counter() - started) * 1000)
        data_out.input = {"code": code}
        data_out.output = {
            "state": data_out.state.value,
            "status": data_out.status,
            "data_source": data_out.data_source,
            "has_analysis_context": bool(data_out.analysis_context),
            "has_realtime_quote": bool(data_out.realtime_quote),
            "retryable": data_out.retryable,
            "partial_ok": data_out.partial_ok,
            "next_action": data_out.next_action,
        }
        return data_out

    def _run_signal_tool(
        self,
        *,
        data_out: DataAgentOutput,
        trade_date: date,
        signal_agent: Any,
        runtime_config: Any,
        started_at: str | None = None,
    ) -> SignalAgentOutput:
        stage_started_at = started_at or local_now().isoformat()
        started = time.perf_counter()
        signal_out = signal_agent.run(data_out, runtime_config=runtime_config)
        signal_out.started_at = stage_started_at
        signal_out.finished_at = local_now().isoformat()
        signal_out.duration_ms = int((time.perf_counter() - started) * 1000)
        signal_out.input = {
            "code": data_out.code,
            "trade_date": trade_date.isoformat(),
            "data_state": data_out.state.value,
            "runtime_llm": bool(runtime_config and getattr(runtime_config, "llm", None) is not None),
        }
        signal_out.output = {
            "operation_advice": signal_out.operation_advice,
            "sentiment_score": signal_out.sentiment_score,
            "trend_signal": signal_out.trend_signal,
            "stop_loss": signal_out.stop_loss,
            "take_profit": signal_out.take_profit,
            "status": signal_out.status,
            "needs_more_data": signal_out.needs_more_data,
            "review_reason": signal_out.review_reason,
            "next_action": signal_out.next_action,
        }
        return signal_out

    def _run_risk_tool(
        self,
        *,
        code: str,
        trade_date: date,
        signal_out: SignalAgentOutput,
        data_out: DataAgentOutput,
        working_snapshot: dict[str, Any],
        risk_agent: Any,
        runtime_config: Any,
        started_at: str | None = None,
    ) -> RiskAgentOutput:
        stage_started_at = started_at or local_now().isoformat()
        current_price = self._resolve_current_price(data_out)
        current_position_value = self._current_position_value(working_snapshot, code)
        started = time.perf_counter()
        risk_out = risk_agent.run(
            code=code,
            trade_date=trade_date,
            current_price=current_price,
            signal_output=signal_out,
            account_snapshot=working_snapshot,
            current_position_value=current_position_value,
            runtime_strategy=(runtime_config.strategy if runtime_config else None),
        )
        risk_out.started_at = stage_started_at
        risk_out.finished_at = local_now().isoformat()
        risk_out.duration_ms = int((time.perf_counter() - started) * 1000)
        risk_out.input = {
            "code": code,
            "current_price": current_price,
            "operation_advice": signal_out.operation_advice,
            "current_position_value": current_position_value,
            "runtime_strategy_applied": bool(runtime_config and getattr(runtime_config, "strategy", None) is not None),
        }
        risk_out.output = {
            "target_weight": risk_out.target_weight,
            "target_notional": risk_out.target_notional,
            "risk_flags": risk_out.risk_flags,
            "risk_level": risk_out.risk_level,
            "execution_allowed": risk_out.execution_allowed,
            "hard_blocks": list(risk_out.hard_blocks or []),
            "soft_flags": list(risk_out.soft_flags or []),
            "review_reason": risk_out.review_reason,
            "effective_stop_loss": risk_out.effective_stop_loss,
            "effective_take_profit": risk_out.effective_take_profit,
            "position_cap_pct": risk_out.position_cap_pct,
            "next_action": risk_out.next_action,
        }
        return risk_out

    def _run_prepare_order_tool(
        self,
        *,
        code: str,
        trade_date: date,
        data_out: DataAgentOutput,
        signal_out: SignalAgentOutput,
        risk_out: RiskAgentOutput,
        working_snapshot: dict[str, Any],
        context: ControllerContext,
        execution_agent: Any,
        started_at: str | None = None,
    ) -> ExecutionAgentOutput:
        stage_started_at = started_at or local_now().isoformat()
        current_price = self._resolve_current_price(data_out)
        started = time.perf_counter()
        if hasattr(execution_agent, "prepare_order"):
            execution_out = execution_agent.prepare_order(
                code=code,
                trade_date=trade_date,
                current_price=current_price,
                risk_output=risk_out,
                account_snapshot=working_snapshot,
                account_name=context.account_name,
                initial_cash_override=context.initial_cash,
                backend_task_id=context.request_id,
                signal_output=signal_out,
                data_output=data_out,
            )
        else:
            execution_out = execution_agent.run(
                run_id=context.request_id or "controller-chat",
                code=code,
                trade_date=trade_date,
                current_price=current_price,
                risk_output=risk_out,
                account_snapshot=working_snapshot,
                account_name=context.account_name,
                initial_cash_override=context.initial_cash,
                runtime_execution=None,
                backend_task_id=context.request_id,
                signal_output=signal_out,
                data_output=data_out,
            )
        execution_out.started_at = stage_started_at
        execution_out.finished_at = local_now().isoformat()
        execution_out.duration_ms = int((time.perf_counter() - started) * 1000)
        execution_out.input = {
            "code": code,
            "account_name": context.account_name,
            "backend_task_id": context.request_id,
            "current_price": current_price,
            "target_weight": risk_out.target_weight,
            "target_notional": risk_out.target_notional,
        }
        execution_out.output = {
            "action": execution_out.action,
            "reason": execution_out.reason,
            "traded_qty": execution_out.traded_qty,
            "position_after": execution_out.position_after,
            "cash_after": execution_out.cash_after,
            "status": execution_out.status,
            "proposal_state": execution_out.proposal_state,
            "next_action": execution_out.next_action,
        }
        return execution_out

    def _apply_requested_order_constraints(
        self,
        *,
        execution_out: ExecutionAgentOutput,
        request_intent: RequestIntent,
    ) -> ExecutionAgentOutput:
        proposed_order = execution_out.proposed_order if isinstance(execution_out.proposed_order, dict) else None
        final_order = execution_out.final_order if isinstance(execution_out.final_order, dict) else None
        if request_intent.requested_quantity and proposed_order:
            current_qty = int(proposed_order.get("quantity") or 0)
            requested_qty = int(request_intent.requested_quantity or 0)
            if 0 < requested_qty <= current_qty:
                proposed_order = dict(proposed_order)
                proposed_order["quantity"] = requested_qty
                proposed_order["target_qty"] = requested_qty
                execution_out.proposed_order = proposed_order
                execution_out.traded_qty = requested_qty
                execution_out.target_qty = requested_qty
                if final_order:
                    final_order = dict(final_order)
                    final_order["quantity"] = requested_qty
                    final_order["target_qty"] = requested_qty
                    execution_out.final_order = final_order
        return execution_out

    def _evaluate_execution_policy(
        self,
        *,
        request_intent: RequestIntent,
        stock_code: str,
        current_price: float,
        risk_out: RiskAgentOutput,
        execution_out: ExecutionAgentOutput,
        account_snapshot: dict[str, Any],
        stage_observer: StageObserver | None,
    ) -> list[dict[str, Any]]:
        evaluations = evaluate_conditions(
            conditions=list(request_intent.conditions or []),
            unsupported_conditions=list(request_intent.unsupported_conditions or []),
            stock_code=stock_code,
            current_price=current_price,
            risk_output=risk_out,
            execution_output=execution_out,
            account_snapshot=account_snapshot,
        )

        proposed_order = execution_out.proposed_order if isinstance(execution_out.proposed_order, dict) else {}
        if request_intent.requested_order_side:
            actual_side = str(proposed_order.get("action") or execution_out.action or "").strip()
            evaluations.append(
                {
                    "stock_code": stock_code,
                    "condition": {
                        "type": "order_side",
                        "value": request_intent.requested_order_side,
                        "operator": "eq",
                        "label": "order_side",
                        "supported": True,
                    },
                    "passed": actual_side == request_intent.requested_order_side,
                    "blocking": True,
                    "reason": "side_match" if actual_side == request_intent.requested_order_side else "side_mismatch",
                    "expected": request_intent.requested_order_side,
                    "actual": actual_side,
                }
            )

        if request_intent.requested_quantity:
            actual_qty = int(proposed_order.get("quantity") or execution_out.traded_qty or 0)
            evaluations.append(
                {
                    "stock_code": stock_code,
                    "condition": {
                        "type": "exact_quantity",
                        "value": int(request_intent.requested_quantity),
                        "operator": "eq",
                        "label": "exact_quantity",
                        "supported": True,
                    },
                    "passed": actual_qty == int(request_intent.requested_quantity),
                    "blocking": True,
                    "reason": "quantity_match" if actual_qty == int(request_intent.requested_quantity) else "quantity_unavailable",
                    "expected": int(request_intent.requested_quantity),
                    "actual": actual_qty,
                }
            )

        for item in evaluations:
            self._emit_planner_event(stage_observer, "condition_eval", item)
        return evaluations

    @staticmethod
    def _evaluations_block_execution(evaluations: list[dict[str, Any]]) -> bool:
        return any(not bool(item.get("passed")) and bool(item.get("blocking", True)) for item in evaluations)

    @staticmethod
    def _first_blocking_reason(evaluations: list[dict[str, Any]]) -> str | None:
        for item in evaluations:
            if not bool(item.get("passed")) and bool(item.get("blocking", True)):
                return str(item.get("reason") or "policy_blocked").strip() or "policy_blocked"
        return None

    @staticmethod
    def _submit_paper_order(
        *,
        stock_code: str,
        execution_out: ExecutionAgentOutput | None,
        submitter: PaperOrderSubmitter | None,
    ) -> dict[str, Any]:
        if submitter is None or execution_out is None:
            return {}
        candidate_order = execution_out.proposed_order if isinstance(execution_out.proposed_order, dict) else {}
        if not candidate_order:
            return {}
        payload = submitter(dict(candidate_order))
        return {
            "stock_code": stock_code,
            "candidate_order": dict(candidate_order),
            **(dict(payload) if isinstance(payload, dict) else {}),
        }

    def _available_tools(
        self,
        *,
        data_out: DataAgentOutput | None,
        signal_out: SignalAgentOutput | None,
        risk_out: RiskAgentOutput | None,
        execution_out: ExecutionAgentOutput | None,
        request_intent: RequestIntent,
        tool_attempts: dict[str, int],
        max_data_retries: int,
        max_signal_retries: int,
        max_execution_retries: int,
        policy_snapshot: dict[str, Any],
        system_state: dict[str, Any],
    ) -> list[str]:
        if data_out is None:
            return self._filter_tools_by_system_state(["fetch_market_data"], system_state=system_state)
        if data_out.state == AgentState.FAILED:
            if tool_attempts["fetch_market_data"] <= max_data_retries:
                return self._filter_tools_by_system_state(["fetch_market_data"], system_state=system_state)
            return []
        if signal_out is None:
            return self._filter_tools_by_system_state(["derive_signal"], system_state=system_state)
        if signal_out.state == AgentState.FAILED:
            if tool_attempts["derive_signal"] <= max_signal_retries:
                return self._filter_tools_by_system_state(["derive_signal"], system_state=system_state)
            return []
        if signal_out.needs_more_data or str(signal_out.next_action or "").strip().lower() == "data":
            if tool_attempts["fetch_market_data"] <= max_data_retries:
                return self._filter_tools_by_system_state(["fetch_market_data"], system_state=system_state)
            return []
        if risk_out is None:
            return self._filter_tools_by_system_state(["evaluate_risk"], system_state=system_state)
        if (risk_out.review_reason or str(risk_out.next_action or "").strip().lower() == "signal") and tool_attempts["derive_signal"] <= max_signal_retries:
            return self._filter_tools_by_system_state(["derive_signal"], system_state=system_state)
        if execution_out is None:
            if risk_out.execution_allowed or risk_out.next_action == "execution" or (
                risk_out.state == AgentState.READY and float(risk_out.target_notional or 0.0) > 0
            ):
                return self._filter_tools_by_system_state(["prepare_order"], system_state=system_state)
            return []
        if request_intent.autonomous_execution_authorized and self._can_submit_paper_order(
            policy_snapshot=policy_snapshot,
            request_intent=request_intent,
            execution_out=execution_out,
            submitter=None,
        ):
            # submitter availability is checked in the actual action path.
            return self._filter_tools_by_system_state(["submit_paper_order"], system_state=system_state)
        if execution_out.state == AgentState.FAILED and tool_attempts["prepare_order"] <= max_execution_retries:
            return self._filter_tools_by_system_state(["prepare_order"], system_state=system_state)
        return []

    @staticmethod
    def _default_action(*, available_tools: list[str]) -> PlannerAction:
        if not available_tools:
            return PlannerAction(kind="finish", reason="planner_finished", summary="No more eligible tools")
        return PlannerAction(
            kind="call_tool",
            tool=available_tools[0],
            reason="rule_selected",
            summary=f"Planner selected {available_tools[0]}",
        )

    def _select_planner_action(
        self,
        *,
        stock_code: str,
        request_intent: RequestIntent,
        available_tools: list[str],
        default_action: PlannerAction,
        data_out: DataAgentOutput | None,
        signal_out: SignalAgentOutput | None,
        risk_out: RiskAgentOutput | None,
        execution_out: ExecutionAgentOutput | None,
        system_state: dict[str, Any],
        account_state: dict[str, Any],
        effective_preferences: dict[str, Any],
        stage_memory: dict[str, Any],
    ) -> PlannerAction:
        if default_action.kind != "call_tool" or not available_tools:
            return default_action
        if not getattr(self.analyzer, "is_available", lambda: False)() or not request_intent.user_message:
            return default_action

        prompt = (
            "你是股票多Agent主控，只能在受限工具集中决定下一步，只输出严格 JSON。\n"
            "允许 kind 只有：call_tool, finish, clarify。\n"
            f"股票代码：{stock_code}\n"
            f"用户目标：{request_intent.user_message}\n"
            f"可用工具：{available_tools}\n"
            f"默认动作：{default_action.to_dict()}\n"
            f"系统状态：{system_state}\n"
            f"账户状态：{account_state}\n"
            f"有效偏好：{effective_preferences}\n"
            f"上轮阶段结果：{stage_memory}\n"
            f"当前数据状态：{data_out.to_dict() if data_out else None}\n"
            f"当前信号状态：{signal_out.to_dict() if signal_out else None}\n"
            f"当前风控状态：{risk_out.to_dict() if risk_out else None}\n"
            f"当前执行状态：{execution_out.to_dict() if execution_out else None}\n"
            "输出字段：kind, tool, reason, summary。"
        )
        try:
            payload = extract_json_object(self.analyzer.generate_text(prompt, temperature=0.0, max_output_tokens=240))
        except Exception as exc:
            logger.warning("planner action fallback: %s", redact_sensitive_text(str(exc)))
            return default_action

        kind = str(payload.get("kind") or "").strip()
        tool = str(payload.get("tool") or "").strip()
        if kind == "finish":
            return PlannerAction(kind="finish", reason=str(payload.get("reason") or "planner_finished"), summary=str(payload.get("summary") or "Planner finished"))
        if kind == "clarify":
            return PlannerAction(kind="clarify", reason=str(payload.get("reason") or "clarify_required"), summary=str(payload.get("summary") or "Planner needs clarification"))
        if kind == "call_tool" and tool in available_tools:
            return PlannerAction(
                kind="call_tool",
                tool=tool,
                reason=str(payload.get("reason") or "llm_selected"),
                summary=str(payload.get("summary") or f"Planner selected {tool}"),
            )
        return default_action

    def _emit_replan_if_needed(
        self,
        *,
        stage_observer: StageObserver | None,
        planner_trace: list[dict[str, Any]],
        stock_code: str,
        from_tool: str,
        next_stage: Any,
        reason: str,
    ) -> int:
        expected_next = STAGE_NEXT_TOOL.get(from_tool)
        actual_next = self._next_stage_to_tool(next_stage)
        if not actual_next or actual_next == expected_next:
            return 0
        payload = {
            "stock_code": stock_code,
            "from_tool": from_tool,
            "to_tool": actual_next,
            "reason": reason or "replan",
        }
        planner_trace.append({"kind": "replan", **payload})
        self._emit_planner_event(stage_observer, "planner_replan", payload)
        return 1

    @staticmethod
    def _next_stage_to_tool(next_stage: Any) -> str | None:
        stage = str(next_stage or "").strip().lower()
        if stage == "data":
            return "fetch_market_data"
        if stage == "signal":
            return "derive_signal"
        if stage == "risk":
            return "evaluate_risk"
        if stage == "execution":
            return "prepare_order"
        if stage == "done":
            return "finish"
        return None

    @staticmethod
    def _build_policy_snapshot(
        *,
        runtime_mode: str,
        autonomous_execution_authorized: bool,
        submitter_available: bool,
    ) -> dict[str, Any]:
        return {
            "execution_scope": "paper_only",
            "runtime_execution_mode": runtime_mode,
            "autonomous_execution_requested": autonomous_execution_authorized,
            "autonomous_execution_allowed": autonomous_execution_authorized and runtime_mode == "paper" and submitter_available,
            "submitter_available": submitter_available,
            "broker_autonomy_enabled": False,
        }

    @staticmethod
    def _resolve_runtime_execution_mode(runtime_config: Any) -> str:
        execution = getattr(runtime_config, "execution", None) if runtime_config is not None else None
        mode = getattr(execution, "mode", None) if execution is not None else None
        return str(mode or "paper").strip().lower() or "paper"

    def _can_submit_paper_order(
        self,
        *,
        policy_snapshot: dict[str, Any],
        request_intent: RequestIntent,
        execution_out: ExecutionAgentOutput | None,
        submitter: PaperOrderSubmitter | None,
    ) -> bool:
        if execution_out is None:
            return False
        if not request_intent.autonomous_execution_authorized:
            return False
        if str(policy_snapshot.get("runtime_execution_mode") or "paper") != "paper":
            return False
        if not isinstance(execution_out.proposed_order, dict) or not execution_out.proposed_order:
            return False
        if str(execution_out.proposal_state or "").strip() not in {"proposed", "submitted", "executed"}:
            return False
        if submitter is None:
            return bool(policy_snapshot.get("submitter_available"))
        return True

    def _emit_stage_trace(self, stage_observer: StageObserver | None, trace: dict[str, Any]) -> None:
        if stage_observer is None:
            return
        stage_observer({"event": "stage_update", **trace})
        for warning in trace.get("warnings") or []:
            text = str(warning or "").strip()
            if text:
                stage_observer(
                    {
                        "event": "warning",
                        "stock_code": trace.get("stock_code"),
                        "stage": trace.get("stage"),
                        "message": text,
                    }
                )

    @staticmethod
    def _emit_planner_event(stage_observer: StageObserver | None, event: str, payload: dict[str, Any]) -> None:
        if stage_observer is None:
            return
        stage_observer({"event": event, **payload})

    def _emit_policy_block(
        self,
        *,
        stage_observer: StageObserver | None,
        stock_code: str,
        reason: str,
    ) -> None:
        self._emit_planner_event(
            stage_observer,
            "policy_block",
            {
                "stock_code": stock_code,
                "reason": reason,
            },
        )

    @staticmethod
    def _collect_trace_warnings(trace: dict[str, Any]) -> list[dict[str, Any]]:
        warnings = trace.get("warnings")
        if not isinstance(warnings, list):
            return []
        items: list[dict[str, Any]] = []
        for item in warnings:
            text = str(item or "").strip()
            if text:
                items.append(
                    {
                        "stock_code": trace.get("stock_code"),
                        "stage": trace.get("stage"),
                        "message": text,
                    }
                )
        return items

    @staticmethod
    def _resolve_current_price(data_out: DataAgentOutput) -> float:
        realtime_price = float((data_out.realtime_quote or {}).get("price") or 0.0)
        if realtime_price > 0:
            return realtime_price
        today = data_out.analysis_context.get("today") if isinstance(data_out.analysis_context, dict) else {}
        fallback_price = float((today or {}).get("close") or 0.0)
        if fallback_price > 0:
            return fallback_price
        return 0.0

    @staticmethod
    def _current_position_value(account_snapshot: dict[str, Any], code: str) -> float:
        for item in account_snapshot.get("positions", []):
            if str(item.get("code") or "") == str(code):
                return float(item.get("market_value") or 0.0)
        return 0.0

    @staticmethod
    def _build_stage_trace(*, code: str, stage: str, output: Any, visit: int) -> dict[str, Any]:
        decision = output.decision if isinstance(getattr(output, "decision", None), dict) else {}
        warnings = [str(item or "").strip() for item in getattr(output, "warnings", []) if str(item or "").strip()]
        summary = str(decision.get("summary") or getattr(output, "error_message", None) or f"{stage} stage completed").strip()
        return {
            "stock_code": code,
            "stage": stage,
            "visit": visit,
            "state": getattr(output, "state", AgentState.SKIPPED).value if hasattr(getattr(output, "state", None), "value") else str(getattr(output, "state", "skipped")),
            "summary": summary,
            "decision": decision,
            "confidence": clamp_confidence(getattr(output, "confidence", None), 0.5),
            "duration_ms": getattr(output, "duration_ms", None),
            "input": copy.deepcopy(getattr(output, "input", None)),
            "output": copy.deepcopy(getattr(output, "output", None)),
            "error_message": str(getattr(output, "error_message", "") or "").strip() or None,
            "warnings": warnings,
            "observations": list(getattr(output, "observations", []) or []),
            "fallback_chain": list(getattr(output, "fallback_chain", []) or []),
            "next_action": str(getattr(output, "next_action", "") or "").strip(),
            "llm_used": bool(getattr(output, "llm_used", False)),
            "started_at": str(getattr(output, "started_at", "") or "").strip() or None,
            "finished_at": str(getattr(output, "finished_at", "") or "").strip() or None,
        }
