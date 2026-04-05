# -*- coding: utf-8 -*-
"""多智能体交易工作流共享的数据契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class AgentState(str, Enum):
    """智能体各阶段的执行状态。"""

    READY = "ready"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class DataAgentOutput:
    """数据智能体产出的标准化市场快照。"""

    code: str
    trade_date: date
    state: AgentState = AgentState.READY
    analysis_context: dict[str, Any] = field(default_factory=dict)
    realtime_quote: dict[str, Any] = field(default_factory=dict)
    data_source: str | None = None
    duration_ms: int | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    error_message: str | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    llm_used: bool = False
    fallback_chain: list[str] = field(default_factory=list)
    next_action: str | None = None
    status: str = "pending"
    retryable: bool = False
    source_attempts: list[dict[str, Any]] = field(default_factory=list)
    partial_ok: bool = False
    suggested_next: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """将数据阶段输出转换为可序列化字典。"""
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["trade_date"] = self.trade_date.isoformat()
        return payload


@dataclass
class SignalAgentOutput:
    """信号智能体产出的操作建议与止盈止损信息。"""

    code: str
    trade_date: date
    state: AgentState = AgentState.READY
    operation_advice: str = "观望"
    sentiment_score: int = 50
    trend_signal: str = "WAIT"
    trend_score: int = 0
    stop_loss: float | None = None
    take_profit: float | None = None
    resolved_stop_loss: float | None = None
    resolved_take_profit: float | None = None
    ai_refreshed: bool = False
    ai_payload: dict[str, Any] = field(default_factory=dict)
    trend_payload: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    error_message: str | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    llm_used: bool = False
    fallback_chain: list[str] = field(default_factory=list)
    next_action: str | None = None
    status: str = "pending"
    needs_more_data: bool = False
    review_reason: str | None = None
    suggested_next: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """将信号阶段输出转换为可序列化字典。"""
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["trade_date"] = self.trade_date.isoformat()
        return payload


@dataclass
class RiskAgentOutput:
    """应用风控约束后的目标仓位结果。"""

    code: str
    trade_date: date
    state: AgentState = AgentState.READY
    target_weight: float = 0.0
    target_notional: float = 0.0
    current_price: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    effective_stop_loss: float | None = None
    effective_take_profit: float | None = None
    position_cap_pct: float | None = None
    strategy_applied: bool = False
    hard_risk_triggered: bool = False
    risk_flags: list[str] = field(default_factory=list)
    duration_ms: int | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    error_message: str | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    llm_used: bool = False
    fallback_chain: list[str] = field(default_factory=list)
    next_action: str | None = None
    status: str = "pending"
    risk_level: str = "unknown"
    execution_allowed: bool = False
    hard_blocks: list[str] = field(default_factory=list)
    soft_flags: list[str] = field(default_factory=list)
    review_reason: str | None = None
    suggested_next: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """将风控阶段输出转换为可序列化字典。"""
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["trade_date"] = self.trade_date.isoformat()
        return payload


@dataclass
class ExecutionAgentOutput:
    """执行阶段产出的成交结果与账户变化。"""

    code: str
    trade_date: date
    state: AgentState = AgentState.READY
    execution_mode: str = "paper"
    backend_task_id: str | None = None
    broker_requested: bool = False
    executed_via: str = "paper"
    broker_ticket_id: str | None = None
    fallback_reason: str | None = None
    action: str = "none"
    reason: str = ""
    order_id: int | None = None
    trade_id: int | None = None
    target_qty: int = 0
    traded_qty: int = 0
    fill_price: float | None = None
    fee: float = 0.0
    tax: float = 0.0
    cash_before: float = 0.0
    cash_after: float = 0.0
    position_before: int = 0
    position_after: int = 0
    account_snapshot: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    error_message: str | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
    llm_used: bool = False
    fallback_chain: list[str] = field(default_factory=list)
    next_action: str | None = None
    original_order: dict[str, Any] | None = None
    final_order: dict[str, Any] | None = None
    adjustment_applied: bool = False
    adjustment_reason: str | None = None
    risk_reduction_only: bool = False
    proposed_order: dict[str, Any] | None = None
    proposal_state: str | None = None
    proposal_reason: str | None = None
    status: str = "pending"
    execution_allowed: bool = False
    paper_submit_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """将执行阶段输出转换为可序列化字典。"""
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["trade_date"] = self.trade_date.isoformat()
        return payload


@dataclass
class StockAgentResult:
    """单只股票在一次编排周期内的各阶段输出。"""

    code: str
    data: DataAgentOutput
    signal: SignalAgentOutput
    risk: RiskAgentOutput
    execution: ExecutionAgentOutput
    planner_trace: list[dict[str, Any]] = field(default_factory=list)
    condition_evaluations: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str | None = None
    replan_count: int = 0
    policy_snapshot: dict[str, Any] = field(default_factory=dict)
    execution_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """将单股票运行结果转换为嵌套字典。"""
        return {
            "code": self.code,
            "data": self.data.to_dict(),
            "signal": self.signal.to_dict(),
            "risk": self.risk.to_dict(),
            "execution": self.execution.to_dict(),
            "planner_trace": self.planner_trace,
            "condition_evaluations": self.condition_evaluations,
            "termination_reason": self.termination_reason,
            "replan_count": self.replan_count,
            "policy_snapshot": self.policy_snapshot,
            "execution_result": self.execution_result,
        }


@dataclass
class AgentRunResult:
    """一次编排周期的顶层运行结果。"""

    run_id: str
    mode: str
    started_at: datetime
    ended_at: datetime
    trade_date: date
    results: list[StockAgentResult] = field(default_factory=list)
    account_snapshot: dict[str, Any] = field(default_factory=dict)
    markdown_report_path: str | None = None
    csv_report_path: str | None = None
    controller_plan: dict[str, Any] = field(default_factory=dict)
    portfolio_decision: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    stage_traces: list[dict[str, Any]] = field(default_factory=list)
    decision_panel: list[dict[str, Any]] = field(default_factory=list)
    planner_trace: list[dict[str, Any]] = field(default_factory=list)
    condition_evaluations: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str | None = None
    replan_count: int = 0
    policy_snapshot: dict[str, Any] = field(default_factory=dict)
    execution_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """将整次运行结果转换为可持久化字典。"""
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "trade_date": self.trade_date.isoformat(),
            "results": [item.to_dict() for item in self.results],
            "account_snapshot": self.account_snapshot,
            "markdown_report_path": self.markdown_report_path,
            "csv_report_path": self.csv_report_path,
            "controller_plan": self.controller_plan,
            "portfolio_decision": self.portfolio_decision,
            "warnings": self.warnings,
            "stage_traces": self.stage_traces,
            "decision_panel": self.decision_panel,
            "planner_trace": self.planner_trace,
            "condition_evaluations": self.condition_evaluations,
            "termination_reason": self.termination_reason,
            "replan_count": self.replan_count,
            "policy_snapshot": self.policy_snapshot,
            "execution_result": self.execution_result,
        }
