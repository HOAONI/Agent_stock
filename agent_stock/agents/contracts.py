# -*- coding: utf-8 -*-
"""多智能体交易工作流共享的数据契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional


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
    analysis_context: Dict[str, Any] = field(default_factory=dict)
    realtime_quote: Dict[str, Any] = field(default_factory=dict)
    data_source: Optional[str] = None
    duration_ms: Optional[int] = None
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    resolved_stop_loss: Optional[float] = None
    resolved_take_profit: Optional[float] = None
    ai_refreshed: bool = False
    ai_payload: Dict[str, Any] = field(default_factory=dict)
    trend_payload: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[int] = None
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    effective_stop_loss: Optional[float] = None
    effective_take_profit: Optional[float] = None
    position_cap_pct: Optional[float] = None
    strategy_applied: bool = False
    hard_risk_triggered: bool = False
    risk_flags: List[str] = field(default_factory=list)
    duration_ms: Optional[int] = None
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
    backend_task_id: Optional[str] = None
    broker_requested: bool = False
    executed_via: str = "paper"
    broker_ticket_id: Optional[str] = None
    fallback_reason: Optional[str] = None
    action: str = "none"
    reason: str = ""
    order_id: Optional[int] = None
    trade_id: Optional[int] = None
    target_qty: int = 0
    traded_qty: int = 0
    fill_price: Optional[float] = None
    fee: float = 0.0
    tax: float = 0.0
    cash_before: float = 0.0
    cash_after: float = 0.0
    position_before: int = 0
    position_after: int = 0
    account_snapshot: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[int] = None
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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

    def to_dict(self) -> Dict[str, Any]:
        """将单股票运行结果转换为嵌套字典。"""
        return {
            "code": self.code,
            "data": self.data.to_dict(),
            "signal": self.signal.to_dict(),
            "risk": self.risk.to_dict(),
            "execution": self.execution.to_dict(),
        }


@dataclass
class AgentRunResult:
    """一次编排周期的顶层运行结果。"""

    run_id: str
    mode: str
    started_at: datetime
    ended_at: datetime
    trade_date: date
    results: List[StockAgentResult] = field(default_factory=list)
    account_snapshot: Dict[str, Any] = field(default_factory=dict)
    markdown_report_path: Optional[str] = None
    csv_report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
        }
