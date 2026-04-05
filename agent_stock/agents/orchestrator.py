# -*- coding: utf-8 -*-
"""多 Agent 串行编排器。

这是主链路里最值得优先阅读的文件之一。它把一次运行拆成四个阶段：
`Data -> Signal -> Risk -> Execution`，并负责把阶段输入/输出、耗时和账户
快照串成一条完整的可观测链路。
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable

from zoneinfo import ZoneInfo

from agent_stock.agents.contracts import AgentRunResult, StockAgentResult
from agent_stock.agents.controller_agent import ControllerAgent, ControllerContext
from agent_stock.agents.data_agent import DataAgent
from agent_stock.agents.execution_agent import ExecutionAgent
from agent_stock.agents.risk_agent import RiskAgent
from agent_stock.agents.signal_agent import SignalAgent
from agent_stock.config import AgentRuntimeConfig, Config, get_config
from agent_stock.protocols import (
    SupportsDataAgent,
    SupportsExecutionAgent,
    SupportsMarketSessionGuard,
    SupportsRiskAgent,
    SupportsSignalAgent,
)
from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.storage import DatabaseManager

logger = logging.getLogger(__name__)


class MarketSessionGuard:
    """A 股实时循环使用的交易时段守卫。"""

    def __init__(self, timezone_name: str, sessions: str) -> None:
        """解析时区和交易时段配置。"""
        self.timezone = ZoneInfo(timezone_name)
        self.sessions = self._parse_sessions(sessions)

    @staticmethod
    def _parse_sessions(sessions: str) -> list[tuple[int, int]]:
        """将 `09:30-11:30` 形式的配置解析为分钟区间。"""
        windows: list[tuple[int, int]] = []
        for item in (sessions or "").split(","):
            block = item.strip()
            if not block or "-" not in block:
                continue
            start_text, end_text = block.split("-", 1)
            start = MarketSessionGuard._to_minutes(start_text)
            end = MarketSessionGuard._to_minutes(end_text)
            if start < end:
                windows.append((start, end))
        return windows

    @staticmethod
    def _to_minutes(hhmm: str) -> int:
        """将 `HH:MM` 文本转换为分钟数。"""
        hour, minute = hhmm.strip().split(":", 1)
        return int(hour) * 60 + int(minute)

    def is_market_open(self, now: datetime | None = None) -> bool:
        """判断当前时间是否落在配置的工作日交易时段内。"""
        now = now or datetime.now(self.timezone)
        if now.tzinfo is None:
            now = now.replace(tzinfo=self.timezone)
        else:
            now = now.astimezone(self.timezone)

        if now.weekday() >= 5:
            return False

        minutes = now.hour * 60 + now.minute
        return any(start <= minutes < end for start, end in self.sessions)


class AgentOrchestrator:
    """串行执行多智能体模拟交易周期的编排器。"""

    def __init__(
        self,
        config: Config | None = None,
        db_manager: DatabaseManager | None = None,
        data_agent: SupportsDataAgent | None = None,
        signal_agent: SupportsSignalAgent | None = None,
        risk_agent: SupportsRiskAgent | None = None,
        execution_agent: SupportsExecutionAgent | None = None,
        controller_agent: ControllerAgent | None = None,
        execution_repo: ExecutionRepository | Any | None = None,
        market_guard: SupportsMarketSessionGuard | None = None,
        now_provider: Callable[[], datetime] | None = None,
        sleep_func: Callable[[float], None] | None = None,
    ) -> None:
        """初始化各阶段智能体、仓储和时段控制器。"""
        self.config = config or get_config()
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = execution_repo or ExecutionRepository(self.db)

        self.data_agent = data_agent or DataAgent(config=self.config, db_manager=self.db)
        self.signal_agent = signal_agent or SignalAgent(
            config=self.config, db_manager=self.db, execution_repo=self.repo
        )
        self.risk_agent = risk_agent or RiskAgent(config=self.config)
        self.execution_agent = execution_agent or ExecutionAgent(
            config=self.config, db_manager=self.db, execution_repo=self.repo
        )
        self.controller_agent = controller_agent or ControllerAgent(config=self.config)

        self.market_guard = market_guard or MarketSessionGuard(
            timezone_name=str(getattr(self.config, "agent_market_timezone", "Asia/Shanghai")),
            sessions=str(getattr(self.config, "agent_market_sessions", "09:30-11:30,13:00-15:00")),
        )

        self._now = now_provider or (lambda: datetime.now(self.market_guard.timezone))
        self._sleep = sleep_func or time.sleep

    def run_cycle(
        self,
        stock_codes: list[str],
        *,
        mode: str,
        request_id: str | None = None,
        account_name: str | None = None,
        initial_cash_override: float | None = None,
        runtime_config: AgentRuntimeConfig | None = None,
        planning_context: dict[str, Any] | None = None,
        paper_order_submitter: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        stage_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentRunResult:
        """按顺序执行一次完整的多股票运行周期。"""
        stock_codes = [str(code or "").strip() for code in stock_codes if str(code or "").strip()]
        if not stock_codes:
            raise ValueError("stock_codes must not be empty")
        run_id = uuid.uuid4().hex
        started_at = self._now()
        trade_date = started_at.date()
        account_name = account_name or str(
            getattr(self.config, "agent_account_name", "paper-default") or "paper-default"
        )
        initial_cash = (
            float(initial_cash_override)
            if initial_cash_override is not None
            else float(getattr(self.config, "agent_initial_cash", 1_000_000.0))
        )
        working_account_snapshot = self._resolve_runtime_account_snapshot(
            runtime_config=runtime_config,
            account_name=account_name,
            initial_cash=initial_cash,
        )

        controller_context = ControllerContext(
            stock_codes=list(stock_codes),
            account_name=account_name,
            initial_cash=initial_cash,
            request_id=request_id,
            runtime_config=runtime_config,
            planning_context=planning_context,
            paper_order_submitter=paper_order_submitter,
        )
        controller_plan = self.controller_agent.build_plan(context=controller_context)
        if stage_observer is not None:
            stage_observer(
                {
                    "event": "supervisor_plan",
                    "goal": controller_plan.get("goal"),
                    "stock_codes": list(stock_codes),
                    "stage_priority": controller_plan.get("stage_priority"),
                    "include_runtime_context": controller_plan.get("include_runtime_context"),
                    "autonomous_execution_authorized": controller_plan.get("autonomous_execution_authorized"),
                    "tool_registry": controller_plan.get("tool_registry"),
                    "policy_snapshot": controller_plan.get("policy_snapshot"),
                }
            )

        # 这份账户快照会在同一轮多股票执行过程中不断递推，模拟同一账户连续处理多只股票。
        per_stock: list[StockAgentResult] = []
        stage_traces: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        decision_panel: list[dict[str, Any]] = []
        planner_trace: list[dict[str, Any]] = []
        condition_evaluations: list[dict[str, Any]] = []
        stock_termination_reasons: list[str] = []
        total_replans = 0
        execution_orders: list[dict[str, Any]] = []
        failed_orders: list[dict[str, Any]] = []

        for raw_code in stock_codes:
            logger.info("[run:%s][%s] controller stock start", run_id, raw_code)
            result, working_account_snapshot, stock_traces, stock_warnings = self.controller_agent.run_stock(
                code=raw_code,
                trade_date=trade_date,
                current_account_snapshot=self._normalize_account_snapshot(
                    working_account_snapshot,
                    account_name=account_name,
                    initial_cash=initial_cash,
                ),
                context=controller_context,
                controller_plan=controller_plan,
                data_agent=self.data_agent,
                signal_agent=self.signal_agent,
                risk_agent=self.risk_agent,
                execution_agent=self.execution_agent,
                stage_observer=stage_observer,
            )
            per_stock.append(result)
            stage_traces.extend(stock_traces)
            warnings.extend(stock_warnings)
            planner_trace.extend([dict(item) for item in result.planner_trace if isinstance(item, dict)])
            condition_evaluations.extend([dict(item) for item in result.condition_evaluations if isinstance(item, dict)])
            stock_termination_reasons.append(str(result.termination_reason or ""))
            total_replans += int(result.replan_count or 0)
            decision_panel.extend(
                [
                    {
                        "stock_code": trace.get("stock_code"),
                        "stage": trace.get("stage"),
                        "summary": trace.get("summary"),
                        "confidence": trace.get("confidence"),
                        "warnings": trace.get("warnings"),
                    }
                    for trace in stock_traces
                ]
            )
            if isinstance(result.execution_result, dict) and result.execution_result:
                status = str(result.execution_result.get("status") or "").strip().lower()
                if status in {"filled", "submitted"}:
                    execution_orders.append(dict(result.execution_result))
                else:
                    failed_orders.append(dict(result.execution_result))
            logger.info("[run:%s][%s] controller stock done", run_id, raw_code)

        ended_at = self._now()
        execution_result: dict[str, Any] | None = None
        if execution_orders or failed_orders:
            execution_result = {
                "mode": "single" if len(execution_orders) + len(failed_orders) == 1 else "batch",
                "executed_count": len(execution_orders),
                "failed_count": len(failed_orders),
                "orders": execution_orders,
                "failed_orders": failed_orders,
                "status": "filled" if execution_orders and not failed_orders else "partial" if execution_orders else "failed",
            }
        return AgentRunResult(
            run_id=run_id,
            mode=mode,
            started_at=started_at,
            ended_at=ended_at,
            trade_date=trade_date,
            results=per_stock,
            account_snapshot=working_account_snapshot,
            controller_plan=controller_plan,
            portfolio_decision={
                "stock_count": len(stock_codes),
                "executed_stock_count": sum(1 for item in per_stock if item.execution.action in {"buy", "sell"}),
                "warning_count": len(warnings),
            },
            warnings=warnings,
            stage_traces=stage_traces,
            decision_panel=decision_panel,
            planner_trace=planner_trace,
            condition_evaluations=condition_evaluations,
            termination_reason="execution_completed" if execution_orders else next((item for item in stock_termination_reasons if item), None),
            replan_count=total_replans,
            policy_snapshot=dict(controller_plan.get("policy_snapshot") or {}),
            execution_result=execution_result,
        )

    def run_once(
        self,
        stock_codes: list[str],
        *,
        request_id: str | None = None,
        account_name: str | None = None,
        initial_cash_override: float | None = None,
        runtime_config: AgentRuntimeConfig | None = None,
        planning_context: dict[str, Any] | None = None,
        paper_order_submitter: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        stage_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentRunResult:
        """执行一次不受交易时段约束的单轮运行。"""
        if account_name is None:
            kwargs = {
                "mode": "once",
                "request_id": request_id,
                "initial_cash_override": initial_cash_override,
                "runtime_config": runtime_config,
                "planning_context": planning_context,
                "stage_observer": stage_observer,
            }
            if paper_order_submitter is not None:
                kwargs["paper_order_submitter"] = paper_order_submitter
            return self.run_cycle(stock_codes, **kwargs)
        kwargs = {
            "mode": "once",
            "request_id": request_id,
            "account_name": account_name,
            "initial_cash_override": initial_cash_override,
            "runtime_config": runtime_config,
            "planning_context": planning_context,
            "stage_observer": stage_observer,
        }
        if paper_order_submitter is not None:
            kwargs["paper_order_submitter"] = paper_order_submitter
        return self.run_cycle(stock_codes, **kwargs)

    def run_realtime(
        self,
        stock_codes: list[str],
        *,
        interval_minutes: int,
        max_cycles: int | None = None,
        heartbeat_sleep: float = 5.0,
        request_id: str | None = None,
        account_name: str | None = None,
        initial_cash_override: float | None = None,
        runtime_config: AgentRuntimeConfig | None = None,
        planning_context: dict[str, Any] | None = None,
        paper_order_submitter: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        stage_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[AgentRunResult]:
        """按配置交易时段执行循环运行。"""
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive")

        results: list[AgentRunResult] = []
        cycles = 0
        next_run_at: datetime | None = None

        while True:
            now = self._now()
            if self.market_guard.is_market_open(now):
                if next_run_at is None or now >= next_run_at:
                    if account_name is None:
                        kwargs = {
                            "mode": "realtime",
                            "request_id": request_id,
                            "initial_cash_override": initial_cash_override,
                            "runtime_config": runtime_config,
                            "planning_context": planning_context,
                            "stage_observer": stage_observer,
                        }
                        if paper_order_submitter is not None:
                            kwargs["paper_order_submitter"] = paper_order_submitter
                        cycle_result = self.run_cycle(stock_codes, **kwargs)
                    else:
                        kwargs = {
                            "mode": "realtime",
                            "request_id": request_id,
                            "account_name": account_name,
                            "initial_cash_override": initial_cash_override,
                            "runtime_config": runtime_config,
                            "planning_context": planning_context,
                            "stage_observer": stage_observer,
                        }
                        if paper_order_submitter is not None:
                            kwargs["paper_order_submitter"] = paper_order_submitter
                        cycle_result = self.run_cycle(stock_codes, **kwargs)
                    results.append(cycle_result)
                    cycles += 1
                    next_run_at = self._next_aligned_time(now, interval_minutes)

                    if max_cycles is not None and cycles >= max_cycles:
                        break
                else:
                    self._sleep(max(0.0, min(heartbeat_sleep, (next_run_at - now).total_seconds())))
            else:
                self._sleep(heartbeat_sleep)

        return results

    @staticmethod
    def _next_aligned_time(now: datetime, interval_minutes: int) -> datetime:
        """向上取整到下一个固定间隔边界。"""
        interval_seconds = interval_minutes * 60
        epoch = int(now.timestamp())
        next_epoch = ((epoch // interval_seconds) + 1) * interval_seconds
        return datetime.fromtimestamp(next_epoch, tz=now.tzinfo)

    @staticmethod
    def _resolve_current_price(data_out) -> float:
        """优先实时价，其次今日收盘价，再退回昨收价。"""
        realtime_price = float(data_out.realtime_quote.get("price") or 0.0)
        if realtime_price > 0:
            return realtime_price

        today = data_out.analysis_context.get("today") if isinstance(data_out.analysis_context, dict) else {}
        fallback_price = float((today or {}).get("close") or 0.0)
        if fallback_price > 0:
            return fallback_price

        yesterday = data_out.analysis_context.get("yesterday") if isinstance(data_out.analysis_context, dict) else {}
        return float((yesterday or {}).get("close") or 0.0)

    @staticmethod
    def _current_position_value(account_snapshot, code: str) -> float:
        """读取某只股票当前持仓市值。"""
        for pos in account_snapshot.get("positions", []):
            if pos.get("code") == code:
                return float(pos.get("market_value") or 0.0)
        return 0.0

    @staticmethod
    def _as_number(value: Any, default: float = 0.0) -> float:
        """将输入安全解析为浮点数。"""
        try:
            num = float(value)
            if num != num:
                return default
            return num
        except Exception:
            return default

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        """将输入安全解析为整数。"""
        try:
            return int(float(value))
        except Exception:
            return default

    @classmethod
    def _normalize_positions(cls, value: Any) -> list[dict[str, Any]]:
        """统一不同来源的持仓结构。"""
        if not isinstance(value, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or item.get("stock_code") or item.get("symbol") or "").strip()
            if not code:
                continue
            quantity = max(0, cls._as_int(item.get("quantity") or item.get("qty") or item.get("volume"), 0))
            available_qty = max(
                0, cls._as_int(item.get("available_qty") or item.get("available") or quantity, quantity)
            )
            avg_cost = cls._as_number(item.get("avg_cost") or item.get("cost_price"), 0.0)
            last_price = cls._as_number(item.get("last_price") or item.get("price"), 0.0)
            market_value = cls._as_number(item.get("market_value"), quantity * last_price)
            if market_value <= 0 and quantity > 0 and last_price > 0:
                market_value = quantity * last_price
            normalized.append(
                {
                    "code": code,
                    "quantity": quantity,
                    "available_qty": min(available_qty, quantity),
                    "avg_cost": avg_cost,
                    "last_price": last_price,
                    "market_value": market_value,
                    "unrealized_pnl": cls._as_number(item.get("unrealized_pnl"), 0.0),
                }
            )
        return normalized

    @classmethod
    def _normalize_account_snapshot(
        cls,
        snapshot: dict[str, Any] | None,
        *,
        account_name: str,
        initial_cash: float,
    ) -> dict[str, Any]:
        """将账户快照归一化为执行链路使用的标准结构。"""
        raw = snapshot if isinstance(snapshot, dict) else {}
        positions = cls._normalize_positions(raw.get("positions"))
        inferred_market_value = sum(float(item.get("market_value") or 0.0) for item in positions)
        cash = cls._as_number(raw.get("cash"), initial_cash)
        total_market_value = cls._as_number(raw.get("total_market_value"), inferred_market_value)
        if total_market_value <= 0 and inferred_market_value > 0:
            total_market_value = inferred_market_value
        total_asset = cls._as_number(raw.get("total_asset"), cash + total_market_value)
        if total_asset <= 0:
            total_asset = cash + total_market_value

        return {
            "account_id": raw.get("account_id"),
            "name": str(raw.get("name") or account_name),
            "cash": round(cash, 4),
            "initial_cash": cls._as_number(raw.get("initial_cash"), initial_cash),
            "total_market_value": round(total_market_value, 4),
            "total_asset": round(total_asset, 4),
            "realized_pnl": cls._as_number(raw.get("realized_pnl"), 0.0),
            "unrealized_pnl": cls._as_number(raw.get("unrealized_pnl"), 0.0),
            "cumulative_fees": cls._as_number(raw.get("cumulative_fees"), 0.0),
            "positions": positions,
            "snapshot_at": raw.get("snapshot_at"),
            "data_source": raw.get("data_source"),
            "broker_account_id": raw.get("broker_account_id"),
            "provider_code": raw.get("provider_code"),
            "provider_name": raw.get("provider_name"),
            "account_uid": raw.get("account_uid"),
            "account_display_name": raw.get("account_display_name"),
        }

    @classmethod
    def _resolve_runtime_account_snapshot(
        cls,
        *,
        runtime_config: AgentRuntimeConfig | None,
        account_name: str,
        initial_cash: float,
    ) -> dict[str, Any]:
        """从运行时上下文中拼出初始账户快照。"""
        seed: dict[str, Any] = {}
        context = runtime_config.context if runtime_config else None
        if context:
            if isinstance(context.account_snapshot, dict):
                seed.update(context.account_snapshot)
            if isinstance(context.summary, dict):
                summary = context.summary
                if "cash" not in seed:
                    seed["cash"] = summary.get("cash") or summary.get("available_cash") or summary.get("availableCash")
                if "total_market_value" not in seed:
                    seed["total_market_value"] = (
                        summary.get("market_value") or summary.get("total_market_value") or summary.get("marketValue")
                    )
                if "total_asset" not in seed:
                    seed["total_asset"] = (
                        summary.get("total_asset") or summary.get("totalAsset") or summary.get("total_equity")
                    )
            if "positions" not in seed and isinstance(context.positions, list):
                seed["positions"] = [item for item in context.positions if isinstance(item, dict)]
        return cls._normalize_account_snapshot(seed, account_name=account_name, initial_cash=initial_cash)
