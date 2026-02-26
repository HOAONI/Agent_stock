# -*- coding: utf-8 -*-
"""Orchestrator for Data -> Signal -> Risk -> Execution agents."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime
from typing import Callable, List, Optional
from zoneinfo import ZoneInfo

from agent_stock.agents.contracts import AgentRunResult, StockAgentResult
from agent_stock.agents.data_agent import DataAgent
from agent_stock.agents.execution_agent import ExecutionAgent
from agent_stock.agents.risk_agent import RiskAgent
from agent_stock.agents.signal_agent import SignalAgent
from src.config import AgentRuntimeConfig, Config, get_config
from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.storage import DatabaseManager

logger = logging.getLogger(__name__)


class MarketSessionGuard:
    """Market-time guard for A-share realtime loops."""

    def __init__(self, timezone_name: str, sessions: str) -> None:
        self.timezone = ZoneInfo(timezone_name)
        self.sessions = self._parse_sessions(sessions)

    @staticmethod
    def _parse_sessions(sessions: str) -> List[tuple[int, int]]:
        windows: List[tuple[int, int]] = []
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
        hour, minute = hhmm.strip().split(":", 1)
        return int(hour) * 60 + int(minute)

    def is_market_open(self, now: Optional[datetime] = None) -> bool:
        """Return True if now falls in configured weekday sessions."""
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
    """Serial orchestrator for multi-agent paper trading cycles."""

    def __init__(
        self,
        config: Optional[Config] = None,
        db_manager: Optional[DatabaseManager] = None,
        data_agent: Optional[DataAgent] = None,
        signal_agent: Optional[SignalAgent] = None,
        risk_agent: Optional[RiskAgent] = None,
        execution_agent: Optional[ExecutionAgent] = None,
        execution_repo: Optional[ExecutionRepository] = None,
        market_guard: Optional[MarketSessionGuard] = None,
        now_provider: Optional[Callable[[], datetime]] = None,
        sleep_func: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.config = config or get_config()
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = execution_repo or ExecutionRepository(self.db)

        self.data_agent = data_agent or DataAgent(config=self.config, db_manager=self.db)
        self.signal_agent = signal_agent or SignalAgent(config=self.config, db_manager=self.db, execution_repo=self.repo)
        self.risk_agent = risk_agent or RiskAgent(config=self.config)
        self.execution_agent = execution_agent or ExecutionAgent(config=self.config, db_manager=self.db, execution_repo=self.repo)

        self.market_guard = market_guard or MarketSessionGuard(
            timezone_name=str(getattr(self.config, "agent_market_timezone", "Asia/Shanghai")),
            sessions=str(getattr(self.config, "agent_market_sessions", "09:30-11:30,13:00-15:00")),
        )

        self._now = now_provider or (lambda: datetime.now(self.market_guard.timezone))
        self._sleep = sleep_func or time.sleep

    def run_cycle(
        self,
        stock_codes: List[str],
        *,
        mode: str,
        request_id: Optional[str] = None,
        account_name: Optional[str] = None,
        initial_cash_override: Optional[float] = None,
        runtime_config: Optional[AgentRuntimeConfig] = None,
    ) -> AgentRunResult:
        """Run one serial cycle across all stock codes."""
        run_id = uuid.uuid4().hex
        started_at = self._now()
        trade_date = started_at.date()
        account_name = account_name or str(getattr(self.config, "agent_account_name", "paper-default") or "paper-default")
        initial_cash = (
            float(initial_cash_override)
            if initial_cash_override is not None
            else float(getattr(self.config, "agent_initial_cash", 1_000_000.0))
        )
        self.repo.get_or_create_account(account_name, initial_cash)

        per_stock: List[StockAgentResult] = []

        for raw_code in stock_codes:
            data_started = time.perf_counter()
            data_out = self.data_agent.run(raw_code)
            data_out.duration_ms = int((time.perf_counter() - data_started) * 1000)
            data_out.input = {"code": raw_code}
            data_out.output = {
                "state": data_out.state.value,
                "data_source": data_out.data_source,
                "has_analysis_context": bool(data_out.analysis_context),
                "has_realtime_quote": bool(data_out.realtime_quote),
            }

            signal_started = time.perf_counter()
            signal_out = self.signal_agent.run(data_out, runtime_config=runtime_config)
            signal_out.duration_ms = int((time.perf_counter() - signal_started) * 1000)
            signal_out.input = {
                "code": data_out.code,
                "trade_date": trade_date.isoformat(),
                "runtime_llm": bool(runtime_config and runtime_config.llm is not None),
            }
            signal_out.output = {
                "operation_advice": signal_out.operation_advice,
                "sentiment_score": signal_out.sentiment_score,
                "trend_signal": signal_out.trend_signal,
                "stop_loss": signal_out.stop_loss,
                "take_profit": signal_out.take_profit,
            }

            current_price = self._resolve_current_price(data_out)
            account_snapshot = self.repo.get_account_snapshot(account_name)
            current_position_value = self._current_position_value(account_snapshot, data_out.code)

            risk_started = time.perf_counter()
            risk_out = self.risk_agent.run(
                code=data_out.code,
                trade_date=trade_date,
                current_price=current_price,
                signal_output=signal_out,
                account_snapshot=account_snapshot,
                current_position_value=current_position_value,
                runtime_strategy=(runtime_config.strategy if runtime_config else None),
            )
            risk_out.duration_ms = int((time.perf_counter() - risk_started) * 1000)
            risk_out.input = {
                "code": data_out.code,
                "current_price": current_price,
                "operation_advice": signal_out.operation_advice,
                "runtime_strategy_applied": bool(runtime_config and runtime_config.strategy is not None),
                "current_position_value": current_position_value,
            }
            risk_out.output = {
                "target_weight": risk_out.target_weight,
                "target_notional": risk_out.target_notional,
                "risk_flags": risk_out.risk_flags,
                "effective_stop_loss": risk_out.effective_stop_loss,
                "effective_take_profit": risk_out.effective_take_profit,
                "position_cap_pct": risk_out.position_cap_pct,
                "strategy_applied": risk_out.strategy_applied,
            }

            execution_started = time.perf_counter()
            execution_out = self.execution_agent.run(
                run_id=run_id,
                code=data_out.code,
                trade_date=trade_date,
                current_price=current_price,
                risk_output=risk_out,
                account_name=account_name,
                initial_cash_override=initial_cash_override,
                runtime_execution=(runtime_config.execution if runtime_config else None),
                backend_task_id=request_id,
            )
            execution_out.duration_ms = int((time.perf_counter() - execution_started) * 1000)
            execution_out.input = {
                "code": data_out.code,
                "account_name": account_name,
                "backend_task_id": request_id,
                "execution_mode": execution_out.execution_mode,
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
                "executed_via": execution_out.executed_via,
                "broker_requested": execution_out.broker_requested,
                "broker_ticket_id": execution_out.broker_ticket_id,
                "fallback_reason": execution_out.fallback_reason,
            }

            per_stock.append(
                StockAgentResult(
                    code=data_out.code,
                    data=data_out,
                    signal=signal_out,
                    risk=risk_out,
                    execution=execution_out,
                )
            )

        ended_at = self._now()
        account_snapshot = self.repo.get_account_snapshot(account_name)
        return AgentRunResult(
            run_id=run_id,
            mode=mode,
            started_at=started_at,
            ended_at=ended_at,
            trade_date=trade_date,
            results=per_stock,
            account_snapshot=account_snapshot,
        )

    def run_once(
        self,
        stock_codes: List[str],
        *,
        request_id: Optional[str] = None,
        account_name: Optional[str] = None,
        initial_cash_override: Optional[float] = None,
        runtime_config: Optional[AgentRuntimeConfig] = None,
    ) -> AgentRunResult:
        """Run one cycle without market-time constraints."""
        if account_name is None:
            return self.run_cycle(
                stock_codes,
                mode="once",
                request_id=request_id,
                initial_cash_override=initial_cash_override,
                runtime_config=runtime_config,
            )
        return self.run_cycle(
            stock_codes,
            mode="once",
            request_id=request_id,
            account_name=account_name,
            initial_cash_override=initial_cash_override,
            runtime_config=runtime_config,
        )

    def run_realtime(
        self,
        stock_codes: List[str],
        *,
        interval_minutes: int,
        max_cycles: Optional[int] = None,
        heartbeat_sleep: float = 5.0,
        request_id: Optional[str] = None,
        account_name: Optional[str] = None,
        initial_cash_override: Optional[float] = None,
        runtime_config: Optional[AgentRuntimeConfig] = None,
    ) -> List[AgentRunResult]:
        """Run loop that executes cycles only during configured market sessions."""
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive")

        results: List[AgentRunResult] = []
        cycles = 0
        next_run_at: Optional[datetime] = None

        while True:
            now = self._now()
            if self.market_guard.is_market_open(now):
                if next_run_at is None or now >= next_run_at:
                    if account_name is None:
                        cycle_result = self.run_cycle(
                            stock_codes,
                            mode="realtime",
                            request_id=request_id,
                            initial_cash_override=initial_cash_override,
                            runtime_config=runtime_config,
                        )
                    else:
                        cycle_result = self.run_cycle(
                            stock_codes,
                            mode="realtime",
                            request_id=request_id,
                            account_name=account_name,
                            initial_cash_override=initial_cash_override,
                            runtime_config=runtime_config,
                        )
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
        """Round up to next interval boundary in local timezone."""
        interval_seconds = interval_minutes * 60
        epoch = int(now.timestamp())
        next_epoch = ((epoch // interval_seconds) + 1) * interval_seconds
        return datetime.fromtimestamp(next_epoch, tz=now.tzinfo)

    @staticmethod
    def _resolve_current_price(data_out) -> float:
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
        for pos in account_snapshot.get("positions", []):
            if pos.get("code") == code:
                return float(pos.get("market_value") or 0.0)
        return 0.0
