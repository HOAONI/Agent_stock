# -*- coding: utf-8 -*-
"""Execution Agent: converts target exposure into paper orders/fills."""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

from data_provider.base import canonical_stock_code

from agent_stock.agents.contracts import AgentState, ExecutionAgentOutput, RiskAgentOutput
from agent_stock.integrations.backend_bridge_client import BackendBridgeClient, BackendBridgeError
from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.storage import DatabaseManager
from src.config import Config, RuntimeExecutionConfig, get_config, redact_sensitive_payload, redact_sensitive_text

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """Paper trading executor with optional broker-bridge runtime context."""

    def __init__(
        self,
        config: Optional[Config] = None,
        db_manager: Optional[DatabaseManager] = None,
        execution_repo: Optional[ExecutionRepository] = None,
        bridge_client: Optional[BackendBridgeClient] = None,
    ) -> None:
        self.config = config or get_config()
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = execution_repo or ExecutionRepository(self.db)
        self.bridge_client = bridge_client or BackendBridgeClient.from_config(self.config)

    def run(
        self,
        *,
        run_id: str,
        code: str,
        trade_date: date,
        current_price: float,
        risk_output: RiskAgentOutput,
        account_name: Optional[str] = None,
        initial_cash_override: Optional[float] = None,
        runtime_execution: Optional[RuntimeExecutionConfig] = None,
        backend_task_id: Optional[str] = None,
    ) -> ExecutionAgentOutput:
        """Execute towards target notional using local paper fills with broker fallback."""
        normalized_code = canonical_stock_code(code)
        execution_mode = self._resolve_execution_mode(runtime_execution)
        broker_requested = execution_mode == "broker"
        broker_ticket_id = self._safe_positive_int(runtime_execution.ticket_id if runtime_execution else None)
        fallback_reason: Optional[str] = None
        bridge_exchange: Optional[Dict[str, Any]] = None

        if broker_requested:
            fallback_reason, bridge_exchange, broker_ticket_id = self._prepare_broker_context(
                runtime_execution=runtime_execution,
                backend_task_id=backend_task_id,
                broker_ticket_id=broker_ticket_id,
            )

        output = self._run_paper_execution(
            run_id=run_id,
            code=normalized_code,
            trade_date=trade_date,
            current_price=current_price,
            risk_output=risk_output,
            account_name=account_name,
            initial_cash_override=initial_cash_override,
        )

        output.execution_mode = execution_mode
        output.backend_task_id = backend_task_id
        output.broker_requested = broker_requested
        output.executed_via = "paper"
        output.broker_ticket_id = broker_ticket_id
        output.fallback_reason = fallback_reason

        if broker_requested and bridge_exchange is not None:
            self._post_execution_fallback_event(
                exchange_payload=bridge_exchange,
                backend_task_id=backend_task_id,
                fallback_reason=fallback_reason or "broker_contract_missing",
                execution_output=output,
            )

        return output

    def _prepare_broker_context(
        self,
        *,
        runtime_execution: Optional[RuntimeExecutionConfig],
        backend_task_id: Optional[str],
        broker_ticket_id: Optional[int],
    ) -> tuple[Optional[str], Optional[Dict[str, Any]], Optional[int]]:
        fallback_reason: Optional[str] = None
        bridge_exchange: Optional[Dict[str, Any]] = None
        credential_ticket = str(runtime_execution.credential_ticket or "").strip() if runtime_execution else ""

        if not credential_ticket:
            return "missing_credential_ticket", None, broker_ticket_id

        try:
            bridge_exchange = self.bridge_client.exchange_credential_ticket(credential_ticket)
            ticket_id = self._safe_positive_int(bridge_exchange.get("ticket_id"))
            if ticket_id is not None:
                broker_ticket_id = ticket_id
        except BackendBridgeError as exc:
            safe_error = redact_sensitive_text(str(exc))
            logger.warning("broker credential exchange failed: task_id=%s error=%s", backend_task_id, safe_error)
            fallback_reason = "credential_exchange_failed"
            return fallback_reason, None, broker_ticket_id
        except Exception as exc:
            safe_error = redact_sensitive_text(str(exc))
            logger.warning("broker credential exchange failed: task_id=%s error=%s", backend_task_id, safe_error)
            fallback_reason = "credential_exchange_failed"
            return fallback_reason, None, broker_ticket_id

        if not self._broker_order_contract_ready():
            fallback_reason = "broker_contract_missing"

        return fallback_reason, bridge_exchange, broker_ticket_id

    @staticmethod
    def _resolve_execution_mode(runtime_execution: Optional[RuntimeExecutionConfig]) -> str:
        mode = str(getattr(runtime_execution, "mode", "paper") or "paper").strip().lower()
        return mode if mode in {"paper", "broker"} else "paper"

    @staticmethod
    def _safe_positive_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _broker_order_contract_ready() -> bool:
        # Broker order gateway contract is not enabled yet in this repository.
        return False

    def _post_execution_fallback_event(
        self,
        *,
        exchange_payload: Dict[str, Any],
        backend_task_id: Optional[str],
        fallback_reason: str,
        execution_output: ExecutionAgentOutput,
    ) -> None:
        user_id = self._safe_positive_int(exchange_payload.get("user_id"))
        broker_account = exchange_payload.get("broker_account") if isinstance(exchange_payload, dict) else None
        broker_account_id = None
        if isinstance(broker_account, dict):
            broker_account_id = self._safe_positive_int(broker_account.get("id"))
        if broker_account_id is None:
            broker_account_id = self._safe_positive_int(exchange_payload.get("broker_account_id"))

        if user_id is None or broker_account_id is None:
            return

        payload = redact_sensitive_payload(
            {
                "backend_task_id": execution_output.backend_task_id,
                "code": execution_output.code,
                "execution_mode": execution_output.execution_mode,
                "executed_via": execution_output.executed_via,
                "action": execution_output.action,
                "traded_qty": execution_output.traded_qty,
                "fill_price": execution_output.fill_price,
                "reason": execution_output.reason,
                "fallback_reason": fallback_reason,
            }
        )

        try:
            self.bridge_client.post_execution_event(
                user_id=user_id,
                broker_account_id=broker_account_id,
                task_id=backend_task_id or str(exchange_payload.get("task_id") or ""),
                event_type="agent_execution_fallback",
                payload=payload if isinstance(payload, dict) else {},
                status="failed",
                error_code=fallback_reason,
            )
        except Exception as exc:
            logger.warning("post execution fallback event failed: %s", redact_sensitive_text(str(exc)))

    def _run_paper_execution(
        self,
        *,
        run_id: str,
        code: str,
        trade_date: date,
        current_price: float,
        risk_output: RiskAgentOutput,
        account_name: Optional[str],
        initial_cash_override: Optional[float],
    ) -> ExecutionAgentOutput:
        if current_price <= 0:
            return ExecutionAgentOutput(
                code=code,
                trade_date=trade_date,
                state=AgentState.FAILED,
                action="none",
                reason="invalid_price",
                error_message="current price is invalid",
            )

        resolved_account_name = account_name or str(getattr(self.config, "agent_account_name", "paper-default") or "paper-default")
        initial_cash = (
            float(initial_cash_override)
            if initial_cash_override is not None
            else float(getattr(self.config, "agent_initial_cash", 1_000_000.0))
        )

        account = self.repo.get_or_create_account(resolved_account_name, initial_cash)
        self.repo.rollover_available_qty(account.id, trade_date)
        account_snapshot = self.repo.recompute_account_metrics(resolved_account_name, {code: current_price})

        position = self._find_position(account_snapshot, code)
        current_qty = int(position.get("quantity", 0))
        available_qty = int(position.get("available_qty", 0))
        cash_before = float(account_snapshot.get("cash") or 0.0)

        lot = max(1, int(getattr(self.config, "agent_min_trade_lot", 100)))
        slippage_bps = float(getattr(self.config, "agent_slippage_bps", 5.0))
        fee_rate = float(getattr(self.config, "agent_fee_rate", 0.0003))
        sell_tax_rate = float(getattr(self.config, "agent_sell_tax_rate", 0.001))

        buy_price = current_price * (1.0 + slippage_bps / 10000.0)
        sell_price = current_price * (1.0 - slippage_bps / 10000.0)

        target_qty = self._target_qty(
            target_notional=float(risk_output.target_notional),
            effective_price=buy_price,
            lot=lot,
        )

        delta = target_qty - current_qty
        if delta == 0:
            self.repo.mark_position_price(account.id, code, current_price)
            refreshed = self.repo.recompute_account_metrics(resolved_account_name, {code: current_price})
            return ExecutionAgentOutput(
                code=code,
                trade_date=trade_date,
                state=AgentState.SKIPPED,
                action="none",
                reason="target_matched",
                target_qty=target_qty,
                traded_qty=0,
                cash_before=cash_before,
                cash_after=float(refreshed.get("cash") or cash_before),
                position_before=current_qty,
                position_after=current_qty,
                account_snapshot=refreshed,
            )

        if delta > 0:
            side = "buy"
            cost_per_share = buy_price * (1.0 + fee_rate)
            affordable_qty = self._target_qty(target_notional=cash_before, effective_price=cost_per_share, lot=lot)
            trade_qty = min(delta, affordable_qty)
            if trade_qty < lot:
                refreshed = self.repo.recompute_account_metrics(resolved_account_name, {code: current_price})
                return ExecutionAgentOutput(
                    code=code,
                    trade_date=trade_date,
                    state=AgentState.SKIPPED,
                    action="none",
                    reason="insufficient_cash",
                    target_qty=target_qty,
                    traded_qty=0,
                    cash_before=cash_before,
                    cash_after=float(refreshed.get("cash") or cash_before),
                    position_before=current_qty,
                    position_after=current_qty,
                    account_snapshot=refreshed,
                )
            fill_price = buy_price
        else:
            side = "sell"
            trade_qty = min(-delta, available_qty)
            trade_qty = (trade_qty // lot) * lot
            if trade_qty < lot:
                refreshed = self.repo.recompute_account_metrics(resolved_account_name, {code: current_price})
                return ExecutionAgentOutput(
                    code=code,
                    trade_date=trade_date,
                    state=AgentState.SKIPPED,
                    action="none",
                    reason="insufficient_available_qty",
                    target_qty=target_qty,
                    traded_qty=0,
                    cash_before=cash_before,
                    cash_after=float(refreshed.get("cash") or cash_before),
                    position_before=current_qty,
                    position_after=current_qty,
                    account_snapshot=refreshed,
                )
            fill_price = sell_price

        gross_amount = fill_price * float(trade_qty)
        fee = gross_amount * fee_rate
        tax = gross_amount * sell_tax_rate if side == "sell" else 0.0

        tx = self.repo.execute_fill(
            run_id=run_id,
            account_name=resolved_account_name,
            code=code,
            side=side,
            qty=trade_qty,
            target_qty=target_qty,
            fill_price=fill_price,
            fee=fee,
            tax=tax,
            slippage_bps=slippage_bps,
            reason="rebalance_to_target",
            trade_date=trade_date,
        )
        refreshed = self.repo.recompute_account_metrics(resolved_account_name, {code: current_price})

        return ExecutionAgentOutput(
            code=code,
            trade_date=trade_date,
            state=AgentState.READY,
            action=side,
            reason="executed",
            order_id=tx.get("order_id"),
            trade_id=tx.get("trade_id"),
            target_qty=target_qty,
            traded_qty=trade_qty,
            fill_price=round(fill_price, 4),
            fee=round(fee, 4),
            tax=round(tax, 4),
            cash_before=float(tx.get("cash_before") if tx.get("cash_before") is not None else cash_before),
            cash_after=float(tx.get("cash_after") if tx.get("cash_after") is not None else cash_before),
            position_before=int(tx.get("position_before") if tx.get("position_before") is not None else current_qty),
            position_after=int(tx.get("position_after") if tx.get("position_after") is not None else current_qty),
            account_snapshot=refreshed,
        )

    @staticmethod
    def _find_position(account_snapshot: Dict[str, Any], code: str) -> Dict[str, Any]:
        for item in account_snapshot.get("positions", []):
            if item.get("code") == code:
                return item
        return {}

    @staticmethod
    def _target_qty(target_notional: float, effective_price: float, lot: int) -> int:
        if target_notional <= 0 or effective_price <= 0 or lot <= 0:
            return 0
        raw_qty = math.floor(target_notional / effective_price)
        return (raw_qty // lot) * lot
