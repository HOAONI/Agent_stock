# -*- coding: utf-8 -*-
"""Execution Agent: simulation intent executor (no local ledger persistence)."""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, Optional

from data_provider.base import canonical_stock_code

from agent_stock.agents.contracts import AgentState, ExecutionAgentOutput, RiskAgentOutput
from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.storage import DatabaseManager
from src.config import Config, RuntimeExecutionConfig, get_config


class ExecutionAgent:
    """Generate execution intent under paper semantics using runtime account snapshot."""

    def __init__(
        self,
        config: Optional[Config] = None,
        db_manager: Optional[DatabaseManager] = None,
        execution_repo: Optional[ExecutionRepository] = None,
    ) -> None:
        self.config = config or get_config()
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = execution_repo or ExecutionRepository(self.db)

    def run(
        self,
        *,
        run_id: str,
        code: str,
        trade_date: date,
        current_price: float,
        risk_output: RiskAgentOutput,
        account_snapshot: Optional[Dict[str, Any]] = None,
        account_name: Optional[str] = None,
        initial_cash_override: Optional[float] = None,
        runtime_execution: Optional[RuntimeExecutionConfig] = None,
        backend_task_id: Optional[str] = None,
    ) -> ExecutionAgentOutput:
        """Execute intent against provided runtime snapshot without writing paper_* tables."""
        del run_id, runtime_execution  # retained for backward-compatible signature
        normalized_code = canonical_stock_code(code)
        resolved_account_name = account_name or str(getattr(self.config, "agent_account_name", "paper-default") or "paper-default")
        initial_cash = (
            float(initial_cash_override)
            if initial_cash_override is not None
            else float(getattr(self.config, "agent_initial_cash", 1_000_000.0))
        )

        output = self._run_paper_intent(
            code=normalized_code,
            trade_date=trade_date,
            current_price=current_price,
            risk_output=risk_output,
            account_name=resolved_account_name,
            initial_cash=initial_cash,
            account_snapshot=account_snapshot,
        )

        output.execution_mode = "paper"
        output.backend_task_id = backend_task_id
        output.broker_requested = False
        output.executed_via = "paper"
        output.broker_ticket_id = None
        output.fallback_reason = None
        return output

    def _run_paper_intent(
        self,
        *,
        code: str,
        trade_date: date,
        current_price: float,
        risk_output: RiskAgentOutput,
        account_name: str,
        initial_cash: float,
        account_snapshot: Optional[Dict[str, Any]],
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

        snapshot_before = self._normalize_snapshot(account_snapshot, account_name=account_name, initial_cash=initial_cash)
        position_before = self._find_position(snapshot_before, code)
        current_qty = int(position_before.get("quantity") or 0)
        cash_before = float(snapshot_before.get("cash") or 0.0)

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
            return ExecutionAgentOutput(
                code=code,
                trade_date=trade_date,
                state=AgentState.SKIPPED,
                action="none",
                reason="target_matched",
                target_qty=target_qty,
                traded_qty=0,
                cash_before=cash_before,
                cash_after=cash_before,
                position_before=current_qty,
                position_after=current_qty,
                account_snapshot=snapshot_before,
            )

        if delta > 0:
            side = "buy"
            affordable_qty = self._target_qty(
                target_notional=cash_before,
                effective_price=buy_price * (1.0 + fee_rate),
                lot=lot,
            )
            trade_qty = min((delta // lot) * lot, affordable_qty)
            if trade_qty < lot:
                return ExecutionAgentOutput(
                    code=code,
                    trade_date=trade_date,
                    state=AgentState.SKIPPED,
                    action="none",
                    reason="insufficient_cash",
                    target_qty=target_qty,
                    traded_qty=0,
                    cash_before=cash_before,
                    cash_after=cash_before,
                    position_before=current_qty,
                    position_after=current_qty,
                    account_snapshot=snapshot_before,
                )
            fill_price = buy_price
        else:
            side = "sell"
            trade_qty = min((-delta // lot) * lot, (current_qty // lot) * lot)
            if trade_qty < lot:
                return ExecutionAgentOutput(
                    code=code,
                    trade_date=trade_date,
                    state=AgentState.SKIPPED,
                    action="none",
                    reason="insufficient_position",
                    target_qty=target_qty,
                    traded_qty=0,
                    cash_before=cash_before,
                    cash_after=cash_before,
                    position_before=current_qty,
                    position_after=current_qty,
                    account_snapshot=snapshot_before,
                )
            fill_price = sell_price

        gross_amount = fill_price * float(trade_qty)
        fee = gross_amount * fee_rate
        tax = gross_amount * sell_tax_rate if side == "sell" else 0.0

        if side == "buy":
            cash_after = cash_before - gross_amount - fee
            position_after_qty = current_qty + trade_qty
            avg_cost_before = float(position_before.get("avg_cost") or 0.0)
            new_avg_cost = (
                ((avg_cost_before * current_qty) + (fill_price * trade_qty) + fee) / position_after_qty
                if position_after_qty > 0
                else 0.0
            )
        else:
            cash_after = cash_before + gross_amount - fee - tax
            position_after_qty = max(0, current_qty - trade_qty)
            new_avg_cost = float(position_before.get("avg_cost") or 0.0)

        snapshot_after = self._project_snapshot_after_trade(
            snapshot=snapshot_before,
            code=code,
            current_price=current_price,
            side=side,
            trade_qty=trade_qty,
            cash_after=cash_after,
            avg_cost_after=new_avg_cost,
            position_after_qty=position_after_qty,
        )

        return ExecutionAgentOutput(
            code=code,
            trade_date=trade_date,
            state=AgentState.READY,
            action=side,
            reason="intent_generated",
            target_qty=target_qty,
            traded_qty=trade_qty,
            fill_price=round(fill_price, 4),
            fee=round(fee, 4),
            tax=round(tax, 4),
            cash_before=round(cash_before, 4),
            cash_after=round(cash_after, 4),
            position_before=current_qty,
            position_after=position_after_qty,
            account_snapshot=snapshot_after,
        )

    @staticmethod
    def _as_number(value: Any, default: float = 0.0) -> float:
        try:
            num = float(value)
            if num != num:
                return default
            return num
        except Exception:
            return default

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    @classmethod
    def _normalize_positions(cls, value: Any) -> list[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        items: list[Dict[str, Any]] = []
        for raw in value:
            if not isinstance(raw, dict):
                continue
            code = str(raw.get("code") or raw.get("stock_code") or raw.get("symbol") or "").strip()
            if not code:
                continue
            qty = max(0, cls._as_int(raw.get("quantity") or raw.get("qty") or raw.get("volume"), 0))
            available_qty = max(0, cls._as_int(raw.get("available_qty") or raw.get("available") or qty, qty))
            avg_cost = cls._as_number(raw.get("avg_cost") or raw.get("cost_price"), 0.0)
            last_price = cls._as_number(raw.get("last_price") or raw.get("price"), 0.0)
            market_value = cls._as_number(raw.get("market_value"), qty * last_price)
            items.append(
                {
                    "code": code,
                    "quantity": qty,
                    "available_qty": min(available_qty, qty),
                    "avg_cost": avg_cost,
                    "last_price": last_price,
                    "market_value": market_value,
                    "unrealized_pnl": cls._as_number(raw.get("unrealized_pnl"), 0.0),
                }
            )
        return items

    @classmethod
    def _normalize_snapshot(
        cls,
        snapshot: Optional[Dict[str, Any]],
        *,
        account_name: str,
        initial_cash: float,
    ) -> Dict[str, Any]:
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

    @staticmethod
    def _find_position(account_snapshot: Dict[str, Any], code: str) -> Dict[str, Any]:
        for item in account_snapshot.get("positions", []):
            if str(item.get("code")) == str(code):
                return item
        return {}

    @classmethod
    def _project_snapshot_after_trade(
        cls,
        *,
        snapshot: Dict[str, Any],
        code: str,
        current_price: float,
        side: str,
        trade_qty: int,
        cash_after: float,
        avg_cost_after: float,
        position_after_qty: int,
    ) -> Dict[str, Any]:
        positions = [dict(item) for item in snapshot.get("positions", []) if isinstance(item, dict)]
        updated = False
        for idx, pos in enumerate(positions):
            if str(pos.get("code")) != str(code):
                continue
            updated = True
            if position_after_qty <= 0:
                positions.pop(idx)
                break
            pos["quantity"] = position_after_qty
            pos["available_qty"] = position_after_qty
            pos["avg_cost"] = round(avg_cost_after, 4)
            pos["last_price"] = round(current_price, 4)
            pos["market_value"] = round(float(position_after_qty) * current_price, 4)
            pos["unrealized_pnl"] = round((current_price - avg_cost_after) * float(position_after_qty), 4)
            break

        if not updated and side == "buy" and position_after_qty > 0:
            positions.append(
                {
                    "code": code,
                    "quantity": position_after_qty,
                    "available_qty": position_after_qty,
                    "avg_cost": round(avg_cost_after, 4),
                    "last_price": round(current_price, 4),
                    "market_value": round(float(position_after_qty) * current_price, 4),
                    "unrealized_pnl": round((current_price - avg_cost_after) * float(position_after_qty), 4),
                }
            )

        total_market_value = sum(float(item.get("market_value") or 0.0) for item in positions)
        total_asset = float(cash_after) + total_market_value

        next_snapshot = dict(snapshot)
        next_snapshot["cash"] = round(float(cash_after), 4)
        next_snapshot["positions"] = positions
        next_snapshot["total_market_value"] = round(total_market_value, 4)
        next_snapshot["total_asset"] = round(total_asset, 4)
        return next_snapshot

    @staticmethod
    def _target_qty(target_notional: float, effective_price: float, lot: int) -> int:
        if target_notional <= 0 or effective_price <= 0 or lot <= 0:
            return 0
        raw_qty = math.floor(target_notional / effective_price)
        return (raw_qty // lot) * lot
