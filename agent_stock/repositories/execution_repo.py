# -*- coding: utf-8 -*-
"""Repository for paper-trading execution, snapshots, and async tasks."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, select

from agent_stock.storage import (
    AgentRun,
    AgentSignalSnapshot,
    AgentTask,
    DatabaseManager,
    PaperAccount,
    PaperOrder,
    PaperPosition,
    PaperTrade,
    StockDaily,
)


class ExecutionRepository:
    """Persistence layer used by Risk/Execution/Orchestrator agents."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager.get_instance()

    def get_or_create_account(self, name: str, initial_cash: float) -> PaperAccount:
        """Get account by name or create with initial cash."""
        with self.db.get_session() as session:
            account = session.execute(
                select(PaperAccount).where(PaperAccount.name == name).limit(1)
            ).scalar_one_or_none()
            if account is not None:
                return account

            account = PaperAccount(
                name=name,
                initial_cash=float(initial_cash),
                cash=float(initial_cash),
                total_market_value=0.0,
                total_asset=float(initial_cash),
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                cumulative_fees=0.0,
                last_rollover_date=None,
            )
            session.add(account)
            session.commit()
            session.refresh(account)
            return account

    def get_account(self, name: str) -> Optional[PaperAccount]:
        """Get account by name."""
        with self.db.get_session() as session:
            return session.execute(
                select(PaperAccount).where(PaperAccount.name == name).limit(1)
            ).scalar_one_or_none()

    def get_account_snapshot(self, name: str) -> Dict[str, Any]:
        """Get account aggregate values with current positions."""
        with self.db.get_session() as session:
            account = session.execute(
                select(PaperAccount).where(PaperAccount.name == name).limit(1)
            ).scalar_one_or_none()
            if account is None:
                return {}

            positions = session.execute(
                select(PaperPosition).where(PaperPosition.account_id == account.id)
            ).scalars().all()
            return {
                "account_id": account.id,
                "name": account.name,
                "cash": float(account.cash or 0.0),
                "initial_cash": float(account.initial_cash or 0.0),
                "total_market_value": float(account.total_market_value or 0.0),
                "total_asset": float(account.total_asset or 0.0),
                "realized_pnl": float(account.realized_pnl or 0.0),
                "unrealized_pnl": float(account.unrealized_pnl or 0.0),
                "cumulative_fees": float(account.cumulative_fees or 0.0),
                "positions": [
                    {
                        "code": p.code,
                        "quantity": int(p.quantity or 0),
                        "available_qty": int(p.available_qty or 0),
                        "avg_cost": float(p.avg_cost or 0.0),
                        "last_price": float(p.last_price or 0.0),
                        "market_value": float(p.market_value or 0.0),
                        "unrealized_pnl": float(p.unrealized_pnl or 0.0),
                    }
                    for p in positions
                ],
            }

    def get_position(self, account_id: int, code: str) -> Optional[PaperPosition]:
        """Get one position by account/code."""
        with self.db.get_session() as session:
            return session.execute(
                select(PaperPosition)
                .where(and_(PaperPosition.account_id == account_id, PaperPosition.code == code))
                .limit(1)
            ).scalar_one_or_none()

    def list_positions(self, account_id: int) -> List[PaperPosition]:
        """List all positions for account."""
        with self.db.get_session() as session:
            rows = session.execute(
                select(PaperPosition)
                .where(PaperPosition.account_id == account_id)
                .order_by(PaperPosition.code)
            ).scalars().all()
            return list(rows)

    def rollover_available_qty(self, account_id: int, trade_date: date) -> None:
        """Apply T+1 rollover at most once per trade date."""
        with self.db.get_session() as session:
            account = session.execute(
                select(PaperAccount).where(PaperAccount.id == account_id).limit(1)
            ).scalar_one_or_none()
            if account is None:
                return

            if account.last_rollover_date == trade_date:
                return

            positions = session.execute(
                select(PaperPosition).where(PaperPosition.account_id == account_id)
            ).scalars().all()
            for pos in positions:
                pos.available_qty = int(pos.quantity or 0)
                pos.updated_at = datetime.now()

            account.last_rollover_date = trade_date
            account.updated_at = datetime.now()
            session.commit()

    def mark_position_price(self, account_id: int, code: str, price: float) -> None:
        """Update last price for one position without changing quantity."""
        with self.db.get_session() as session:
            pos = session.execute(
                select(PaperPosition)
                .where(and_(PaperPosition.account_id == account_id, PaperPosition.code == code))
                .limit(1)
            ).scalar_one_or_none()
            if pos is None:
                return
            pos.last_price = float(price)
            pos.market_value = float(price) * float(pos.quantity or 0)
            pos.unrealized_pnl = (float(price) - float(pos.avg_cost or 0.0)) * float(pos.quantity or 0)
            pos.updated_at = datetime.now()
            session.commit()

    def execute_fill(
        self,
        *,
        run_id: str,
        account_name: str,
        code: str,
        side: str,
        qty: int,
        target_qty: int,
        fill_price: float,
        fee: float,
        tax: float,
        slippage_bps: float,
        reason: str,
        trade_date: date,
    ) -> Dict[str, Any]:
        """Persist one order/fill and update account+position atomically."""
        with self.db.get_session() as session:
            account = session.execute(
                select(PaperAccount).where(PaperAccount.name == account_name).limit(1)
            ).scalar_one_or_none()
            if account is None:
                account = PaperAccount(
                    name=account_name,
                    initial_cash=0.0,
                    cash=0.0,
                    total_market_value=0.0,
                    total_asset=0.0,
                    realized_pnl=0.0,
                    unrealized_pnl=0.0,
                    cumulative_fees=0.0,
                )
                session.add(account)
                session.flush()

            position = session.execute(
                select(PaperPosition)
                .where(and_(PaperPosition.account_id == account.id, PaperPosition.code == code))
                .limit(1)
            ).scalar_one_or_none()

            cash_before = float(account.cash or 0.0)
            quantity_before = int(position.quantity) if position is not None else 0
            available_before = int(position.available_qty) if position is not None else 0
            avg_cost_before = float(position.avg_cost) if position is not None else 0.0

            order = PaperOrder(
                run_id=run_id,
                account_id=account.id,
                code=code,
                side=side,
                qty=int(qty),
                target_qty=int(target_qty),
                limit_price=float(fill_price),
                status="filled",
                reason=reason,
                created_at=datetime.now(),
            )
            session.add(order)
            session.flush()

            gross_amount = float(fill_price) * float(qty)
            trade = PaperTrade(
                run_id=run_id,
                order_id=order.id,
                account_id=account.id,
                code=code,
                side=side,
                qty=int(qty),
                fill_price=float(fill_price),
                gross_amount=float(gross_amount),
                fee=float(fee),
                tax=float(tax),
                slippage_bps=float(slippage_bps),
                trade_date=trade_date,
                created_at=datetime.now(),
            )
            session.add(trade)

            if position is None:
                position = PaperPosition(
                    account_id=account.id,
                    code=code,
                    quantity=0,
                    available_qty=0,
                    avg_cost=0.0,
                    last_price=float(fill_price),
                    market_value=0.0,
                    unrealized_pnl=0.0,
                )
                session.add(position)
                session.flush()

            if side == "buy":
                new_qty = quantity_before + int(qty)
                total_cost_before = avg_cost_before * float(quantity_before)
                total_cost_after = total_cost_before + gross_amount + float(fee)
                position.quantity = new_qty
                position.available_qty = available_before
                position.avg_cost = (total_cost_after / new_qty) if new_qty > 0 else 0.0
                position.last_price = float(fill_price)
                account.cash = cash_before - gross_amount - float(fee)
            else:
                sell_qty = int(qty)
                new_qty = max(0, quantity_before - sell_qty)
                position.quantity = new_qty
                position.available_qty = max(0, available_before - sell_qty)
                position.last_price = float(fill_price)
                account.cash = cash_before + gross_amount - float(fee) - float(tax)
                realized_delta = (float(fill_price) - avg_cost_before) * float(sell_qty) - float(fee) - float(tax)
                account.realized_pnl = float(account.realized_pnl or 0.0) + realized_delta
                if new_qty == 0:
                    position.avg_cost = 0.0
                if new_qty == 0 and position.available_qty == 0:
                    session.delete(position)

            account.cumulative_fees = float(account.cumulative_fees or 0.0) + float(fee) + float(tax)
            account.updated_at = datetime.now()

            self._recompute_account_metrics_in_session(
                session=session,
                account=account,
                price_overrides={code: float(fill_price)},
            )

            session.commit()

            return {
                "order_id": order.id,
                "trade_id": trade.id,
                "cash_before": cash_before,
                "cash_after": float(account.cash or 0.0),
                "position_before": quantity_before,
                "position_after": max(0, quantity_before + int(qty) if side == "buy" else quantity_before - int(qty)),
            }

    def recompute_account_metrics(self, account_name: str, price_overrides: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Recompute account market value and unrealized pnl."""
        with self.db.get_session() as session:
            account = session.execute(
                select(PaperAccount).where(PaperAccount.name == account_name).limit(1)
            ).scalar_one_or_none()
            if account is None:
                return {}

            self._recompute_account_metrics_in_session(session=session, account=account, price_overrides=price_overrides)
            session.commit()

        return self.get_account_snapshot(account_name)

    def _recompute_account_metrics_in_session(
        self,
        *,
        session,
        account: PaperAccount,
        price_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        """Internal helper to recompute account totals inside an open session."""
        price_overrides = price_overrides or {}
        positions = session.execute(
            select(PaperPosition).where(PaperPosition.account_id == account.id)
        ).scalars().all()

        total_mv = 0.0
        total_unrealized = 0.0

        for pos in positions:
            latest_price = float(price_overrides.get(pos.code) or pos.last_price or 0.0)
            if latest_price <= 0:
                latest_price = self._get_latest_close_price_in_session(session=session, code=pos.code) or 0.0

            pos.last_price = latest_price
            pos.market_value = latest_price * float(pos.quantity or 0)
            pos.unrealized_pnl = (latest_price - float(pos.avg_cost or 0.0)) * float(pos.quantity or 0)
            pos.updated_at = datetime.now()

            total_mv += float(pos.market_value or 0.0)
            total_unrealized += float(pos.unrealized_pnl or 0.0)

        account.total_market_value = total_mv
        account.unrealized_pnl = total_unrealized
        account.total_asset = float(account.cash or 0.0) + total_mv
        account.updated_at = datetime.now()

    @staticmethod
    def _get_latest_close_price_in_session(session, code: str) -> Optional[float]:
        row = session.execute(
            select(StockDaily)
            .where(StockDaily.code == code)
            .order_by(desc(StockDaily.date))
            .limit(1)
        ).scalar_one_or_none()
        if row is None or row.close is None:
            return None
        return float(row.close)

    def save_agent_run(
        self,
        *,
        run_id: str,
        mode: str,
        trade_date: date,
        stock_codes: List[str],
        account_name: str,
        status: str,
        data_snapshot: Dict[str, Any],
        signal_snapshot: Dict[str, Any],
        risk_snapshot: Dict[str, Any],
        execution_snapshot: Dict[str, Any],
        account_snapshot: Dict[str, Any],
        report_path: Optional[str] = None,
        error_message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> None:
        """Persist one orchestrator cycle."""
        payload = AgentRun(
            run_id=run_id,
            mode=mode,
            trade_date=trade_date,
            stock_codes=",".join(stock_codes),
            account_name=account_name,
            status=status,
            data_snapshot=self._safe_json(data_snapshot),
            signal_snapshot=self._safe_json(signal_snapshot),
            risk_snapshot=self._safe_json(risk_snapshot),
            execution_snapshot=self._safe_json(execution_snapshot),
            account_snapshot=self._safe_json(account_snapshot),
            report_path=report_path,
            error_message=error_message,
            started_at=started_at,
            ended_at=ended_at,
            created_at=datetime.now(),
        )

        with self.db.get_session() as session:
            existing = session.execute(
                select(AgentRun).where(AgentRun.run_id == run_id).limit(1)
            ).scalar_one_or_none()
            if existing is None:
                session.add(payload)
            else:
                existing.mode = payload.mode
                existing.trade_date = payload.trade_date
                existing.stock_codes = payload.stock_codes
                existing.account_name = payload.account_name
                existing.status = payload.status
                existing.data_snapshot = payload.data_snapshot
                existing.signal_snapshot = payload.signal_snapshot
                existing.risk_snapshot = payload.risk_snapshot
                existing.execution_snapshot = payload.execution_snapshot
                existing.account_snapshot = payload.account_snapshot
                existing.report_path = payload.report_path
                existing.error_message = payload.error_message
                existing.started_at = payload.started_at
                existing.ended_at = payload.ended_at
                existing.created_at = datetime.now()
            session.commit()

    def get_agent_run(self, run_id: str) -> Dict[str, Any]:
        """Get one persisted run by run_id."""
        with self.db.get_session() as session:
            row = session.execute(
                select(AgentRun).where(AgentRun.run_id == run_id).limit(1)
            ).scalar_one_or_none()
            if row is None:
                return {}
            return self._run_to_dict(row)

    def list_agent_runs(
        self,
        *,
        limit: int = 20,
        status: Optional[str] = None,
        trade_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """List recent persisted runs."""
        with self.db.get_session() as session:
            query = select(AgentRun)
            if status:
                query = query.where(AgentRun.status == status)
            if trade_date:
                query = query.where(AgentRun.trade_date == trade_date)

            rows = session.execute(
                query.order_by(desc(AgentRun.created_at)).limit(max(1, min(limit, 200)))
            ).scalars().all()
            return [self._run_to_dict(row) for row in rows]

    def create_agent_task(
        self,
        *,
        task_id: str,
        stock_codes: List[str],
        account_name: str,
        request_id: Optional[str] = None,
        status: str = "pending",
    ) -> None:
        """Persist a new async task row."""
        with self.db.get_session() as session:
            row = AgentTask(
                task_id=task_id,
                request_id=request_id,
                status=status,
                stock_codes=",".join(stock_codes),
                account_name=account_name,
                run_id=None,
                error_message=None,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            session.add(row)
            session.commit()

    def get_agent_task(self, task_id: str) -> Dict[str, Any]:
        """Get task by task_id."""
        with self.db.get_session() as session:
            row = session.execute(
                select(AgentTask).where(AgentTask.task_id == task_id).limit(1)
            ).scalar_one_or_none()
            if row is None:
                return {}
            return self._task_to_dict(row)

    def get_agent_task_by_request_id(self, request_id: str) -> Dict[str, Any]:
        """Get task by idempotency request_id."""
        if not request_id:
            return {}
        with self.db.get_session() as session:
            row = session.execute(
                select(AgentTask).where(AgentTask.request_id == request_id).limit(1)
            ).scalar_one_or_none()
            if row is None:
                return {}
            return self._task_to_dict(row)

    def update_agent_task(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        run_id: Optional[str] = None,
        error_message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """Update task state fields."""
        with self.db.get_session() as session:
            row = session.execute(
                select(AgentTask).where(AgentTask.task_id == task_id).limit(1)
            ).scalar_one_or_none()
            if row is None:
                return

            if status is not None:
                row.status = status
            if run_id is not None:
                row.run_id = run_id
            if error_message is not None:
                row.error_message = error_message
            if started_at is not None:
                row.started_at = started_at
            if completed_at is not None:
                row.completed_at = completed_at
            row.updated_at = datetime.now()
            session.commit()

    def mark_inflight_tasks_failed(self, reason: str = "service_restarted") -> int:
        """Mark pending/processing tasks as failed."""
        with self.db.get_session() as session:
            rows = session.execute(
                select(AgentTask).where(AgentTask.status.in_(["pending", "processing"]))
            ).scalars().all()
            if not rows:
                return 0

            now = datetime.now()
            for row in rows:
                row.status = "failed"
                row.error_message = reason
                row.completed_at = now
                row.updated_at = now
            session.commit()
            return len(rows)

    def upsert_signal_snapshot(
        self,
        *,
        code: str,
        trade_date: date,
        signal_payload: Dict[str, Any],
        ai_payload: Dict[str, Any],
    ) -> None:
        """Upsert per-code daily signal cache."""
        with self.db.get_session() as session:
            row = session.execute(
                select(AgentSignalSnapshot)
                .where(and_(AgentSignalSnapshot.code == code, AgentSignalSnapshot.trade_date == trade_date))
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                row = AgentSignalSnapshot(
                    code=code,
                    trade_date=trade_date,
                    signal_payload=self._safe_json(signal_payload),
                    ai_payload=self._safe_json(ai_payload),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
                session.add(row)
            else:
                row.signal_payload = self._safe_json(signal_payload)
                row.ai_payload = self._safe_json(ai_payload)
                row.updated_at = datetime.now()
            session.commit()

    def get_signal_snapshot(self, *, code: str, trade_date: date) -> Dict[str, Any]:
        """Get per-code daily signal cache."""
        with self.db.get_session() as session:
            row = session.execute(
                select(AgentSignalSnapshot)
                .where(and_(AgentSignalSnapshot.code == code, AgentSignalSnapshot.trade_date == trade_date))
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return {}
            return {
                "code": row.code,
                "trade_date": row.trade_date.isoformat() if row.trade_date else None,
                "signal_payload": self._safe_loads(row.signal_payload),
                "ai_payload": self._safe_loads(row.ai_payload),
            }

    def _run_to_dict(self, row: AgentRun) -> Dict[str, Any]:
        return {
            "run_id": row.run_id,
            "mode": row.mode,
            "trade_date": row.trade_date.isoformat() if row.trade_date else None,
            "stock_codes": [item for item in str(row.stock_codes or "").split(",") if item],
            "account_name": row.account_name,
            "status": row.status,
            "data_snapshot": self._safe_loads(row.data_snapshot),
            "signal_snapshot": self._safe_loads(row.signal_snapshot),
            "risk_snapshot": self._safe_loads(row.risk_snapshot),
            "execution_snapshot": self._safe_loads(row.execution_snapshot),
            "account_snapshot": self._safe_loads(row.account_snapshot),
            "report_path": row.report_path,
            "error_message": row.error_message,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "ended_at": row.ended_at.isoformat() if row.ended_at else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    @staticmethod
    def _task_to_dict(row: AgentTask) -> Dict[str, Any]:
        return {
            "task_id": row.task_id,
            "request_id": row.request_id,
            "status": row.status,
            "stock_codes": [item for item in str(row.stock_codes or "").split(",") if item],
            "account_name": row.account_name,
            "run_id": row.run_id,
            "error_message": row.error_message,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    @staticmethod
    def _safe_json(payload: Any) -> str:
        try:
            return json.dumps(payload or {}, ensure_ascii=False)
        except Exception:
            return "{}"

    @staticmethod
    def _safe_loads(raw: Optional[str]) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            if isinstance(value, dict):
                return value
            return {}
        except Exception:
            return {}
