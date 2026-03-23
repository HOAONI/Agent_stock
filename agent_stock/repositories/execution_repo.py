# -*- coding: utf-8 -*-
"""执行链路仓储层。

它位于 ORM 表结构之上，向 Agent/Service 暴露更贴近业务的读写接口。涉及账户、
持仓、订单、成交、运行结果和异步任务的持久化时，通常都从这里进入。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, cast


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
from agent_stock.time_utils import local_now


def _mapped_int(value: Any) -> int:
    """将 ORM 读取值明确收窄为整数，便于静态分析识别实例字段。"""
    return int(cast(Any, value))


def _mapped_float(value: Any) -> float:
    """将 ORM 读取值明确收窄为浮点数，便于静态分析识别实例字段。"""
    return float(cast(Any, value))


class ExecutionRepository:
    """供 Risk/Execution/Orchestrator 使用的持久化层。"""

    def __init__(self, db_manager: DatabaseManager | None = None):
        """初始化数据库管理器。"""
        self.db = db_manager or DatabaseManager.get_instance()

    def get_or_create_account(self, name: str, initial_cash: float) -> PaperAccount:
        """按名称获取账户，若不存在则按初始资金创建。"""
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

    def get_account(self, name: str) -> PaperAccount | None:
        """按名称获取账户。"""
        with self.db.get_session() as session:
            return session.execute(
                select(PaperAccount).where(PaperAccount.name == name).limit(1)
            ).scalar_one_or_none()

    def get_account_snapshot(self, name: str) -> dict[str, Any]:
        """获取包含当前持仓的账户聚合值。"""
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

    def get_latest_runtime_account_snapshot(self, name: str) -> dict[str, Any]:
        """获取账户最新的持久化运行快照（轻量状态来源）。"""
        with self.db.get_session() as session:
            row = session.execute(
                select(AgentRun)
                .where(AgentRun.account_name == name)
                .order_by(desc(AgentRun.created_at))
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return {}
            snapshot = self._safe_loads(row.account_snapshot)
            if not snapshot:
                return {}
            # 运行时快照来自历史 AgentRun 记录，适合在轻量模式下快速恢复账户状态。
            payload = dict(snapshot)
            payload.setdefault("name", name)
            payload.setdefault(
                "snapshot_at",
                row.ended_at.isoformat() if row.ended_at else (row.created_at.isoformat() if row.created_at else None),
            )
            return payload

    def get_position(self, account_id: int, code: str) -> PaperPosition | None:
        """按账户和代码获取单个持仓。"""
        with self.db.get_session() as session:
            return session.execute(
                select(PaperPosition)
                .where(and_(PaperPosition.account_id == account_id, PaperPosition.code == code))
                .limit(1)
            ).scalar_one_or_none()

    def list_positions(self, account_id: int) -> list[PaperPosition]:
        """列出账户下的全部持仓。"""
        with self.db.get_session() as session:
            rows = session.execute(
                select(PaperPosition)
                .where(PaperPosition.account_id == account_id)
                .order_by(PaperPosition.code)
            ).scalars().all()
            return list(rows)

    def rollover_available_qty(self, account_id: int, trade_date: date) -> None:
        """在每个交易日内最多执行一次 T+1 结转。"""
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
                # 本地模拟账户把“总持仓 -> 当日可卖”结转拆到日初统一完成，避免重复放量。
                pos.available_qty = int(pos.quantity or 0)
                pos.updated_at = local_now()

            account.last_rollover_date = trade_date
            account.updated_at = local_now()
            session.commit()

    def mark_position_price(self, account_id: int, code: str, price: float) -> None:
        """在不改变数量的前提下更新单个持仓的最新价格。"""
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
            pos.updated_at = local_now()
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
    ) -> dict[str, Any]:
        """原子化持久化单笔订单/成交并更新账户与持仓。"""
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

            account_id = _mapped_int(account.id)
            cash_before = _mapped_float(account.cash or 0.0)
            quantity_before = _mapped_int(position.quantity) if position is not None else 0
            available_before = _mapped_int(position.available_qty) if position is not None else 0
            avg_cost_before = _mapped_float(position.avg_cost) if position is not None else 0.0

            order = PaperOrder(
                run_id=run_id,
                account_id=account_id,
                code=code,
                side=side,
                qty=int(qty),
                target_qty=int(target_qty),
                limit_price=float(fill_price),
                status="filled",
                reason=reason,
                created_at=local_now(),
            )
            session.add(order)
            session.flush()
            order_id = _mapped_int(order.id)

            gross_amount = float(fill_price) * float(qty)
            trade = PaperTrade(
                run_id=run_id,
                order_id=order_id,
                account_id=account_id,
                code=code,
                side=side,
                qty=int(qty),
                fill_price=float(fill_price),
                gross_amount=float(gross_amount),
                fee=float(fee),
                tax=float(tax),
                slippage_bps=float(slippage_bps),
                trade_date=trade_date,
                created_at=local_now(),
            )
            session.add(trade)

            if position is None:
                position = PaperPosition(
                    account_id=account_id,
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
            account.updated_at = local_now()

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

    def recompute_account_metrics(self, account_name: str, price_overrides: dict[str, float] | None = None) -> dict[str, Any]:
        """重新计算账户市值与未实现盈亏。"""
        with self.db.get_session() as session:
            account = session.execute(
                select(PaperAccount).where(PaperAccount.name == account_name).limit(1)
            ).scalar_one_or_none()
            if account is None:
                return {}

            self._recompute_account_metrics_in_session(session=session, account=account, price_overrides=price_overrides)
            session.commit()

        return self.get_account_snapshot(account_name)

    def add_funds(self, account_name: str, amount: float) -> dict[str, Any]:
        """原子化增加账户现金与初始资金。"""
        amount_num = float(amount)
        if amount_num <= 0:
            raise ValueError("amount must be > 0")

        with self.db.get_session() as session:
            account = session.execute(
                select(PaperAccount).where(PaperAccount.name == account_name).limit(1)
            ).scalar_one_or_none()
            if account is None:
                raise ValueError("account not found")

            cash_before = float(account.cash or 0.0)
            initial_cash_before = float(account.initial_cash or 0.0)

            account.cash = cash_before + amount_num
            account.initial_cash = initial_cash_before + amount_num
            account.updated_at = local_now()

            self._recompute_account_metrics_in_session(session=session, account=account)
            session.commit()

            return {
                "cash_before": cash_before,
                "cash_after": float(account.cash or 0.0),
                "initial_cash_before": initial_cash_before,
                "initial_cash_after": float(account.initial_cash or 0.0),
            }

    def _recompute_account_metrics_in_session(
        self,
        *,
        session,
        account: PaperAccount,
        price_overrides: dict[str, float] | None = None,
    ) -> None:
        """在打开会话中重新计算账户总计的内部辅助逻辑。"""
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
            pos.updated_at = local_now()

            total_mv += float(pos.market_value or 0.0)
            total_unrealized += float(pos.unrealized_pnl or 0.0)

        account.total_market_value = total_mv
        account.unrealized_pnl = total_unrealized
        account.total_asset = float(account.cash or 0.0) + total_mv
        account.updated_at = local_now()

    @staticmethod
    def _get_latest_close_price_in_session(session, code: str) -> float | None:
        """在当前会话内读取指定股票最近收盘价。"""
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
        stock_codes: list[str],
        account_name: str,
        status: str,
        data_snapshot: dict[str, Any],
        signal_snapshot: dict[str, Any],
        risk_snapshot: dict[str, Any],
        execution_snapshot: dict[str, Any],
        account_snapshot: dict[str, Any],
        report_path: str | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
    ) -> None:
        """持久化一次编排周期。"""
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
            created_at=local_now(),
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
                existing.created_at = local_now()
            session.commit()

    def get_agent_run(self, run_id: str) -> dict[str, Any]:
        """按 `run_id` 获取单条持久化运行记录。"""
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
        status: str | None = None,
        trade_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """列出最近的持久化运行记录。"""
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
        stock_codes: list[str],
        account_name: str,
        request_id: str | None = None,
        status: str = "pending",
    ) -> None:
        """持久化一条新的异步任务记录。"""
        with self.db.get_session() as session:
            row = AgentTask(
                task_id=task_id,
                request_id=request_id,
                status=status,
                stock_codes=",".join(stock_codes),
                account_name=account_name,
                run_id=None,
                error_message=None,
                created_at=local_now(),
                updated_at=local_now(),
            )
            session.add(row)
            session.commit()

    def get_agent_task(self, task_id: str) -> dict[str, Any]:
        """按 `task_id` 获取任务。"""
        with self.db.get_session() as session:
            row = session.execute(
                select(AgentTask).where(AgentTask.task_id == task_id).limit(1)
            ).scalar_one_or_none()
            if row is None:
                return {}
            return self._task_to_dict(row)

    def get_agent_task_by_request_id(self, request_id: str) -> dict[str, Any]:
        """按幂等 `request_id` 获取任务。"""
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
        status: str | None = None,
        run_id: str | None = None,
        error_message: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        """更新任务状态字段。"""
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
            row.updated_at = local_now()
            session.commit()

    def mark_inflight_tasks_failed(self, reason: str = "service_restarted") -> int:
        """将 pending/processing 任务标记为失败。"""
        with self.db.get_session() as session:
            rows = session.execute(
                select(AgentTask).where(AgentTask.status.in_(["pending", "processing"]))
            ).scalars().all()
            if not rows:
                return 0

            now = local_now()
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
        signal_payload: dict[str, Any],
        ai_payload: dict[str, Any],
    ) -> None:
        """按代码插入或更新每日信号缓存。"""
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
                    created_at=local_now(),
                    updated_at=local_now(),
                )
                session.add(row)
            else:
                row.signal_payload = self._safe_json(signal_payload)
                row.ai_payload = self._safe_json(ai_payload)
                row.updated_at = local_now()
            session.commit()

    def get_signal_snapshot(self, *, code: str, trade_date: date) -> dict[str, Any]:
        """获取按代码划分的每日信号缓存。"""
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

    def _run_to_dict(self, row: AgentRun) -> dict[str, Any]:
        """将运行记录 ORM 对象转换为接口字典。"""
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
    def _task_to_dict(row: AgentTask) -> dict[str, Any]:
        """将任务 ORM 对象转换为接口字典。"""
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
        """安全序列化字典载荷。"""
        try:
            return json.dumps(payload or {}, ensure_ascii=False)
        except Exception:
            return "{}"

    @staticmethod
    def _safe_loads(raw: str | None) -> dict[str, Any]:
        """安全反序列化字典载荷。"""
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            if isinstance(value, dict):
                return value
            return {}
        except Exception:
            return {}
