# -*- coding: utf-8 -*-
"""基于本地账本表实现的模拟交易运行时服务。"""

from __future__ import annotations

import math
import os
import random
from datetime import date
from typing import Any


from sqlalchemy import desc, select

from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.storage import DatabaseManager, PaperAccount, PaperOrder, PaperTrade
from agent_stock.time_utils import local_now
from data_provider import DataFetcherManager
from data_provider.base import canonical_stock_code, normalize_stock_code


def _as_dict(value: Any) -> dict[str, Any]:
    """将任意值安全转换为字典，不可转换时返回空字典。"""
    if isinstance(value, dict):
        return value
    return {}


def _as_positive_float(value: Any) -> float | None:
    """将输入解析为大于 0 的浮点数。"""
    try:
        num = float(value)
    except Exception:
        return None
    if not math.isfinite(num) or num <= 0:
        return None
    return num


def _as_non_negative_float(value: Any) -> float | None:
    """将输入解析为大于等于 0 的浮点数。"""
    try:
        num = float(value)
    except Exception:
        return None
    if not math.isfinite(num) or num < 0:
        return None
    return num


def _extract_price(quote: Any) -> float | None:
    """从字典或对象形式的实时行情中提取价格。"""
    if quote is None:
        return None
    if isinstance(quote, dict):
        return _as_positive_float(quote.get("price") or quote.get("current_price"))
    for key in ("price", "current_price"):
        if hasattr(quote, key):
            value = _as_positive_float(getattr(quote, key))
            if value is not None:
                return value
    return None


class BacktraderRuntimeService:
    """为 Backend_stock 提供本地模拟账户的运行时操作。"""

    def __init__(
        self,
        repo: ExecutionRepository | None = None,
        db_manager: DatabaseManager | None = None,
        fetcher_manager: DataFetcherManager | None = None,
    ):
        """初始化仓储、数据库和默认交易参数。"""
        self.repo = repo or ExecutionRepository()
        self.db = db_manager or DatabaseManager.get_instance()
        self.fetcher = fetcher_manager or DataFetcherManager()
        self.default_initial_capital = float(max(1.0, float(os.environ.get("BACKTRADER_DEFAULT_INITIAL_CAPITAL", "100000"))))
        self.default_commission = float(max(0.0, float(os.environ.get("BACKTRADER_DEFAULT_COMMISSION", "0.0003"))))
        self.default_slippage_bps = float(max(0.0, float(os.environ.get("BACKTRADER_DEFAULT_SLIPPAGE_BPS", "2"))))
        self.default_stamp_duty = float(max(0.0, float(os.environ.get("BACKTRADER_DEFAULT_STAMP_DUTY", "0.001"))))

    @staticmethod
    def account_name_from_broker_id(broker_account_id: int) -> str:
        """将 Backend 的 broker_account_id 映射为本地账户名。"""
        return f"bt-{int(broker_account_id)}"

    def _resolve_initial_capital(self, req: dict[str, Any]) -> float:
        """从多种兼容字段中解析初始资金。"""
        payload = _as_dict(req.get("payload"))
        credentials = _as_dict(req.get("credentials"))
        value = (
            req.get("initial_capital")
            or payload.get("initial_capital")
            or payload.get("initialCapital")
            or credentials.get("initial_capital")
            or credentials.get("initialCapital")
            or self.default_initial_capital
        )
        initial_capital = _as_positive_float(value)
        if initial_capital is None:
            raise ValueError("initial_capital must be > 0")
        return float(round(initial_capital, 2))

    def _resolve_commission_rate(self, req: dict[str, Any]) -> float:
        """解析手续费率，缺省时回退到服务默认值。"""
        payload = _as_dict(req.get("payload"))
        credentials = _as_dict(req.get("credentials"))
        value = (
            payload.get("commission_rate")
            or payload.get("commissionRate")
            or credentials.get("commission_rate")
            or credentials.get("commissionRate")
            or self.default_commission
        )
        rate = _as_non_negative_float(value)
        return float(rate if rate is not None else self.default_commission)

    def _resolve_slippage_bps(self, req: dict[str, Any]) -> float:
        """解析滑点基点数，缺省时回退到服务默认值。"""
        payload = _as_dict(req.get("payload"))
        credentials = _as_dict(req.get("credentials"))
        value = (
            payload.get("slippage_bps")
            or payload.get("slippageBps")
            or credentials.get("slippage_bps")
            or credentials.get("slippageBps")
            or self.default_slippage_bps
        )
        bps = _as_non_negative_float(value)
        return float(bps if bps is not None else self.default_slippage_bps)

    def _resolve_add_funds_amount(self, req: dict[str, Any]) -> float:
        """解析入金金额。"""
        payload = _as_dict(req.get("payload"))
        value = payload.get("amount") or req.get("amount")
        amount = _as_positive_float(value)
        if amount is None:
            raise ValueError("payload.amount must be > 0")
        return float(round(amount, 2))

    def _resolve_account(self, req: dict[str, Any]) -> PaperAccount:
        """按请求中的 broker_account_id 获取或创建本地账户。"""
        broker_account_id = int(req.get("broker_account_id") or 0)
        if broker_account_id <= 0:
            raise ValueError("broker_account_id must be >= 1")
        account_name = self.account_name_from_broker_id(broker_account_id)
        initial_capital = self._resolve_initial_capital(req)
        return self.repo.get_or_create_account(account_name, initial_capital)

    def _summary_payload(self, req: dict[str, Any]) -> dict[str, Any]:
        """构造账户汇总响应，必要时先重算账户指标。"""
        broker_account_id = int(req.get("broker_account_id") or 0)
        account_name = self.account_name_from_broker_id(broker_account_id)
        snapshot = self.repo.recompute_account_metrics(account_name)
        if not snapshot:
            account = self._resolve_account(req)
            snapshot = self.repo.get_account_snapshot(account.name)
        return {
            "broker_account_id": broker_account_id,
            "engine": "backtrader",
            "total_asset": float(snapshot.get("total_asset") or 0.0),
            "cash": float(snapshot.get("cash") or 0.0),
            "market_value": float(snapshot.get("total_market_value") or 0.0),
            "pnl_total": float(snapshot.get("realized_pnl") or 0.0) + float(snapshot.get("unrealized_pnl") or 0.0),
            "return_pct": (
                ((float(snapshot.get("total_asset") or 0.0) - float(snapshot.get("initial_cash") or 0.0))
                 / float(snapshot.get("initial_cash") or 1.0)) * 100.0
                if float(snapshot.get("initial_cash") or 0.0) > 0
                else 0.0
            ),
            "initial_capital": float(snapshot.get("initial_cash") or 0.0),
            "realized_pnl": float(snapshot.get("realized_pnl") or 0.0),
            "unrealized_pnl": float(snapshot.get("unrealized_pnl") or 0.0),
            "cumulative_fees": float(snapshot.get("cumulative_fees") or 0.0),
            "snapshot_at": local_now().isoformat(),
        }

    def provision_account(self, req: dict[str, Any]) -> dict[str, Any]:
        """为请求中的账户准备本地模拟账户。"""
        account = self._resolve_account(req)
        return {
            "verified": True,
            "engine": "backtrader",
            "account_name": account.name,
            "account_id": int(account.id),
            "initial_capital": float(account.initial_cash or 0.0),
            "message": "backtrader local account ready",
        }

    def get_account_summary(self, req: dict[str, Any]) -> dict[str, Any]:
        """返回账户汇总信息。"""
        return self._summary_payload(req)

    def add_funds(self, req: dict[str, Any]) -> dict[str, Any]:
        """向本地模拟账户追加资金。"""
        account = self._resolve_account(req)
        payload = _as_dict(req.get("payload"))
        amount = self._resolve_add_funds_amount(req)
        change = self.repo.add_funds(account.name, amount)
        summary = self._summary_payload(req)

        return {
            "account_name": account.name,
            "amount": amount,
            "cash_before": float(change.get("cash_before") or 0.0),
            "cash_after": float(change.get("cash_after") or 0.0),
            "initial_capital_before": float(change.get("initial_cash_before") or 0.0),
            "initial_capital_after": float(change.get("initial_cash_after") or 0.0),
            "note": str(payload.get("note") or "").strip() or None,
            "summary": summary,
            "snapshot_at": local_now().isoformat(),
        }

    def get_positions(self, req: dict[str, Any]) -> list[dict[str, Any]]:
        """读取账户当前持仓列表。"""
        broker_account_id = int(req.get("broker_account_id") or 0)
        account_name = self.account_name_from_broker_id(broker_account_id)
        snapshot = self.repo.recompute_account_metrics(account_name) or self.repo.get_account_snapshot(account_name)
        positions = snapshot.get("positions") or []
        items: list[dict[str, Any]] = []
        for row in positions:
            if not isinstance(row, dict):
                continue
            items.append(
                {
                    "stock_code": row.get("code"),
                    "quantity": int(row.get("quantity") or 0),
                    "available_qty": int(row.get("available_qty") or 0),
                    "avg_cost": float(row.get("avg_cost") or 0.0),
                    "last_price": float(row.get("last_price") or 0.0),
                    "market_value": float(row.get("market_value") or 0.0),
                    "unrealized_pnl": float(row.get("unrealized_pnl") or 0.0),
                },
            )
        return items

    def get_orders(self, req: dict[str, Any]) -> list[dict[str, Any]]:
        """读取账户最近的订单记录。"""
        account = self._resolve_account(req)
        with self.db.get_session() as session:
            rows = session.execute(
                select(PaperOrder)
                .where(PaperOrder.account_id == account.id)
                .order_by(desc(PaperOrder.created_at))
                .limit(200)
            ).scalars().all()
        return [
            {
                "order_id": f"bt-order-{row.id}",
                "code": row.code,
                "direction": row.side,
                "type": "market",
                "price": float(row.limit_price or 0.0),
                "quantity": int(row.qty or 0),
                "filled_quantity": int(row.qty or 0),
                "filled_price": float(row.limit_price or 0.0),
                "status": str(row.status or "filled"),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]

    def get_trades(self, req: dict[str, Any]) -> list[dict[str, Any]]:
        """读取账户最近的成交记录。"""
        account = self._resolve_account(req)
        with self.db.get_session() as session:
            rows = session.execute(
                select(PaperTrade)
                .where(PaperTrade.account_id == account.id)
                .order_by(desc(PaperTrade.created_at))
                .limit(200)
            ).scalars().all()
        return [
            {
                "trade_id": f"bt-trade-{row.id}",
                "order_id": f"bt-order-{row.order_id}",
                "stock_code": row.code,
                "direction": row.side,
                "price": float(row.fill_price or 0.0),
                "quantity": int(row.qty or 0),
                "amount": float(row.gross_amount or 0.0),
                "fee": float(row.fee or 0.0),
                "tax": float(row.tax or 0.0),
                "traded_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]

    def _resolve_price(self, stock_code: str, req: dict[str, Any], order_payload: dict[str, Any]) -> float:
        """优先使用显式价格；缺失时回退到实时行情。"""
        price = _as_positive_float(order_payload.get("price"))
        if price is not None:
            return price

        quote = None
        try:
            quote = self.fetcher.get_realtime_quote(stock_code)
        except Exception:
            quote = None
        quote_price = _extract_price(quote)
        if quote_price is None:
            raise ValueError("price is required when realtime quote is unavailable")
        return float(quote_price)

    def place_order(self, req: dict[str, Any]) -> dict[str, Any]:
        """在本地账本中模拟一次立即成交的市价单。"""
        account = self._resolve_account(req)
        order_payload = _as_dict(req.get("payload"))
        stock_code_raw = str(order_payload.get("stock_code") or "").strip()
        if not stock_code_raw:
            raise ValueError("payload.stock_code is required")
        stock_code = normalize_stock_code(canonical_stock_code(stock_code_raw))

        side = str(order_payload.get("direction") or order_payload.get("side") or "").strip().lower()
        if side not in {"buy", "sell"}:
            raise ValueError("payload.direction must be buy|sell")

        qty_value = _as_positive_float(order_payload.get("quantity"))
        if qty_value is None:
            raise ValueError("payload.quantity must be > 0")
        qty = int(max(1, math.floor(qty_value)))

        base_price = self._resolve_price(stock_code, req, order_payload)
        slippage_bps = self._resolve_slippage_bps(req)
        slip_ratio = slippage_bps / 10000.0
        # 统一在撮合价上施加滑点，确保和回测侧成本模型一致。
        fill_price = base_price * (1.0 + slip_ratio) if side == "buy" else max(0.01, base_price * (1.0 - slip_ratio))
        fill_price = float(round(fill_price, 4))

        commission_rate = self._resolve_commission_rate(req)
        gross_amount = fill_price * qty
        fee = float(round(gross_amount * commission_rate, 6))
        tax = float(round(gross_amount * self.default_stamp_duty, 6)) if side == "sell" else 0.0

        snapshot = self.repo.get_account_snapshot(account.name)
        cash = float(snapshot.get("cash") or 0.0)
        positions = snapshot.get("positions") or []
        pos_map = {
            str(item.get("code")): int(item.get("available_qty") or item.get("quantity") or 0)
            for item in positions
            if isinstance(item, dict)
        }

        if side == "buy":
            # 本地模拟账户按资金足额校验买入，不做融资融券。
            if cash + 1e-6 < gross_amount + fee:
                return {
                    "order_id": None,
                    "trade_id": None,
                    "status": "rejected",
                    "provider_status": "rejected",
                    "provider_order_id": None,
                    "filled_quantity": 0,
                    "filled_price": None,
                    "fee": 0.0,
                    "tax": 0.0,
                    "message": "可用资金不足",
                }
        else:
            # 卖出必须校验可用持仓，避免把总持仓误当成 T+0 可卖数量。
            available_qty = pos_map.get(stock_code, 0)
            if available_qty < qty:
                return {
                    "order_id": None,
                    "trade_id": None,
                    "status": "rejected",
                    "provider_status": "rejected",
                    "provider_order_id": None,
                    "filled_quantity": 0,
                    "filled_price": None,
                    "fee": 0.0,
                    "tax": 0.0,
                    "message": "可用持仓不足",
                }

        run_id = f"bt-{local_now().strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"
        result = self.repo.execute_fill(
            run_id=run_id,
            account_name=account.name,
            code=stock_code,
            side=side,
            qty=qty,
            target_qty=qty,
            fill_price=fill_price,
            fee=fee,
            tax=tax,
            slippage_bps=slippage_bps,
            reason="backtrader_local",
            trade_date=date.today(),
        )

        order_id = int(result.get("order_id") or 0)
        return {
            "order_id": order_id,
            "trade_id": int(result.get("trade_id") or 0),
            "status": "filled",
            "provider_status": "filled",
            "provider_order_id": f"bt-order-{order_id}",
            "filled_quantity": qty,
            "filled_price": fill_price,
            "fee": fee,
            "tax": tax,
            "cash_before": float(result.get("cash_before") or cash),
            "cash_after": float(result.get("cash_after") or 0.0),
            "position_before": int(result.get("position_before") or pos_map.get(stock_code, 0)),
            "position_after": int(result.get("position_after") or 0),
            "submitted_at": local_now().isoformat(),
            "message": "ok",
        }

    def cancel_order(self, req: dict[str, Any]) -> dict[str, Any]:
        """尝试取消未成交的本地模拟订单。"""
        account = self._resolve_account(req)
        payload = _as_dict(req.get("payload"))
        order_id_raw = str(payload.get("order_id") or "").strip()
        if not order_id_raw:
            raise ValueError("payload.order_id is required")

        numeric_id = order_id_raw
        if order_id_raw.startswith("bt-order-"):
            numeric_id = order_id_raw.replace("bt-order-", "", 1)
        if not numeric_id.isdigit():
            raise ValueError("payload.order_id is invalid")
        order_id = int(numeric_id)

        with self.db.get_session() as session:
            row = session.execute(
                select(PaperOrder)
                .where(PaperOrder.id == order_id, PaperOrder.account_id == account.id)
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return {
                    "order_id": order_id_raw,
                    "provider_order_id": order_id_raw,
                    "provider_status": "not_found",
                    "status": "not_found",
                    "cancelled_at": local_now().isoformat(),
                    "message": "order not found",
                }
            if str(row.status).lower() == "filled":
                return {
                    "order_id": order_id_raw,
                    "provider_order_id": order_id_raw,
                    "provider_status": "cannot_cancel_filled",
                    "status": "cannot_cancel_filled",
                    "cancelled_at": local_now().isoformat(),
                    "message": "filled order cannot be cancelled",
                }
            row.status = "cancelled"
            session.commit()

        return {
            "order_id": order_id_raw,
            "provider_order_id": order_id_raw,
            "provider_status": "cancelled",
            "status": "cancelled",
            "cancelled_at": local_now().isoformat(),
            "message": "cancelled",
        }


_backtrader_runtime_service: BacktraderRuntimeService | None = None


def get_backtrader_runtime_service() -> BacktraderRuntimeService:
    """返回运行时服务单例。"""
    global _backtrader_runtime_service
    if _backtrader_runtime_service is None:
        _backtrader_runtime_service = BacktraderRuntimeService()
    return _backtrader_runtime_service
