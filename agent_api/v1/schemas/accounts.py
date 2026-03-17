# -*- coding: utf-8 -*-
"""Agent API 的账户相关数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PositionPayload(BaseModel):
    """账户快照中的持仓载荷。"""

    code: str
    quantity: int
    available_qty: int
    avg_cost: float
    last_price: float
    market_value: float
    unrealized_pnl: float


class AccountSnapshotResponse(BaseModel):
    """账户快照响应载荷。"""

    account_id: int | None = None
    name: str | None = None
    cash: float = 0.0
    initial_cash: float = 0.0
    total_market_value: float = 0.0
    total_asset: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    cumulative_fees: float = 0.0
    positions: list[PositionPayload] = Field(default_factory=list)
