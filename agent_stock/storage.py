# -*- coding: utf-8 -*-
"""Storage layer for Agent_stock runtime."""

from __future__ import annotations

import atexit
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    desc,
    select,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.config import get_config

logger = logging.getLogger(__name__)

Base = declarative_base()


class StockDaily(Base):
    """Daily OHLCV data used by DataAgent and account repricing."""

    __tablename__ = "stock_daily"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    amount = Column(Float)
    pct_chg = Column(Float)

    ma5 = Column(Float)
    ma10 = Column(Float)
    ma20 = Column(Float)
    volume_ratio = Column(Float)

    data_source = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        UniqueConstraint("code", "date", name="uix_code_date"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "date": self.date.isoformat() if self.date else None,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "amount": self.amount,
            "pct_chg": self.pct_chg,
            "ma5": self.ma5,
            "ma10": self.ma10,
            "ma20": self.ma20,
            "volume_ratio": self.volume_ratio,
            "data_source": self.data_source,
        }


class PaperAccount(Base):
    """Paper-trading account aggregate state."""

    __tablename__ = "paper_accounts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False, unique=True, index=True)

    initial_cash = Column(Float, nullable=False, default=0.0)
    cash = Column(Float, nullable=False, default=0.0)
    total_market_value = Column(Float, nullable=False, default=0.0)
    total_asset = Column(Float, nullable=False, default=0.0)

    realized_pnl = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    cumulative_fees = Column(Float, nullable=False, default=0.0)

    last_rollover_date = Column(Date)

    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)


class PaperPosition(Base):
    """Per-stock position state for a paper account."""

    __tablename__ = "paper_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(Integer, ForeignKey("paper_accounts.id"), nullable=False, index=True)

    code = Column(String(10), nullable=False, index=True)
    quantity = Column(Integer, nullable=False, default=0)
    available_qty = Column(Integer, nullable=False, default=0)

    avg_cost = Column(Float, nullable=False, default=0.0)
    last_price = Column(Float, nullable=False, default=0.0)
    market_value = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)

    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        UniqueConstraint("account_id", "code", name="uix_paper_position_account_code"),
    )


class PaperOrder(Base):
    """Paper-trading order ledger."""

    __tablename__ = "paper_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), index=True)
    account_id = Column(Integer, ForeignKey("paper_accounts.id"), nullable=False, index=True)

    code = Column(String(10), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    qty = Column(Integer, nullable=False, default=0)
    target_qty = Column(Integer, nullable=False, default=0)
    limit_price = Column(Float, nullable=False, default=0.0)
    status = Column(String(16), nullable=False, default="filled")
    reason = Column(String(255))

    created_at = Column(DateTime, default=datetime.now, nullable=False, index=True)


class PaperTrade(Base):
    """Paper-trading trade ledger."""

    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), index=True)
    order_id = Column(Integer, ForeignKey("paper_orders.id"), nullable=False, index=True)
    account_id = Column(Integer, ForeignKey("paper_accounts.id"), nullable=False, index=True)

    code = Column(String(10), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    qty = Column(Integer, nullable=False, default=0)
    fill_price = Column(Float, nullable=False, default=0.0)
    gross_amount = Column(Float, nullable=False, default=0.0)
    fee = Column(Float, nullable=False, default=0.0)
    tax = Column(Float, nullable=False, default=0.0)
    slippage_bps = Column(Float, nullable=False, default=0.0)
    trade_date = Column(Date, nullable=False, index=True)

    created_at = Column(DateTime, default=datetime.now, nullable=False, index=True)


class AgentRun(Base):
    """One orchestrator cycle snapshot."""

    __tablename__ = "agent_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, unique=True, index=True)
    mode = Column(String(16), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    stock_codes = Column(String(1000), nullable=False)
    account_name = Column(String(128), nullable=False, default="paper-default", index=True)
    status = Column(String(16), nullable=False, default="completed", index=True)

    data_snapshot = Column(Text)
    signal_snapshot = Column(Text)
    risk_snapshot = Column(Text)
    execution_snapshot = Column(Text)
    account_snapshot = Column(Text)
    report_path = Column(String(1024))
    error_message = Column(Text)

    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False, index=True)


class AgentSignalSnapshot(Base):
    """Daily AI/signal cache by stock code."""

    __tablename__ = "agent_signal_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)

    signal_payload = Column(Text)
    ai_payload = Column(Text)

    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        UniqueConstraint("code", "trade_date", name="uix_agent_signal_code_date"),
    )


class AgentTask(Base):
    """Persisted task state for asynchronous execution."""

    __tablename__ = "agent_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), nullable=False, unique=True, index=True)
    request_id = Column(String(128), nullable=True, unique=True, index=True)
    status = Column(String(16), nullable=False, default="pending", index=True)
    stock_codes = Column(String(1000), nullable=False)
    account_name = Column(String(128), nullable=False, default="paper-default", index=True)
    run_id = Column(String(64), nullable=True, index=True)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.now, nullable=False, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, index=True)


class DatabaseManager:
    """Singleton database manager used by Agent_stock."""

    _instance: Optional["DatabaseManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_url: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return

        if db_url is None:
            config = get_config()
            db_url = config.get_db_url()

        engine_kwargs: Dict[str, Any] = {
            "echo": False,
            "pool_pre_ping": True,
            "future": True,
        }
        if str(db_url).startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        self._engine = create_engine(db_url, **engine_kwargs)
        self._SessionLocal = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

        Base.metadata.create_all(self._engine)
        self._initialized = True
        self._recover_inflight_tasks()
        atexit.register(DatabaseManager._cleanup_engine, self._engine)

    @property
    def engine(self):
        """Expose SQLAlchemy engine for migration tooling."""
        return self._engine

    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        if cls._instance is not None:
            if hasattr(cls._instance, "_engine") and cls._instance._engine is not None:
                cls._instance._engine.dispose()
            cls._instance._initialized = False
            cls._instance = None

    @classmethod
    def _cleanup_engine(cls, engine) -> None:
        try:
            if engine is not None:
                engine.dispose()
        except Exception as exc:
            logger.debug("Engine cleanup skipped: %s", exc)

    def _recover_inflight_tasks(self) -> None:
        """Mark pending/processing tasks as failed when service restarts."""
        with self.get_session() as session:
            rows = session.execute(
                select(AgentTask).where(AgentTask.status.in_(["pending", "processing"]))
            ).scalars().all()
            if not rows:
                return
            now = datetime.now()
            for row in rows:
                row.status = "failed"
                row.error_message = "service_restarted"
                row.completed_at = now
                row.updated_at = now
            session.commit()

    def get_session(self) -> Session:
        if not getattr(self, "_initialized", False) or not hasattr(self, "_SessionLocal"):
            raise RuntimeError("DatabaseManager is not initialized")
        return self._SessionLocal()

    def ping(self) -> bool:
        """Check whether the database connection is healthy."""
        try:
            with self.get_session() as session:
                session.execute(select(1))
            return True
        except Exception:
            return False

    def has_today_data(self, code: str, target_date: Optional[date] = None) -> bool:
        if target_date is None:
            target_date = date.today()

        with self.get_session() as session:
            result = session.execute(
                select(StockDaily).where(
                    and_(
                        StockDaily.code == code,
                        StockDaily.date == target_date,
                    )
                )
            ).scalar_one_or_none()
            return result is not None

    def get_latest_data(self, code: str, days: int = 2) -> List[StockDaily]:
        with self.get_session() as session:
            rows = session.execute(
                select(StockDaily)
                .where(StockDaily.code == code)
                .order_by(desc(StockDaily.date))
                .limit(days)
            ).scalars().all()
            return list(rows)

    def save_daily_data(self, df: pd.DataFrame, code: str, data_source: str = "Unknown") -> int:
        if df is None or df.empty:
            return 0

        saved_count = 0
        with self.get_session() as session:
            try:
                for _, row in df.iterrows():
                    row_date = row.get("date")
                    if isinstance(row_date, str):
                        row_date = datetime.strptime(row_date, "%Y-%m-%d").date()
                    elif isinstance(row_date, datetime):
                        row_date = row_date.date()
                    elif isinstance(row_date, pd.Timestamp):
                        row_date = row_date.date()

                    existing = session.execute(
                        select(StockDaily).where(
                            and_(
                                StockDaily.code == code,
                                StockDaily.date == row_date,
                            )
                        )
                    ).scalar_one_or_none()

                    if existing is None:
                        session.add(
                            StockDaily(
                                code=code,
                                date=row_date,
                                open=row.get("open"),
                                high=row.get("high"),
                                low=row.get("low"),
                                close=row.get("close"),
                                volume=row.get("volume"),
                                amount=row.get("amount"),
                                pct_chg=row.get("pct_chg"),
                                ma5=row.get("ma5"),
                                ma10=row.get("ma10"),
                                ma20=row.get("ma20"),
                                volume_ratio=row.get("volume_ratio"),
                                data_source=data_source,
                            )
                        )
                        saved_count += 1
                    else:
                        existing.open = row.get("open")
                        existing.high = row.get("high")
                        existing.low = row.get("low")
                        existing.close = row.get("close")
                        existing.volume = row.get("volume")
                        existing.amount = row.get("amount")
                        existing.pct_chg = row.get("pct_chg")
                        existing.ma5 = row.get("ma5")
                        existing.ma10 = row.get("ma10")
                        existing.ma20 = row.get("ma20")
                        existing.volume_ratio = row.get("volume_ratio")
                        existing.data_source = data_source
                        existing.updated_at = datetime.now()

                session.commit()
            except Exception:
                session.rollback()
                raise

        return saved_count

    def get_analysis_context(
        self,
        code: str,
        target_date: Optional[date] = None,
        history_days: int = 60,
    ) -> Optional[Dict[str, Any]]:
        if target_date is None:
            target_date = date.today()

        recent_data = self.get_latest_data(code, days=2)
        if not recent_data:
            return None

        today_data = recent_data[0]
        yesterday_data = recent_data[1] if len(recent_data) > 1 else None

        context: Dict[str, Any] = {
            "code": code,
            "date": today_data.date.isoformat(),
            "today": today_data.to_dict(),
        }

        if yesterday_data is not None:
            context["yesterday"] = yesterday_data.to_dict()

            if yesterday_data.volume and yesterday_data.volume > 0:
                context["volume_change_ratio"] = round(today_data.volume / yesterday_data.volume, 2)

            if yesterday_data.close and yesterday_data.close > 0:
                context["price_change_ratio"] = round(
                    (today_data.close - yesterday_data.close) / yesterday_data.close * 100,
                    2,
                )

            context["ma_status"] = self._analyze_ma_status(today_data)

        bars = self.get_latest_data(code, days=max(2, min(int(history_days), 240)))
        if bars:
            raw_data: List[Dict[str, Any]] = []
            for item in reversed(bars):
                row = item.to_dict()
                row_date = row.get("date")
                if isinstance(row_date, date):
                    row["date"] = row_date.isoformat()
                raw_data.append(row)
            context["raw_data"] = raw_data

        return context

    def _analyze_ma_status(self, data: StockDaily) -> str:
        close = data.close or 0
        ma5 = data.ma5 or 0
        ma10 = data.ma10 or 0
        ma20 = data.ma20 or 0

        if close > ma5 > ma10 > ma20 > 0:
            return "多头排列 📈"
        if close < ma5 < ma10 < ma20 and ma20 > 0:
            return "空头排列 📉"
        if close > ma5 and ma5 > ma10:
            return "短期向好 🔼"
        return "震荡整理"


def get_db() -> DatabaseManager:
    """Return the singleton database manager."""
    return DatabaseManager.get_instance()
