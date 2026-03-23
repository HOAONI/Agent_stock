# -*- coding: utf-8 -*-
"""Agent_stock 的 ORM 存储层。

本文件集中定义数据库表结构和 `DatabaseManager` 单例。阅读存储链路时可以把
它理解为“事实来源”：运行结果、账户台账、任务状态、搜索情报和分析历史最终
都沉淀在这里。
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import re
from datetime import date, datetime
from typing import Any, ClassVar

import pandas as pd
from sqlalchemy import (
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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from agent_stock.config import get_config
from agent_stock.time_utils import local_now

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """全项目共享的 SQLAlchemy 2.0 Declarative Base。"""


class StockDaily(Base):
    """供 DataAgent、回测和账户重估使用的日线 OHLCV 数据。"""

    __tablename__ = "stock_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    amount: Mapped[float | None] = mapped_column(Float)
    pct_chg: Mapped[float | None] = mapped_column(Float)

    ma5: Mapped[float | None] = mapped_column(Float)
    ma10: Mapped[float | None] = mapped_column(Float)
    ma20: Mapped[float | None] = mapped_column(Float)
    volume_ratio: Mapped[float | None] = mapped_column(Float)

    data_source: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime | None] = mapped_column(DateTime, default=local_now)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, default=local_now, onupdate=local_now)

    __table_args__ = (UniqueConstraint("code", "date", name="uix_code_date"),)

    def to_dict(self) -> dict[str, Any]:
        """将日线记录转换为字典。"""
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


class NewsIntel(Base):
    """持久化的新闻/搜索情报，供后续分析复盘使用。"""

    __tablename__ = "news_intel"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[str | None] = mapped_column(String(64), index=True)
    code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(50))
    dimension: Mapped[str | None] = mapped_column(String(32), index=True)
    query: Mapped[str | None] = mapped_column(String(255))
    provider: Mapped[str | None] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    snippet: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String(1000), nullable=False)
    source: Mapped[str | None] = mapped_column(String(100))
    published_date: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime, default=local_now, index=True)
    query_source: Mapped[str | None] = mapped_column(String(32), index=True)
    requester_platform: Mapped[str | None] = mapped_column(String(20))
    requester_user_id: Mapped[str | None] = mapped_column(String(64))
    requester_user_name: Mapped[str | None] = mapped_column(String(64))
    requester_chat_id: Mapped[str | None] = mapped_column(String(64))
    requester_message_id: Mapped[str | None] = mapped_column(String(64))
    requester_query: Mapped[str | None] = mapped_column(String(255))

    __table_args__ = (UniqueConstraint("url", name="uix_news_url"),)


class AnalysisHistory(Base):
    """单次股票分析请求的持久化结果。"""

    __tablename__ = "analysis_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[str | None] = mapped_column(String(64), index=True)
    code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(50))
    report_type: Mapped[str | None] = mapped_column(String(16), index=True)
    sentiment_score: Mapped[int | None] = mapped_column(Integer)
    operation_advice: Mapped[str | None] = mapped_column(String(20))
    trend_prediction: Mapped[str | None] = mapped_column(String(50))
    analysis_summary: Mapped[str | None] = mapped_column(Text)
    raw_result: Mapped[str | None] = mapped_column(Text)
    news_content: Mapped[str | None] = mapped_column(Text)
    context_snapshot: Mapped[str | None] = mapped_column(Text)
    ideal_buy: Mapped[float | None] = mapped_column(Float)
    secondary_buy: Mapped[float | None] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float)
    take_profit: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, default=local_now, index=True)


class PaperAccount(Base):
    """模拟交易账户的聚合状态。"""

    __tablename__ = "paper_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)

    initial_cash: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cash: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_market_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_asset: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cumulative_fees: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    last_rollover_date: Mapped[date | None] = mapped_column(Date)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, onupdate=local_now, nullable=False)


class PaperPosition(Base):
    """模拟交易账户下单只股票的持仓状态。"""

    __tablename__ = "paper_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, ForeignKey("paper_accounts.id"), nullable=False, index=True)

    code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    available_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    avg_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    market_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, onupdate=local_now, nullable=False)

    __table_args__ = (UniqueConstraint("account_id", "code", name="uix_paper_position_account_code"),)


class PaperOrder(Base):
    """模拟交易订单台账。"""

    __tablename__ = "paper_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    account_id: Mapped[int] = mapped_column(Integer, ForeignKey("paper_accounts.id"), nullable=False, index=True)

    code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    target_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    limit_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="filled")
    reason: Mapped[str | None] = mapped_column(String(255))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, nullable=False, index=True)


class PaperTrade(Base):
    """模拟交易成交台账。"""

    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    order_id: Mapped[int] = mapped_column(Integer, ForeignKey("paper_orders.id"), nullable=False, index=True)
    account_id: Mapped[int] = mapped_column(Integer, ForeignKey("paper_accounts.id"), nullable=False, index=True)

    code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fill_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    gross_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tax: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    slippage_bps: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, nullable=False, index=True)


class AgentRun(Base):
    """一次编排运行周期的快照记录。"""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    stock_codes: Mapped[str] = mapped_column(String(1000), nullable=False)
    account_name: Mapped[str] = mapped_column(String(128), nullable=False, default="paper-default", index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="completed", index=True)

    data_snapshot: Mapped[str | None] = mapped_column(Text)
    signal_snapshot: Mapped[str | None] = mapped_column(Text)
    risk_snapshot: Mapped[str | None] = mapped_column(Text)
    execution_snapshot: Mapped[str | None] = mapped_column(Text)
    account_snapshot: Mapped[str | None] = mapped_column(Text)
    report_path: Mapped[str | None] = mapped_column(String(1024))
    error_message: Mapped[str | None] = mapped_column(Text)

    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, nullable=False, index=True)


class AgentSignalSnapshot(Base):
    """按股票和交易日缓存的 AI/信号快照。"""

    __tablename__ = "agent_signal_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    signal_payload: Mapped[str | None] = mapped_column(Text)
    ai_payload: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, onupdate=local_now, nullable=False)

    __table_args__ = (UniqueConstraint("code", "trade_date", name="uix_agent_signal_code_date"),)


class AgentTask(Base):
    """异步执行任务的持久化状态。"""

    __tablename__ = "agent_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    stock_codes: Mapped[str] = mapped_column(String(1000), nullable=False)
    account_name: Mapped[str] = mapped_column(String(128), nullable=False, default="paper-default", index=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, nullable=False, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=local_now, onupdate=local_now, nullable=False, index=True)


class DatabaseManager:
    """Agent_stock 使用的数据库管理器单例。"""

    _instance: ClassVar[DatabaseManager | None] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> DatabaseManager:
        """确保 DatabaseManager 维持单例。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        assert cls._instance is not None
        return cls._instance

    def __init__(self, db_url: str | None = None):
        """初始化数据库引擎、会话工厂和基础表结构。"""
        if getattr(self, "_initialized", False):
            return

        if db_url is None:
            config = get_config()
            db_url = config.get_db_url()

        engine_kwargs: dict[str, Any] = {
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
        """向迁移工具暴露 SQLAlchemy engine。"""
        return self._engine

    @classmethod
    def get_instance(cls) -> DatabaseManager:
        """返回数据库管理器单例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置数据库管理器单例。"""
        if cls._instance is not None:
            if hasattr(cls._instance, "_engine") and cls._instance._engine is not None:
                cls._instance._engine.dispose()
            cls._instance._initialized = False
            cls._instance = None

    @classmethod
    def _cleanup_engine(cls, engine) -> None:
        """在进程退出时尽力释放数据库引擎资源。"""
        try:
            if engine is not None:
                engine.dispose()
        except Exception as exc:
            logger.debug("Engine cleanup skipped: %s", exc)

    def _recover_inflight_tasks(self) -> None:
        """在服务重启时将 pending/processing 任务标记为 failed。"""
        with self.get_session() as session:
            rows = session.execute(
                select(AgentTask).where(AgentTask.status.in_(["pending", "processing"]))
            ).scalars().all()
            if not rows:
                return
            now = local_now()
            for row in rows:
                row.status = "failed"
                row.error_message = "service_restarted"
                row.completed_at = now
                row.updated_at = now
            session.commit()

    def get_session(self) -> Session:
        """创建一个新的 SQLAlchemy 会话。"""
        if not getattr(self, "_initialized", False) or not hasattr(self, "_SessionLocal"):
            raise RuntimeError("DatabaseManager is not initialized")
        return self._SessionLocal()

    def ping(self) -> bool:
        """检查数据库连接是否健康。"""
        try:
            with self.get_session() as session:
                session.execute(select(1))
            return True
        except Exception:
            return False

    def has_today_data(self, code: str, target_date: date | None = None) -> bool:
        """判断指定股票在目标日期是否已有日线数据。"""
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

    def get_latest_data(self, code: str, days: int = 2) -> list[StockDaily]:
        """读取指定股票最近若干天的日线数据。"""
        with self.get_session() as session:
            rows = session.execute(
                select(StockDaily)
                .where(StockDaily.code == code)
                .order_by(desc(StockDaily.date))
                .limit(days)
            ).scalars().all()
            return list(rows)

    def save_daily_data(self, df: pd.DataFrame, code: str, data_source: str = "Unknown") -> int:
        """批量写入或更新日线数据。"""
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
                        existing.updated_at = local_now()

                session.commit()
            except Exception:
                session.rollback()
                raise

        return saved_count

    def save_news_intel(
        self,
        code: str,
        name: str,
        dimension: str,
        query: str,
        response: Any,
        query_context: dict[str, str] | None = None,
    ) -> int:
        """基于 URL 去重并持久化单条搜索响应载荷。"""
        if not response or not getattr(response, "results", None):
            return 0

        saved_count = 0
        query_ctx = query_context or {}
        current_query_id = str(query_ctx.get("query_id") or "").strip()

        with self.get_session() as session:
            try:
                for item in response.results:
                    title = str(getattr(item, "title", "") or "").strip()
                    url = str(getattr(item, "url", "") or "").strip()
                    source = str(getattr(item, "source", "") or "").strip()
                    snippet = str(getattr(item, "snippet", "") or "").strip()
                    published_date = self._parse_published_date(getattr(item, "published_date", None))
                    if not title and not url:
                        continue

                    url_key = url or self._build_fallback_url_key(
                        code=code,
                        title=title,
                        source=source,
                        published_date=published_date,
                    )

                    existing = session.execute(select(NewsIntel).where(NewsIntel.url == url_key)).scalar_one_or_none()
                    if existing is not None:
                        existing.name = name or existing.name
                        existing.dimension = dimension or existing.dimension
                        existing.query = query or existing.query
                        existing.provider = getattr(response, "provider", None) or existing.provider
                        existing.snippet = snippet or existing.snippet
                        existing.source = source or existing.source
                        existing.published_date = published_date or existing.published_date
                        existing.fetched_at = local_now()
                        if query_context:
                            if not existing.query_id and current_query_id:
                                existing.query_id = current_query_id
                            existing.query_source = query_ctx.get("query_source") or existing.query_source
                            existing.requester_platform = query_ctx.get("requester_platform") or existing.requester_platform
                            existing.requester_user_id = query_ctx.get("requester_user_id") or existing.requester_user_id
                            existing.requester_user_name = query_ctx.get("requester_user_name") or existing.requester_user_name
                            existing.requester_chat_id = query_ctx.get("requester_chat_id") or existing.requester_chat_id
                            existing.requester_message_id = (
                                query_ctx.get("requester_message_id") or existing.requester_message_id
                            )
                            existing.requester_query = query_ctx.get("requester_query") or existing.requester_query
                        continue

                    try:
                        with session.begin_nested():
                            session.add(
                                NewsIntel(
                                    code=code,
                                    name=name,
                                    dimension=dimension,
                                    query=query,
                                    provider=getattr(response, "provider", None),
                                    title=title,
                                    snippet=snippet,
                                    url=url_key,
                                    source=source,
                                    published_date=published_date,
                                    fetched_at=local_now(),
                                    query_id=current_query_id or None,
                                    query_source=query_ctx.get("query_source"),
                                    requester_platform=query_ctx.get("requester_platform"),
                                    requester_user_id=query_ctx.get("requester_user_id"),
                                    requester_user_name=query_ctx.get("requester_user_name"),
                                    requester_chat_id=query_ctx.get("requester_chat_id"),
                                    requester_message_id=query_ctx.get("requester_message_id"),
                                    requester_query=query_ctx.get("requester_query"),
                                )
                            )
                            session.flush()
                        saved_count += 1
                    except IntegrityError:
                        logger.debug("Duplicate news intel skipped: %s %s", code, url_key)

                session.commit()
            except Exception:
                session.rollback()
                raise

        return saved_count

    def save_analysis_history(
        self,
        result: Any,
        query_id: str,
        report_type: str,
        news_content: str | None,
        context_snapshot: dict[str, Any] | None = None,
        save_snapshot: bool = True,
    ) -> int:
        """为审计与复盘持久化单条分析结果。"""
        if result is None:
            return 0

        sniper_points = self._extract_sniper_points(result)
        context_text = None
        if save_snapshot and context_snapshot is not None:
            context_text = self._safe_json_dumps(context_snapshot)

        record = AnalysisHistory(
            query_id=query_id,
            code=result.code,
            name=result.name,
            report_type=report_type,
            sentiment_score=result.sentiment_score,
            operation_advice=result.operation_advice,
            trend_prediction=result.trend_prediction,
            analysis_summary=result.analysis_summary,
            raw_result=self._safe_json_dumps(self._build_raw_result(result)),
            news_content=news_content,
            context_snapshot=context_text,
            ideal_buy=sniper_points.get("ideal_buy"),
            secondary_buy=sniper_points.get("secondary_buy"),
            stop_loss=sniper_points.get("stop_loss"),
            take_profit=sniper_points.get("take_profit"),
            created_at=local_now(),
        )

        with self.get_session() as session:
            try:
                session.add(record)
                session.commit()
                return 1
            except Exception:
                session.rollback()
                logger.exception("Failed to save analysis history")
                return 0

    def get_analysis_context(
        self,
        code: str,
        target_date: date | None = None,
        history_days: int = 60,
    ) -> dict[str, Any] | None:
        """构建分析流程使用的上下文字典。"""
        if target_date is None:
            target_date = date.today()

        recent_data = self.get_latest_data(code, days=2)
        if not recent_data:
            return None

        today_data = recent_data[0]
        yesterday_data = recent_data[1] if len(recent_data) > 1 else None

        context: dict[str, Any] = {
            "code": code,
            "date": today_data.date.isoformat(),
            "today": today_data.to_dict(),
        }

        if yesterday_data is not None:
            context["yesterday"] = yesterday_data.to_dict()

            today_volume = float(today_data.volume or 0.0)
            yesterday_volume = float(yesterday_data.volume or 0.0)
            today_close = float(today_data.close or 0.0)
            yesterday_close = float(yesterday_data.close or 0.0)

            if yesterday_volume > 0:
                context["volume_change_ratio"] = round(today_volume / yesterday_volume, 2)

            if yesterday_close > 0:
                context["price_change_ratio"] = round(
                    (today_close - yesterday_close) / yesterday_close * 100,
                    2,
                )

            context["ma_status"] = self._analyze_ma_status(today_data)

        bars = self.get_latest_data(code, days=max(2, min(int(history_days), 240)))
        if bars:
            raw_data: list[dict[str, Any]] = []
            for item in reversed(bars):
                row = item.to_dict()
                row_date = row.get("date")
                if isinstance(row_date, date):
                    row["date"] = row_date.isoformat()
                raw_data.append(row)
            context["raw_data"] = raw_data

        return context

    def _analyze_ma_status(self, data: StockDaily) -> str:
        """根据均线关系生成简要均线状态描述。"""
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

    @staticmethod
    def _parse_published_date(value: str | None) -> datetime | None:
        """解析新闻发布时间文本。"""
        if not value:
            return None
        if isinstance(value, datetime):
            return value

        text = str(value).strip()
        if not text:
            return None

        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass

        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
        ):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _safe_json_dumps(data: Any) -> str:
        """安全序列化 JSON。"""
        try:
            return json.dumps(data, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps(str(data), ensure_ascii=False)

    @staticmethod
    def _build_raw_result(result: Any) -> dict[str, Any]:
        """构造分析结果原始落库存档。"""
        data = result.to_dict() if hasattr(result, "to_dict") else {}
        data.update(
            {
                "data_sources": getattr(result, "data_sources", ""),
                "raw_response": getattr(result, "raw_response", None),
            }
        )
        return data

    @staticmethod
    def _parse_sniper_value(value: Any) -> float | None:
        """从字符串或数字中提取狙击点价格。"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).replace(",", "").strip()
        if not text:
            return None

        try:
            return float(text)
        except ValueError:
            pass

        colon_pos = max(text.rfind("："), text.rfind(":"))
        yuan_pos = text.find("元", colon_pos + 1 if colon_pos != -1 else 0)
        if yuan_pos == -1:
            return None

        segment_start = colon_pos + 1 if colon_pos != -1 else 0
        segment = text[segment_start:yuan_pos]
        matches = list(re.finditer(r"-?\d+(?:\.\d+)?", segment))
        valid_numbers: list[str] = []
        for match in matches:
            if match.start() >= 2 and segment[match.start() - 2 : match.start()].upper() == "MA":
                continue
            valid_numbers.append(match.group())

        if not valid_numbers:
            return None

        try:
            return float(valid_numbers[-1])
        except ValueError:
            return None

    def _extract_sniper_points(self, result: Any) -> dict[str, float | None]:
        """提取理想买点、次优买点、止损和止盈。"""
        raw_points = result.get_sniper_points() if hasattr(result, "get_sniper_points") else {}
        raw_points = raw_points or {}
        return {
            "ideal_buy": self._parse_sniper_value(raw_points.get("ideal_buy")),
            "secondary_buy": self._parse_sniper_value(raw_points.get("secondary_buy")),
            "stop_loss": self._parse_sniper_value(raw_points.get("stop_loss")),
            "take_profit": self._parse_sniper_value(raw_points.get("take_profit")),
        }

    @staticmethod
    def _build_fallback_url_key(
        *,
        code: str,
        title: str,
        source: str,
        published_date: datetime | None,
    ) -> str:
        """在新闻没有 URL 时构造稳定去重键。"""
        date_str = published_date.isoformat() if published_date else ""
        raw_key = f"{code}|{title}|{source}|{date_str}"
        digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
        return f"no-url:{code}:{digest}"


def get_db() -> DatabaseManager:
    """返回数据库管理器单例。"""
    return DatabaseManager.get_instance()
