# -*- coding: utf-8 -*-
"""Unit tests for Agent historical replay backtest service."""

from __future__ import annotations

from datetime import date, timedelta
import unittest

import pandas as pd

from agent_stock.services.agent_historical_backtest_service import AgentHistoricalBacktestService
from src.stock_analyzer import BuySignal, TrendAnalysisResult


def _build_days(count: int) -> list[date]:
    start = date(2024, 1, 2)
    return [start + timedelta(days=index) for index in range(count)]


def _build_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


class _FakeFetcher:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_daily_data(self, code, start_date=None, end_date=None, days=5000):  # noqa: ARG002
        return self.frame.copy(), "fake"


class _RecordingTrendAnalyzer:
    def __init__(self) -> None:
        self.max_dates: list[str] = []

    def analyze(self, df: pd.DataFrame, code: str):  # noqa: ARG002
        self.max_dates.append(str(pd.to_datetime(df["date"]).max().date()))
        return TrendAnalysisResult(code=code, buy_signal=BuySignal.WAIT, signal_score=50)


class _FakeAiAnalyzer:
    def __init__(self, counter: dict[str, int]) -> None:
        self.counter = counter

    def analyze(self, context, news_context=None):  # noqa: ARG002
        self.counter["calls"] = self.counter.get("calls", 0) + 1

        class _Result:
            operation_advice = "买入"
            sentiment_score = 88
            trend_prediction = "看多"
            analysis_summary = "cached"

            @staticmethod
            def get_sniper_points():
                return {}

        return _Result()


class AgentHistoricalBacktestServiceTestCase(unittest.TestCase):
    def test_rolling_window_never_uses_future_bars(self):
        days = _build_days(5)
        frame = _build_frame(
            [
                {
                    "date": day.isoformat(),
                    "open": 10 + index,
                    "high": 11 + index,
                    "low": 9 + index,
                    "close": 10.5 + index,
                    "volume": 1000 + index * 10,
                    "amount": 10000 + index * 100,
                }
                for index, day in enumerate(days)
            ]
        )
        trend = _RecordingTrendAnalyzer()
        service = AgentHistoricalBacktestService(
            fetcher_manager=_FakeFetcher(frame),
            trend_analyzer=trend,
        )

        result = service.run(
            {
                "code": "600519",
                "start_date": days[0].isoformat(),
                "end_date": days[-1].isoformat(),
                "phase": "fast",
            }
        )

        self.assertEqual(len(result["daily_steps"]), 5)
        self.assertEqual(trend.max_dates, [day.isoformat() for day in days])

    def test_last_day_never_schedules_new_entry(self):
        days = _build_days(2)
        frame = _build_frame(
            [
                {"date": days[0].isoformat(), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000, "amount": 100000},
                {"date": days[1].isoformat(), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000, "amount": 100000},
            ]
        )
        cached_snapshots = [
            {
                "trade_date": days[0].isoformat(),
                "decision_source": "fast_rule",
                "llm_used": False,
                "factor_payload": {},
                "archived_news_payload": [],
                "signal_payload": {"operation_advice": "观望", "sentiment_score": 50, "trend_signal": "WAIT", "trend_score": 0},
                "ai_overlay": {},
            },
            {
                "trade_date": days[1].isoformat(),
                "decision_source": "fast_rule",
                "llm_used": False,
                "factor_payload": {},
                "archived_news_payload": [],
                "signal_payload": {"operation_advice": "买入", "sentiment_score": 70, "trend_signal": "BUY", "trend_score": 80},
                "ai_overlay": {},
            },
        ]
        service = AgentHistoricalBacktestService(fetcher_manager=_FakeFetcher(frame))

        result = service.run(
            {
                "code": "600519",
                "start_date": days[0].isoformat(),
                "end_date": days[-1].isoformat(),
                "phase": "fast",
                "cached_snapshots": cached_snapshots,
            }
        )

        last_execution = result["daily_steps"][-1]["execution_payload"]
        self.assertEqual(last_execution["pending_action"], "none")
        self.assertEqual(last_execution["pending_reason"], "last_day_no_new_entry")

    def test_stop_loss_wins_when_stop_and_take_hit_same_day(self):
        days = _build_days(2)
        frame = _build_frame(
            [
                {"date": days[0].isoformat(), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000, "amount": 100000},
                {"date": days[1].isoformat(), "open": 100, "high": 110, "low": 90, "close": 100, "volume": 1000, "amount": 100000},
            ]
        )
        cached_snapshots = [
            {
                "trade_date": days[0].isoformat(),
                "decision_source": "fast_rule",
                "llm_used": False,
                "factor_payload": {},
                "archived_news_payload": [],
                "signal_payload": {"operation_advice": "买入", "sentiment_score": 80, "trend_signal": "BUY", "trend_score": 80},
                "ai_overlay": {},
            },
            {
                "trade_date": days[1].isoformat(),
                "decision_source": "fast_rule",
                "llm_used": False,
                "factor_payload": {},
                "archived_news_payload": [],
                "signal_payload": {"operation_advice": "持有", "sentiment_score": 60, "trend_signal": "HOLD", "trend_score": 60},
                "ai_overlay": {},
            },
        ]
        service = AgentHistoricalBacktestService(fetcher_manager=_FakeFetcher(frame))

        result = service.run(
            {
                "code": "600519",
                "start_date": days[0].isoformat(),
                "end_date": days[-1].isoformat(),
                "phase": "fast",
                "runtime_strategy": {
                    "position_max_pct": 30,
                    "stop_loss_pct": 5,
                    "take_profit_pct": 5,
                },
                "cached_snapshots": cached_snapshots,
            }
        )

        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"][0]["exit_reason"], "stop_loss")

    def test_refine_uses_cached_anchor_without_calling_ai(self):
        days = _build_days(1)
        frame = _build_frame(
            [
                {"date": days[0].isoformat(), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000, "amount": 100000},
            ]
        )
        calls = {"calls": 0}
        cached_snapshots = [
            {
                "trade_date": days[0].isoformat(),
                "decision_source": "llm_anchor",
                "llm_used": True,
                "confidence": 0.88,
                "factor_payload": {},
                "archived_news_payload": [],
                "signal_payload": {"operation_advice": "买入", "sentiment_score": 88, "trend_signal": "BUY", "trend_score": 80},
                "ai_overlay": {"operation_advice": "买入", "sentiment_score": 88},
            },
        ]
        service = AgentHistoricalBacktestService(
            fetcher_manager=_FakeFetcher(frame),
            ai_analyzer_factory=lambda: _FakeAiAnalyzer(calls),
        )

        result = service.run(
            {
                "code": "600519",
                "start_date": days[0].isoformat(),
                "end_date": days[0].isoformat(),
                "phase": "refine",
                "cached_snapshots": cached_snapshots,
            }
        )

        self.assertEqual(calls["calls"], 0)
        self.assertEqual(result["daily_steps"][0]["decision_source"], "llm_anchor")


if __name__ == "__main__":
    unittest.main()
