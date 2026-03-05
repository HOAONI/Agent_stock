# -*- coding: utf-8 -*-
"""Signal Agent: computes trend signal and refreshes AI signal snapshot."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Callable, Dict, Optional

import pandas as pd

from agent_stock.agents.contracts import AgentState, DataAgentOutput, SignalAgentOutput
from src.config import AgentRuntimeConfig, Config, get_config, redact_sensitive_text
from src.core.pipeline import StockAnalysisPipeline
from src.enums import ReportType
from agent_stock.repositories.execution_repo import ExecutionRepository
from src.stock_analyzer import BuySignal, StockTrendAnalyzer, TrendAnalysisResult
from agent_stock.storage import DatabaseManager

logger = logging.getLogger(__name__)


class SignalAgent:
    """Generate operation advice and risk price levels for one stock."""

    def __init__(
        self,
        config: Optional[Config] = None,
        db_manager: Optional[DatabaseManager] = None,
        trend_analyzer: Optional[StockTrendAnalyzer] = None,
        execution_repo: Optional[ExecutionRepository] = None,
        ai_resolver: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.config = config or get_config()
        self.db = db_manager or DatabaseManager.get_instance()
        self.trend_analyzer = trend_analyzer or StockTrendAnalyzer()
        self.repo = execution_repo or ExecutionRepository(self.db)
        self._ai_resolver = ai_resolver
        self._pipeline: Optional[StockAnalysisPipeline] = None

    def run(
        self,
        data_output: DataAgentOutput,
        *,
        runtime_config: Optional[AgentRuntimeConfig] = None,
    ) -> SignalAgentOutput:
        """Run trend + AI policy with daily cache."""
        code = data_output.code
        trade_date = data_output.trade_date

        trend_result = self._build_trend_result(data_output)
        trend_payload = self._trend_to_payload(trend_result)

        cached = self.repo.get_signal_snapshot(code=code, trade_date=trade_date)
        ai_refreshed = False

        ai_policy = str(getattr(self.config, "agent_ai_refresh_policy", "daily_once") or "daily_once").lower()
        ai_payload = cached.get("ai_payload", {}) if cached else {}

        if ai_policy != "daily_once" or not ai_payload:
            ai_result = self._resolve_ai(code, runtime_config=runtime_config)
            if ai_result is not None:
                ai_payload = self._extract_ai_payload(ai_result)
                ai_refreshed = True
            elif not ai_payload:
                ai_payload = {}

        operation_advice = str(ai_payload.get("operation_advice") or self._fallback_operation_advice(trend_result))
        sentiment_score = int(ai_payload.get("sentiment_score") or trend_payload.get("signal_score") or 50)
        stop_loss = self._parse_price(ai_payload.get("stop_loss"))
        take_profit = self._parse_price(ai_payload.get("take_profit"))

        self.repo.upsert_signal_snapshot(
            code=code,
            trade_date=trade_date,
            signal_payload={
                "operation_advice": operation_advice,
                "sentiment_score": sentiment_score,
                "trend_payload": trend_payload,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "resolved_stop_loss": stop_loss,
                "resolved_take_profit": take_profit,
            },
            ai_payload=ai_payload,
        )

        state = AgentState.READY
        error_message = None
        if not trend_payload and not ai_payload:
            state = AgentState.FAILED
            error_message = "signal generation failed: no trend payload and no AI payload"

        return SignalAgentOutput(
            code=code,
            trade_date=trade_date,
            state=state,
            operation_advice=operation_advice,
            sentiment_score=sentiment_score,
            trend_signal=str(trend_payload.get("buy_signal") or "WAIT"),
            trend_score=int(trend_payload.get("signal_score") or 0),
            stop_loss=stop_loss,
            take_profit=take_profit,
            resolved_stop_loss=stop_loss,
            resolved_take_profit=take_profit,
            ai_refreshed=ai_refreshed,
            ai_payload=ai_payload,
            trend_payload=trend_payload,
            error_message=error_message,
        )

    def _resolve_ai(self, code: str, *, runtime_config: Optional[AgentRuntimeConfig] = None) -> Any:
        """Resolve AI output via injected resolver or default pipeline."""
        if self._ai_resolver is not None:
            try:
                return self._ai_resolver(code)
            except Exception as exc:
                logger.warning("[%s] custom AI resolver failed: %s", code, redact_sensitive_text(str(exc)))
                return None

        if self._pipeline is None:
            self._pipeline = StockAnalysisPipeline(
                config=self.config,
                query_id=uuid.uuid4().hex,
                query_source="agent",
            )
        pipeline = self._pipeline

        runtime_account_context = self._build_runtime_account_context(runtime_config=runtime_config)

        try:
            return pipeline.analyze_stock(
                code=code,
                report_type=ReportType.FULL,
                query_id=uuid.uuid4().hex,
                runtime_account_context=runtime_account_context,
            )
        except Exception as exc:
            logger.warning("[%s] default AI resolver failed: %s", code, redact_sensitive_text(str(exc)))
            return None

    @staticmethod
    def _build_runtime_account_context(*, runtime_config: Optional[AgentRuntimeConfig]) -> Optional[Dict[str, Any]]:
        if not runtime_config or not runtime_config.context:
            return None

        context = runtime_config.context
        account_snapshot = context.account_snapshot if isinstance(context.account_snapshot, dict) else {}
        summary = context.summary if isinstance(context.summary, dict) else {}
        positions = context.positions if isinstance(context.positions, list) else []

        payload: Dict[str, Any] = {}
        if account_snapshot:
            payload["account_snapshot"] = dict(account_snapshot)
        if summary:
            payload["summary"] = dict(summary)
        if positions:
            payload["positions"] = [dict(item) for item in positions if isinstance(item, dict)]
        return payload or None

    def _build_trend_result(self, data_output: DataAgentOutput) -> Optional[TrendAnalysisResult]:
        """Compute trend signal for every cycle from context raw data."""
        context = data_output.analysis_context or {}
        raw_data = context.get("raw_data")
        if not isinstance(raw_data, list) or not raw_data:
            return None

        try:
            df = pd.DataFrame(raw_data)
            if df.empty:
                return None
            return self.trend_analyzer.analyze(df, data_output.code)
        except Exception as exc:
            logger.warning("[%s] trend analyzer failed: %s", data_output.code, exc)
            return None

    @staticmethod
    def _trend_to_payload(trend_result: Optional[TrendAnalysisResult]) -> Dict[str, Any]:
        if trend_result is None:
            return {}

        return {
            "trend_status": trend_result.trend_status.value,
            "buy_signal": trend_result.buy_signal.value,
            "signal_score": trend_result.signal_score,
            "signal_reasons": trend_result.signal_reasons,
            "risk_factors": trend_result.risk_factors,
            "bias_ma5": trend_result.bias_ma5,
            "volume_status": trend_result.volume_status.value,
        }

    def _extract_ai_payload(self, ai_result: Any) -> Dict[str, Any]:
        """Extract normalized fields from AnalysisResult-like object."""
        payload: Dict[str, Any] = {
            "operation_advice": getattr(ai_result, "operation_advice", None),
            "sentiment_score": getattr(ai_result, "sentiment_score", None),
            "trend_prediction": getattr(ai_result, "trend_prediction", None),
            "analysis_summary": getattr(ai_result, "analysis_summary", None),
        }

        stop_loss = None
        take_profit = None
        if hasattr(ai_result, "get_sniper_points"):
            points = ai_result.get_sniper_points() or {}
            stop_loss = points.get("stop_loss")
            take_profit = points.get("take_profit")

        payload["stop_loss"] = self._parse_price(stop_loss)
        payload["take_profit"] = self._parse_price(take_profit)
        return payload

    @staticmethod
    def _parse_price(value: Any) -> Optional[float]:
        """Parse numeric price from numeric/string fields."""
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

        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not matches:
            return None
        try:
            return float(matches[-1])
        except ValueError:
            return None

    @staticmethod
    def _fallback_operation_advice(trend_result: Optional[TrendAnalysisResult]) -> str:
        """Fallback operation advice from trend signal when AI is unavailable."""
        if trend_result is None:
            return "观望"

        if trend_result.buy_signal in (BuySignal.STRONG_BUY, BuySignal.BUY):
            return "买入"
        if trend_result.buy_signal == BuySignal.HOLD:
            return "持有"
        if trend_result.buy_signal in (BuySignal.SELL, BuySignal.STRONG_SELL):
            return "卖出"
        return "观望"
