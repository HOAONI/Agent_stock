# -*- coding: utf-8 -*-
"""信号智能体。

它站在“规则分析”和“AI 分析”之间做汇总：先从历史上下文构建趋势信号，再决定
是否刷新 AI 结果，最后把两者压平成统一的 `SignalAgentOutput`。
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Callable, Dict, Optional

import pandas as pd

from agent_stock.agents.contracts import AgentState, DataAgentOutput, SignalAgentOutput
from agent_stock.analyzer import LlmRequestTimeoutError
from agent_stock.config import AgentRuntimeConfig, Config, get_config, redact_sensitive_text
from agent_stock.core.pipeline import StockAnalysisPipeline
from agent_stock.enums import ReportType
from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.stock_analyzer import BuySignal, StockTrendAnalyzer, TrendAnalysisResult
from agent_stock.storage import DatabaseManager

logger = logging.getLogger(__name__)


class SignalAgent:
    """为单只股票生成操作建议与风险价格位。"""

    def __init__(
        self,
        config: Optional[Config] = None,
        db_manager: Optional[DatabaseManager] = None,
        trend_analyzer: Optional[StockTrendAnalyzer] = None,
        execution_repo: Optional[ExecutionRepository] = None,
        ai_resolver: Optional[Callable[[str], Any]] = None,
    ) -> None:
        """初始化信号生成所需的趋势分析、缓存和 AI 依赖。"""
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
        """结合趋势分析与 AI 结果生成每日信号。"""
        code = data_output.code
        trade_date = data_output.trade_date
        has_runtime_llm = bool(runtime_config and runtime_config.llm is not None)

        trend_result = self._build_trend_result(data_output)
        trend_payload = self._trend_to_payload(trend_result)

        # 请求级 LLM 覆盖会绕过日级缓存，确保本次运行真正使用调用方指定的模型参数。
        cached = None if has_runtime_llm else self.repo.get_signal_snapshot(code=code, trade_date=trade_date)
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

        # 无论本次是否调用 AI，最终都会把规则结果和 AI 结果按同一结构落入快照表。
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
        """通过注入的解析器或默认分析管线获取 AI 结果。"""
        if self._ai_resolver is not None:
            try:
                return self._ai_resolver(code)
            except LlmRequestTimeoutError:
                raise
            except Exception as exc:
                logger.warning("[%s] custom AI resolver failed: %s", code, redact_sensitive_text(str(exc)))
                return None

        runtime_llm = runtime_config.llm if runtime_config else None
        if runtime_llm is not None:
            pipeline = StockAnalysisPipeline(
                config=self.config,
                query_id=uuid.uuid4().hex,
                query_source="agent",
                runtime_llm=runtime_llm,
            )
        else:
            if self._pipeline is None:
                self._pipeline = StockAnalysisPipeline(
                    config=self.config,
                    query_id=uuid.uuid4().hex,
                    query_source="agent",
                )
            pipeline = self._pipeline

        runtime_account_context = self._build_runtime_account_context(runtime_config=runtime_config)

        try:
            logger.info("[%s] signal stage AI analysis start (runtime_llm=%s)", code, bool(runtime_llm is not None))
            return pipeline.analyze_stock(
                code=code,
                report_type=ReportType.FULL,
                query_id=uuid.uuid4().hex,
                runtime_account_context=runtime_account_context,
            )
        except LlmRequestTimeoutError:
            logger.error("[%s] signal stage AI analysis timed out", code)
            raise
        except Exception as exc:
            logger.warning("[%s] default AI resolver failed: %s", code, redact_sensitive_text(str(exc)))
            return None

    @staticmethod
    def _build_runtime_account_context(*, runtime_config: Optional[AgentRuntimeConfig]) -> Optional[Dict[str, Any]]:
        """将运行时账户上下文整理为 AI 分析可消费的载荷。"""
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
        """基于上下文原始数据计算趋势分析结果。"""
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
        """将趋势分析结果压平成可存储字典。"""
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
        """从 AI 分析结果中提取标准化字段。"""
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
        """从数字或字符串字段中提取价格。"""
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
        """在 AI 不可用时，根据趋势信号回退生成操作建议。"""
        if trend_result is None:
            return "观望"

        if trend_result.buy_signal in (BuySignal.STRONG_BUY, BuySignal.BUY):
            return "买入"
        if trend_result.buy_signal == BuySignal.HOLD:
            return "持有"
        if trend_result.buy_signal in (BuySignal.SELL, BuySignal.STRONG_SELL):
            return "卖出"
        return "观望"
