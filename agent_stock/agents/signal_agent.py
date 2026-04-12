# -*- coding: utf-8 -*-
"""信号智能体。

它站在“规则分析”和“AI 分析”两者之间做汇总：先从历史上下文构建趋势信号，再决定
是否刷新 AI 结果，最后把两者压平成统一的 `SignalAgentOutput`。
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Callable


import pandas as pd

from agent_stock.agents.contracts import AgentState, DataAgentOutput, SignalAgentOutput
from agent_stock.agents.agentic_decision import generate_structured_decision
from agent_stock.analyzer import LlmRequestTimeoutError
from agent_stock.analyzer import get_analyzer
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
        config: Config | None = None,
        db_manager: DatabaseManager | None = None,
        trend_analyzer: StockTrendAnalyzer | None = None,
        execution_repo: ExecutionRepository | None = None,
        ai_resolver: Callable[[str], Any] | None = None,
        analyzer=None,
    ) -> None:
        """初始化信号生成所需的趋势分析、缓存和 AI 依赖。"""
        self.config = config or get_config()
        self.db = db_manager or DatabaseManager.get_instance()
        self.trend_analyzer = trend_analyzer or StockTrendAnalyzer()
        self.repo = execution_repo or ExecutionRepository(self.db)
        self._ai_resolver = ai_resolver
        self._pipeline: StockAnalysisPipeline | None = None
        self.analyzer = analyzer or get_analyzer()

    def run(
        self,
        data_output: DataAgentOutput,
        *,
        runtime_config: AgentRuntimeConfig | None = None,
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

        observations = [
            {
                "data_state": data_output.state.value,
                "data_source": data_output.data_source,
                "data_confidence": data_output.confidence,
                "has_raw_data": bool((data_output.analysis_context or {}).get("raw_data")),
                "has_realtime_quote": bool(data_output.realtime_quote),
                "trend_ready": bool(trend_payload),
                "ai_ready": bool(ai_payload),
                "ai_refreshed": ai_refreshed,
                "data_warnings": list(data_output.warnings or []),
            }
        ]
        fallback_chain: list[str] = []
        if trend_payload:
            fallback_chain.append("trend_rule")
        if ai_payload:
            fallback_chain.append("ai_analysis")

        if state == AgentState.FAILED:
            default_decision = {
                "action": "abort",
                "summary": "信号阶段未能形成可靠的趋势或 AI 结论，本轮不建议继续执行。",
                "reason": "signal_unavailable",
                "next_action": "abort",
                "confidence": 0.08,
                "warnings": list(data_output.warnings or []),
            }
        elif not trend_payload and bool((data_output.analysis_context or {}).get("raw_data")):
            default_decision = {
                "action": "request_more_data",
                "summary": "当前仅拿到部分有效信号，建议先回到数据阶段补强后再继续。",
                "reason": "trend_missing",
                "next_action": "data",
                "confidence": 0.35,
                "warnings": list(data_output.warnings or []),
            }
        else:
            base_confidence = 0.82 if trend_payload and ai_payload else 0.62 if (trend_payload or ai_payload) else 0.3
            if data_output.warnings:
                base_confidence = max(0.25, base_confidence - 0.18)
            default_decision = {
                "action": "continue_risk",
                "summary": f"已形成 {operation_advice} 信号，可进入风控阶段继续决策。",
                "reason": "signal_ready",
                "next_action": "risk",
                "confidence": base_confidence,
                "warnings": list(data_output.warnings or []),
            }

        decision, planner_llm_used = generate_structured_decision(
            analyzer=self.analyzer,
            stage="signal",
            prompt=self._build_signal_stage_prompt(
                code=code,
                data_output=data_output,
                trend_payload=trend_payload,
                ai_payload=ai_payload,
                operation_advice=operation_advice,
                sentiment_score=sentiment_score,
            ),
            allowed_actions={"continue_risk", "request_more_data", "abort"},
            default_decision=default_decision,
        )
        warnings = list(data_output.warnings or [])
        for item in decision.get("warnings") or []:
            if isinstance(item, str) and item not in warnings:
                warnings.append(item)

        action = str(decision.get("action") or default_decision["action"]).strip() or default_decision["action"]
        next_action = str(decision.get("next_action") or default_decision.get("next_action") or "risk")
        needs_more_data = action == "request_more_data"
        review_reason = str(decision.get("reason") or "").strip() or None
        status = "failed" if state == AgentState.FAILED else "needs_more_data" if needs_more_data else "ready"

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
            observations=observations,
            decision=decision,
            confidence=float(decision.get("confidence") or default_decision["confidence"]),
            warnings=warnings,
            llm_used=bool(ai_payload) or planner_llm_used,
            fallback_chain=fallback_chain,
            next_action=next_action,
            status=status,
            needs_more_data=needs_more_data,
            review_reason=review_reason if needs_more_data or state == AgentState.FAILED else None,
            suggested_next=next_action,
        )

    @staticmethod
    def _build_signal_stage_prompt(
        *,
        code: str,
        data_output: DataAgentOutput,
        trend_payload: dict[str, Any],
        ai_payload: dict[str, Any],
        operation_advice: str,
        sentiment_score: int,
    ) -> str:
        return (
            "你是股票信号研判代理，只输出严格 JSON，不要输出解释、Markdown 或代码块。\n"
            "允许 action 只有：continue_risk, request_more_data, abort。\n"
            "规则：1. 趋势或 AI 至少有一项有效时，优先 continue_risk；2. 数据明显不足时可 request_more_data；"
            "3. 没有任何可用信号时必须 abort。\n\n"
            f"股票代码：{code}\n"
            f"数据阶段输出：{data_output.to_dict()}\n"
            f"趋势结果：{trend_payload}\n"
            f"AI 结果：{ai_payload}\n"
            f"当前操作建议：{operation_advice}\n"
            f"当前情绪分：{sentiment_score}\n\n"
            "输出 JSON 字段：action, summary, reason, next_action, confidence, warnings。"
        )

    def _resolve_ai(self, code: str, *, runtime_config: AgentRuntimeConfig | None = None) -> Any:
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
        runtime_market_source = runtime_config.data_source.market_source if runtime_config and runtime_config.data_source else None
        if runtime_llm is not None or runtime_market_source is not None:
            pipeline = StockAnalysisPipeline(
                config=self.config,
                query_id=uuid.uuid4().hex,
                query_source="agent",
                runtime_llm=runtime_llm,
                runtime_market_source=runtime_market_source,
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
    def _build_runtime_account_context(*, runtime_config: AgentRuntimeConfig | None) -> dict[str, Any] | None:
        """将运行时账户上下文整理为 AI 分析可消费的载荷。"""
        if not runtime_config or not runtime_config.context:
            return None

        context = runtime_config.context
        account_snapshot = context.account_snapshot if isinstance(context.account_snapshot, dict) else {}
        summary = context.summary if isinstance(context.summary, dict) else {}
        positions = context.positions if isinstance(context.positions, list) else []

        payload: dict[str, Any] = {}
        if account_snapshot:
            payload["account_snapshot"] = dict(account_snapshot)
        if summary:
            payload["summary"] = dict(summary)
        if positions:
            payload["positions"] = [dict(item) for item in positions if isinstance(item, dict)]
        return payload or None

    def _build_trend_result(self, data_output: DataAgentOutput) -> TrendAnalysisResult | None:
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
    def _trend_to_payload(trend_result: TrendAnalysisResult | None) -> dict[str, Any]:
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

    def _extract_ai_payload(self, ai_result: Any) -> dict[str, Any]:
        """从 AI 分析结果中提取标准化字段。"""
        payload: dict[str, Any] = {
            "operation_advice": getattr(ai_result, "operation_advice", None),
            "sentiment_score": getattr(ai_result, "sentiment_score", None),
            "trend_prediction": getattr(ai_result, "trend_prediction", None),
            "analysis_summary": getattr(ai_result, "analysis_summary", None),
            "news_summary": getattr(ai_result, "news_summary", None),
            "search_performed": bool(getattr(ai_result, "search_performed", False)),
        }
        news_items = getattr(ai_result, "news_items", None)
        if isinstance(news_items, list):
            payload["news_items"] = [dict(item) for item in news_items if isinstance(item, dict)]

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
    def _parse_price(value: Any) -> float | None:
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
    def _fallback_operation_advice(trend_result: TrendAnalysisResult | None) -> str:
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
