# -*- coding: utf-8 -*-
"""数据智能体。

它负责把“原始行情抓取”转换成后续阶段能直接消费的标准化输入，包括：
1. 拉取并缓存近 60 个交易日的日线数据
2. 尝试补充实时行情
3. 组装 `analysis_context` 供 Signal/Analyzer 使用
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from data_provider import DataFetcherManager
from data_provider.base import canonical_stock_code
from data_provider.base import DataSourceUnavailableError
from agent_stock.agents.contracts import AgentState, DataAgentOutput
from agent_stock.agents.agentic_decision import generate_structured_decision
from agent_stock.analyzer import get_analyzer
from agent_stock.config import AgentRuntimeConfig, Config, get_config
from agent_stock.protocols import SupportsRealtimeQuoteFetcher
from agent_stock.services.runtime_market_service import QUOTE_FALLBACK_ORDER
from agent_stock.storage import DatabaseManager, get_db

logger = logging.getLogger(__name__)


class DataAgent:
    """抓取日线与实时行情，并准备分析所需上下文。"""

    def __init__(
        self,
        config: Config | None = None,
        fetcher_manager: SupportsRealtimeQuoteFetcher | None = None,
        db_manager: DatabaseManager | None = None,
        analyzer=None,
    ) -> None:
        """初始化数据抓取器与存储依赖。"""
        self.config = config or get_config()
        self.fetcher_manager = fetcher_manager or DataFetcherManager()
        self.db = db_manager or get_db()
        self.analyzer = analyzer or get_analyzer()

    @staticmethod
    def _analyze_ma_status(today: dict) -> str:
        close = float(today.get("close") or 0)
        ma5 = float(today.get("ma5") or 0)
        ma10 = float(today.get("ma10") or 0)
        ma20 = float(today.get("ma20") or 0)

        if close > ma5 > ma10 > ma20 > 0:
            return "多头排列 📈"
        if close < ma5 < ma10 < ma20 and ma20 > 0:
            return "空头排列 📉"
        if close > ma5 and ma5 > ma10:
            return "短期向好 🔼"
        return "震荡整理"

    def _build_analysis_context_from_daily_frame(self, code: str, daily_df) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for _, row in daily_df.tail(60).iterrows():
            rows.append(
                {
                    "date": str(row.get("date"))[:10],
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                    "amount": row.get("amount"),
                    "pct_chg": row.get("pct_chg"),
                    "ma5": row.get("ma5"),
                    "ma10": row.get("ma10"),
                    "ma20": row.get("ma20"),
                    "volume_ratio": row.get("volume_ratio"),
                    "data_source": row.get("data_source"),
                }
            )

        if not rows:
            return {}

        today = dict(rows[-1])
        yesterday = dict(rows[-2]) if len(rows) > 1 else None
        # 分析上下文里既有行情原始字典，也有衍生出的数值字段，因此这里统一放宽为 Any。
        context: dict[str, Any] = {
            "code": code,
            "date": str(today.get("date") or ""),
            "today": today,
            "raw_data": rows,
        }

        if yesterday is not None:
            context["yesterday"] = yesterday
            yesterday_volume = float(yesterday.get("volume") or 0)
            yesterday_close = float(yesterday.get("close") or 0)
            today_volume = float(today.get("volume") or 0)
            today_close = float(today.get("close") or 0)
            if yesterday_volume > 0:
                context["volume_change_ratio"] = round(today_volume / yesterday_volume, 2)
            if yesterday_close > 0:
                context["price_change_ratio"] = round((today_close - yesterday_close) / yesterday_close * 100, 2)
            context["ma_status"] = self._analyze_ma_status(today)

        return context

    def _resolve_requested_source(self, runtime_config: AgentRuntimeConfig | None) -> str:
        fixed_source = runtime_config.data_source.market_source if runtime_config and runtime_config.data_source else None
        if fixed_source:
            return fixed_source
        priorities = [item.strip().lower() for item in str(getattr(self.config, "realtime_source_priority", "") or "").split(",") if item.strip()]
        for source in priorities:
            if source in QUOTE_FALLBACK_ORDER:
                return source
        return "tencent"

    @staticmethod
    def _dedupe_sources(sources: list[str]) -> list[str]:
        deduped: list[str] = []
        for source in sources:
            if source and source not in deduped:
                deduped.append(source)
        return deduped

    def _build_candidate_sources(self, requested_source: str, max_source_switches: int = 2) -> list[str]:
        sources = self._dedupe_sources([requested_source, *QUOTE_FALLBACK_ORDER.get(requested_source, ())])
        return sources[: max_source_switches + 1]

    def _build_data_stage_prompt(
        self,
        *,
        code: str,
        attempts: list[dict[str, Any]],
        has_analysis_context: bool,
        has_realtime_quote: bool,
    ) -> str:
        return (
            "你是股票数据代理，只输出严格 JSON，不要输出解释、Markdown 或代码块。\n"
            "允许 action 只有：continue, continue_with_partial, abort。\n"
            "规则：1. 数据完整时优先 continue；2. 只有至少有可用分析上下文时才允许 continue_with_partial；"
            "3. 无有效历史数据时必须 abort。\n\n"
            f"股票代码：{code}\n"
            f"阶段观测：{attempts}\n"
            f"是否有可用分析上下文：{has_analysis_context}\n"
            f"是否有可用实时行情：{has_realtime_quote}\n\n"
            "输出 JSON 字段：action, summary, reason, confidence, warnings。"
        )

    def run(self, code: str, *, runtime_config: AgentRuntimeConfig | None = None) -> DataAgentOutput:
        """为单只股票执行一次数据采集。"""
        normalized_code = canonical_stock_code(code)
        trade_date = date.today()
        requested_source = self._resolve_requested_source(runtime_config)
        candidate_sources = self._build_candidate_sources(requested_source, max_source_switches=2)
        attempts: list[dict[str, Any]] = []
        warnings: list[str] = []
        fallback_chain: list[str] = []
        selected_source = requested_source
        selected_daily_df = None
        selected_quote_dict: dict[str, Any] = {}
        error_message = None
        partial_candidate: tuple[Any, str] | None = None

        for source in candidate_sources:
            attempt: dict[str, Any] = {"source": source, "daily_ready": False, "quote_ready": False}
            fallback_chain.append(source)
            daily_df = None
            quote_dict: dict[str, Any] = {}

            try:
                daily_df, _effective_source = self.fetcher_manager.get_daily_data(
                    normalized_code,
                    days=60,
                    fixed_source=source,
                )
                if daily_df is None or daily_df.empty:
                    raise DataSourceUnavailableError(f"{source} returned no usable history rows")
                self.db.save_daily_data(daily_df, normalized_code, source)
                attempt["daily_ready"] = True
                attempt["daily_rows"] = int(len(daily_df.index))
            except Exception as exc:
                attempt["daily_error"] = str(exc).strip() or "unknown error"

            try:
                quote = self.fetcher_manager.get_realtime_quote(normalized_code, fixed_source=source)
                if quote is None or (hasattr(quote, "has_basic_data") and not quote.has_basic_data()):
                    raise DataSourceUnavailableError(f"{source} realtime quote returned no usable data")
                quote_dict = quote.to_dict() if hasattr(quote, "to_dict") else dict(quote.__dict__)
                attempt["quote_ready"] = bool(quote_dict)
            except Exception as exc:
                attempt["quote_error"] = str(exc).strip() or "unknown error"

            attempts.append(attempt)

            if attempt["daily_ready"] and attempt["quote_ready"]:
                selected_source = source
                selected_daily_df = daily_df
                selected_quote_dict = quote_dict
                break

            if attempt["daily_ready"] and partial_candidate is None:
                partial_candidate = (daily_df, source)

        if selected_daily_df is None and partial_candidate is not None:
            selected_daily_df, selected_source = partial_candidate
            warnings.append(f"实时行情暂不可用，已基于 {selected_source} 的日线数据继续分析。")

        if selected_source != requested_source:
            warnings.append(f"主数据源 {requested_source} 不可用，已自动切换到 {selected_source}。")

        analysis_context = self.db.get_analysis_context(normalized_code, history_days=60) or {}
        if "raw_data" not in analysis_context and selected_daily_df is not None and not selected_daily_df.empty:
            analysis_context = self._build_analysis_context_from_daily_frame(normalized_code, selected_daily_df)
        elif selected_daily_df is not None and not selected_daily_df.empty:
            analysis_context = self._build_analysis_context_from_daily_frame(normalized_code, selected_daily_df)

        has_analysis_context = bool(analysis_context and analysis_context.get("raw_data"))
        has_realtime_quote = bool(selected_quote_dict)

        default_action = "continue" if has_analysis_context and has_realtime_quote else "continue_with_partial" if has_analysis_context else "abort"
        default_decision = {
            "action": default_action,
            "summary": (
                f"已完成 {selected_source} 数据准备。"
                if default_action == "continue"
                else f"已拿到 {selected_source} 的部分可用数据，后续分析将降置信度继续。"
                if default_action == "continue_with_partial"
                else "当前无法获得足够可靠的数据，建议停止本轮分析。"
            ),
            "reason": "data_ready" if default_action == "continue" else "partial_data_ready" if default_action == "continue_with_partial" else "data_unavailable",
            "confidence": 0.88 if default_action == "continue" else 0.42 if default_action == "continue_with_partial" else 0.1,
            "warnings": warnings,
        }
        decision, llm_used = generate_structured_decision(
            analyzer=self.analyzer,
            stage="data",
            prompt=self._build_data_stage_prompt(
                code=normalized_code,
                attempts=attempts,
                has_analysis_context=has_analysis_context,
                has_realtime_quote=has_realtime_quote,
            ),
            allowed_actions={"continue", "continue_with_partial", "abort"},
            default_decision=default_decision,
        )
        decision_warnings = [item for item in decision.get("warnings") or [] if isinstance(item, str)]
        for item in decision_warnings:
            if item not in warnings:
                warnings.append(item)

        action = str(decision.get("action") or default_action).strip() or default_action
        if action == "abort" and has_analysis_context:
            action = "continue_with_partial"

        state = AgentState.READY if action != "abort" and has_analysis_context else AgentState.FAILED
        if state == AgentState.FAILED:
            error_message = "data collection failed: no usable analysis context"
            logger.warning("[%s] %s", normalized_code, error_message)

        return DataAgentOutput(
            code=normalized_code,
            trade_date=trade_date,
            state=state,
            analysis_context=analysis_context if state != AgentState.FAILED else {},
            realtime_quote=selected_quote_dict,
            data_source=selected_source,
            error_message=error_message,
            observations=attempts,
            decision=decision,
            confidence=float(decision.get("confidence") or default_decision["confidence"]),
            warnings=warnings,
            llm_used=llm_used,
            fallback_chain=fallback_chain,
            next_action="signal" if state == AgentState.READY else "abort",
            status="ready" if has_analysis_context and has_realtime_quote else "partial" if has_analysis_context else "failed",
            retryable=state == AgentState.FAILED and bool(attempts),
            source_attempts=attempts,
            partial_ok=has_analysis_context,
            suggested_next="signal" if state == AgentState.READY else "abort",
        )
