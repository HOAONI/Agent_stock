# -*- coding: utf-8 -*-
"""Signal Agent 使用的核心股票分析管线。"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any


from agent_stock.analyzer import AnalysisResult, GeminiAnalyzer, LlmRequestTimeoutError, STOCK_NAME_MAP
from agent_stock.config import Config, RuntimeLlmConfig, get_config
from agent_stock.enums import ReportType
from agent_stock.search_service import SearchService
from agent_stock.stock_analyzer import StockTrendAnalyzer, TrendAnalysisResult
from agent_stock.storage import get_db
from data_provider import DataFetcherManager
from data_provider.realtime_types import ChipDistribution

logger = logging.getLogger(__name__)


class StockAnalysisPipeline:
    """整合行情、搜索和上下文后执行一次 AI 分析。"""

    def __init__(
        self,
        config: Config | None = None,
        query_id: str | None = None,
        query_source: str | None = None,
        save_context_snapshot: bool | None = None,
        runtime_llm: RuntimeLlmConfig | None = None,
        runtime_market_source: str | None = None,
    ) -> None:
        """初始化分析管线依赖。"""
        self.config = config or get_config()
        self.query_id = query_id
        self.query_source = self._resolve_query_source(query_source)
        self.save_context_snapshot = (
            self.config.save_context_snapshot if save_context_snapshot is None else bool(save_context_snapshot)
        )
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        self.runtime_market_source = str(runtime_market_source or "").strip().lower() or None
        self.trend_analyzer = StockTrendAnalyzer()
        self.analyzer = GeminiAnalyzer(config=self.config, runtime_llm=runtime_llm)
        self.search_service = SearchService(
            bocha_keys=self.config.bocha_api_keys,
            tavily_keys=self.config.tavily_api_keys,
            brave_keys=self.config.brave_api_keys,
            serpapi_keys=self.config.serpapi_keys,
            news_max_age_days=self.config.news_max_age_days,
        )

    def analyze_stock(
        self,
        code: str,
        report_type: ReportType,
        query_id: str,
        runtime_account_context: dict[str, Any] | None = None,
    ) -> AnalysisResult | None:
        """执行一次端到端分析，并按需落库。"""
        try:
            logger.info("[%s] analysis pipeline start", code)
            stock_name = STOCK_NAME_MAP.get(code, "")

            realtime_quote = None
            try:
                if self.runtime_market_source:
                    realtime_quote = self.fetcher_manager.get_realtime_quote(code, fixed_source=self.runtime_market_source)
                else:
                    realtime_quote = self.fetcher_manager.get_realtime_quote(code)
                if realtime_quote and getattr(realtime_quote, "name", None):
                    stock_name = realtime_quote.name
            except Exception as exc:
                logger.warning("[%s] realtime quote failed: %s", code, exc)

            if not stock_name:
                stock_name = f"股票{code}"

            chip_data = None
            if self.runtime_market_source is None:
                try:
                    chip_data = self.fetcher_manager.get_chip_distribution(code)
                except Exception as exc:
                    logger.warning("[%s] chip distribution failed: %s", code, exc)

            trend_result = self._resolve_trend_result(code)

            news_context = None
            if self.search_service.is_available:
                try:
                    intel_results = self.search_service.search_comprehensive_intel(
                        stock_code=code,
                        stock_name=stock_name,
                        max_searches=5,
                    )
                    if intel_results:
                        news_context = self.search_service.format_intel_report(intel_results, stock_name)
                        query_context = self._build_query_context(query_id=query_id)
                        for dimension, response in intel_results.items():
                            if response and response.success and response.results:
                                self.db.save_news_intel(
                                    code=code,
                                    name=stock_name,
                                    dimension=dimension,
                                    query=response.query,
                                    response=response,
                                    query_context=query_context,
                                )
                except Exception as exc:
                    logger.warning("[%s] intel search failed: %s", code, exc)

            context = self.db.get_analysis_context(code)
            if context is None:
                context = {
                    "code": code,
                    "stock_name": stock_name,
                    "date": date.today().isoformat(),
                    "data_missing": True,
                    "today": {},
                    "yesterday": {},
                }

            enhanced_context = self._enhance_context(
                context,
                realtime_quote,
                chip_data,
                trend_result,
                stock_name=stock_name,
                runtime_account_context=runtime_account_context,
            )
            result = self.analyzer.analyze(enhanced_context, news_context=news_context)

            if result:
                realtime_data = enhanced_context.get("realtime", {})
                result.current_price = realtime_data.get("price")
                result.change_pct = realtime_data.get("change_pct")
                try:
                    self.db.save_analysis_history(
                        result=result,
                        query_id=query_id,
                        report_type=report_type.value,
                        news_content=news_context,
                        context_snapshot=self._build_context_snapshot(
                            enhanced_context=enhanced_context,
                            news_content=news_context,
                            realtime_quote=realtime_quote,
                            chip_data=chip_data,
                        ),
                        save_snapshot=self.save_context_snapshot,
                    )
                except Exception as exc:
                    logger.warning("[%s] save analysis history failed: %s", code, exc)

            return result
        except LlmRequestTimeoutError:
            logger.error("[%s] analysis timed out", code)
            raise
        except Exception as exc:
            logger.error("[%s] analysis failed: %s", code, exc)
            logger.exception("[%s] detailed analysis error", code)
            return None

    def _resolve_trend_result(self, code: str) -> TrendAnalysisResult | None:
        """从历史上下文中恢复趋势分析结果。"""
        context = self.db.get_analysis_context(code)
        if not context or "raw_data" not in context:
            return None
        raw_data = context["raw_data"]
        if not isinstance(raw_data, list) or not raw_data:
            return None

        try:
            import pandas as pd

            frame = pd.DataFrame(raw_data)
            if frame.empty:
                return None
            return self.trend_analyzer.analyze(frame, code)
        except Exception as exc:
            logger.warning("[%s] trend analysis failed: %s", code, exc)
            return None

    def _enhance_context(
        self,
        context: dict[str, Any],
        realtime_quote: Any,
        chip_data: ChipDistribution | None,
        trend_result: TrendAnalysisResult | None,
        *,
        stock_name: str,
        runtime_account_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """将实时行情、筹码和运行时账户信息叠加到上下文中。"""
        enhanced = dict(context)

        if stock_name:
            enhanced["stock_name"] = stock_name
        elif realtime_quote and getattr(realtime_quote, "name", None):
            enhanced["stock_name"] = realtime_quote.name

        if realtime_quote:
            volume_ratio = getattr(realtime_quote, "volume_ratio", None)
            realtime_payload = {
                "name": getattr(realtime_quote, "name", ""),
                "price": getattr(realtime_quote, "price", None),
                "change_pct": getattr(realtime_quote, "change_pct", None),
                "volume_ratio": volume_ratio,
                "volume_ratio_desc": self._describe_volume_ratio(volume_ratio) if volume_ratio else "无数据",
                "turnover_rate": getattr(realtime_quote, "turnover_rate", None),
                "pe_ratio": getattr(realtime_quote, "pe_ratio", None),
                "pb_ratio": getattr(realtime_quote, "pb_ratio", None),
                "total_mv": getattr(realtime_quote, "total_mv", None),
                "circ_mv": getattr(realtime_quote, "circ_mv", None),
                "change_60d": getattr(realtime_quote, "change_60d", None),
                "source": getattr(realtime_quote, "source", None),
            }
            enhanced["realtime"] = {key: value for key, value in realtime_payload.items() if value is not None}

        if chip_data:
            current_price = getattr(realtime_quote, "price", 0) if realtime_quote else 0
            enhanced["chip"] = {
                "profit_ratio": chip_data.profit_ratio,
                "avg_cost": chip_data.avg_cost,
                "concentration_90": chip_data.concentration_90,
                "concentration_70": chip_data.concentration_70,
                "chip_status": chip_data.get_chip_status(current_price or 0),
            }

        if trend_result:
            enhanced["trend_analysis"] = {
                "trend_status": trend_result.trend_status.value,
                "ma_alignment": trend_result.ma_alignment,
                "trend_strength": trend_result.trend_strength,
                "bias_ma5": trend_result.bias_ma5,
                "bias_ma10": trend_result.bias_ma10,
                "volume_status": trend_result.volume_status.value,
                "volume_trend": trend_result.volume_trend,
                "buy_signal": trend_result.buy_signal.value,
                "signal_score": trend_result.signal_score,
                "signal_reasons": trend_result.signal_reasons,
                "risk_factors": trend_result.risk_factors,
            }

        if isinstance(runtime_account_context, dict):
            enhanced["runtime_account"] = self._build_runtime_account_payload(enhanced.get("code"), runtime_account_context)

        return enhanced

    @staticmethod
    def _build_runtime_account_payload(current_code: Any, runtime_account_context: dict[str, Any]) -> dict[str, Any]:
        """抽取当前股票维度最相关的账户信息给 AI 使用。"""
        account_snapshot = runtime_account_context.get("account_snapshot")
        summary = runtime_account_context.get("summary")
        positions = runtime_account_context.get("positions")
        account_snapshot = account_snapshot if isinstance(account_snapshot, dict) else {}
        summary = summary if isinstance(summary, dict) else {}
        position_list = positions if isinstance(positions, list) else []

        def pick_number(*values):
            """从多个候选值中挑选第一个可用数字。"""
            for value in values:
                try:
                    number = float(value)
                    if number == number:
                        return number
                except Exception:
                    continue
            return None

        current_code_text = str(current_code or "").strip()
        current_position = None
        for row in position_list:
            if not isinstance(row, dict):
                continue
            row_code = str(row.get("code") or row.get("stock_code") or row.get("symbol") or "").strip()
            if row_code != current_code_text:
                continue
            current_position = {
                "code": row_code,
                "quantity": int(float(row.get("quantity") or row.get("qty") or 0) or 0),
                "available_qty": int(
                    float(row.get("available_qty") or row.get("available") or row.get("quantity") or row.get("qty") or 0)
                    or 0
                ),
                "market_value": pick_number(row.get("market_value")),
            }
            break

        payload = {
            "cash": pick_number(
                account_snapshot.get("cash"),
                summary.get("cash"),
                summary.get("available_cash"),
                summary.get("availableCash"),
            ),
            "total_asset": pick_number(
                account_snapshot.get("total_asset"),
                summary.get("total_asset"),
                summary.get("totalAsset"),
                summary.get("total_equity"),
            ),
            "total_market_value": pick_number(
                account_snapshot.get("total_market_value"),
                summary.get("market_value"),
                summary.get("total_market_value"),
                summary.get("marketValue"),
            ),
            "position": current_position,
            "snapshot_at": account_snapshot.get("snapshot_at"),
            "data_source": account_snapshot.get("data_source"),
        }
        return {key: value for key, value in payload.items() if value is not None}

    @staticmethod
    def _describe_volume_ratio(volume_ratio: float) -> str:
        """把量比转换为更易读的中文描述。"""
        if volume_ratio < 0.5:
            return "极度萎缩"
        if volume_ratio < 0.8:
            return "明显萎缩"
        if volume_ratio < 1.2:
            return "正常"
        if volume_ratio < 2.0:
            return "温和放量"
        if volume_ratio < 3.0:
            return "明显放量"
        return "巨量"

    def _build_context_snapshot(
        self,
        *,
        enhanced_context: dict[str, Any],
        news_content: str | None,
        realtime_quote: Any,
        chip_data: ChipDistribution | None,
    ) -> dict[str, Any]:
        """构建可持久化的上下文快照。"""
        return {
            "enhanced_context": enhanced_context,
            "news_content": news_content,
            "realtime_quote_raw": self._safe_to_dict(realtime_quote),
            "chip_distribution_raw": self._safe_to_dict(chip_data),
        }

    @staticmethod
    def _safe_to_dict(value: Any) -> dict[str, Any] | None:
        """安全地将对象转换为字典。"""
        if value is None:
            return None
        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except Exception:
                return None
        if hasattr(value, "__dict__"):
            try:
                return dict(value.__dict__)
            except Exception:
                return None
        return None

    def _resolve_query_source(self, query_source: str | None) -> str:
        """推断本次查询来源。"""
        if query_source:
            return query_source
        if self.query_id:
            return "web"
        return "system"

    def _build_query_context(self, query_id: str | None = None) -> dict[str, str]:
        """构造搜索结果持久化所需的查询上下文。"""
        return {
            "query_id": query_id or self.query_id or "",
            "query_source": self.query_source or "",
        }
