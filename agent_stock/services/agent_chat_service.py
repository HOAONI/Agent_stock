# -*- coding: utf-8 -*-
"""Agent 问股聊天服务。"""

from __future__ import annotations

from calendar import monthrange
import asyncio
import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Awaitable, Callable

from data_provider.base import canonical_stock_code

from agent_stock.agents.chat_planner_agent import (
    AgentAccountState,
    AgentSystemState,
    ChatContextBundle,
    ChatPlannerAgent,
    ChatPlannerPlan,
    EffectiveUserPreferences,
    StageMemorySnapshot,
)
from agent_stock.agents.contracts import AgentRunResult, StockAgentResult
from agent_stock.agents.planner_runtime import compile_message_conditions
from agent_stock.analyzer import STOCK_NAME_MAP, get_analyzer
from agent_stock.config import Config, RuntimeLlmConfig, get_config, redact_sensitive_text
from agent_stock.repositories.chat_repo import AgentChatRepository
from agent_stock.services.backtest_interpretation_service import BacktestInterpretationService
from agent_stock.services.agent_chat_monitor_service import AgentChatMonitorService
from agent_stock.services.agent_service import AgentService
from agent_stock.services.agent_task_service import get_agent_task_service
from agent_stock.services.backend_agent_chat_client import BackendAgentChatClient
from agent_stock.services.runtime_market_service import RuntimeMarketService
from agent_stock.services.strategy_rule_dsl import (
    RULE_DSL_TEMPLATE_CODE,
    build_rule_dsl_from_text,
    build_rule_dsl_strategy_name,
)
from agent_stock.storage import DatabaseManager

logger = logging.getLogger(__name__)

_CHINESE_NUMERAL_DIGITS: dict[str, int] = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CHINESE_NUMERAL_UNITS: dict[str, int] = {
    "十": 10,
    "百": 100,
    "千": 1000,
    "万": 10000,
}

ChatEventHandler = Callable[[str, dict[str, Any]], Awaitable[None] | None]
BoardCatalogProvider = Callable[[str], list[dict[str, Any]]]
BoardConstituentsProvider = Callable[[str, str], list[dict[str, Any]]]

_OUTSIDE_TRADING_SESSION_REASON = "outside_trading_session"
_SUCCESSFUL_EXECUTION_STATUSES = {"filled", "submitted", "partial_filled", "partial"}

_A_SHARE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_HK_SHARE_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")
_US_SHARE_RE = re.compile(r"\b([A-Z]{2,5})\b")

_ANALYSIS_KEYWORDS = ("分析", "行情", "走势", "组合", "建议", "研判", "判断")
_ANALYSIS_VIEW_KEYWORDS = ("看看", "看下", "看一下")
_IMPLICIT_ANALYSIS_KEYWORDS = (
    "风险大吗",
    "风险高吗",
    "风险不大吧",
    "能买吗",
    "能不能买",
    "值不值得买",
    "适合上车吗",
    "适不适合买",
    "我该不该买",
    "现在能买",
    "可以买吗",
)
_ORDER_KEYWORDS = ("下单", "执行", "成交", "买入", "卖出", "买进", "卖掉", "模拟盘")
_BUY_KEYWORDS = ("买入", "买进", "买", "加仓")
_SELL_KEYWORDS = ("卖出", "卖掉", "卖", "减仓")
_HISTORY_KEYWORDS = ("历史分析", "分析记录", "最近分析", "上次分析", "之前分析")
_BACKTEST_KEYWORDS = ("回测", "胜率", "收益", "策略表现")
_STRATEGY_COMPARE_KEYWORDS = ("对比", "比较", "哪个好", "哪个更好", "优劣", "胜出")
_STRATEGY_BACKTEST_SIGNAL_KEYWORDS = (
    "macd",
    "金叉",
    "死叉",
    "均线",
    "ma",
    "rsi",
    "超卖",
    "超买",
)
_STRATEGY_BACKTEST_PERFORMANCE_KEYWORDS = (
    "回测",
    "历史",
    "过去",
    "最近",
    "近一年",
    "近半年",
    "近三个月",
    "近3个月",
    "近一个月",
    "近30天",
    "一年",
    "半年",
    "一个月",
    "三个月",
    "收益",
    "胜率",
    "回撤",
    "夏普",
    "表现",
    "怎样",
    "怎么样",
    "如何",
)
_STRATEGY_BACKTEST_STOCK_REF_IGNORE_TOKENS = frozenset(
    {
        "MACD",
        "RSI",
        "MA",
        "EMA",
        "KDJ",
        "BOLL",
        "DIF",
        "DEA",
        "金叉",
        "死叉",
        "均线",
        "超卖",
        "超买",
        "止损",
        "止盈",
        "收益",
        "回测",
        "胜率",
    }
)
_ACCOUNT_KEYWORDS = ("持仓", "账户", "仓位", "资金", "模拟盘情况", "现金")
_PORTFOLIO_HEALTH_KEYWORDS = (
    "仓位健康",
    "持仓健康",
    "组合健康",
    "投资组合",
    "整体风险",
    "风险分布",
    "再平衡",
    "行业过重",
    "行业集中",
    "集中度",
    "最大回撤",
    "回撤",
    "夏普",
    "夏普比率",
    "收益率",
)
_SAVE_KEYWORDS = ("保存本轮分析", "保存分析", "保存这轮分析")
_MARKET_WIDE_SCOPE_KEYWORDS = (
    "全市场",
    "所有股票",
    "全部股票",
    "全a股",
    "整个a股",
    "整个市场",
    "沪深两市",
)
_MARKET_WIDE_SELECTION_KEYWORDS = (
    "哪只最值得买",
    "哪几只最值得买",
    "哪个最值得买",
    "哪些最值得买",
    "最值得买",
    "最值得购买",
    "最该买",
    "推荐一只",
    "推荐几只",
    "选一只",
    "选几只",
    "最看好哪只",
    "最看好哪些",
)
_AUTONOMOUS_EXECUTION_KEYWORDS = (
    "根据结果决定是否去下单",
    "根据结果决定是否下单",
    "根据结果决定要不要下单",
    "根据分析结果决定是否下单",
    "你来决定是否下单",
    "你来决定要不要下单",
    "你来判断是否下单",
    "由你决定是否下单",
    "帮我决定是否下单",
    "如果合适就下单",
    "合适就下单",
    "适合就下单",
    "可以的话就下单",
    "可以就下单",
    "分析后直接下单",
    "分析完直接下单",
    "分析后如果合适就下单",
    "值得买就买",
)
_CONDITIONAL_EXECUTION_KEYWORDS = (
    "风险低的话",
    "风险不大的话",
    "风险可控的话",
    "如果风险低",
    "如果风险不大",
    "如果可以买",
    "可以买的话",
    "可以就买",
    "适合就买",
    "值得买就买",
    "合适就买",
)
_CONFIRMATION_SHORTCUT_KEYWORDS = (
    "确认",
    "确认买入",
    "确认卖出",
    "就按刚才的来",
    "按刚才的来",
    "就这么来",
    "那就买吧",
    "那就卖掉",
    "去下单吧",
)
_ORDER_ALL_KEYWORDS = (
    "全部",
    "都下",
    "都执行",
    "全部执行",
    "一起下",
    "一起执行",
    "按组合执行",
    "组合都下",
    "刚才那几笔",
    "这些都下",
    "全部候选单",
)
_ORDER_BEST_KEYWORDS = (
    "最看好",
    "最优",
    "最好的一笔",
    "最强的一笔",
    "最有把握",
)
_STRUCTURED_OUTPUT_LINE_RE = re.compile(r'^\s*"[\w\u4e00-\u9fff\s-]+"\s*:')
_STOCK_NAME_SPLIT_RE = re.compile(r"(?:、|,|，|/|以及|还有|和|及|与|跟)")
_STOCK_NAME_CONTEXT_SUFFIX_RE = re.compile(
    r"(?:的?(?:行情|走势|情况|表现|风险|机会|标的|研报|财报|基本面|技术面|消息面|股价)|怎么样|如何|咋样|大吗|高吗|呢|吗|吧|呀|啊|啦|了|？|\?)+$"
)
_STOCK_NAME_ALIAS_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "控股集团",
    "控股股份",
    "集团股份",
    "股份",
    "集团",
    "控股",
    "有限公司",
    "公司",
)
_DIRECT_STOCK_QUERY_FILLER_TOKENS = (
    "帮我分析一下",
    "帮我分析",
    "分析一下",
    "分析",
    "看一下",
    "看下",
    "看看",
    "研究一下",
    "研究",
    "聊聊",
    "聊下",
    "今天的",
    "今天",
    "当前的",
    "当前",
    "这个",
    "那个",
    "这只",
    "那只",
    "帮我",
    "给我",
    "请",
    "一下",
    "再",
    "呢",
    "吗",
    "吧",
    "呀",
    "啊",
    "啦",
    "了",
    "如何",
    "怎么样",
    "咋样",
)
_BOARD_SCOPE_SUFFIX_RE = re.compile(r"(?:行业板块|概念板块|行业|板块|概念|赛道)+$")
_BOARD_NAME_COLUMNS = ("板块名称", "板块", "名称", "name", "label", "board_name")
_BOARD_CODE_COLUMNS = ("板块代码", "代码", "board_code", "symbol", "code", "board_symbol")
_BOARD_STOCK_CODE_COLUMNS = ("代码", "股票代码", "证券代码", "code", "stock_code")
_BOARD_STOCK_NAME_COLUMNS = ("名称", "股票名称", "证券简称", "name", "stock_name")
_BOARD_TOTAL_MV_COLUMNS = ("总市值", "总市值-动态", "total_mv", "total_market_value")
_BOARD_AMOUNT_COLUMNS = ("成交额", "amount", "turnover", "成交金额")
_BOARD_COMPONENT_LIMIT = 10
_BOARD_AMBIGUOUS_LIMIT = 5
_ALLOWED_CHAT_RUNTIME_LLM_PROVIDERS = frozenset({"gemini", "anthropic", "openai", "deepseek", "custom"})
_GENERIC_STOCK_NAME_EXACT_BLOCKLIST = {
    "你自行决定",
    "自行决定",
    "帮我决定",
    "根据结果决定是否去下单",
    "根据结果决定是否下单",
    "根据结果决定要不要下单",
    "请明确您的指令",
    "账户情况",
    "账户",
    "持仓",
    "仓位",
    "资金",
    "模拟盘",
    "模拟盘情况",
    "最近分析记录",
    "分析记录",
    "历史分析",
    "最近分析",
    "上次分析",
    "之前分析",
    "回测",
    "胜率",
    "收益",
    "策略表现",
    "确认",
    "去下单吧",
    "就按刚才的来",
    "按刚才的来",
    "试一次",
    "再试一次",
    "再来一次",
}
_GENERIC_STOCK_NAME_FRAGMENT_KEYWORDS = (
    "分析",
    "行情",
    "走势",
    "情况",
    "表现",
    "风险",
    "机会",
    "标的",
    "建议",
    "判断",
    "研判",
    "记录",
    "历史",
    "账户",
    "持仓",
    "仓位",
    "资金",
    "模拟盘",
    "回测",
    "胜率",
    "收益",
    "策略",
    "下单",
    "确认",
    "结果",
    "决定",
    "刚才",
    "上一轮",
    "这轮",
    "今天",
)


@dataclass
class ChatPlan:
    """聊天消息的执行计划。"""

    primary_intent: str
    stock_codes: list[str] = field(default_factory=list)
    include_runtime_context: bool = False
    include_history: bool = False
    include_backtest: bool = False
    target_candidate_order: dict[str, Any] | None = None
    target_candidate_orders: list[dict[str, Any]] = field(default_factory=list)
    clarification: str | None = None
    save_requested: bool = False
    autonomous_execution_authorized: bool = False
    planner_source: str = "rule"
    requested_order_side: str | None = None
    requested_quantity: int | None = None
    conditions: list[dict[str, Any]] = field(default_factory=list)
    followup_reference: str | None = None
    intent_resolution: dict[str, Any] = field(default_factory=dict)
    pending_actions: list[dict[str, Any]] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    intent_source: str = "llm"
    stock_scope: dict[str, Any] = field(default_factory=dict)
    followup_target: dict[str, Any] = field(default_factory=dict)
    session_preference_overrides: dict[str, Any] = field(default_factory=dict)
    blocked_code: str | None = None


@dataclass(frozen=True)
class ChatLlmSelection:
    """单次聊天请求实际采用的聊天侧 LLM 解析结果。"""

    analyzer: Any
    source: str
    provider: str = ""
    base_url: str = ""
    model: str = ""


class AgentChatHandledError(RuntimeError):
    """表示本轮失败已被持久化为 assistant 消息，可安全回传给客户端。"""

    def __init__(self, message: str, final_payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.final_payload = dict(final_payload)


class AgentChatService:
    """负责会话、多轮上下文、工具调用和结果汇总。"""

    def __init__(
        self,
        *,
        config: Config | None = None,
        db_manager: DatabaseManager | None = None,
        chat_repo: AgentChatRepository | None = None,
        agent_service: AgentService | None = None,
        backend_client: BackendAgentChatClient | None = None,
        monitor_service: AgentChatMonitorService | None = None,
        backtest_interpretation_service: BacktestInterpretationService | None = None,
        analyzer=None,
        analyzer_factory: Callable[[RuntimeLlmConfig | None], Any] | None = None,
        board_catalog_provider: BoardCatalogProvider | None = None,
        board_constituents_provider: BoardConstituentsProvider | None = None,
    ) -> None:
        self.config = config or get_config()
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = chat_repo or AgentChatRepository(self.db)
        self.monitor_service = monitor_service or AgentChatMonitorService(self.repo)
        self.agent_service = agent_service or AgentService(config=self.config, db_manager=self.db)
        self.backend_client = backend_client or BackendAgentChatClient(config=self.config)
        self.backtest_interpretation_service = backtest_interpretation_service or BacktestInterpretationService(
            config=self.config,
        )
        self._analyzer_factory = analyzer_factory or (
            lambda runtime_llm=None: get_analyzer(config=self.config, runtime_llm=runtime_llm)
        )
        self.default_analyzer = analyzer or self._analyzer_factory(None)
        self.chat_planner = ChatPlannerAgent(config=self.config)
        self.runtime_market_service = RuntimeMarketService(config=self.config)
        self._stock_name_lookup_lock = threading.Lock()
        self._board_lookup_lock = threading.Lock()
        self._static_stock_name_profiles = self._build_static_stock_name_profiles()
        self._static_stock_name_index = self._build_static_stock_name_index()
        self._dynamic_stock_name_profiles: dict[str, dict[str, Any]] | None = None
        self._dynamic_stock_name_index: dict[str, list[str]] | None = None
        self._stock_name_lookup_manager: Any | None = None
        self._board_catalog_provider = board_catalog_provider or self._default_board_catalog_provider
        self._board_constituents_provider = board_constituents_provider or self._default_board_constituents_provider
        self._board_catalog_cache: dict[str, list[dict[str, Any]]] = {}
        self._board_constituents_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.tools = {
            "load_system_state": self._tool_load_system_state,
            "load_account_state": self._tool_load_account_state,
            "load_portfolio_health": self._tool_load_portfolio_health,
            "load_session_memory": self._tool_load_session_memory,
            "load_user_preferences": self._tool_load_user_preferences,
            "load_stage_memory": self._tool_load_stage_memory,
            "load_history": self._tool_load_history,
            "load_backtest": self._tool_load_backtest,
            "run_strategy_backtest": self._tool_run_strategy_backtest,
            "run_multi_stock_analysis": self._tool_run_multi_stock_analysis,
            "place_simulated_order": self._tool_place_simulated_order,
            "batch_execute_candidate_orders": self._tool_batch_execute_candidate_orders,
        }

    def _extract_runtime_llm(self, runtime_config: dict[str, Any] | None) -> RuntimeLlmConfig | None:
        """从请求级 runtime_config 中提取聊天侧要复用的 LLM 配置。"""
        if not isinstance(runtime_config, dict):
            return None

        llm_raw = runtime_config.get("llm")
        if not isinstance(llm_raw, dict):
            return None

        provider = str(llm_raw.get("provider") or "").strip().lower()
        base_url = str(llm_raw.get("base_url") or "").strip()
        model = str(llm_raw.get("model") or "").strip()
        if provider not in _ALLOWED_CHAT_RUNTIME_LLM_PROVIDERS or not base_url or not model:
            return None

        api_token = str(llm_raw.get("api_token") or "").strip() or None
        return RuntimeLlmConfig(
            provider=provider,
            base_url=base_url,
            model=model,
            api_token=api_token,
            has_token=bool(llm_raw.get("has_token") or api_token),
        )

    def _log_chat_llm_selection(self, selection: ChatLlmSelection, *, reason: str) -> None:
        """记录聊天请求最终采用的 LLM 来源，不输出 token。"""
        logger.info(
            "chat llm resolved (%s): source=%s provider=%s base_url=%s model=%s",
            reason,
            selection.source,
            selection.provider or "unknown",
            selection.base_url or "",
            selection.model or "",
        )

    def _resolve_chat_llm_selection(
        self,
        runtime_config: dict[str, Any] | None,
        *,
        reason: str,
        log_resolution: bool = True,
    ) -> ChatLlmSelection:
        """优先使用请求级 runtime llm；缺省时回退服务默认 analyzer。"""
        runtime_llm = self._extract_runtime_llm(runtime_config)
        if runtime_llm is not None:
            selection = ChatLlmSelection(
                analyzer=self._analyzer_factory(runtime_llm),
                source="runtime",
                provider=runtime_llm.provider,
                base_url=runtime_llm.base_url or "",
                model=runtime_llm.model or "",
            )
            if log_resolution:
                self._log_chat_llm_selection(selection, reason=reason)
            return selection

        default_meta = self.config.resolve_default_runtime_llm()
        selection = ChatLlmSelection(
            analyzer=self.default_analyzer,
            source="default",
            provider=default_meta.provider if default_meta else "",
            base_url=default_meta.base_url if default_meta else "",
            model=default_meta.model if default_meta else "",
        )
        if log_resolution:
            self._log_chat_llm_selection(selection, reason=reason)
        return selection

    async def handle_chat(
        self,
        payload: dict[str, Any],
        *,
        event_handler: ChatEventHandler | None = None,
    ) -> dict[str, Any]:
        """处理一轮聊天并返回统一 done 载荷。"""
        owner_user_id = int(payload.get("owner_user_id") or 0)
        if owner_user_id <= 0:
            raise ValueError("owner_user_id is required")

        message = str(payload.get("message") or "").strip()
        if not message:
            raise ValueError("message is required")

        username = str(payload.get("username") or "").strip() or f"u{owner_user_id}"
        session_id = str(payload.get("session_id") or "").strip() or uuid.uuid4().hex
        incoming_context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        runtime_config = payload.get("runtime_config") if isinstance(payload.get("runtime_config"), dict) else {}
        chat_llm = self._resolve_chat_llm_selection(runtime_config, reason="handle_chat")

        existing_session = self.repo.get_session(owner_user_id, session_id)
        existing_context = existing_session.get("context") if isinstance(existing_session, dict) else {}
        frontend_context = self._merge_session_context(
            existing_context if isinstance(existing_context, dict) else {},
            incoming_context,
        )
        conversation_state = self._normalize_conversation_state(frontend_context.get("conversation_state"))
        agent_preferences = self._normalize_agent_preferences(frontend_context.get("agent_chat_preferences"))

        self.repo.ensure_session(
            owner_user_id=owner_user_id,
            session_id=session_id,
            title=frontend_context.get("title") if isinstance(frontend_context, dict) else None,
            context=frontend_context,
        )
        self.repo.add_message(
            owner_user_id=owner_user_id,
            session_id=session_id,
            role="user",
            content=message,
            meta={"username": username},
        )
        session_header = self.repo.get_session(owner_user_id, session_id) or {}
        self.monitor_service.start_run(
            owner_user_id=owner_user_id,
            session_id=session_id,
            title=str(session_header.get("title") or ""),
            user_message=message,
        )

        client_event_handler = event_handler

        async def emit_event(event_name: str, event_payload: dict[str, Any]) -> None:
            self.monitor_service.record_event(
                owner_user_id=owner_user_id,
                session_id=session_id,
                event_name=event_name,
                payload=event_payload,
            )
            if client_event_handler is None:
                return
            maybe_result = client_event_handler(event_name, event_payload)
            if asyncio.iscoroutine(maybe_result):
                await maybe_result

        event_handler = emit_event

        try:
            await self._emit(event_handler, "thinking", {"message": "正在理解你的问题"})
            recent_assistant_messages = self.repo.list_recent_assistant_messages(owner_user_id, session_id, limit=5)
            latest_assistant_message = recent_assistant_messages[0] if recent_assistant_messages else None
            latest_assistant_meta = latest_assistant_message.get("meta") if latest_assistant_message else {}
            plan = await asyncio.to_thread(
                self._build_llm_plan,
                message=message,
                frontend_context=frontend_context,
                latest_assistant_meta=latest_assistant_meta if isinstance(latest_assistant_meta, dict) else {},
                recent_assistant_messages=recent_assistant_messages,
                conversation_state=conversation_state,
                llm_selection=chat_llm,
            )

            if plan.primary_intent in {"clarify", "unsupported"}:
                content = plan.clarification or "请告诉我你想分析哪只股票，或者明确说明要执行哪一笔候选订单。"
                structured_intent = "unsupported" if plan.primary_intent == "unsupported" else "clarify"
                final_payload = {
                    "session_id": session_id,
                    "content": content,
                    "structured_result": {
                        "intent": structured_intent,
                        "message": content,
                        "blocked_code": plan.blocked_code,
                        "supported": structured_intent != "unsupported",
                        "intent_source": plan.intent_source,
                        "loaded_context_keys": [],
                        "effective_preferences": {},
                        "stage_memory": {},
                    },
                    "candidate_orders": [],
                    "execution_result": None,
                    "status": "blocked",
                }
                return await self._finalize_chat_response(
                    owner_user_id=owner_user_id,
                    session_id=session_id,
                    frontend_context=frontend_context,
                    conversation_state=conversation_state,
                    plan=plan,
                    user_message=message,
                    final_payload=final_payload,
                )

            if plan.save_requested:
                content = "当前版本暂未开放“保存本轮分析”为正式分析记录，但聊天会话和候选订单已经保留。"
                final_payload = {
                    "session_id": session_id,
                    "content": content,
                    "structured_result": {"intent": "save_analysis", "supported": False},
                    "candidate_orders": [],
                    "execution_result": None,
                    "status": "blocked",
                }
                return await self._finalize_chat_response(
                    owner_user_id=owner_user_id,
                    session_id=session_id,
                    frontend_context=frontend_context,
                    conversation_state=conversation_state,
                    plan=plan,
                    user_message=message,
                    final_payload=final_payload,
                )

            tool_context: dict[str, Any] = {
                "owner_user_id": owner_user_id,
                "username": username,
                "session_id": session_id,
                "runtime_config": runtime_config,
                "frontend_context": frontend_context,
                "event_handler": event_handler,
                "conversation_state": conversation_state,
                "recent_assistant_messages": recent_assistant_messages,
            }

            context_bundle = await self._load_context_bundle(
                plan=plan,
                tool_context=tool_context,
                event_handler=event_handler,
            )
            account_state_payload = context_bundle.account_state.to_dict() if context_bundle.account_state else {}
            portfolio_health_payload = dict(context_bundle.portfolio_health or {})
            runtime_context_payload = account_state_payload.get("runtime_context") if isinstance(account_state_payload.get("runtime_context"), dict) else None
            history_payload = context_bundle.history if context_bundle.history else None
            backtest_payload = context_bundle.backtest if context_bundle.backtest else None
            effective_preferences_payload = context_bundle.effective_user_preferences.to_dict()
            stage_memory_payload = context_bundle.stage_memory.to_dict()

            if plan.primary_intent == "order_followup_single":
                execution_result = await self._run_tool(
                    "place_simulated_order",
                    {
                        "owner_user_id": owner_user_id,
                        "session_id": session_id,
                        "candidate_order": plan.target_candidate_order or {},
                    },
                    tool_context,
                    event_handler,
                )
                next_stage_memory = self._merge_stage_memory(
                    stage_memory_payload,
                    self._build_stage_memory_from_structured_result({}, execution_result),
                )
                content = self._render_execution_content(execution_result)
                final_payload = {
                    "session_id": session_id,
                    "content": content,
                    "structured_result": {
                        "intent": "order_followup_single",
                        "candidate_order": plan.target_candidate_order,
                        "loaded_context_keys": list(context_bundle.loaded_keys),
                        "effective_preferences": effective_preferences_payload,
                        "stage_memory": next_stage_memory,
                        "intent_source": plan.intent_source,
                    },
                    "candidate_orders": [plan.target_candidate_order] if plan.target_candidate_order else [],
                    "execution_result": execution_result,
                    "status": self._resolve_execution_status(execution_result),
                }
            elif plan.primary_intent == "order_followup_all":
                execution_result = await self._run_tool(
                    "batch_execute_candidate_orders",
                    {
                        "owner_user_id": owner_user_id,
                        "session_id": session_id,
                        "candidate_orders": plan.target_candidate_orders,
                    },
                    tool_context,
                    event_handler,
                )
                next_stage_memory = self._merge_stage_memory(
                    stage_memory_payload,
                    self._build_stage_memory_from_structured_result({}, execution_result),
                )
                content = self._render_execution_content(execution_result)
                final_payload = {
                    "session_id": session_id,
                    "content": content,
                    "structured_result": {
                        "intent": "order_followup_all",
                        "candidate_orders": plan.target_candidate_orders,
                        "loaded_context_keys": list(context_bundle.loaded_keys),
                        "effective_preferences": effective_preferences_payload,
                        "stage_memory": next_stage_memory,
                        "intent_source": plan.intent_source,
                    },
                    "candidate_orders": list(plan.target_candidate_orders),
                    "execution_result": execution_result,
                    "status": self._resolve_execution_status(execution_result),
                }
            elif plan.primary_intent == "portfolio_health":
                portfolio_health_result = portfolio_health_payload.get("portfolio_health") if isinstance(portfolio_health_payload.get("portfolio_health"), dict) else {}
                portfolio_stock_codes = self._extract_portfolio_health_stock_codes(
                    account_state_payload=account_state_payload,
                    portfolio_health=portfolio_health_result,
                )
                analysis_payload: dict[str, Any] | None = None
                candidate_orders: list[dict[str, Any]] = []
                if portfolio_stock_codes:
                    analysis_payload = await self._run_tool(
                        "run_multi_stock_analysis",
                        {
                            "stock_codes": portfolio_stock_codes,
                            "runtime_context_payload": runtime_context_payload,
                            "runtime_config": runtime_config,
                            "planning_context": {
                                "message": message,
                                "user_message": message,
                                "intent": plan.primary_intent,
                                "primary_intent": plan.primary_intent,
                                "loaded_context": context_bundle.to_dict(),
                                "session_overrides": dict(plan.session_preference_overrides),
                                "stage_memory": dict(stage_memory_payload),
                                "intent_resolution": dict(plan.intent_resolution or {}),
                            },
                        },
                        tool_context,
                        event_handler,
                    )
                    candidate_orders = self._normalize_candidate_orders(analysis_payload.get("candidate_orders"))
                portfolio_stage_memory = self._build_portfolio_health_stage_memory(
                    portfolio_health=portfolio_health_result,
                )
                next_stage_memory = self._merge_stage_memory(stage_memory_payload, portfolio_stage_memory)
                if isinstance(analysis_payload, dict):
                    next_stage_memory = self._merge_stage_memory(
                        next_stage_memory,
                        self._build_stage_memory_from_structured_result(
                            analysis_payload.get("structured_result") if isinstance(analysis_payload.get("structured_result"), dict) else {},
                            None,
                        ),
                    )
                content = self._render_portfolio_health_content(
                    portfolio_health=portfolio_health_result,
                    analysis_payload=analysis_payload,
                    effective_preferences=effective_preferences_payload,
                )
                final_payload = {
                    "session_id": session_id,
                    "content": content,
                    "structured_result": {
                        "intent": "portfolio_health",
                        "portfolio_health": portfolio_health_result,
                        "account_state": account_state_payload if account_state_payload else None,
                        "analysis": analysis_payload.get("structured_result") if isinstance(analysis_payload, dict) else None,
                        "loaded_context_keys": list(context_bundle.loaded_keys),
                        "effective_preferences": effective_preferences_payload,
                        "stage_memory": next_stage_memory,
                        "intent_source": plan.intent_source,
                    },
                    "candidate_orders": candidate_orders,
                    "execution_result": None,
                    "status": "analysis_only",
                }
            elif plan.primary_intent == "backtest" and "run_strategy_backtest" in plan.required_tools:
                try:
                    strategy_backtest_request = self._build_strategy_backtest_request(
                        message=message,
                        stock_codes=plan.stock_codes,
                        runtime_config=runtime_config,
                        account_state_payload=account_state_payload,
                    )
                except ValueError as exc:
                    content = self._build_strategy_backtest_clarification(str(exc))
                    final_payload = {
                        "session_id": session_id,
                        "content": content,
                        "structured_result": {
                            "intent": "clarify",
                            "message": content,
                            "blocked_code": "strategy_backtest_request_invalid",
                            "supported": True,
                            "loaded_context_keys": list(context_bundle.loaded_keys),
                            "effective_preferences": effective_preferences_payload,
                            "stage_memory": stage_memory_payload,
                            "intent_source": plan.intent_source,
                        },
                        "candidate_orders": [],
                        "execution_result": None,
                        "status": "blocked",
                    }
                else:
                    strategy_backtest_payload = await self._run_tool(
                        "run_strategy_backtest",
                        {
                            "owner_user_id": owner_user_id,
                            "code": strategy_backtest_request["code"],
                            "start_date": strategy_backtest_request["start_date"],
                            "end_date": strategy_backtest_request["end_date"],
                            "strategies": strategy_backtest_request["strategies"],
                            "initial_capital": strategy_backtest_request.get("initial_capital"),
                        },
                        tool_context,
                        event_handler,
                    )
                    interpretation_payload = self._interpret_strategy_backtest_result(
                        strategy_backtest_payload,
                        runtime_config=runtime_config,
                    )
                    next_stage_memory = self._merge_stage_memory(
                        stage_memory_payload,
                        self._build_strategy_backtest_stage_memory(
                            backtest_request=strategy_backtest_request,
                            backtest_result=strategy_backtest_payload,
                        ),
                    )
                    content = self._render_strategy_backtest_content(
                        backtest_request=strategy_backtest_request,
                        backtest_result=strategy_backtest_payload,
                        interpretation_payload=interpretation_payload,
                    )
                    final_payload = {
                        "session_id": session_id,
                        "content": content,
                        "structured_result": {
                            "intent": "backtest",
                            "backtest_mode": "strategy_run",
                            "backtest_request": strategy_backtest_request,
                            "strategy_backtest": strategy_backtest_payload,
                            "interpretation": interpretation_payload,
                            "loaded_context_keys": list(context_bundle.loaded_keys),
                            "effective_preferences": effective_preferences_payload,
                            "stage_memory": next_stage_memory,
                            "intent_source": plan.intent_source,
                        },
                        "candidate_orders": [],
                        "execution_result": None,
                        "status": "analysis_only",
                    }
            elif plan.primary_intent in {"history", "backtest", "account"}:
                content = self._render_context_only_content(
                    account_state_payload=account_state_payload,
                    runtime_context_payload=runtime_context_payload,
                    history_payload=history_payload,
                    backtest_payload=backtest_payload,
                )
                final_payload = {
                    "session_id": session_id,
                    "content": content,
                    "structured_result": {
                        "intent": plan.primary_intent,
                        "account_state": account_state_payload if account_state_payload else None,
                        "runtime_context": runtime_context_payload,
                        "history": history_payload,
                        "backtest": backtest_payload,
                        "loaded_context_keys": list(context_bundle.loaded_keys),
                        "effective_preferences": effective_preferences_payload,
                        "stage_memory": stage_memory_payload,
                        "intent_source": plan.intent_source,
                    },
                    "candidate_orders": [],
                    "execution_result": None,
                    "status": "analysis_only",
                }
            else:
                analysis_stock_codes = self._normalize_stock_codes(plan.stock_codes)
                if not analysis_stock_codes:
                    content = "请告诉我你想分析的具体股票名称、行业板块、概念板块或 6 位股票代码。"
                    final_payload = {
                        "session_id": session_id,
                        "content": content,
                        "structured_result": {
                            "intent": "clarify",
                            "message": content,
                            "blocked_code": "analysis_scope_empty",
                            "supported": True,
                            "loaded_context_keys": list(context_bundle.loaded_keys),
                            "effective_preferences": effective_preferences_payload,
                            "stage_memory": stage_memory_payload,
                            "intent_source": plan.intent_source,
                        },
                        "candidate_orders": [],
                        "execution_result": None,
                        "status": "blocked",
                    }
                else:
                    analysis_payload = await self._run_tool(
                        "run_multi_stock_analysis",
                        {
                            "stock_codes": analysis_stock_codes,
                            "runtime_context_payload": runtime_context_payload,
                            "runtime_config": runtime_config,
                            "planning_context": {
                                "message": message,
                                "user_message": message,
                                "intent": plan.primary_intent,
                                "autonomous_execution_authorized": plan.autonomous_execution_authorized,
                                "primary_intent": plan.primary_intent,
                                "requested_order_side": plan.requested_order_side,
                                "requested_quantity": plan.requested_quantity,
                                "constraints": [dict(item) for item in plan.conditions if isinstance(item, dict)],
                                "conditions": [dict(item) for item in plan.conditions if isinstance(item, dict)],
                                "loaded_context": context_bundle.to_dict(),
                                "session_overrides": dict(plan.session_preference_overrides),
                                "stage_memory": dict(stage_memory_payload),
                                "intent_resolution": dict(plan.intent_resolution or {}),
                            },
                        },
                        tool_context,
                        event_handler,
                    )
                    candidate_orders = self._resolve_candidate_orders_for_request(
                        plan=plan,
                        analysis_payload=analysis_payload,
                    )
                    if plan.primary_intent == "analysis_then_execute":
                        planner_execution_result = analysis_payload.get("execution_result") if isinstance(analysis_payload.get("execution_result"), dict) else None
                        execution_result: dict[str, Any] | None = planner_execution_result
                        if execution_result:
                            autonomous_execution = self._build_autonomous_execution_payload(
                                authorized=plan.autonomous_execution_authorized,
                                candidate_orders=candidate_orders,
                                execution_result=execution_result,
                                default_reason=str(execution_result.get("reason") or execution_result.get("status") or "execution_failed"),
                                default_gate_passed=False,
                                default_gate_message=str(execution_result.get("message") or ""),
                            )
                        else:
                            execution_gate = self._evaluate_analysis_execution_gate(
                                plan=plan,
                                analysis_payload=analysis_payload,
                                candidate_orders=candidate_orders,
                                runtime_config=runtime_config,
                            )
                            candidate_orders = self._normalize_candidate_orders(execution_gate.get("candidate_orders"))
                            autonomous_execution = self._build_autonomous_execution_payload(
                                authorized=plan.autonomous_execution_authorized,
                                candidate_orders=candidate_orders,
                                execution_result=None,
                                default_reason=str(execution_gate.get("reason") or "no_candidate_orders"),
                                default_gate_passed=bool(execution_gate.get("eligible")),
                                default_gate_message=str(execution_gate.get("message") or ""),
                            )
                            if plan.autonomous_execution_authorized and bool(execution_gate.get("eligible")) and candidate_orders:
                                execution_result = await self._run_tool(
                                    "batch_execute_candidate_orders",
                                    {
                                        "owner_user_id": owner_user_id,
                                        "session_id": session_id,
                                        "candidate_orders": candidate_orders,
                                    },
                                    tool_context,
                                    event_handler,
                                )
                                autonomous_execution = self._build_autonomous_execution_payload(
                                    authorized=plan.autonomous_execution_authorized,
                                    candidate_orders=candidate_orders,
                                    execution_result=execution_result,
                                    default_reason=str(execution_gate.get("reason") or "no_candidate_orders"),
                                    default_gate_passed=bool(execution_gate.get("eligible")),
                                    default_gate_message=str(execution_gate.get("message") or ""),
                                )
                        content = await self._render_analysis_then_execute_content(
                            analysis_payload=analysis_payload,
                            candidate_orders=candidate_orders,
                            execution_result=execution_result,
                            autonomous_execution=autonomous_execution,
                            runtime_context_payload=runtime_context_payload,
                            history_payload=history_payload,
                            backtest_payload=backtest_payload,
                            original_message=message,
                            llm_selection=chat_llm,
                        )
                        final_payload = {
                            "session_id": session_id,
                            "content": content,
                            "structured_result": {
                                "intent": "analysis_then_execute",
                                "analysis": analysis_payload.get("structured_result"),
                                "account_state": account_state_payload if account_state_payload else None,
                                "runtime_context": runtime_context_payload,
                                "history": history_payload,
                                "backtest": backtest_payload,
                                "autonomous_execution": autonomous_execution,
                                "loaded_context_keys": list(context_bundle.loaded_keys),
                                "effective_preferences": effective_preferences_payload,
                                "stage_memory": self._build_stage_memory_from_structured_result(
                                    analysis_payload.get("structured_result") if isinstance(analysis_payload.get("structured_result"), dict) else {},
                                    execution_result,
                                ),
                                "intent_source": plan.intent_source,
                            },
                            "candidate_orders": candidate_orders,
                            "execution_result": execution_result,
                            "status": self._resolve_execution_status(execution_result) if execution_result else "analysis_only",
                        }
                    else:
                        content = await self._render_analysis_content(
                            analysis_payload=analysis_payload,
                            runtime_context_payload=runtime_context_payload,
                            history_payload=history_payload,
                            backtest_payload=backtest_payload,
                            original_message=message,
                            llm_selection=chat_llm,
                        )
                        final_payload = {
                            "session_id": session_id,
                            "content": content,
                            "structured_result": {
                                "intent": "analysis",
                                "analysis": analysis_payload.get("structured_result"),
                                "account_state": account_state_payload if account_state_payload else None,
                                "runtime_context": runtime_context_payload,
                                "history": history_payload,
                                "backtest": backtest_payload,
                                "loaded_context_keys": list(context_bundle.loaded_keys),
                                "effective_preferences": effective_preferences_payload,
                                "stage_memory": self._build_stage_memory_from_structured_result(
                                    analysis_payload.get("structured_result") if isinstance(analysis_payload.get("structured_result"), dict) else {},
                                    None,
                                ),
                                "intent_source": plan.intent_source,
                            },
                            "candidate_orders": candidate_orders,
                            "execution_result": None,
                            "status": "analysis_only",
                        }

            await self._emit_outside_trading_session_warning_if_needed(
                event_handler,
                final_payload.get("execution_result") if isinstance(final_payload.get("execution_result"), dict) else None,
            )
            await self._emit_assistant_message_content(
                event_handler,
                str(final_payload.get("content") or ""),
            )
            return await self._finalize_chat_response(
                owner_user_id=owner_user_id,
                session_id=session_id,
                frontend_context=frontend_context,
                conversation_state=conversation_state,
                plan=plan,
                user_message=message,
                final_payload=final_payload,
            )
        except Exception as exc:
            safe_message = redact_sensitive_text(str(exc))
            logger.exception("Agent chat failed: session=%s owner=%s", session_id, owner_user_id)
            content = f"本轮 Agent 问股执行失败：{safe_message}"
            final_payload = {
                "session_id": session_id,
                "content": content,
                "structured_result": {"intent": "error", "message": safe_message},
                "candidate_orders": [],
                "execution_result": None,
                "status": "blocked",
            }
            await self._emit_assistant_message_content(event_handler, content)
            self.repo.add_message(
                owner_user_id=owner_user_id,
                session_id=session_id,
                role="assistant",
                content=content,
                meta=final_payload,
            )
            self.monitor_service.finalize_run(owner_user_id, session_id)
            raise AgentChatHandledError(safe_message, final_payload) from exc

    @staticmethod
    def _extract_analysis_result_for_mirror(structured_result: dict[str, Any]) -> dict[str, Any]:
        intent = str(structured_result.get("intent") or "").strip()
        if intent not in {"analysis", "analysis_then_execute", "portfolio_health"}:
            return {}
        analysis = structured_result.get("analysis")
        if not isinstance(analysis, dict):
            return {}
        stocks = analysis.get("stocks")
        if not isinstance(stocks, list) or not stocks:
            return {}
        return dict(analysis)

    @staticmethod
    def _extract_strategy_backtest_interpretation_for_mirror(structured_result: dict[str, Any]) -> dict[str, Any]:
        intent = str(structured_result.get("intent") or "").strip()
        backtest_mode = str(structured_result.get("backtest_mode") or "").strip()
        if intent != "backtest" or backtest_mode != "strategy_run":
            return {}

        strategy_backtest = structured_result.get("strategy_backtest")
        if not isinstance(strategy_backtest, dict):
            return {}
        run_group_id = int(strategy_backtest.get("run_group_id") or 0)
        if run_group_id <= 0:
            return {}

        interpretation = structured_result.get("interpretation")
        if not isinstance(interpretation, dict):
            return {}
        items = [dict(item) for item in interpretation.get("items") or [] if isinstance(item, dict)]
        if not items:
            return {}

        return {
            "run_group_id": run_group_id,
            "items": items,
        }

    async def _mirror_analysis_history_if_needed(
        self,
        *,
        owner_user_id: int,
        session_id: str,
        assistant_message_id: int,
        structured_result: dict[str, Any],
    ) -> None:
        analysis_result = self._extract_analysis_result_for_mirror(structured_result)
        if not analysis_result:
            return
        news_items_by_stock = self._extract_news_items_by_stock_for_mirror(analysis_result)

        try:
            await self.backend_client.save_analysis_records(
                owner_user_id=owner_user_id,
                session_id=session_id,
                assistant_message_id=assistant_message_id,
                analysis_result=analysis_result,
                news_items_by_stock=news_items_by_stock or None,
            )
        except Exception as exc:
            logger.warning(
                "Mirror analysis history failed: session=%s owner=%s assistant_message_id=%s error=%s",
                session_id,
                owner_user_id,
                assistant_message_id,
                redact_sensitive_text(str(exc)),
            )

    async def _mirror_strategy_backtest_interpretation_if_needed(
        self,
        *,
        owner_user_id: int,
        session_id: str,
        structured_result: dict[str, Any],
    ) -> dict[str, Any]:
        mirror_payload = self._extract_strategy_backtest_interpretation_for_mirror(structured_result)
        if not mirror_payload:
            return {}

        try:
            return await self.backend_client.save_strategy_backtest_interpretation(
                owner_user_id=owner_user_id,
                run_group_id=int(mirror_payload.get("run_group_id") or 0),
                items=[dict(item) for item in mirror_payload.get("items") or [] if isinstance(item, dict)],
            )
        except Exception as exc:
            logger.warning(
                "Mirror strategy backtest interpretation failed: session=%s owner=%s run_group_id=%s error=%s",
                session_id,
                owner_user_id,
                int(mirror_payload.get("run_group_id") or 0),
                redact_sensitive_text(str(exc)),
            )
            return {}

    async def _finalize_chat_response(
        self,
        *,
        owner_user_id: int,
        session_id: str,
        frontend_context: dict[str, Any],
        conversation_state: dict[str, Any],
        plan: ChatPlan,
        user_message: str,
        final_payload: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_orders = self._normalize_candidate_orders(final_payload.get("candidate_orders"))
        final_payload["candidate_orders"] = candidate_orders
        structured_result = final_payload.get("structured_result") if isinstance(final_payload.get("structured_result"), dict) else {}
        structured_result = dict(structured_result)
        structured_result.setdefault("intent_source", plan.intent_source)
        structured_result.setdefault("loaded_context_keys", [])
        structured_result.setdefault("effective_preferences", {})
        structured_result.setdefault("stage_memory", {})
        structured_result["intent_resolution"] = dict(plan.intent_resolution or {})
        next_state = self._build_next_conversation_state(
            current_state=conversation_state,
            plan=plan,
            user_message=str(user_message or "").strip(),
            response_status=str(final_payload.get("status") or "").strip() or "analysis_only",
            structured_result=structured_result,
            candidate_orders=candidate_orders,
            execution_result=final_payload.get("execution_result") if isinstance(final_payload.get("execution_result"), dict) else None,
        )
        structured_result["conversation_state"] = next_state
        structured_result["pending_actions"] = list(next_state.get("pending_actions") or [])
        final_payload["structured_result"] = structured_result

        merged_context = dict(frontend_context or {})
        merged_context["conversation_state"] = next_state
        self.repo.update_session_context(
            owner_user_id=owner_user_id,
            session_id=session_id,
            context=merged_context,
        )
        assistant_message = self.repo.add_message(
            owner_user_id=owner_user_id,
            session_id=session_id,
            role="assistant",
            content=str(final_payload.get("content") or ""),
            meta=final_payload,
        )
        assistant_message_id = int(assistant_message.get("id") or 0)
        if assistant_message_id > 0:
            await self._mirror_analysis_history_if_needed(
                owner_user_id=owner_user_id,
                session_id=session_id,
                assistant_message_id=assistant_message_id,
                structured_result=structured_result,
            )
            backtest_mirror_result = await self._mirror_strategy_backtest_interpretation_if_needed(
                owner_user_id=owner_user_id,
                session_id=session_id,
                structured_result=structured_result,
            )
            if backtest_mirror_result:
                strategy_backtest = structured_result.get("strategy_backtest")
                if isinstance(strategy_backtest, dict):
                    strategy_backtest["ai_interpretation_status"] = str(
                        backtest_mirror_result.get("ai_interpretation_status") or "completed"
                    ).strip() or "completed"
        self.monitor_service.finalize_run(owner_user_id, session_id)
        return final_payload

    def _build_next_conversation_state(
        self,
        *,
        current_state: dict[str, Any],
        plan: ChatPlan,
        user_message: str,
        response_status: str,
        structured_result: dict[str, Any],
        candidate_orders: list[dict[str, Any]],
        execution_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        next_state = self._normalize_conversation_state(current_state)
        next_state["last_intent"] = str(structured_result.get("intent") or plan.primary_intent or "").strip()
        next_state["last_requested_constraints"] = {
            "order_side": plan.requested_order_side,
            "quantity": plan.requested_quantity,
            "conditions": [dict(item) for item in plan.conditions if isinstance(item, dict)],
            "followup_reference": plan.followup_reference,
        }
        next_state["last_execution_result"] = dict(execution_result) if isinstance(execution_result, dict) else None
        merged_session_overrides = dict(next_state.get("session_preference_overrides") or {})
        merged_session_overrides.update(self._normalize_session_preference_overrides(plan.session_preference_overrides))
        next_state["session_preference_overrides"] = merged_session_overrides

        focus_codes = self._normalize_stock_codes(plan.stock_codes)
        if not focus_codes:
            analysis_focus = self._extract_analysis_focus_codes(structured_result)
            focus_codes = analysis_focus or self._normalize_stock_codes(next_state.get("focus_stocks"))
        next_state["focus_stocks"] = focus_codes

        analysis_summary = self._build_analysis_summary_snapshot(structured_result)
        if analysis_summary:
            next_state["last_analysis_summary"] = analysis_summary

        pending_actions = self._resolve_next_pending_actions(
            current_pending=self._normalize_pending_actions(next_state.get("pending_actions")),
            plan=plan,
            candidate_orders=candidate_orders,
            execution_result=execution_result,
            intent=str(structured_result.get("intent") or plan.primary_intent or "").strip(),
        )
        next_state["pending_actions"] = pending_actions
        next_state["last_candidate_snapshot"] = {
            "candidate_orders": [dict(item) for item in candidate_orders],
            "pending_action_count": len(pending_actions),
        }
        next_stage_memory = self._merge_stage_memory(
            next_state.get("last_stage_memory"),
            structured_result.get("stage_memory"),
        )
        next_state["last_stage_memory"] = next_stage_memory
        last_execution_result = next_state.get("last_execution_result") if isinstance(next_state.get("last_execution_result"), dict) else {}
        failure_reason = (
            str(next_stage_memory.get("failure_reason") or "").strip()
            or str(last_execution_result.get("message") or "").strip()
            or str(last_execution_result.get("status") or "").strip()
            or None
        )
        next_state["current_task"] = {
            "intent": str(structured_result.get("intent") or plan.primary_intent or "").strip(),
            "user_message": str(user_message or "").strip(),
            "status": str(response_status or "analysis_only").strip() or "analysis_only",
            "focus_stocks": list(focus_codes),
            "required_tools": list(plan.required_tools or []),
            "failure_reason": failure_reason,
        }
        next_state["last_tool_failures"] = self._build_last_tool_failures(
            structured_result=structured_result,
            execution_result=execution_result,
        )
        return next_state

    def _resolve_next_pending_actions(
        self,
        *,
        current_pending: list[dict[str, Any]],
        plan: ChatPlan,
        candidate_orders: list[dict[str, Any]],
        execution_result: dict[str, Any] | None,
        intent: str,
    ) -> list[dict[str, Any]]:
        if intent in {"analysis", "analysis_then_execute", "portfolio_health"}:
            if candidate_orders:
                failed_orders = []
                if isinstance(execution_result, dict):
                    failed_orders = self._normalize_candidate_orders(
                        [item.get("candidate_order") for item in execution_result.get("failed_orders") or [] if isinstance(item, dict)]
                    )
                if failed_orders:
                    return [dict(item) for item in failed_orders]
                executed_count = int(execution_result.get("executed_count") or 0) if isinstance(execution_result, dict) else 0
                if executed_count > 0 and not failed_orders:
                    return []
                return [dict(item) for item in candidate_orders]
            if plan.stock_codes or intent == "portfolio_health":
                return []
            return [dict(item) for item in current_pending]

        if intent == "order_followup_single":
            if self._is_outside_trading_session_execution(execution_result):
                if current_pending:
                    return [dict(item) for item in current_pending]
                target = plan.target_candidate_order if isinstance(plan.target_candidate_order, dict) else {}
                return [dict(target)] if target else []
            target = plan.target_candidate_order if isinstance(plan.target_candidate_order, dict) else {}
            target_key = self._candidate_order_key(target)
            return [dict(item) for item in current_pending if self._candidate_order_key(item) != target_key]

        if intent == "order_followup_all":
            if self._is_outside_trading_session_execution(execution_result):
                if current_pending:
                    return [dict(item) for item in current_pending]
                return [dict(item) for item in plan.target_candidate_orders if isinstance(item, dict)]
            return []

        return [dict(item) for item in current_pending]

    @staticmethod
    def _extract_analysis_focus_codes(structured_result: dict[str, Any]) -> list[str]:
        analysis = structured_result.get("analysis") if isinstance(structured_result.get("analysis"), dict) else structured_result
        stocks = analysis.get("stocks") if isinstance(analysis, dict) and isinstance(analysis.get("stocks"), list) else []
        codes: list[str] = []
        for item in stocks:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip()
            if code and code not in codes:
                codes.append(code)
        return codes

    @staticmethod
    def _build_analysis_summary_snapshot(structured_result: dict[str, Any]) -> dict[str, Any]:
        analysis = structured_result.get("analysis") if isinstance(structured_result.get("analysis"), dict) else structured_result
        if not isinstance(analysis, dict):
            return {}
        stocks = analysis.get("stocks") if isinstance(analysis.get("stocks"), list) else []
        portfolio_summary = analysis.get("portfolio_summary") if isinstance(analysis.get("portfolio_summary"), dict) else {}
        lead_stock = stocks[0] if stocks and isinstance(stocks[0], dict) else {}
        return {
            "trade_date": analysis.get("trade_date"),
            "stock_count": int(portfolio_summary.get("stock_count") or len(stocks)),
            "candidate_order_count": int(portfolio_summary.get("candidate_order_count") or 0),
            "lead_stock": {
                "code": lead_stock.get("code"),
                "name": lead_stock.get("name"),
                "operation_advice": lead_stock.get("operation_advice"),
                "sentiment_score": lead_stock.get("sentiment_score"),
            } if lead_stock else {},
        }

    def list_sessions(self, owner_user_id: int, limit: int = 50) -> dict[str, Any]:
        items = self.repo.list_sessions(owner_user_id, limit=limit)
        return {"total": len(items), "items": items}

    def get_session_detail(self, owner_user_id: int, session_id: str) -> dict[str, Any] | None:
        header = self.repo.get_session(owner_user_id, session_id)
        if not header:
            return None
        return {
            **header,
            "messages": self.repo.list_messages(owner_user_id, session_id),
        }

    def get_monitor_snapshot(self, owner_user_id: int) -> dict[str, Any]:
        return self.monitor_service.get_snapshot(owner_user_id)

    def subscribe_monitor(self, owner_user_id: int) -> asyncio.Queue[dict[str, Any]]:
        return self.monitor_service.subscribe(owner_user_id)

    def unsubscribe_monitor(self, owner_user_id: int, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.monitor_service.unsubscribe(owner_user_id, queue)

    def delete_session(self, owner_user_id: int, session_id: str) -> bool:
        return self.repo.delete_session(owner_user_id, session_id)

    def _build_plan(
        self,
        *,
        message: str,
        frontend_context: dict[str, Any],
        latest_assistant_meta: dict[str, Any],
        recent_assistant_messages: list[dict[str, Any]],
        conversation_state: dict[str, Any],
        agent_preferences: dict[str, Any],
        llm_selection: ChatLlmSelection | None = None,
    ) -> ChatPlan:
        normalized = message.strip()
        explicit_stock_codes, unresolved_stock_names = self._extract_explicit_stock_references(normalized)
        has_explicit_stock_reference = bool(explicit_stock_codes or unresolved_stock_names)
        latest_codes = self._extract_default_stock_codes(latest_assistant_meta, frontend_context, conversation_state)
        resolved_codes = explicit_stock_codes or ([] if has_explicit_stock_reference else latest_codes)
        pending_actions = self._normalize_pending_actions(conversation_state.get("pending_actions"))
        requested_order_side = self._extract_requested_order_side(normalized)
        requested_quantity = self._extract_requested_quantity(normalized)
        conditions = self._extract_conditions(normalized)

        contains_order = self._contains_order_intent(normalized) or bool(requested_order_side and conditions)
        contains_history = any(keyword in normalized for keyword in _HISTORY_KEYWORDS)
        contains_backtest = any(keyword in normalized for keyword in _BACKTEST_KEYWORDS)
        contains_account = any(keyword in normalized for keyword in _ACCOUNT_KEYWORDS)
        contains_save = any(keyword in normalized for keyword in _SAVE_KEYWORDS)
        contains_analysis = self._contains_analysis_intent(normalized, resolved_codes)
        confirmation_shortcuts_enabled = bool(agent_preferences.get("confirmation_shortcuts_enabled", True))
        followup_focus_enabled = bool(agent_preferences.get("followup_focus_resolution_enabled", True))
        execution_policy = str(agent_preferences.get("execution_policy") or "auto_execute_if_condition_met").strip()
        confirmation_shortcut = confirmation_shortcuts_enabled and self._message_is_confirmation_shortcut(normalized)
        conditional_trade_requested = bool(resolved_codes and requested_order_side and conditions)
        autonomous_execution_authorized = (
            self._contains_autonomous_execution_authorization(normalized)
            or (conditional_trade_requested and execution_policy == "auto_execute_if_condition_met")
        )
        candidate_snapshots = self._extract_candidate_snapshots(recent_assistant_messages)
        planner_hint = self._build_llm_planner_hint(
            message=normalized,
            explicit_stock_codes=explicit_stock_codes,
            fallback_focus_codes=latest_codes,
            resolved_codes=resolved_codes,
            candidate_snapshots=candidate_snapshots,
            contains_analysis=contains_analysis,
            contains_order=contains_order,
            contains_history=contains_history,
            contains_backtest=contains_backtest,
            contains_account=contains_account,
            autonomous_execution_authorized=autonomous_execution_authorized,
            analyzer=(llm_selection.analyzer if llm_selection is not None else self.default_analyzer),
        )
        hint_intent = str(planner_hint.get("intent") or "").strip()
        hint_codes = self._normalize_stock_codes(planner_hint.get("stock_codes"))
        plan_codes = explicit_stock_codes or hint_codes or ([] if has_explicit_stock_reference else latest_codes)
        include_history = contains_history or bool(planner_hint.get("include_history"))
        include_backtest = contains_backtest or bool(planner_hint.get("include_backtest"))
        include_runtime_context = bool(planner_hint.get("include_runtime_context")) or contains_order or conditional_trade_requested
        planner_source = str(planner_hint.get("planner_source") or "rule")
        planner_clarification = str(planner_hint.get("clarification") or "").strip()

        selected_orders, followup_reference, followup_ambiguous = self._resolve_followup_orders(
            message=normalized,
            stock_codes=explicit_stock_codes,
            pending_actions=pending_actions,
            candidate_snapshots=candidate_snapshots,
            prefer_all=hint_intent == "order_followup_all" or self._message_requests_all_orders(normalized),
            prefer_best=self._message_requests_best_order(normalized),
            allow_shortcut=confirmation_shortcut or contains_order,
        )
        if unresolved_stock_names and contains_analysis:
            clarification = self._build_stock_name_clarification(unresolved_stock_names)
            return ChatPlan(
                primary_intent="clarify",
                clarification=clarification,
                planner_source="rule",
                requested_order_side=requested_order_side,
                requested_quantity=requested_quantity,
                conditions=conditions,
                intent_resolution=self._build_intent_resolution(
                    intent="clarify",
                    stock_codes=[],
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    confidence=0.96,
                    missing_slots=["stock_codes"],
                    source="rule",
                ),
                pending_actions=pending_actions,
            )

        if not plan_codes and followup_focus_enabled and not has_explicit_stock_reference:
            focus_codes = self._normalize_stock_codes(conversation_state.get("focus_stocks"))
            if focus_codes:
                plan_codes = focus_codes

        if contains_save:
            return ChatPlan(
                primary_intent="save_analysis",
                save_requested=True,
                planner_source=planner_source,
                requested_order_side=requested_order_side,
                requested_quantity=requested_quantity,
                conditions=conditions,
                intent_resolution=self._build_intent_resolution(
                    intent="save_analysis",
                    stock_codes=plan_codes,
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    confidence=0.98,
                    missing_slots=[],
                    source=planner_source,
                ),
                pending_actions=pending_actions,
            )

        if followup_ambiguous:
            clarification = self._build_followup_clarification(pending_actions)
            return ChatPlan(
                primary_intent="clarify",
                clarification=clarification,
                planner_source=planner_source,
                requested_order_side=requested_order_side,
                requested_quantity=requested_quantity,
                conditions=conditions,
                followup_reference="pending_actions",
                intent_resolution=self._build_intent_resolution(
                    intent="clarify",
                    stock_codes=plan_codes,
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference="pending_actions",
                    confidence=0.92,
                    missing_slots=["pending_action_target"],
                    source=planner_source,
                ),
                pending_actions=pending_actions,
            )

        if hint_intent == "clarify" and planner_clarification and not plan_codes and not selected_orders:
            return ChatPlan(
                primary_intent="clarify",
                clarification=planner_clarification,
                planner_source=planner_source,
                requested_order_side=requested_order_side,
                requested_quantity=requested_quantity,
                conditions=conditions,
                intent_resolution=self._build_intent_resolution(
                    intent="clarify",
                    stock_codes=plan_codes,
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    confidence=0.55,
                    missing_slots=["stock_codes"],
                    source=planner_source,
                ),
                pending_actions=pending_actions,
            )

        if hint_intent == "history" and not contains_order and not contains_analysis:
            return ChatPlan(
                primary_intent="history",
                stock_codes=plan_codes,
                include_history=True,
                planner_source=planner_source,
                intent_resolution=self._build_intent_resolution(
                    intent="history",
                    stock_codes=plan_codes,
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    confidence=0.88,
                    missing_slots=[],
                    source=planner_source,
                ),
                pending_actions=pending_actions,
            )
        if hint_intent == "backtest" and not contains_order and not contains_analysis:
            return ChatPlan(
                primary_intent="backtest",
                stock_codes=plan_codes,
                include_backtest=True,
                planner_source=planner_source,
                intent_resolution=self._build_intent_resolution(
                    intent="backtest",
                    stock_codes=plan_codes,
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    confidence=0.88,
                    missing_slots=[],
                    source=planner_source,
                ),
                pending_actions=pending_actions,
            )
        if hint_intent == "account" and not contains_order and not contains_analysis:
            return ChatPlan(
                primary_intent="account",
                stock_codes=plan_codes,
                include_runtime_context=True,
                planner_source=planner_source,
                intent_resolution=self._build_intent_resolution(
                    intent="account",
                    stock_codes=plan_codes,
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    confidence=0.88,
                    missing_slots=[],
                    source=planner_source,
                ),
                pending_actions=pending_actions,
            )

        if selected_orders:
            if len(selected_orders) == 1 and hint_intent != "order_followup_all":
                target = dict(selected_orders[0])
                return ChatPlan(
                    primary_intent="order_followup_single",
                    stock_codes=[str(target.get("code") or "")],
                    target_candidate_order=target,
                    target_candidate_orders=[target],
                    planner_source=planner_source,
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    intent_resolution=self._build_intent_resolution(
                        intent="order_followup_single",
                        stock_codes=[str(target.get("code") or "")],
                        requested_order_side=requested_order_side or str(target.get("action") or "") or None,
                        requested_quantity=requested_quantity or int(target.get("quantity") or 0) or None,
                        conditions=conditions,
                        followup_reference=followup_reference,
                        confidence=0.96,
                        missing_slots=[],
                        source=planner_source,
                    ),
                    pending_actions=pending_actions,
                )
            return ChatPlan(
                primary_intent="order_followup_all",
                stock_codes=[str(item.get("code") or "") for item in selected_orders if str(item.get("code") or "").strip()],
                target_candidate_orders=[dict(item) for item in selected_orders],
                planner_source=planner_source,
                requested_order_side=requested_order_side,
                requested_quantity=requested_quantity,
                conditions=conditions,
                followup_reference=followup_reference,
                intent_resolution=self._build_intent_resolution(
                    intent="order_followup_all",
                    stock_codes=[str(item.get("code") or "") for item in selected_orders if str(item.get("code") or "").strip()],
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    confidence=0.95,
                    missing_slots=[],
                    source=planner_source,
                ),
                pending_actions=pending_actions,
            )

        if contains_history and (not plan_codes or not contains_analysis):
            return ChatPlan(
                primary_intent="history",
                stock_codes=plan_codes,
                include_history=True,
                planner_source=planner_source,
                intent_resolution=self._build_intent_resolution(
                    intent="history",
                    stock_codes=plan_codes,
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    confidence=0.92,
                    missing_slots=[],
                    source=planner_source,
                ),
                pending_actions=pending_actions,
            )

        if contains_backtest and (not plan_codes or not contains_analysis):
            return ChatPlan(
                primary_intent="backtest",
                stock_codes=plan_codes,
                include_backtest=True,
                planner_source=planner_source,
                intent_resolution=self._build_intent_resolution(
                    intent="backtest",
                    stock_codes=plan_codes,
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    confidence=0.92,
                    missing_slots=[],
                    source=planner_source,
                ),
                pending_actions=pending_actions,
            )

        if contains_account and (not plan_codes or not contains_analysis):
            return ChatPlan(
                primary_intent="account",
                stock_codes=plan_codes,
                include_runtime_context=True,
                planner_source=planner_source,
                intent_resolution=self._build_intent_resolution(
                    intent="account",
                    stock_codes=plan_codes,
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    confidence=0.92,
                    missing_slots=[],
                    source=planner_source,
                ),
                pending_actions=pending_actions,
            )

        if plan_codes:
            if conditional_trade_requested or (autonomous_execution_authorized and (hint_intent == "analysis_then_execute" or contains_order)):
                return ChatPlan(
                    primary_intent="analysis_then_execute",
                    stock_codes=plan_codes,
                    include_runtime_context=True,
                    include_history=include_history,
                    include_backtest=include_backtest,
                    autonomous_execution_authorized=True,
                    planner_source=planner_source,
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    intent_resolution=self._build_intent_resolution(
                        intent="analysis_then_execute",
                        stock_codes=plan_codes,
                        requested_order_side=requested_order_side,
                        requested_quantity=requested_quantity,
                        conditions=conditions,
                        followup_reference=followup_reference,
                        confidence=0.9,
                        missing_slots=[],
                        source=planner_source,
                    ),
                    pending_actions=pending_actions,
                )
            return ChatPlan(
                primary_intent="analysis",
                stock_codes=plan_codes,
                include_runtime_context=include_runtime_context or contains_analysis or contains_order,
                include_history=include_history,
                include_backtest=include_backtest,
                planner_source=planner_source,
                requested_order_side=requested_order_side,
                requested_quantity=requested_quantity,
                conditions=conditions,
                followup_reference=followup_reference,
                intent_resolution=self._build_intent_resolution(
                    intent="analysis",
                    stock_codes=plan_codes,
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    confidence=0.9 if contains_analysis else 0.78,
                    missing_slots=[],
                    source=planner_source,
                ),
                pending_actions=pending_actions,
            )

        if contains_analysis:
            return ChatPlan(
                primary_intent="clarify",
                clarification="请告诉我股票代码，例如“帮我分析一下今天的 600519 行情”。",
                planner_source=planner_source,
                requested_order_side=requested_order_side,
                requested_quantity=requested_quantity,
                conditions=conditions,
                intent_resolution=self._build_intent_resolution(
                    intent="clarify",
                    stock_codes=[],
                    requested_order_side=requested_order_side,
                    requested_quantity=requested_quantity,
                    conditions=conditions,
                    followup_reference=followup_reference,
                    confidence=0.85,
                    missing_slots=["stock_codes"],
                    source=planner_source,
                ),
                pending_actions=pending_actions,
            )

        return ChatPlan(
            primary_intent="clarify",
            clarification="请直接告诉我股票代码和需求，例如“分析 600519”“把刚才那几笔都下了”，或“根据结果决定是否下单”。",
            planner_source=planner_source,
            requested_order_side=requested_order_side,
            requested_quantity=requested_quantity,
            conditions=conditions,
            intent_resolution=self._build_intent_resolution(
                intent="clarify",
                stock_codes=[],
                requested_order_side=requested_order_side,
                requested_quantity=requested_quantity,
                conditions=conditions,
                followup_reference=followup_reference,
                confidence=0.4,
                missing_slots=["stock_codes"],
                source=planner_source,
            ),
            pending_actions=pending_actions,
        )

    def _build_llm_planner_hint(
        self,
        *,
        message: str,
        explicit_stock_codes: list[str],
        fallback_focus_codes: list[str],
        resolved_codes: list[str],
        candidate_snapshots: list[dict[str, Any]],
        contains_analysis: bool,
        contains_order: bool,
        contains_history: bool,
        contains_backtest: bool,
        contains_account: bool,
        autonomous_execution_authorized: bool,
        analyzer=None,
    ) -> dict[str, Any]:
        active_analyzer = analyzer or self.default_analyzer
        if not getattr(active_analyzer, "is_available", lambda: False)():
            return {}

        prompt = self._build_planner_prompt(
            message=message,
            explicit_stock_codes=explicit_stock_codes,
            fallback_focus_codes=fallback_focus_codes,
            resolved_codes=resolved_codes,
            candidate_snapshots=candidate_snapshots,
            contains_analysis=contains_analysis,
            contains_order=contains_order,
            contains_history=contains_history,
            contains_backtest=contains_backtest,
            contains_account=contains_account,
            autonomous_execution_authorized=autonomous_execution_authorized,
        )
        try:
            raw = active_analyzer.generate_text(
                prompt,
                temperature=0.0,
                max_output_tokens=400,
            )
        except Exception as exc:
            logger.warning("LLM chat planner fallback: %s", redact_sensitive_text(str(exc)))
            return {}

        payload = self._extract_json_object(raw)
        if not payload:
            return {}

        intent = str(payload.get("intent") or "").strip()
        if intent not in {"analysis", "analysis_then_execute", "order_followup_single", "order_followup_all", "history", "backtest", "account", "clarify"}:
            return {}

        hint_codes = self._normalize_stock_codes(payload.get("stock_codes"))
        if intent in {"analysis", "analysis_then_execute"} and not (hint_codes or resolved_codes):
            return {}
        if intent == "analysis_then_execute" and not autonomous_execution_authorized:
            return {}
        if intent in {"order_followup_single", "order_followup_all"} and not candidate_snapshots:
            return {}

        clarification = str(payload.get("clarification") or "").strip()
        return {
            "intent": intent,
            "stock_codes": hint_codes,
            "include_runtime_context": bool(payload.get("include_runtime_context")),
            "include_history": bool(payload.get("include_history")),
            "include_backtest": bool(payload.get("include_backtest")),
            "clarification": clarification,
            "planner_source": "llm",
        }

    @staticmethod
    def _build_planner_prompt(
        *,
        message: str,
        explicit_stock_codes: list[str],
        fallback_focus_codes: list[str],
        resolved_codes: list[str],
        candidate_snapshots: list[dict[str, Any]],
        contains_analysis: bool,
        contains_order: bool,
        contains_history: bool,
        contains_backtest: bool,
        contains_account: bool,
        autonomous_execution_authorized: bool,
    ) -> str:
        snapshot_lines: list[str] = []
        for index, snapshot in enumerate(candidate_snapshots[:3], start=1):
            codes = "、".join(str(item.get("code") or "") for item in snapshot.get("candidate_orders") or [])
            snapshot_lines.append(f"{index}. 候选单 {len(snapshot.get('candidate_orders') or [])} 笔：{codes}")
        candidate_summary = "\n".join(snapshot_lines) if snapshot_lines else "无可用候选单记忆"
        return (
            "你是一个中文股票交易聊天意图规划器，只输出严格 JSON，不要输出解释、Markdown 或代码块。\n"
            "允许的 intent 只有：analysis, analysis_then_execute, order_followup_single, order_followup_all, history, backtest, account, clarify。\n"
            "规则：1. 只有用户明确授权 Agent 根据分析结果自主决定是否下模拟盘单时，才能返回 analysis_then_execute；"
            "2. order_followup_* 只用于引用历史候选单；3. 无法确定时返回 clarify；"
            "4. 如果本轮消息已经明确提到新的股票，不得沿用上一轮焦点股票。\n\n"
            f"用户消息：{message}\n"
            f"本轮显式股票代码：{explicit_stock_codes}\n"
            f"上一轮焦点股票代码：{fallback_focus_codes}\n"
            f"预解析股票代码：{resolved_codes}\n"
            f"是否包含分析诉求：{contains_analysis}\n"
            f"是否包含下单诉求：{contains_order}\n"
            f"是否包含历史分析诉求：{contains_history}\n"
            f"是否包含回测诉求：{contains_backtest}\n"
            f"是否包含账户诉求：{contains_account}\n"
            f"是否已明确授权自主决定下单：{autonomous_execution_authorized}\n"
            f"最近候选单记忆：\n{candidate_summary}\n\n"
            "输出 JSON 字段：intent, stock_codes, include_runtime_context, include_history, include_backtest, clarification。"
        )

    @staticmethod
    def _extract_json_object(content: Any) -> dict[str, Any]:
        raw = AgentChatService._strip_code_fence(str(content or "").strip())
        if not raw:
            return {}
        candidates = [raw]
        matched = re.search(r"\{[\s\S]*\}", raw)
        if matched:
            candidates.append(matched.group(0))
        for item in candidates:
            try:
                value = json.loads(item)
            except Exception:
                continue
            if isinstance(value, dict):
                return value
        return {}

    def _normalize_stock_codes(self, values: Any) -> list[str]:
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return []
        normalized: list[str] = []
        for item in values:
            text = str(item or "").strip().upper()
            if not text:
                continue
            try:
                code = canonical_stock_code(text)
            except Exception:
                code = text
            if code not in normalized:
                normalized.append(code)
        return normalized

    @staticmethod
    def _contains_analysis_intent(message: str, stock_codes: list[str]) -> bool:
        if any(keyword in message for keyword in _ANALYSIS_KEYWORDS):
            return True
        if bool(stock_codes) and any(keyword in message for keyword in _ANALYSIS_VIEW_KEYWORDS):
            return True
        if bool(stock_codes) and any(keyword in message for keyword in _IMPLICIT_ANALYSIS_KEYWORDS):
            return True
        return bool(stock_codes) and "风险" in message and any(token in message for token in ("吗", "？", "?"))

    @classmethod
    def _contains_portfolio_health_intent(cls, message: str) -> bool:
        compact = cls._compact_message_text(message)
        if not compact:
            return False
        portfolio_scope_tokens = ("投资组合", "组合", "持仓", "仓位", "账户")
        if not any(token in compact for token in portfolio_scope_tokens):
            return False
        return any(token in compact for token in _PORTFOLIO_HEALTH_KEYWORDS)

    @staticmethod
    def _compact_message_text(message: str) -> str:
        return re.sub(r"\s+", "", str(message or "")).lower()

    @classmethod
    def _is_market_wide_selection_request(cls, message: str) -> bool:
        compact = cls._compact_message_text(message)
        if not compact:
            return False
        has_market_scope = any(keyword in compact for keyword in _MARKET_WIDE_SCOPE_KEYWORDS)
        if not has_market_scope:
            return False
        return any(keyword in compact for keyword in _MARKET_WIDE_SELECTION_KEYWORDS)

    @classmethod
    def _contains_strategy_backtest_run_intent(cls, message: str) -> bool:
        compact = cls._compact_message_text(message)
        if not compact:
            return False
        has_strategy_signal = any(keyword in compact for keyword in _STRATEGY_BACKTEST_SIGNAL_KEYWORDS)
        if not has_strategy_signal:
            return False
        return any(keyword in compact for keyword in _STRATEGY_BACKTEST_PERFORMANCE_KEYWORDS)

    @staticmethod
    def _parse_strategy_backtest_date_token(value: str) -> date | None:
        text = str(value or "").strip()
        if not text:
            return None
        for pattern in (
            r"(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})",
            r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日?",
        ):
            matched = re.search(pattern, text)
            if not matched:
                continue
            try:
                return date(
                    int(matched.group("year")),
                    int(matched.group("month")),
                    int(matched.group("day")),
                )
            except Exception:
                return None
        return None

    @staticmethod
    def _parse_chinese_numeral_integer(value: str) -> int | None:
        text = str(value or "").strip()
        if not text:
            return None

        total = 0
        section = 0
        number = 0
        seen_token = False
        for char in text:
            if char in _CHINESE_NUMERAL_DIGITS:
                number = _CHINESE_NUMERAL_DIGITS[char]
                seen_token = True
                continue
            unit = _CHINESE_NUMERAL_UNITS.get(char)
            if unit is None:
                return None
            seen_token = True
            if unit == 10000:
                section = (section + number) or 1
                total += section * unit
                section = 0
                number = 0
                continue
            current = number or 1
            section += current * unit
            number = 0

        if not seen_token:
            return None
        return total + section + number

    @classmethod
    def _parse_strategy_backtest_relative_quantity(cls, value: str) -> float | None:
        text = str(value or "").strip()
        if not text:
            return None
        if re.fullmatch(r"\d+", text):
            return float(int(text))
        if text == "半":
            return 0.5
        parsed = cls._parse_chinese_numeral_integer(text)
        return float(parsed) if parsed is not None else None

    @staticmethod
    def _subtract_calendar_months(base_date: date, months: int) -> date:
        if months <= 0:
            return base_date
        total_months = base_date.year * 12 + (base_date.month - 1) - months
        year = total_months // 12
        month = total_months % 12 + 1
        day = min(base_date.day, monthrange(year, month)[1])
        return date(year, month, day)

    @classmethod
    def _build_strategy_backtest_relative_window(
        cls,
        matched: re.Match[str],
        today: date,
    ) -> dict[str, str] | None:
        quantity_text = str(matched.group("quantity") or "").strip()
        quantity = cls._parse_strategy_backtest_relative_quantity(quantity_text)
        if quantity is None or quantity <= 0:
            return None

        unit = str(matched.group("unit") or "").strip()
        if unit in {"天", "日", "交易日"}:
            start_date = today - timedelta(days=max(1, int(round(quantity))))
            window_label = f"过去{quantity_text}天"
        elif unit in {"周", "星期"}:
            start_date = today - timedelta(days=max(1, int(round(quantity * 7))))
            window_label = f"过去{quantity_text}周"
        elif unit in {"个月", "月"}:
            if quantity == 0.5:
                start_date = today - timedelta(days=15)
                window_label = "过去半个月"
            elif quantity.is_integer():
                start_date = cls._subtract_calendar_months(today, int(quantity))
                window_label = f"过去{quantity_text}个月"
            else:
                return None
        elif unit == "年":
            if quantity == 0.5:
                start_date = cls._subtract_calendar_months(today, 6)
                window_label = "过去半年"
            elif quantity.is_integer():
                start_date = cls._subtract_calendar_months(today, int(quantity) * 12)
                window_label = f"过去{quantity_text}年"
            else:
                return None
        else:
            return None

        return {
            "start_date": start_date.isoformat(),
            "end_date": today.isoformat(),
            "window_label": window_label,
        }

    def _extract_strategy_backtest_window(self, message: str) -> dict[str, str]:
        today = date.today()
        absolute_range = re.search(
            r"((?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|(?:\d{4}年\d{1,2}月\d{1,2}日?))\s*(?:到|至|-|~|—)\s*"
            r"((?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|(?:\d{4}年\d{1,2}月\d{1,2}日?))",
            str(message or ""),
        )
        if absolute_range:
            start_date = self._parse_strategy_backtest_date_token(absolute_range.group(1))
            end_date = self._parse_strategy_backtest_date_token(absolute_range.group(2))
            if start_date is not None and end_date is not None and start_date <= end_date:
                return {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "window_label": f"{start_date.isoformat()} 到 {end_date.isoformat()}",
                }

        compact = self._compact_message_text(message)
        relative_range = re.search(
            r"(?:(?:过去|最近|近))(?P<quantity>\d+|[零〇一二两三四五六七八九十百千万半]+)"
            r"(?P<connector>个)?(?P<unit>交易日|天|日|周|星期|个月|月|年)",
            compact,
        )
        if relative_range:
            window = self._build_strategy_backtest_relative_window(relative_range, today)
            if window is not None:
                return window

        start_date = self._subtract_calendar_months(today, 12)
        return {
            "start_date": start_date.isoformat(),
            "end_date": today.isoformat(),
            "window_label": "过去一年",
        }

    @staticmethod
    def _extract_first_number(patterns: list[str], message: str) -> int | None:
        for pattern in patterns:
            matched = re.search(pattern, message, flags=re.I)
            if matched:
                try:
                    return int(matched.group(1))
                except Exception:
                    continue
        return None

    def _extract_strategy_definitions_from_message(self, message: str) -> list[dict[str, Any]]:
        compact = self._compact_message_text(message)
        raw_message = str(message or "")
        compare_requested = any(keyword in compact for keyword in _STRATEGY_COMPARE_KEYWORDS)
        if not compare_requested:
            dsl_params = build_rule_dsl_from_text(raw_message)
            if dsl_params is not None:
                return [
                    {
                        "strategy_name": build_rule_dsl_strategy_name(dsl_params),
                        "template_code": RULE_DSL_TEMPLATE_CODE,
                        "params": dsl_params,
                    }
                ]

        strategies: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add_strategy(template_code: str, strategy_name: str, params: dict[str, Any]) -> None:
            if template_code in seen:
                return
            seen.add(template_code)
            strategies.append(
                {
                    "strategy_name": strategy_name,
                    "template_code": template_code,
                    "params": params,
                }
            )

        macd_tuple = re.search(r"macd[\(\[]?\s*(\d{1,2})\s*[,/，]\s*(\d{1,3})\s*[,/，]\s*(\d{1,2})", compact, flags=re.I)
        if "macd" in compact and any(token in compact for token in ("金叉", "死叉", "上穿", "下穿")):
            macd_fast = int(macd_tuple.group(1)) if macd_tuple else 12
            macd_slow = int(macd_tuple.group(2)) if macd_tuple else 26
            macd_signal = int(macd_tuple.group(3)) if macd_tuple else 9
            add_strategy(
                "macd_cross",
                "MACD 金叉",
                {
                    "macdFast": macd_fast,
                    "macdSlow": macd_slow,
                    "macdSignal": macd_signal,
                },
            )

        if "rsi" in compact or ("超卖" in compact and "超买" in compact):
            rsi_period = self._extract_first_number([r"rsi\s*(\d{1,2})"], compact) or 14
            oversold_threshold = self._extract_first_number(
                [r"超卖(?:阈值)?\D{0,4}(\d{1,2})", r"低于\D{0,2}(\d{1,2})买入"],
                str(message or ""),
            ) or 30
            overbought_threshold = self._extract_first_number(
                [r"超买(?:阈值)?\D{0,4}(\d{1,2})", r"高于\D{0,2}(\d{1,2})卖出"],
                str(message or ""),
            ) or 70
            add_strategy(
                "rsi_threshold",
                "RSI 阈值",
                {
                    "rsiPeriod": rsi_period,
                    "oversoldThreshold": oversold_threshold,
                    "overboughtThreshold": overbought_threshold,
                },
            )

        has_ma_token = "均线" in compact or bool(re.search(r"(?i)\bma(?:\s*\d{1,3})?\b", raw_message))
        if has_ma_token and any(
            token in compact for token in ("上穿", "跌破", "交叉", "金叉", "死叉", "突破")
        ):
            ma_window = self._extract_first_number(
                [
                    r"(\d{1,3})\s*(?:日|天)?均线",
                    r"ma\s*(\d{1,3})",
                ],
                str(message or ""),
            ) or 20
            add_strategy(
                "ma_cross",
                f"MA{ma_window} 交叉",
                {"maWindow": ma_window},
            )

        return strategies

    def _extract_strategy_backtest_initial_capital(
        self,
        *,
        message: str,
        runtime_config: dict[str, Any] | None,
        account_state_payload: dict[str, Any] | None,
    ) -> float | None:
        capital_match = re.search(r"本金\s*(\d+(?:\.\d+)?)\s*(万|元)?", str(message or ""))
        if not capital_match:
            capital_match = re.search(r"(\d+(?:\.\d+)?)\s*(万|元)\s*本金", str(message or ""))
        if capital_match:
            base = float(capital_match.group(1))
            unit = str(capital_match.group(2) or "元").strip()
            return base * 10000.0 if unit == "万" else base

        runtime_account = runtime_config.get("account") if isinstance(runtime_config, dict) and isinstance(runtime_config.get("account"), dict) else {}
        runtime_initial = self._safe_number(runtime_account.get("initial_cash") or runtime_account.get("initialCapital"))
        if runtime_initial is not None and runtime_initial > 0:
            return runtime_initial

        account_state = account_state_payload.get("account_state") if isinstance(account_state_payload, dict) and isinstance(account_state_payload.get("account_state"), dict) else {}
        runtime_context = account_state_payload.get("runtime_context") if isinstance(account_state_payload, dict) and isinstance(account_state_payload.get("runtime_context"), dict) else {}
        runtime_summary = runtime_context.get("summary") if isinstance(runtime_context.get("summary"), dict) else {}
        for candidate in (
            runtime_summary.get("initial_capital"),
            runtime_summary.get("initial_cash"),
            account_state.get("total_asset"),
        ):
            value = self._safe_number(candidate)
            if value is not None and value > 0:
                return value
        return None

    def _build_strategy_backtest_request(
        self,
        *,
        message: str,
        stock_codes: list[str],
        runtime_config: dict[str, Any] | None,
        account_state_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized_codes = self._normalize_stock_codes(stock_codes)
        if len(normalized_codes) != 1:
            raise ValueError("strategy_backtest_requires_single_stock")

        strategies = self._extract_strategy_definitions_from_message(message)
        if not strategies:
            raise ValueError("strategy_backtest_strategy_unrecognized")

        window = self._extract_strategy_backtest_window(message)
        initial_capital = self._extract_strategy_backtest_initial_capital(
            message=message,
            runtime_config=runtime_config,
            account_state_payload=account_state_payload,
        )
        return {
            "code": normalized_codes[0],
            "start_date": window["start_date"],
            "end_date": window["end_date"],
            "window_label": window["window_label"],
            "strategies": strategies,
            "initial_capital": initial_capital,
            "compare_requested": any(keyword in self._compact_message_text(message) for keyword in _STRATEGY_COMPARE_KEYWORDS) or len(strategies) > 1,
        }

    @classmethod
    def _stock_scope_refs_hit_market_wide_boundary(cls, stock_refs: Any) -> bool:
        refs = stock_refs if isinstance(stock_refs, list) else []
        for item in refs:
            compact = cls._compact_message_text(str(item or ""))
            if compact and any(keyword in compact for keyword in _MARKET_WIDE_SCOPE_KEYWORDS):
                return True
        return False

    @staticmethod
    def _build_market_wide_scope_unsupported_message() -> str:
        return (
            "这个问题我目前做不到。"
            " 当前 Agent 只支持分析你明确提供的股票代码、股票名称、行业板块或概念板块，"
            "暂不支持直接对 A 股全市场做实时扫描、排序，或直接选出“今天最值得买”的股票。"
            " 你可以改成给我一个候选范围，例如“分析半导体板块里最值得买的股票”或“分析比亚迪、宁德时代、贵州茅台，告诉我今天最值得买的是哪只”。"
        )

    @staticmethod
    def _as_bool(value: Any, fallback: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
        return fallback

    def _normalize_agent_preferences(self, value: Any) -> dict[str, Any]:
        source = value if isinstance(value, dict) else {}
        execution_policy = str(
            source.get("execution_policy")
            or source.get("executionPolicy")
            or "auto_execute_if_condition_met"
        ).strip()
        response_style = str(source.get("response_style") or source.get("responseStyle") or "concise_factual").strip()
        return {
            "execution_policy": "confirm_before_execute" if execution_policy == "confirm_before_execute" else "auto_execute_if_condition_met",
            "confirmation_shortcuts_enabled": self._as_bool(
                source.get("confirmation_shortcuts_enabled", source.get("confirmationShortcutsEnabled")),
                True,
            ),
            "followup_focus_resolution_enabled": self._as_bool(
                source.get("followup_focus_resolution_enabled", source.get("followupFocusResolutionEnabled")),
                True,
            ),
            "response_style": response_style if response_style in {"balanced", "detailed"} else "concise_factual",
        }

    def _normalize_conversation_state(self, value: Any) -> dict[str, Any]:
        source = value if isinstance(value, dict) else {}
        focus_stocks = self._normalize_stock_codes(source.get("focus_stocks"))
        last_candidate_snapshot = source.get("last_candidate_snapshot") if isinstance(source.get("last_candidate_snapshot"), dict) else {}
        last_requested_constraints = source.get("last_requested_constraints") if isinstance(source.get("last_requested_constraints"), dict) else {}
        last_analysis_summary = source.get("last_analysis_summary") if isinstance(source.get("last_analysis_summary"), dict) else {}
        last_execution_result = source.get("last_execution_result") if isinstance(source.get("last_execution_result"), dict) else None
        current_task = source.get("current_task") if isinstance(source.get("current_task"), dict) else {}
        last_stage_memory = source.get("last_stage_memory") if isinstance(source.get("last_stage_memory"), dict) else {}
        return {
            "focus_stocks": focus_stocks,
            "last_intent": str(source.get("last_intent") or "").strip(),
            "last_analysis_summary": dict(last_analysis_summary),
            "pending_actions": self._normalize_pending_actions(source.get("pending_actions")),
            "last_requested_constraints": dict(last_requested_constraints),
            "last_candidate_snapshot": {
                **dict(last_candidate_snapshot),
                "candidate_orders": self._normalize_candidate_orders(last_candidate_snapshot.get("candidate_orders")),
            },
            "last_execution_result": dict(last_execution_result) if isinstance(last_execution_result, dict) else None,
            "session_preference_overrides": self._normalize_session_preference_overrides(source.get("session_preference_overrides")),
            "current_task": dict(current_task),
            "last_stage_memory": dict(last_stage_memory),
            "last_tool_failures": self._normalize_tool_failures(source.get("last_tool_failures")),
        }

    @staticmethod
    def _merge_stage_memory(base: Any, overlay: Any) -> dict[str, Any]:
        merged = dict(base) if isinstance(base, dict) else {}
        if not isinstance(overlay, dict):
            return merged
        for key, value in overlay.items():
            if value is None:
                continue
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {
                    **dict(merged.get(key) or {}),
                    **dict(value),
                }
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _normalize_tool_failures(value: Any) -> list[dict[str, Any]]:
        failures = value if isinstance(value, list) else []
        normalized: list[dict[str, Any]] = []
        for item in failures:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "tool": str(item.get("tool") or "").strip() or None,
                    "stage": str(item.get("stage") or "").strip() or None,
                    "reason": str(item.get("reason") or "").strip() or None,
                    "code": str(item.get("code") or "").strip() or None,
                    "message": str(item.get("message") or "").strip() or None,
                }
            )
        return normalized

    def _build_last_tool_failures(
        self,
        *,
        structured_result: dict[str, Any],
        execution_result: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        for item in structured_result.get("condition_evaluations") or []:
            if not isinstance(item, dict) or bool(item.get("passed")):
                continue
            failures.append(
                {
                    "tool": "execution_policy",
                    "stage": "execution",
                    "reason": str(item.get("reason") or "policy_blocked").strip() or "policy_blocked",
                    "code": str(item.get("stock_code") or "").strip() or None,
                    "message": str(item.get("reason") or "policy_blocked").strip() or "policy_blocked",
                }
            )
        if isinstance(execution_result, dict):
            for item in execution_result.get("failed_orders") or []:
                if not isinstance(item, dict):
                    continue
                candidate_order = item.get("candidate_order") if isinstance(item.get("candidate_order"), dict) else {}
                failures.append(
                    {
                        "tool": "execution",
                        "stage": "execution",
                        "reason": str(item.get("message") or execution_result.get("status") or "execution_failed").strip() or "execution_failed",
                        "code": str(candidate_order.get("code") or "").strip() or None,
                        "message": str(item.get("message") or "").strip() or None,
                    }
                )
            if not failures and str(execution_result.get("status") or "").strip().lower() == "failed":
                failures.append(
                    {
                        "tool": "execution",
                        "stage": "execution",
                        "reason": str(execution_result.get("status") or "execution_failed").strip() or "execution_failed",
                        "code": None,
                        "message": str(execution_result.get("message") or "").strip() or None,
                    }
                )
        if not failures:
            stage_memory = structured_result.get("stage_memory") if isinstance(structured_result.get("stage_memory"), dict) else {}
            failure_reason = str(stage_memory.get("failure_reason") or "").strip()
            if failure_reason:
                failures.append(
                    {
                        "tool": "planner",
                        "stage": None,
                        "reason": failure_reason,
                        "code": None,
                        "message": failure_reason,
                    }
                )
        return failures

    def _merge_session_context(
        self,
        existing_context: dict[str, Any],
        incoming_context: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(existing_context or {})
        for key, value in (incoming_context or {}).items():
            merged[key] = value
        if "conversation_state" not in incoming_context and "conversation_state" in existing_context:
            merged["conversation_state"] = existing_context.get("conversation_state")
        return merged

    @staticmethod
    def _message_is_confirmation_shortcut(message: str) -> bool:
        normalized = "".join(str(message or "").split())
        return any(keyword in normalized for keyword in _CONFIRMATION_SHORTCUT_KEYWORDS)

    @staticmethod
    def _extract_requested_order_side(message: str) -> str | None:
        normalized = str(message or "")
        buy_index = max((normalized.rfind(keyword) for keyword in _BUY_KEYWORDS if keyword in normalized), default=-1)
        sell_index = max((normalized.rfind(keyword) for keyword in _SELL_KEYWORDS if keyword in normalized), default=-1)
        if buy_index < 0 and sell_index < 0:
            return None
        return "sell" if sell_index > buy_index else "buy"

    @staticmethod
    def _extract_requested_quantity(message: str) -> int | None:
        matched = re.search(r"(?<!\d)(\d{1,7})\s*股", str(message or ""))
        if not matched:
            return None
        value = int(matched.group(1))
        return value if value > 0 else None

    @staticmethod
    def _extract_conditions(message: str) -> list[dict[str, Any]]:
        supported, _unsupported = compile_message_conditions(str(message or ""))
        return [item.to_dict() for item in supported]

    @staticmethod
    def _extract_unsupported_conditions(message: str) -> list[dict[str, Any]]:
        _supported, unsupported = compile_message_conditions(str(message or ""))
        return [item.to_dict() for item in unsupported]

    def _build_intent_resolution(
        self,
        *,
        intent: str,
        stock_codes: list[str],
        requested_order_side: str | None,
        requested_quantity: int | None,
        conditions: list[dict[str, Any]],
        followup_reference: str | None,
        confidence: float,
        missing_slots: list[str],
        source: str,
    ) -> dict[str, Any]:
        return {
            "intent": intent,
            "stock_codes": list(stock_codes),
            "order_side": requested_order_side,
            "quantity": requested_quantity,
            "conditions": [dict(item) for item in conditions if isinstance(item, dict)],
            "followup_reference": followup_reference,
            "confidence": round(float(confidence), 4),
            "missing_slots": list(missing_slots),
            "source": source,
        }

    @staticmethod
    def _contains_autonomous_execution_authorization(message: str) -> bool:
        return any(keyword in message for keyword in _AUTONOMOUS_EXECUTION_KEYWORDS)

    @staticmethod
    def _contains_order_intent(message: str) -> bool:
        if any(keyword in message for keyword in _ORDER_KEYWORDS):
            return True
        compact = "".join(str(message or "").split())
        if re.search(r"下[\u4e00-\u9fffA-Z0-9]*单", compact):
            return True
        if "候选单" in compact:
            return True
        if "下" in compact and ("那笔" in compact or "这笔" in compact):
            return True
        return False

    @staticmethod
    def _message_requests_all_orders(message: str) -> bool:
        return any(keyword in message for keyword in _ORDER_ALL_KEYWORDS)

    @staticmethod
    def _message_requests_best_order(message: str) -> bool:
        return any(keyword in message for keyword in _ORDER_BEST_KEYWORDS)

    def _normalize_pending_actions(self, pending_actions: Any) -> list[dict[str, Any]]:
        return self._normalize_candidate_orders(pending_actions)

    def _resolve_followup_orders(
        self,
        *,
        message: str,
        stock_codes: list[str],
        pending_actions: list[dict[str, Any]],
        candidate_snapshots: list[dict[str, Any]],
        prefer_all: bool = False,
        prefer_best: bool = False,
        allow_shortcut: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None, bool]:
        state_orders = [dict(item) for item in self._normalize_pending_actions(pending_actions)]
        if prefer_all and state_orders:
            return state_orders, "pending_actions", False
        if prefer_best and candidate_snapshots:
            best = self._pick_best_candidate_order(candidate_snapshots[0])
            return ([best] if best else []), "candidate_snapshot", False
        if prefer_best and state_orders:
            best = self._pick_best_candidate_order({"candidate_orders": state_orders, "structured_result": {}})
            return ([best] if best else []), "pending_actions", False
        if stock_codes and state_orders:
            matches: list[dict[str, Any]] = []
            seen_keys: set[str] = set()
            for code in stock_codes:
                for item in state_orders:
                    if str(item.get("code") or "").strip() != str(code).strip():
                        continue
                    candidate_key = self._candidate_order_key(item)
                    if candidate_key in seen_keys:
                        continue
                    seen_keys.add(candidate_key)
                    matches.append(dict(item))
            if matches:
                return matches, "pending_actions", False
        if state_orders and allow_shortcut:
            if len(state_orders) == 1:
                return [dict(state_orders[0])], "pending_actions", False
            return [], "pending_actions", True

        fallback_orders = self._resolve_followup_candidate_orders(
            message=message,
            stock_codes=stock_codes,
            candidate_snapshots=candidate_snapshots,
            prefer_all=prefer_all,
            prefer_best=prefer_best,
        )
        if not fallback_orders:
            return [], None, False
        if allow_shortcut and len(fallback_orders) > 1 and not (prefer_all or prefer_best or stock_codes):
            return [], "candidate_snapshot", True
        return fallback_orders, "candidate_snapshot", False

    @staticmethod
    def _build_followup_clarification(pending_actions: list[dict[str, Any]]) -> str:
        codes = [str(item.get("code") or "").strip() for item in pending_actions if str(item.get("code") or "").strip()]
        code_text = "、".join(codes[:5]) if codes else "股票代码"
        return (
            "当前有多笔待确认动作。"
            f" 请直接告诉我要下哪一笔，例如“下 {code_text} 的单”“下最看好的那笔”，"
            "或者说“把刚才那几笔都下了”。"
        )

    @classmethod
    def _extract_candidate_snapshots(cls, recent_assistant_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        for message in recent_assistant_messages:
            meta = message.get("meta")
            if not isinstance(meta, dict):
                continue
            candidate_orders = cls._normalize_candidate_orders(meta.get("candidate_orders"))
            if not candidate_orders:
                structured_result = meta.get("structured_result") if isinstance(meta.get("structured_result"), dict) else {}
                candidate_orders = cls._normalize_candidate_orders(structured_result.get("pending_actions"))
            if not candidate_orders:
                continue
            structured_result = meta.get("structured_result") if isinstance(meta.get("structured_result"), dict) else {}
            snapshots.append(
                {
                    "message_id": message.get("id"),
                    "created_at": message.get("created_at"),
                    "candidate_orders": candidate_orders,
                    "structured_result": structured_result,
                }
            )
        return snapshots

    @staticmethod
    def _normalize_candidate_orders(candidate_orders: Any) -> list[dict[str, Any]]:
        if not isinstance(candidate_orders, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in candidate_orders:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip()
            if not code:
                continue
            normalized.append(dict(item))
        return normalized

    def _resolve_followup_candidate_orders(
        self,
        *,
        message: str,
        stock_codes: list[str],
        candidate_snapshots: list[dict[str, Any]],
        prefer_all: bool = False,
        prefer_best: bool = False,
    ) -> list[dict[str, Any]]:
        if not candidate_snapshots:
            return []

        latest_orders = [dict(item) for item in candidate_snapshots[0].get("candidate_orders") or [] if isinstance(item, dict)]
        if prefer_all and latest_orders:
            return latest_orders

        if prefer_best and latest_orders:
            best = self._pick_best_candidate_order(candidate_snapshots[0])
            return [best] if best else []

        if stock_codes:
            matches: list[dict[str, Any]] = []
            seen_keys: set[str] = set()
            for code in stock_codes:
                match = self._find_candidate_order_by_code(candidate_snapshots, code)
                if match is None:
                    continue
                candidate_key = self._candidate_order_key(match)
                if candidate_key in seen_keys:
                    continue
                seen_keys.add(candidate_key)
                matches.append(match)
            return matches

        if len(latest_orders) == 1:
            return [dict(latest_orders[0])]
        if len(latest_orders) > 1:
            return latest_orders

        for snapshot in candidate_snapshots:
            for item in snapshot.get("candidate_orders") or []:
                code = str(item.get("code") or "").strip()
                if code and code in message:
                    return [dict(item)]
        return []

    def _find_candidate_order_by_code(
        self,
        candidate_snapshots: list[dict[str, Any]],
        code: str,
    ) -> dict[str, Any] | None:
        normalized_code = str(code or "").strip()
        if not normalized_code:
            return None
        for snapshot in candidate_snapshots:
            for item in snapshot.get("candidate_orders") or []:
                if str(item.get("code") or "").strip() == normalized_code:
                    return dict(item)
        return None

    @staticmethod
    def _candidate_order_key(candidate_order: dict[str, Any]) -> str:
        return ":".join(
            [
                str(candidate_order.get("code") or "").strip(),
                str(candidate_order.get("action") or "").strip(),
                str(candidate_order.get("quantity") or "").strip(),
                str(candidate_order.get("price") or "").strip(),
            ]
        )

    def _pick_best_candidate_order(self, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        candidate_orders = [dict(item) for item in snapshot.get("candidate_orders") or [] if isinstance(item, dict)]
        if not candidate_orders:
            return None

        structured_result = snapshot.get("structured_result")
        analysis = structured_result.get("analysis") if isinstance(structured_result, dict) and isinstance(structured_result.get("analysis"), dict) else {}
        stocks = analysis.get("stocks") if isinstance(analysis.get("stocks"), list) else []
        stock_map: dict[str, dict[str, Any]] = {}
        for item in stocks:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip()
            if code:
                stock_map[code] = item

        def score(order: dict[str, Any]) -> tuple[float, float, float]:
            stock = stock_map.get(str(order.get("code") or "").strip(), {})
            sentiment = float(stock.get("sentiment_score") or 0)
            target_weight = float(stock.get("target_weight") or 0.0)
            current_price = float(order.get("current_price") or order.get("price") or 0.0)
            return (sentiment, target_weight, current_price)

        return max(candidate_orders, key=score)

    def _extract_stock_codes(self, message: str) -> list[str]:
        raw_matches: list[str] = []
        raw_matches.extend(_A_SHARE_RE.findall(message))
        raw_matches.extend(_HK_SHARE_RE.findall(message))
        raw_matches.extend(_US_SHARE_RE.findall(message.upper()))

        normalized: list[str] = []
        for item in raw_matches:
            text = str(item or "").strip().upper()
            if not text:
                continue
            if text in {"AND", "OR", "THE", "TODAY", "CHAT"}:
                continue
            try:
                code = canonical_stock_code(text)
            except Exception:
                code = text
            if code not in normalized:
                normalized.append(code)
        return normalized

    @staticmethod
    def _normalize_stock_name_key(value: Any) -> str:
        return re.sub(r"\s+", "", str(value or "").strip()).upper()

    @classmethod
    def _build_stock_name_aliases(cls, value: Any) -> list[str]:
        raw_name = str(value or "").strip()
        if not raw_name:
            return []

        aliases: list[str] = []
        seen: set[str] = set()

        def push(alias: str) -> None:
            key = cls._normalize_stock_name_key(alias)
            if key and key not in seen:
                seen.add(key)
                aliases.append(key)

        push(raw_name)

        compact = re.sub(r"\s+", "", raw_name)
        push(compact)

        current = compact
        while len(current) >= 2:
            stripped = current
            for suffix in _STOCK_NAME_ALIAS_SUFFIXES:
                if stripped.endswith(suffix) and len(stripped) > len(suffix):
                    stripped = stripped[: -len(suffix)].strip()
                    break
            if stripped == current:
                break
            current = stripped
            push(current)

        return aliases

    @classmethod
    def _build_stock_name_profile(cls, raw_code: Any, raw_name: Any) -> dict[str, Any] | None:
        code = canonical_stock_code(str(raw_code or "").strip())
        name = str(raw_name or "").strip()
        aliases = cls._build_stock_name_aliases(name)
        if not code or not name or not aliases:
            return None
        return {
            "code": code,
            "name": name,
            "aliases": aliases,
        }

    @staticmethod
    def _append_stock_name_index(index: dict[str, list[str]], alias: str, code: str) -> None:
        if not alias or not code:
            return
        bucket = index.setdefault(alias, [])
        if code not in bucket:
            bucket.append(code)

    @classmethod
    def _build_stock_name_index_from_profiles(cls, profiles: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for profile in profiles.values():
            code = str(profile.get("code") or "").strip()
            for alias in profile.get("aliases") or []:
                cls._append_stock_name_index(index, str(alias or "").strip(), code)
        return index

    @classmethod
    def _build_static_stock_name_profiles(cls) -> dict[str, dict[str, Any]]:
        profiles: dict[str, dict[str, Any]] = {}
        for raw_code, raw_name in STOCK_NAME_MAP.items():
            profile = cls._build_stock_name_profile(raw_code, raw_name)
            if profile is None:
                continue
            code = str(profile.get("code") or "").strip()
            if code and code not in profiles:
                profiles[code] = profile
        return profiles

    @classmethod
    def _build_static_stock_name_index(cls) -> dict[str, list[str]]:
        return cls._build_stock_name_index_from_profiles(cls._build_static_stock_name_profiles())

    def _get_stock_name_lookup_manager(self) -> Any | None:
        if self._stock_name_lookup_manager is not None:
            return self._stock_name_lookup_manager

        orchestrator = getattr(self.agent_service, "orchestrator", None)
        data_agent = getattr(orchestrator, "data_agent", None)
        fetcher_manager = getattr(data_agent, "fetcher_manager", None)
        if fetcher_manager is not None:
            self._stock_name_lookup_manager = fetcher_manager
            return self._stock_name_lookup_manager

        try:
            from data_provider import DataFetcherManager

            self._stock_name_lookup_manager = DataFetcherManager()
        except Exception as exc:
            logger.warning("stock name lookup manager unavailable: %s", redact_sensitive_text(str(exc)))
            self._stock_name_lookup_manager = None
        return self._stock_name_lookup_manager

    def _ensure_dynamic_stock_name_lookup_cache(self) -> None:
        if self._dynamic_stock_name_index is not None and self._dynamic_stock_name_profiles is not None:
            return

        with self._stock_name_lookup_lock:
            if self._dynamic_stock_name_index is not None and self._dynamic_stock_name_profiles is not None:
                return

            dynamic_profiles: dict[str, dict[str, Any]] = {}
            dynamic_index: dict[str, list[str]] = {}
            fetcher_manager = self._get_stock_name_lookup_manager()
            if fetcher_manager is None or not hasattr(fetcher_manager, "get_stock_list"):
                self._dynamic_stock_name_profiles = dynamic_profiles
                self._dynamic_stock_name_index = dynamic_index
                return

            try:
                stock_list = fetcher_manager.get_stock_list()
            except Exception as exc:
                logger.warning("stock list lazy load failed: %s", redact_sensitive_text(str(exc)))
                self._dynamic_stock_name_profiles = dynamic_profiles
                self._dynamic_stock_name_index = dynamic_index
                return

            if stock_list is None or getattr(stock_list, "empty", True):
                self._dynamic_stock_name_profiles = dynamic_profiles
                self._dynamic_stock_name_index = dynamic_index
                return

            for _index, row in stock_list.iterrows():
                profile = self._build_stock_name_profile(row.get("code"), row.get("name"))
                if profile is None:
                    continue
                code = str(profile.get("code") or "").strip()
                if not code or code in dynamic_profiles:
                    continue
                dynamic_profiles[code] = profile
                for alias in profile.get("aliases") or []:
                    self._append_stock_name_index(dynamic_index, str(alias or "").strip(), code)

            self._dynamic_stock_name_profiles = dynamic_profiles
            self._dynamic_stock_name_index = dynamic_index

    def _get_dynamic_stock_name_profiles(self) -> dict[str, dict[str, Any]]:
        self._ensure_dynamic_stock_name_lookup_cache()
        return dict(self._dynamic_stock_name_profiles or {})

    def _get_dynamic_stock_name_index(self) -> dict[str, list[str]]:
        self._ensure_dynamic_stock_name_lookup_cache()
        return dict(self._dynamic_stock_name_index or {})

    def _get_stock_name_index(self) -> dict[str, list[str]]:
        index = dict(self._static_stock_name_index)
        for alias, codes in self._get_dynamic_stock_name_index().items():
            bucket = index.setdefault(alias, [])
            for code in codes:
                text = str(code or "").strip()
                if text and text not in bucket:
                    bucket.append(text)
        return index

    def _get_stock_name_profiles(self) -> dict[str, dict[str, Any]]:
        profiles = dict(self._static_stock_name_profiles)
        for code, profile in self._get_dynamic_stock_name_profiles().items():
            if code not in profiles:
                profiles[code] = dict(profile)
        return profiles

    def _get_stock_name_profile(self, code: str) -> dict[str, Any] | None:
        normalized = canonical_stock_code(str(code or "").strip())
        if not normalized:
            return None
        profile = self._static_stock_name_profiles.get(normalized)
        if profile is not None:
            return dict(profile)
        dynamic_profiles = self._get_dynamic_stock_name_profiles()
        profile = dynamic_profiles.get(normalized)
        return dict(profile) if profile is not None else None

    @staticmethod
    def _score_stock_name_alias_match(query_key: str, alias_key: str) -> int | None:
        if not query_key or not alias_key:
            return None
        if alias_key == query_key:
            return 0
        if len(query_key) < 2:
            return None
        if alias_key.startswith(query_key) or query_key.startswith(alias_key):
            return 1
        if query_key in alias_key or alias_key in query_key:
            return 2
        return None

    def _collect_stock_name_match_candidates(self, raw_name: str) -> list[dict[str, Any]]:
        name = str(raw_name or "").strip()
        query_key = self._normalize_stock_name_key(name)
        if not query_key:
            return []

        exact_codes: list[str] = []
        for index in (self._static_stock_name_index, self._get_dynamic_stock_name_index()):
            for code in index.get(query_key) or []:
                text = str(code or "").strip()
                if text and text not in exact_codes:
                    exact_codes.append(text)

        if exact_codes:
            candidates: list[dict[str, Any]] = []
            for code in exact_codes:
                profile = self._get_stock_name_profile(code)
                candidates.append(
                    {
                        "code": code,
                        "name": str((profile or {}).get("name") or code).strip() or code,
                        "score": 0,
                    }
                )
            return candidates

        candidates_by_code: dict[str, dict[str, Any]] = {}
        for code, profile in self._get_stock_name_profiles().items():
            best_score: int | None = None
            for alias in profile.get("aliases") or []:
                score = self._score_stock_name_alias_match(query_key, str(alias or "").strip())
                if score is None:
                    continue
                if best_score is None or score < best_score:
                    best_score = score
            if best_score is None:
                continue
            candidates_by_code[code] = {
                "code": code,
                "name": str(profile.get("name") or code).strip() or code,
                "score": best_score,
            }

        candidates = list(candidates_by_code.values())
        candidates.sort(key=lambda item: (int(item.get("score") or 9), len(str(item.get("name") or "")), str(item.get("code") or "")))
        return candidates

    def _resolve_stock_name_reference(self, raw_name: str) -> dict[str, Any]:
        name = str(raw_name or "").strip()
        if not name:
            return {
                "raw_name": "",
                "normalized_name": "",
                "status": "unknown",
                "stock_codes": [],
                "candidate_matches": [],
            }

        candidates = self._collect_stock_name_match_candidates(name)
        if len(candidates) == 1:
            candidate = dict(candidates[0])
            return {
                "raw_name": name,
                "normalized_name": self._normalize_stock_name_key(name),
                "status": "resolved",
                "stock_codes": [str(candidate.get("code") or "").strip()],
                "candidate_matches": [candidate],
                "matched_name": str(candidate.get("name") or "").strip(),
            }
        if len(candidates) > 1:
            return {
                "raw_name": name,
                "normalized_name": self._normalize_stock_name_key(name),
                "status": "ambiguous",
                "stock_codes": [],
                "candidate_matches": [dict(item) for item in candidates[:_BOARD_AMBIGUOUS_LIMIT]],
            }
        return {
            "raw_name": name,
            "normalized_name": self._normalize_stock_name_key(name),
            "status": "unknown",
            "stock_codes": [],
            "candidate_matches": [],
        }

    @classmethod
    def _normalize_board_name_key(cls, value: Any) -> str:
        normalized = re.sub(r"\s+", "", str(value or "").strip()).upper()
        if not normalized:
            return ""
        compact = normalized
        while True:
            stripped = _BOARD_SCOPE_SUFFIX_RE.sub("", compact)
            if stripped == compact or not stripped:
                break
            compact = stripped
        return compact or normalized

    @staticmethod
    def _pick_first_nonempty_value(payload: dict[str, Any], columns: tuple[str, ...]) -> Any:
        for column in columns:
            value = payload.get(column)
            if value not in {None, ""}:
                return value
        return None

    @staticmethod
    def _coerce_records(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]

        iterrows = getattr(value, "iterrows", None)
        if callable(iterrows):
            records: list[dict[str, Any]] = []
            try:
                for _index, row in value.iterrows():
                    if isinstance(row, dict):
                        records.append(dict(row))
                    elif hasattr(row, "to_dict"):
                        records.append(dict(row.to_dict()))
            except Exception:
                return []
            return records
        return []

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value in {None, ""}:
            return None
        try:
            text = str(value).replace(",", "").strip()
            if not text or text in {"-", "--", "nan", "None"}:
                return None
            return float(text)
        except (TypeError, ValueError):
            return None

    def _coerce_board_catalog_entries(self, board_type: str, value: Any) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in self._coerce_records(value):
            board_name = str(self._pick_first_nonempty_value(row, _BOARD_NAME_COLUMNS) or "").strip()
            board_symbol = str(self._pick_first_nonempty_value(row, _BOARD_CODE_COLUMNS) or board_name).strip()
            normalized_name = self._normalize_board_name_key(board_name)
            normalized_symbol = self._normalize_board_name_key(board_symbol)
            if not board_name or not normalized_name:
                continue
            dedupe_key = (str(board_type).strip(), board_symbol or board_name, normalized_name)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            entries.append(
                {
                    "entity_type": str(board_type).strip(),
                    "board_name": board_name,
                    "board_symbol": board_symbol or board_name,
                    "normalized_name": normalized_name,
                    "normalized_symbol": normalized_symbol,
                }
            )
        entries.sort(key=lambda item: (str(item.get("board_name") or ""), str(item.get("board_symbol") or "")))
        return entries

    def _coerce_board_constituents(self, value: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self._coerce_records(value):
            raw_code = self._pick_first_nonempty_value(row, _BOARD_STOCK_CODE_COLUMNS)
            code = canonical_stock_code(str(raw_code or "").strip())
            if not re.fullmatch(r"\d{6}", code or ""):
                continue
            rows.append(
                {
                    "code": code,
                    "name": str(self._pick_first_nonempty_value(row, _BOARD_STOCK_NAME_COLUMNS) or "").strip(),
                    "total_mv": self._safe_float(self._pick_first_nonempty_value(row, _BOARD_TOTAL_MV_COLUMNS)),
                    "amount": self._safe_float(self._pick_first_nonempty_value(row, _BOARD_AMOUNT_COLUMNS)),
                }
            )
        rows.sort(
            key=lambda item: (
                -(item["total_mv"] if item["total_mv"] is not None else -1.0),
                -(item["amount"] if item["amount"] is not None else -1.0),
                str(item["code"] or ""),
            )
        )
        deduped: list[dict[str, Any]] = []
        seen_codes: set[str] = set()
        for row in rows:
            code = str(row.get("code") or "").strip()
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            deduped.append(row)
        return deduped[:_BOARD_COMPONENT_LIMIT]

    def _default_board_catalog_provider(self, board_type: str) -> list[dict[str, Any]]:
        import akshare as ak

        board_kind = str(board_type or "").strip()
        if board_kind == "industry_board":
            raw = ak.stock_board_industry_name_em()
        elif board_kind == "concept_board":
            raw = ak.stock_board_concept_name_em()
        else:
            return []
        return self._coerce_board_catalog_entries(board_kind, raw)

    def _default_board_constituents_provider(self, board_type: str, board_symbol: str) -> list[dict[str, Any]]:
        import akshare as ak

        board_kind = str(board_type or "").strip()
        symbol = str(board_symbol or "").strip()
        if not symbol:
            return []
        if board_kind == "industry_board":
            raw = ak.stock_board_industry_cons_em(symbol=symbol)
        elif board_kind == "concept_board":
            raw = ak.stock_board_concept_cons_em(symbol=symbol)
        else:
            return []
        return self._coerce_board_constituents(raw)

    def _load_board_catalog(self, board_type: str) -> list[dict[str, Any]]:
        board_kind = str(board_type or "").strip()
        if not board_kind:
            return []
        if board_kind in self._board_catalog_cache:
            return list(self._board_catalog_cache[board_kind])

        with self._board_lookup_lock:
            if board_kind in self._board_catalog_cache:
                return list(self._board_catalog_cache[board_kind])
            try:
                catalog = self._board_catalog_provider(board_kind)
            except Exception as exc:
                logger.warning("board catalog load failed: %s", redact_sensitive_text(str(exc)))
                catalog = []
            normalized_catalog = self._coerce_board_catalog_entries(board_kind, catalog)
            self._board_catalog_cache[board_kind] = normalized_catalog
            return list(normalized_catalog)

    def _load_board_constituents(self, board_type: str, board_symbol: str) -> list[dict[str, Any]]:
        board_kind = str(board_type or "").strip()
        symbol = str(board_symbol or "").strip()
        cache_key = (board_kind, symbol)
        if cache_key in self._board_constituents_cache:
            return [dict(item) for item in self._board_constituents_cache[cache_key]]

        with self._board_lookup_lock:
            if cache_key in self._board_constituents_cache:
                return [dict(item) for item in self._board_constituents_cache[cache_key]]
            try:
                constituents = self._board_constituents_provider(board_kind, symbol)
            except Exception as exc:
                logger.warning("board constituents load failed: %s", redact_sensitive_text(str(exc)))
                constituents = []
            normalized = self._coerce_board_constituents(constituents)
            self._board_constituents_cache[cache_key] = normalized
            return [dict(item) for item in normalized]

    def _match_board_entries(self, normalized_ref: str) -> list[dict[str, Any]]:
        if not normalized_ref:
            return []

        entries = [*self._load_board_catalog("industry_board"), *self._load_board_catalog("concept_board")]
        exact_matches = [
            dict(item)
            for item in entries
            if normalized_ref in {str(item.get("normalized_name") or ""), str(item.get("normalized_symbol") or "")}
        ]
        if exact_matches:
            return exact_matches

        partial_matches = [
            dict(item)
            for item in entries
            if normalized_ref in str(item.get("normalized_name") or "")
            or str(item.get("normalized_name") or "") in normalized_ref
        ]
        return partial_matches

    def _resolve_scope_entity(self, ref: str) -> dict[str, Any]:
        raw_ref = str(ref or "").strip()
        normalized_ref = self._normalize_board_name_key(raw_ref)
        if self._stock_scope_refs_hit_market_wide_boundary([raw_ref]):
            return {
                "raw_ref": raw_ref,
                "normalized_ref": normalized_ref,
                "entity_type": "market_wide",
                "status": "blocked",
                "stock_codes": [],
                "candidate_matches": [],
                "message": self._build_market_wide_scope_unsupported_message(),
            }

        stock_ref_resolution = self._extract_explicit_stock_reference_resolution(raw_ref)
        explicit_codes = [str(code or "").strip() for code in stock_ref_resolution.get("stock_codes") or [] if str(code or "").strip()]
        ambiguous_refs = [dict(item) for item in stock_ref_resolution.get("ambiguous_refs") or [] if isinstance(item, dict)]
        unresolved_names = [str(item or "").strip() for item in stock_ref_resolution.get("unresolved_refs") or [] if str(item or "").strip()]

        if explicit_codes:
            return {
                "raw_ref": raw_ref,
                "normalized_ref": normalized_ref,
                "entity_type": "stock",
                "status": "resolved",
                "stock_codes": list(explicit_codes),
                "candidate_matches": [
                    {
                        "code": str(item.get("code") or "").strip(),
                        "name": str(item.get("name") or "").strip(),
                    }
                    for item in (
                        self._resolve_stock_name_reference(raw_ref).get("candidate_matches")
                        if not self._extract_stock_codes(raw_ref)
                        else []
                    )
                    if isinstance(item, dict)
                ],
                "unresolved_tokens": list(unresolved_names),
            }
        if ambiguous_refs:
            first_match = ambiguous_refs[0]
            candidate_matches = [
                {
                    "code": str(item.get("code") or "").strip(),
                    "name": str(item.get("name") or "").strip(),
                }
                for item in first_match.get("candidate_matches") or []
                if isinstance(item, dict)
            ]
            return {
                "raw_ref": raw_ref,
                "normalized_ref": normalized_ref,
                "entity_type": "stock",
                "status": "ambiguous",
                "stock_codes": [],
                "candidate_matches": candidate_matches,
                "message": self._build_stock_name_ambiguous_clarification(
                    str(first_match.get("raw_name") or raw_ref).strip(),
                    candidate_matches,
                ),
            }

        matches = self._match_board_entries(normalized_ref)
        if len(matches) > 1:
            return {
                "raw_ref": raw_ref,
                "normalized_ref": normalized_ref,
                "entity_type": "unknown",
                "status": "ambiguous",
                "stock_codes": [],
                "candidate_matches": [
                    str(item.get("board_name") or item.get("board_symbol") or "").strip()
                    for item in matches[:_BOARD_AMBIGUOUS_LIMIT]
                ],
                "message": self._build_board_scope_ambiguous_clarification(raw_ref, matches),
            }
        if len(matches) == 1:
            match = dict(matches[0])
            constituents = self._load_board_constituents(
                str(match.get("entity_type") or "").strip(),
                str(match.get("board_symbol") or "").strip(),
            )
            stock_codes = [str(item.get("code") or "").strip() for item in constituents if str(item.get("code") or "").strip()]
            if not stock_codes:
                return {
                    "raw_ref": raw_ref,
                    "normalized_ref": normalized_ref,
                    "entity_type": str(match.get("entity_type") or "unknown"),
                    "status": "unavailable",
                    "matched_name": str(match.get("board_name") or "").strip(),
                    "matched_symbol": str(match.get("board_symbol") or "").strip(),
                    "stock_codes": [],
                    "candidate_matches": [],
                    "message": self._build_board_constituents_unavailable_clarification(
                        raw_ref,
                        str(match.get("board_name") or "").strip(),
                    ),
                }
            return {
                "raw_ref": raw_ref,
                "normalized_ref": normalized_ref,
                "entity_type": str(match.get("entity_type") or "unknown"),
                "status": "resolved",
                "matched_name": str(match.get("board_name") or "").strip(),
                "matched_symbol": str(match.get("board_symbol") or "").strip(),
                "stock_codes": stock_codes,
                "candidate_matches": [],
            }

        return {
            "raw_ref": raw_ref,
            "normalized_ref": normalized_ref,
            "entity_type": "unknown",
            "status": "unknown",
            "stock_codes": [],
            "candidate_matches": [],
            "unresolved_tokens": list(unresolved_names),
        }

    def _extract_stock_name_candidates(self, message: str) -> list[str]:
        normalized = re.sub(r"\s+", "", str(message or "").strip())
        if not normalized:
            return []

        target_segments: list[str] = []
        if normalized.startswith("那") and normalized.endswith(("呢", "呢？", "呢?")):
            target_segments.append(normalized[1:])
        else:
            for marker in ("分析一下", "分析", "看一下", "看下", "看看", "研究一下", "研究", "聊聊", "聊下"):
                index = normalized.find(marker)
                if index >= 0:
                    target_segments.append(normalized[index + len(marker):])
                    break

        if not target_segments:
            if re.fullmatch(r"[\u4e00-\u9fff、，,和及与跟/]{2,24}", normalized):
                target_segments.append(normalized)
            else:
                return []

        candidates: list[str] = []
        seen: set[str] = set()
        for segment in target_segments:
            working = segment
            working = re.sub(r"^(?:一下|今天的|今天|当前的|当前|这个|那个|这只|那只)+", "", working)
            working = _STOCK_NAME_CONTEXT_SUFFIX_RE.sub("", working)
            for token in _STOCK_NAME_SPLIT_RE.split(working):
                cleaned = re.sub(r"^(?:一下|今天的|今天|当前的|当前|这个|那个|这只|那只|再|那)+", "", str(token or "").strip())
                cleaned = _STOCK_NAME_CONTEXT_SUFFIX_RE.sub("", cleaned).strip()
                if not self._looks_like_stock_name_candidate(cleaned):
                    continue
                if cleaned not in seen:
                    seen.add(cleaned)
                    candidates.append(cleaned)
        return candidates

    @staticmethod
    def _looks_like_stock_name_candidate(token: str) -> bool:
        text = str(token or "").strip()
        if not text:
            return False
        if text in _GENERIC_STOCK_NAME_EXACT_BLOCKLIST:
            return False
        if any(keyword in text for keyword in _GENERIC_STOCK_NAME_FRAGMENT_KEYWORDS):
            return False
        if text.isdigit():
            return False
        if not re.fullmatch(r"[\u4e00-\u9fffA-Z0-9]{2,12}", text.upper()):
            return False
        return True

    def _resolve_stock_codes_from_names(self, names: list[str]) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        if not names:
            return [], [], []

        codes: list[str] = []
        ambiguous_refs: list[dict[str, Any]] = []
        unresolved: list[str] = []

        for raw_name in names:
            resolution = self._resolve_stock_name_reference(raw_name)
            status = str(resolution.get("status") or "").strip()
            if status == "resolved":
                for code in resolution.get("stock_codes") or []:
                    text = str(code or "").strip()
                    if text and text not in codes:
                        codes.append(text)
                continue
            if status == "ambiguous":
                ambiguous_refs.append(
                    {
                        "raw_name": str(resolution.get("raw_name") or "").strip(),
                        "candidate_matches": [
                            {
                                "code": str(item.get("code") or "").strip(),
                                "name": str(item.get("name") or "").strip(),
                            }
                            for item in resolution.get("candidate_matches") or []
                            if isinstance(item, dict)
                        ],
                    }
                )
                continue

            unresolved_name = str(resolution.get("raw_name") or "").strip()
            if unresolved_name:
                unresolved.append(unresolved_name)

        return codes, ambiguous_refs, unresolved

    def _extract_explicit_stock_reference_resolution(self, message: str) -> dict[str, Any]:
        explicit_codes = self._extract_stock_codes(message)
        name_candidates = self._extract_stock_name_candidates(message)
        name_codes, ambiguous_refs, unresolved_names = self._resolve_stock_codes_from_names(name_candidates)

        merged_codes: list[str] = []
        for code in [*explicit_codes, *name_codes]:
            text = str(code or "").strip()
            if text and text not in merged_codes:
                merged_codes.append(text)

        raw_refs: list[str] = []
        for ref in [*explicit_codes, *name_candidates]:
            text = str(ref or "").strip()
            if text and text not in raw_refs:
                raw_refs.append(text)

        return {
            "stock_codes": merged_codes,
            "ambiguous_refs": ambiguous_refs,
            "unresolved_refs": unresolved_names,
            "name_candidates": name_candidates,
            "raw_refs": raw_refs,
        }

    def _extract_strategy_backtest_stock_refs(self, message: str) -> list[str]:
        raw_message = str(message or "")
        refs: list[str] = []
        seen: set[str] = set()

        def push(ref: str) -> None:
            text = str(ref or "").strip()
            if text and text not in seen:
                seen.add(text)
                refs.append(text)

        for raw_code in _A_SHARE_RE.findall(raw_message):
            code = canonical_stock_code(str(raw_code or "").strip())
            if re.fullmatch(r"\d{6}", code or ""):
                push(code)
        if refs:
            return refs

        compact = self._normalize_stock_name_key(raw_message)
        if not compact:
            return []

        alias_matches: list[tuple[int, int, str]] = []
        for alias in self._get_stock_name_index():
            normalized_alias = str(alias or "").strip()
            if len(normalized_alias) < 2 or normalized_alias in _STRATEGY_BACKTEST_STOCK_REF_IGNORE_TOKENS:
                continue
            position = compact.find(normalized_alias)
            if position < 0:
                continue
            alias_matches.append((position, -len(normalized_alias), normalized_alias))

        alias_matches.sort(key=lambda item: (item[0], item[1], item[2]))
        occupied_spans: list[tuple[int, int]] = []
        for position, neg_length, alias in alias_matches:
            end = position - neg_length
            if any(not (end <= start or position >= stop) for start, stop in occupied_spans):
                continue
            occupied_spans.append((position, end))
            push(alias)

        return refs

    def _extract_explicit_stock_references(self, message: str) -> tuple[list[str], list[str]]:
        resolution = self._extract_explicit_stock_reference_resolution(message)
        unresolved_names = [
            *[str(item.get("raw_name") or "").strip() for item in resolution.get("ambiguous_refs") or [] if isinstance(item, dict)],
            *[str(item or "").strip() for item in resolution.get("unresolved_refs") or []],
        ]
        return list(resolution.get("stock_codes") or []), [item for item in unresolved_names if item]

    @staticmethod
    def _build_stock_name_clarification(unresolved_names: list[str]) -> str:
        joined = "、".join(str(item or "").strip() for item in unresolved_names if str(item or "").strip())
        return (
            f"我识别到你提到了 {joined}，但当前还没法把它准确映射到股票、行业板块或概念板块。"
            "请补充更具体的板块名，或直接给我 6 位股票代码，例如“601899 紫金矿业”。"
        )

    @staticmethod
    def _format_stock_candidate_matches(candidate_matches: list[dict[str, Any]]) -> str:
        return "、".join(
            f"{str(item.get('code') or '').strip()} {str(item.get('name') or '').strip()}".strip()
            for item in candidate_matches[:_BOARD_AMBIGUOUS_LIMIT]
            if isinstance(item, dict) and (str(item.get("code") or "").strip() or str(item.get("name") or "").strip())
        )

    @classmethod
    def _build_stock_name_ambiguous_clarification(cls, raw_name: str, candidate_matches: list[dict[str, Any]]) -> str:
        candidates = cls._format_stock_candidate_matches(candidate_matches)
        return (
            f"我识别到你提到了 {str(raw_name or '').strip()}，但它可能对应多只股票：{candidates}。"
            "请再说得更具体一点，或直接给我股票代码。"
        )

    @staticmethod
    def _remove_direct_stock_query_fillers(message: str, ref_tokens: list[str]) -> str:
        working = re.sub(r"\s+", "", str(message or "").strip())
        for token in sorted((str(item or "").strip() for item in ref_tokens), key=len, reverse=True):
            if token:
                working = working.replace(token, "")
        working = re.sub(r"(?:、|,|，|/|以及|还有|和|及|与|跟)+", "", working)
        for token in _DIRECT_STOCK_QUERY_FILLER_TOKENS:
            if token:
                working = working.replace(token, "")
        return re.sub(r"[，。！？!?；;：:\-—_~`'\"()\[\]{}<>]", "", working)

    def _is_direct_stock_reference_message(self, message: str, ref_tokens: list[str]) -> bool:
        if not ref_tokens:
            return False
        return not self._remove_direct_stock_query_fillers(message, ref_tokens)

    @staticmethod
    def _build_board_scope_ambiguous_clarification(ref: str, matches: list[dict[str, Any]]) -> str:
        candidates = "、".join(
            str(item.get("board_name") or item.get("board_symbol") or "").strip()
            for item in matches[:_BOARD_AMBIGUOUS_LIMIT]
            if str(item.get("board_name") or item.get("board_symbol") or "").strip()
        )
        return (
            f"我识别到你提到了 {str(ref or '').strip()}，但它可能对应多个行业/概念板块：{candidates}。"
            "请说得更具体一点，或直接给我股票代码。"
        )

    @staticmethod
    def _build_board_constituents_unavailable_clarification(ref: str, matched_name: str) -> str:
        board_name = matched_name or str(ref or "").strip()
        return (
            f"我识别到你说的是板块“{board_name}”，但当前无法拉取它的成分股。"
            "你可以稍后再试，或直接告诉我几只想分析的股票代码。"
        )

    def _extract_default_stock_codes(
        self,
        latest_assistant_meta: dict[str, Any],
        frontend_context: dict[str, Any],
        conversation_state: dict[str, Any],
    ) -> list[str]:
        codes: list[str] = []
        query_ref = str(
            frontend_context.get("stock_ref")
            or frontend_context.get("stockRef")
            or frontend_context.get("stock_code")
            or frontend_context.get("stockCode")
            or ""
        ).strip()
        query_code = str(frontend_context.get("stock_code") or frontend_context.get("stockCode") or "").strip()
        if query_ref:
            resolution = self._extract_explicit_stock_reference_resolution(query_ref)
            for code in resolution.get("stock_codes") or []:
                text = str(code or "").strip()
                if text and text not in codes:
                    codes.append(text)
        elif query_code:
            codes.append(query_code)

        structured_result = latest_assistant_meta.get("structured_result")
        if isinstance(structured_result, dict):
            analysis = structured_result.get("analysis")
            if isinstance(analysis, dict):
                stocks = analysis.get("stocks")
                if isinstance(stocks, list):
                    for item in stocks:
                        if not isinstance(item, dict):
                            continue
                        code = str(item.get("code") or "").strip()
                        if code and code not in codes:
                            codes.append(code)
            pending_actions = structured_result.get("pending_actions")
            if isinstance(pending_actions, list):
                for item in pending_actions:
                    if not isinstance(item, dict):
                        continue
                    code = str(item.get("code") or "").strip()
                    if code and code not in codes:
                        codes.append(code)
        for code in conversation_state.get("focus_stocks") or []:
            resolution = self._extract_explicit_stock_reference_resolution(str(code or "").strip())
            for resolved_code in resolution.get("stock_codes") or []:
                text = str(resolved_code or "").strip()
                if text and text not in codes:
                    codes.append(text)
        return codes

    def _build_llm_plan(
        self,
        *,
        message: str,
        frontend_context: dict[str, Any],
        latest_assistant_meta: dict[str, Any],
        recent_assistant_messages: list[dict[str, Any]],
        conversation_state: dict[str, Any],
        llm_selection: ChatLlmSelection,
    ) -> ChatPlan:
        pending_actions = self._normalize_pending_actions(conversation_state.get("pending_actions"))
        candidate_snapshots = self._extract_candidate_snapshots(recent_assistant_messages)
        default_focus_codes = self._extract_default_stock_codes(latest_assistant_meta, frontend_context, conversation_state)
        explicit_ref_resolution = self._extract_explicit_stock_reference_resolution(message)
        explicit_stock_refs = [
            str(ref or "").strip()
            for ref in explicit_ref_resolution.get("raw_refs") or []
            if str(ref or "").strip()
        ]
        direct_stock_reference_message = self._is_direct_stock_reference_message(message, explicit_stock_refs)
        if self._contains_portfolio_health_intent(message):
            intent_resolution = self._build_intent_resolution(
                intent="portfolio_health",
                stock_codes=[],
                requested_order_side=None,
                requested_quantity=None,
                conditions=[],
                followup_reference=None,
                confidence=0.98,
                missing_slots=[],
                source="rule",
            )
            intent_resolution["scope_resolution"] = {
                "mode": "none",
                "requested_refs": [],
                "resolved_stock_codes": [],
                "unresolved_refs": [],
                "blocked_code": None,
                "resolver": "portfolio_health_rule",
            }
            intent_resolution["resolved_scope_entities"] = []
            return ChatPlan(
                primary_intent="portfolio_health",
                include_runtime_context=True,
                planner_source="rule",
                intent_source="rule",
                intent_resolution=intent_resolution,
                pending_actions=pending_actions,
                required_tools=["load_account_state", "load_portfolio_health", "load_user_preferences"],
                stock_scope={"mode": "none", "stock_refs": []},
                followup_target={"mode": "none", "stock_refs": []},
            )
        if self._contains_strategy_backtest_run_intent(message):
            strategy_requested_refs: list[str] = []
            for ref in self._extract_strategy_backtest_stock_refs(message):
                text = str(ref or "").strip()
                if text and text not in strategy_requested_refs:
                    strategy_requested_refs.append(text)
            stock_scope_mode = "explicit" if strategy_requested_refs else "focus"
            requested_refs = strategy_requested_refs if strategy_requested_refs else []
            scope_resolution = self._resolve_stock_codes_from_refs(
                requested_refs,
                fallback_focus_codes=default_focus_codes,
                stock_scope_mode=stock_scope_mode,
                pending_actions=pending_actions,
            )
            if scope_resolution.get("clarification"):
                clarification = str(scope_resolution.get("clarification") or "").strip()
                return ChatPlan(
                    primary_intent="clarify",
                    clarification=clarification,
                    planner_source="rule",
                    intent_source="rule",
                    intent_resolution=self._build_intent_resolution(
                        intent="clarify",
                        stock_codes=[],
                        requested_order_side=None,
                        requested_quantity=None,
                        conditions=[],
                        followup_reference=None,
                        confidence=0.94,
                        missing_slots=["stock_codes"],
                        source="rule",
                    ),
                    pending_actions=pending_actions,
                )
            stock_codes = [
                code
                for code in self._normalize_stock_codes(scope_resolution.get("stock_codes"))
                if re.fullmatch(r"\d{6}", str(code or "").strip())
            ]
            if len(stock_codes) != 1:
                clarification = (
                    "策略回测当前一次只支持 1 只股票。请直接告诉我一只股票代码，"
                    "或者先选定一只股票后再问“这只股票用 MACD 金叉策略过去一年收益怎样”。"
                )
                return ChatPlan(
                    primary_intent="clarify",
                    clarification=clarification,
                    planner_source="rule",
                    intent_source="rule",
                    intent_resolution=self._build_intent_resolution(
                        intent="clarify",
                        stock_codes=stock_codes,
                        requested_order_side=None,
                        requested_quantity=None,
                        conditions=[],
                        followup_reference=None,
                        confidence=0.9,
                        missing_slots=["stock_codes"],
                        source="rule",
                    ),
                    pending_actions=pending_actions,
                )
            intent_resolution = self._build_intent_resolution(
                intent="backtest",
                stock_codes=stock_codes,
                requested_order_side=None,
                requested_quantity=None,
                conditions=[],
                followup_reference=None,
                confidence=0.98,
                missing_slots=[],
                source="rule",
            )
            intent_resolution["scope_resolution"] = dict(scope_resolution.get("scope_resolution") or {})
            intent_resolution["resolved_scope_entities"] = list(scope_resolution.get("resolved_scope_entities") or [])
            return ChatPlan(
                primary_intent="backtest",
                stock_codes=stock_codes,
                include_runtime_context=True,
                planner_source="rule",
                intent_source="rule",
                intent_resolution=intent_resolution,
                pending_actions=pending_actions,
                required_tools=["load_account_state", "run_strategy_backtest"],
                stock_scope={"mode": stock_scope_mode, "stock_refs": requested_refs},
                followup_target={"mode": "none", "stock_refs": []},
            )
        if self._is_market_wide_selection_request(message):
            clarification = self._build_market_wide_scope_unsupported_message()
            intent_resolution = self._build_intent_resolution(
                intent="unsupported",
                stock_codes=[],
                requested_order_side=None,
                requested_quantity=None,
                conditions=[],
                followup_reference=None,
                confidence=0.99,
                missing_slots=[],
                source="rule",
            )
            intent_resolution["scope_resolution"] = {
                "mode": "explicit",
                "requested_refs": [message],
                "resolved_stock_codes": [],
                "unresolved_refs": [],
                "blocked_code": "market_wide_stock_selection_unsupported",
                "resolver": "scope_resolver_v2",
            }
            intent_resolution["resolved_scope_entities"] = [
                {
                    "raw_ref": message,
                    "normalized_ref": self._compact_message_text(message),
                    "entity_type": "market_wide",
                    "status": "blocked",
                    "stock_codes": [],
                    "candidate_matches": [],
                }
            ]
            return ChatPlan(
                primary_intent="unsupported",
                clarification=clarification,
                planner_source="rule",
                intent_source="rule",
                intent_resolution=intent_resolution,
                pending_actions=pending_actions,
                stock_scope={"mode": "none", "stock_refs": []},
                followup_target={"mode": "none", "stock_refs": []},
                blocked_code="market_wide_stock_selection_unsupported",
            )
        if direct_stock_reference_message and explicit_stock_refs and explicit_ref_resolution.get("stock_codes"):
            scope_resolution = self._resolve_stock_codes_from_refs(
                explicit_stock_refs,
                fallback_focus_codes=default_focus_codes,
                stock_scope_mode="explicit",
                pending_actions=pending_actions,
            )
            direct_stock_codes = self._normalize_stock_codes(scope_resolution.get("stock_codes"))
            blocked_code = str(scope_resolution.get("blocked_code") or "").strip() or None
            clarification = str(scope_resolution.get("clarification") or "").strip()
            if blocked_code == "market_wide_stock_selection_unsupported":
                intent_resolution = self._build_intent_resolution(
                    intent="unsupported",
                    stock_codes=[],
                    requested_order_side=None,
                    requested_quantity=None,
                    conditions=[],
                    followup_reference=None,
                    confidence=0.99,
                    missing_slots=[],
                    source="rule",
                )
                intent_resolution["scope_resolution"] = dict(scope_resolution.get("scope_resolution") or {})
                intent_resolution["resolved_scope_entities"] = list(scope_resolution.get("resolved_scope_entities") or [])
                return ChatPlan(
                    primary_intent="unsupported",
                    clarification=clarification or self._build_market_wide_scope_unsupported_message(),
                    planner_source="rule",
                    intent_source="rule",
                    intent_resolution=intent_resolution,
                    pending_actions=pending_actions,
                    stock_scope={"mode": "explicit", "stock_refs": explicit_stock_refs},
                    followup_target={"mode": "none", "stock_refs": []},
                    blocked_code=blocked_code,
                )
            if clarification:
                intent_resolution = self._build_intent_resolution(
                    intent="clarify",
                    stock_codes=direct_stock_codes,
                    requested_order_side=None,
                    requested_quantity=None,
                    conditions=[],
                    followup_reference=None,
                    confidence=0.96,
                    missing_slots=["stock_codes"],
                    source="rule",
                )
                intent_resolution["scope_resolution"] = dict(scope_resolution.get("scope_resolution") or {})
                intent_resolution["resolved_scope_entities"] = list(scope_resolution.get("resolved_scope_entities") or [])
                return ChatPlan(
                    primary_intent="clarify",
                    clarification=clarification,
                    planner_source="rule",
                    intent_source="rule",
                    intent_resolution=intent_resolution,
                    pending_actions=pending_actions,
                    stock_scope={"mode": "explicit", "stock_refs": explicit_stock_refs},
                    followup_target={"mode": "none", "stock_refs": []},
                )
            if direct_stock_codes:
                intent_resolution = self._build_intent_resolution(
                    intent="analysis",
                    stock_codes=direct_stock_codes,
                    requested_order_side=None,
                    requested_quantity=None,
                    conditions=[],
                    followup_reference=None,
                    confidence=0.98,
                    missing_slots=[],
                    source="rule",
                )
                intent_resolution["scope_resolution"] = dict(scope_resolution.get("scope_resolution") or {})
                intent_resolution["resolved_scope_entities"] = list(scope_resolution.get("resolved_scope_entities") or [])
                return ChatPlan(
                    primary_intent="analysis",
                    stock_codes=direct_stock_codes,
                    planner_source="rule",
                    intent_source="rule",
                    intent_resolution=intent_resolution,
                    pending_actions=pending_actions,
                    required_tools=["run_multi_stock_analysis"],
                    stock_scope={"mode": "explicit", "stock_refs": explicit_stock_refs},
                    followup_target={"mode": "none", "stock_refs": []},
                )
        session_summary = {
            "focus_stocks": list(conversation_state.get("focus_stocks") or default_focus_codes),
            "last_intent": conversation_state.get("last_intent"),
            "pending_action_count": len(pending_actions),
            "last_stage_memory": dict(conversation_state.get("last_stage_memory") or {}),
            "current_task": dict(conversation_state.get("current_task") or {}),
        }
        planner_plan = self.chat_planner.plan(
            message=message,
            session_summary=session_summary,
            pending_action_summary=[dict(item) for item in pending_actions[:5]],
            tool_registry=list(self.tools.keys()),
            analyzer=llm_selection.analyzer,
        )
        resolved_stock_scope = dict(planner_plan.stock_scope)
        resolved_followup_target = dict(planner_plan.followup_target)
        resolved_intent_resolution = planner_plan.to_intent_resolution()
        resolved_intent_resolution.setdefault(
            "scope_resolution",
            {
                "mode": str(resolved_stock_scope.get("mode") or "none"),
                "requested_refs": list(resolved_stock_scope.get("stock_refs") or []),
                "resolved_stock_codes": [],
                "unresolved_refs": [],
                "blocked_code": None,
                "resolver": "scope_resolver_v2",
            },
        )
        resolved_intent_resolution.setdefault("resolved_scope_entities", [])
        resolved_session_overrides = dict(planner_plan.session_preference_overrides)

        if planner_plan.intent == "clarify":
            if explicit_stock_refs and direct_stock_reference_message:
                fallback_scope_resolution = self._resolve_stock_codes_from_refs(
                    explicit_stock_refs,
                    fallback_focus_codes=default_focus_codes,
                    stock_scope_mode="explicit",
                    pending_actions=pending_actions,
                )
                fallback_stock_codes = self._normalize_stock_codes(fallback_scope_resolution.get("stock_codes"))
                resolved_intent_resolution["scope_resolution"] = dict(fallback_scope_resolution.get("scope_resolution") or {})
                resolved_intent_resolution["resolved_scope_entities"] = [
                    dict(item)
                    for item in fallback_scope_resolution.get("resolved_scope_entities") or []
                    if isinstance(item, dict)
                ]
                if fallback_scope_resolution.get("blocked_code") == "market_wide_stock_selection_unsupported":
                    return ChatPlan(
                        primary_intent="unsupported",
                        clarification=str(
                            fallback_scope_resolution.get("clarification") or self._build_market_wide_scope_unsupported_message()
                        ).strip(),
                        planner_source="rule",
                        intent_source="rule",
                        intent_resolution=resolved_intent_resolution,
                        pending_actions=pending_actions,
                        required_tools=list(planner_plan.required_tools),
                        stock_scope={"mode": "explicit", "stock_refs": explicit_stock_refs},
                        followup_target=resolved_followup_target,
                        session_preference_overrides=resolved_session_overrides,
                        blocked_code="market_wide_stock_selection_unsupported",
                    )
                if fallback_stock_codes:
                    return ChatPlan(
                        primary_intent="analysis",
                        stock_codes=fallback_stock_codes,
                        planner_source="rule",
                        intent_source="rule",
                        intent_resolution=resolved_intent_resolution,
                        pending_actions=pending_actions,
                        required_tools=["run_multi_stock_analysis"],
                        stock_scope={"mode": "explicit", "stock_refs": explicit_stock_refs},
                        followup_target={"mode": "none", "stock_refs": []},
                        session_preference_overrides=resolved_session_overrides,
                    )
                fallback_clarification = str(fallback_scope_resolution.get("clarification") or "").strip()
                if fallback_clarification:
                    return ChatPlan(
                        primary_intent="clarify",
                        clarification=fallback_clarification,
                        planner_source="rule",
                        intent_source="rule",
                        intent_resolution=resolved_intent_resolution,
                        pending_actions=pending_actions,
                        required_tools=list(planner_plan.required_tools),
                        stock_scope={"mode": "explicit", "stock_refs": explicit_stock_refs},
                        followup_target=resolved_followup_target,
                        session_preference_overrides=resolved_session_overrides,
                    )
            return ChatPlan(
                primary_intent="clarify",
                clarification=planner_plan.clarification,
                planner_source=planner_plan.planner_source,
                intent_source=planner_plan.planner_source,
                intent_resolution=resolved_intent_resolution,
                pending_actions=pending_actions,
                required_tools=list(planner_plan.required_tools),
                stock_scope=resolved_stock_scope,
                followup_target=resolved_followup_target,
                session_preference_overrides=resolved_session_overrides,
            )

        if (
            planner_plan.intent in {"analysis", "analysis_then_execute"}
            and explicit_stock_refs
            and not resolved_stock_scope.get("stock_refs")
        ):
            resolved_stock_scope["mode"] = "explicit"
            resolved_stock_scope["stock_refs"] = list(explicit_stock_refs)
            resolved_intent_resolution["stock_scope"] = dict(resolved_stock_scope)
        elif (
            planner_plan.intent in {"analysis", "analysis_then_execute"}
            and str(resolved_stock_scope.get("mode") or "none") == "explicit"
            and not resolved_stock_scope.get("stock_refs")
        ):
            fallback_codes, fallback_unresolved = self._extract_explicit_stock_references(message)
            fallback_refs = [*fallback_codes, *fallback_unresolved]
            if fallback_refs:
                resolved_stock_scope["mode"] = "explicit"
                resolved_stock_scope["stock_refs"] = fallback_refs
                resolved_intent_resolution["stock_scope"] = dict(resolved_stock_scope)
            else:
                return ChatPlan(
                    primary_intent="clarify",
                    clarification="请告诉我你想分析哪只股票，最好直接提供股票名称或 6 位股票代码。",
                    planner_source=planner_plan.planner_source,
                    intent_source=planner_plan.planner_source,
                    intent_resolution=resolved_intent_resolution,
                    pending_actions=pending_actions,
                    required_tools=list(planner_plan.required_tools),
                    stock_scope=resolved_stock_scope,
                    followup_target=resolved_followup_target,
                    session_preference_overrides=resolved_session_overrides,
                )

        scope_resolution = self._resolve_stock_codes_from_refs(
            resolved_stock_scope.get("stock_refs"),
            fallback_focus_codes=default_focus_codes,
            stock_scope_mode=str(resolved_stock_scope.get("mode") or "none"),
            pending_actions=pending_actions,
        )
        resolved_intent_resolution["scope_resolution"] = dict(scope_resolution.get("scope_resolution") or {})
        resolved_intent_resolution["resolved_scope_entities"] = [
            dict(item) for item in scope_resolution.get("resolved_scope_entities") or [] if isinstance(item, dict)
        ]
        stock_codes = [str(code or "").strip() for code in scope_resolution.get("stock_codes") or [] if str(code or "").strip()]
        if scope_resolution.get("blocked_code") == "market_wide_stock_selection_unsupported":
            clarification = str(scope_resolution.get("clarification") or "").strip() or self._build_market_wide_scope_unsupported_message()
            return ChatPlan(
                primary_intent="unsupported",
                clarification=clarification,
                planner_source=planner_plan.planner_source,
                intent_source=planner_plan.planner_source,
                intent_resolution=resolved_intent_resolution,
                pending_actions=pending_actions,
                required_tools=list(planner_plan.required_tools),
                stock_scope=resolved_stock_scope,
                followup_target=resolved_followup_target,
                session_preference_overrides=resolved_session_overrides,
                blocked_code="market_wide_stock_selection_unsupported",
            )
        if scope_resolution.get("clarification"):
            return ChatPlan(
                primary_intent="clarify",
                clarification=str(scope_resolution.get("clarification") or "").strip(),
                planner_source=planner_plan.planner_source,
                intent_source=planner_plan.planner_source,
                intent_resolution=resolved_intent_resolution,
                pending_actions=pending_actions,
                required_tools=list(planner_plan.required_tools),
                stock_scope=resolved_stock_scope,
                followup_target=resolved_followup_target,
                session_preference_overrides=resolved_session_overrides,
            )

        if planner_plan.intent in {"analysis", "analysis_then_execute"} and not stock_codes:
            if self._is_market_wide_selection_request(message) or self._stock_scope_refs_hit_market_wide_boundary(
                resolved_stock_scope.get("stock_refs"),
            ):
                clarification = self._build_market_wide_scope_unsupported_message()
                return ChatPlan(
                    primary_intent="unsupported",
                    clarification=clarification,
                    planner_source=planner_plan.planner_source,
                    intent_source=planner_plan.planner_source,
                    intent_resolution=self._build_intent_resolution(
                        intent="unsupported",
                        stock_codes=[],
                        requested_order_side=None,
                        requested_quantity=None,
                        conditions=[],
                        followup_reference=None,
                        confidence=0.99,
                        missing_slots=[],
                        source=planner_plan.planner_source,
                    ),
                    pending_actions=pending_actions,
                    required_tools=list(planner_plan.required_tools),
                    stock_scope=resolved_stock_scope,
                    followup_target=resolved_followup_target,
                    session_preference_overrides=resolved_session_overrides,
                    blocked_code="market_wide_stock_selection_unsupported",
                )
            return ChatPlan(
                primary_intent="clarify",
                clarification="请告诉我你想分析的具体股票名称或 6 位股票代码；当前我不能在没有候选范围的情况下直接发起分析。",
                planner_source=planner_plan.planner_source,
                intent_source=planner_plan.planner_source,
                intent_resolution=resolved_intent_resolution,
                pending_actions=pending_actions,
                required_tools=list(planner_plan.required_tools),
                stock_scope=resolved_stock_scope,
                followup_target=resolved_followup_target,
                session_preference_overrides=resolved_session_overrides,
                blocked_code="analysis_scope_empty",
            )

        requested_order_side, requested_quantity = self._extract_requested_values_from_constraints(planner_plan.constraints)
        selected_orders = self._resolve_followup_target_orders(
            followup_target=planner_plan.followup_target,
            stock_codes=stock_codes,
            pending_actions=pending_actions,
            candidate_snapshots=candidate_snapshots,
        )

        primary_intent = planner_plan.intent
        target_candidate_order: dict[str, Any] | None = None
        target_candidate_orders: list[dict[str, Any]] = []
        followup_reference: str | None = None
        if planner_plan.intent == "order_followup":
            if not selected_orders:
                if len(pending_actions) > 1:
                    return ChatPlan(
                        primary_intent="clarify",
                        clarification="当前有多笔待确认动作，请明确告诉我要执行哪只股票，或直接说“把刚才那几笔都下了”。",
                        planner_source=planner_plan.planner_source,
                        intent_source=planner_plan.planner_source,
                        intent_resolution=resolved_intent_resolution,
                        pending_actions=pending_actions,
                        required_tools=list(planner_plan.required_tools),
                        stock_scope=resolved_stock_scope,
                        followup_target=resolved_followup_target,
                        session_preference_overrides=resolved_session_overrides,
                    )
                return ChatPlan(
                    primary_intent="clarify",
                    clarification="当前没有可执行的历史候选单，请先完成一轮分析，或明确告诉我要执行哪只股票。",
                    planner_source=planner_plan.planner_source,
                    intent_source=planner_plan.planner_source,
                    intent_resolution=resolved_intent_resolution,
                    pending_actions=pending_actions,
                    required_tools=list(planner_plan.required_tools),
                    stock_scope=resolved_stock_scope,
                    followup_target=resolved_followup_target,
                    session_preference_overrides=resolved_session_overrides,
                )
            followup_reference = "pending_actions" if pending_actions else "candidate_snapshot"
            if len(selected_orders) == 1:
                primary_intent = "order_followup_single"
                target_candidate_order = dict(selected_orders[0])
                target_candidate_orders = [dict(selected_orders[0])]
                stock_codes = [str(target_candidate_order.get("code") or "").strip()]
            else:
                primary_intent = "order_followup_all"
                target_candidate_orders = [dict(item) for item in selected_orders]
                stock_codes = [str(item.get("code") or "").strip() for item in target_candidate_orders if str(item.get("code") or "").strip()]

        return ChatPlan(
            primary_intent=primary_intent,
            stock_codes=stock_codes,
            include_runtime_context="load_account_state" in planner_plan.required_tools,
            include_history="load_history" in planner_plan.required_tools,
            include_backtest="load_backtest" in planner_plan.required_tools,
            target_candidate_order=target_candidate_order,
            target_candidate_orders=target_candidate_orders,
            clarification=planner_plan.clarification,
            autonomous_execution_authorized=planner_plan.execution_authorized,
            planner_source=planner_plan.planner_source,
            requested_order_side=requested_order_side,
            requested_quantity=requested_quantity,
            conditions=[dict(item) for item in planner_plan.constraints if isinstance(item, dict)],
            followup_reference=followup_reference,
            intent_resolution=resolved_intent_resolution,
            pending_actions=pending_actions,
            required_tools=list(planner_plan.required_tools),
            intent_source=planner_plan.planner_source,
            stock_scope=resolved_stock_scope,
            followup_target=resolved_followup_target,
            session_preference_overrides=self._normalize_session_preference_overrides(
                resolved_session_overrides,
            ),
        )

    def _resolve_stock_codes_from_refs(
        self,
        stock_refs: Any,
        *,
        fallback_focus_codes: list[str],
        stock_scope_mode: str,
        pending_actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        requested_refs = [str(item or "").strip() for item in (stock_refs if isinstance(stock_refs, list) else []) if str(item or "").strip()]
        if stock_scope_mode == "focus":
            stock_codes = [str(code or "").strip() for code in fallback_focus_codes if str(code or "").strip()]
            return {
                "stock_codes": stock_codes,
                "unresolved_refs": [],
                "blocked_code": None,
                "clarification": None,
                "scope_resolution": {
                    "mode": "focus",
                    "requested_refs": [],
                    "resolved_stock_codes": stock_codes,
                    "unresolved_refs": [],
                    "blocked_code": None,
                    "resolver": "scope_resolver_v2",
                },
                "resolved_scope_entities": [],
            }
        if stock_scope_mode == "pending_actions":
            pending_codes = [
                str(item.get("code") or "").strip()
                for item in (pending_actions or [])
                if isinstance(item, dict) and str(item.get("code") or "").strip()
            ]
            if not pending_codes:
                pending_codes = [str(code or "").strip() for code in fallback_focus_codes if str(code or "").strip()]
            return {
                "stock_codes": pending_codes,
                "unresolved_refs": [],
                "blocked_code": None,
                "clarification": None,
                "scope_resolution": {
                    "mode": "pending_actions",
                    "requested_refs": [],
                    "resolved_stock_codes": pending_codes,
                    "unresolved_refs": [],
                    "blocked_code": None,
                    "resolver": "scope_resolver_v2",
                },
                "resolved_scope_entities": [],
            }

        explicit_codes: list[str] = []
        unresolved_refs: list[str] = []
        resolved_scope_entities: list[dict[str, Any]] = []
        blocked_code: str | None = None
        clarification: str | None = None
        refs = stock_refs if isinstance(stock_refs, list) else []

        for item in refs:
            ref = str(item or "").strip()
            if not ref:
                continue
            entity = self._resolve_scope_entity(ref)
            resolved_scope_entities.append(dict(entity))
            for code in entity.get("stock_codes") or []:
                text = str(code or "").strip()
                if text and text not in explicit_codes:
                    explicit_codes.append(text)
            status = str(entity.get("status") or "").strip()
            if status == "blocked":
                blocked_code = "market_wide_stock_selection_unsupported"
                clarification = str(entity.get("message") or "").strip() or self._build_market_wide_scope_unsupported_message()
                break
            if status in {"ambiguous", "unavailable"} and not clarification:
                clarification = str(entity.get("message") or "").strip() or None
            if status == "unknown":
                unresolved_ref = str(entity.get("raw_ref") or "").strip()
                if unresolved_ref and unresolved_ref not in unresolved_refs:
                    unresolved_refs.append(unresolved_ref)

        if not clarification and unresolved_refs:
            clarification = self._build_stock_name_clarification(unresolved_refs)

        return {
            "stock_codes": explicit_codes,
            "unresolved_refs": unresolved_refs,
            "blocked_code": blocked_code,
            "clarification": clarification,
            "scope_resolution": {
                "mode": "explicit",
                "requested_refs": requested_refs,
                "resolved_stock_codes": explicit_codes,
                "unresolved_refs": unresolved_refs,
                "blocked_code": blocked_code,
                "resolver": "scope_resolver_v2",
            },
            "resolved_scope_entities": resolved_scope_entities,
        }

    @staticmethod
    def _extract_requested_values_from_constraints(constraints: list[dict[str, Any]]) -> tuple[str | None, int | None]:
        requested_order_side: str | None = None
        requested_quantity: int | None = None
        for item in constraints:
            if not isinstance(item, dict):
                continue
            type_name = str(item.get("type") or "").strip()
            if type_name == "order_side":
                value = str(item.get("value") or "").strip().lower()
                if value in {"buy", "sell"}:
                    requested_order_side = value
            elif type_name == "exact_quantity":
                try:
                    quantity = int(float(item.get("value")))
                except (TypeError, ValueError):
                    quantity = 0
                if quantity > 0:
                    requested_quantity = quantity
        return requested_order_side, requested_quantity

    def _resolve_followup_target_orders(
        self,
        *,
        followup_target: dict[str, Any],
        stock_codes: list[str],
        pending_actions: list[dict[str, Any]],
        candidate_snapshots: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        mode = str(followup_target.get("mode") or "none").strip().lower()
        refs = self._normalize_stock_codes(followup_target.get("stock_refs"))
        if mode == "all":
            if pending_actions:
                return [dict(item) for item in pending_actions]
            if candidate_snapshots:
                return [dict(item) for item in candidate_snapshots[0].get("candidate_orders") or [] if isinstance(item, dict)]
            return []
        if mode == "best":
            if pending_actions:
                best = self._pick_best_candidate_order({"candidate_orders": pending_actions, "structured_result": {}})
                return [best] if best else []
            if candidate_snapshots:
                best = self._pick_best_candidate_order(candidate_snapshots[0])
                return [best] if best else []
            return []
        if mode == "single":
            scoped_codes = refs or list(stock_codes)
            if scoped_codes:
                resolved = self._resolve_followup_candidate_orders(
                    message="",
                    stock_codes=scoped_codes,
                    candidate_snapshots=candidate_snapshots,
                    prefer_all=False,
                    prefer_best=False,
                )
                if resolved:
                    return resolved[:1]
                for item in pending_actions:
                    if str(item.get("code") or "").strip() in scoped_codes:
                        return [dict(item)]
            if len(pending_actions) == 1:
                return [dict(pending_actions[0])]
            if candidate_snapshots:
                latest = [dict(item) for item in candidate_snapshots[0].get("candidate_orders") or [] if isinstance(item, dict)]
                if len(latest) == 1:
                    return latest
            return []
        return []

    @staticmethod
    def _normalize_session_preference_overrides(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, Any] = {}
        for key in (
            "riskProfile",
            "analysisStrategy",
            "maxSingleTradeAmount",
            "positionMaxPct",
            "stopLossPct",
            "takeProfitPct",
            "executionPolicy",
            "responseStyle",
        ):
            if key in value:
                normalized[key] = value[key]
        return normalized

    def _build_session_memory_payload(
        self,
        *,
        conversation_state: dict[str, Any],
        frontend_context: dict[str, Any],
        recent_assistant_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        latest_assistant_message = recent_assistant_messages[0] if recent_assistant_messages else None
        latest_meta = latest_assistant_message.get("meta") if isinstance(latest_assistant_message, dict) else {}
        default_focus_codes = self._extract_default_stock_codes(
            latest_meta if isinstance(latest_meta, dict) else {},
            frontend_context,
            conversation_state,
        )
        pending_actions = self._normalize_pending_actions(conversation_state.get("pending_actions"))
        current_task = conversation_state.get("current_task") if isinstance(conversation_state.get("current_task"), dict) else {}
        last_stage_memory = conversation_state.get("last_stage_memory") if isinstance(conversation_state.get("last_stage_memory"), dict) else {}
        last_analysis_summary = conversation_state.get("last_analysis_summary") if isinstance(conversation_state.get("last_analysis_summary"), dict) else {}
        failure_reason = (
            str(current_task.get("failure_reason") or "").strip()
            or str(last_stage_memory.get("failure_reason") or "").strip()
            or str(last_stage_memory.get("termination_reason") or "").strip()
            or None
        )
        return {
            "focus_stocks": list(conversation_state.get("focus_stocks") or default_focus_codes),
            "last_intent": str(conversation_state.get("last_intent") or "").strip(),
            "current_task": dict(current_task),
            "last_user_goal": str(current_task.get("user_message") or last_analysis_summary.get("lead_stock", {}).get("code") or "").strip() or None,
            "last_failure_reason": failure_reason,
            "pending_actions": [dict(item) for item in pending_actions],
            "session_preference_overrides": dict(conversation_state.get("session_preference_overrides") or {}),
            "last_stage_memory": dict(last_stage_memory),
        }

    @staticmethod
    def _build_stage_memory_from_structured_result(
        structured_result: dict[str, Any],
        execution_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(structured_result, dict):
            if isinstance(execution_result, dict) and execution_result:
                return {
                    "execution": {
                        "status": execution_result.get("status"),
                        "summary": execution_result.get("summary"),
                    },
                    "failure_reason": execution_result.get("status") if execution_result.get("status") == "failed" else None,
                }
            return {}

        stage_traces = structured_result.get("stage_traces") if isinstance(structured_result.get("stage_traces"), list) else []
        stage_map: dict[str, dict[str, Any]] = {}
        for trace in stage_traces:
            if not isinstance(trace, dict):
                continue
            stage = str(trace.get("stage") or "").strip()
            if not stage:
                continue
            stage_map[stage] = {
                "summary": trace.get("summary"),
                "confidence": trace.get("confidence"),
                "warnings": list(trace.get("warnings") or []),
                "next_action": trace.get("next_action"),
            }

        failed_condition = None
        for item in structured_result.get("condition_evaluations") or []:
            if isinstance(item, dict) and not bool(item.get("passed")) and bool(item.get("blocking", True)):
                failed_condition = str(item.get("reason") or "policy_blocked").strip() or "policy_blocked"
                break

        return {
            "data_quality": stage_map.get("data"),
            "signal": stage_map.get("signal"),
            "risk": stage_map.get("risk"),
            "execution": {
                **(stage_map.get("execution") or {}),
                "result": dict(execution_result or {}) if isinstance(execution_result, dict) else structured_result.get("execution_result"),
            },
            "termination_reason": structured_result.get("termination_reason"),
            "failure_reason": failed_condition or structured_result.get("termination_reason"),
            "replan_count": structured_result.get("replan_count"),
            "policy_snapshot": dict(structured_result.get("policy_snapshot") or {}),
        }

    async def _load_context_bundle(
        self,
        *,
        plan: ChatPlan,
        tool_context: dict[str, Any],
        event_handler: ChatEventHandler | None,
    ) -> ChatContextBundle:
        loaded_keys: list[str] = []
        system_state_payload: dict[str, Any] = {}
        account_state_payload: dict[str, Any] = {}
        portfolio_health_payload: dict[str, Any] = {}
        session_memory_payload: dict[str, Any] = {}
        user_preferences_payload: dict[str, Any] = {}
        stage_memory_payload: dict[str, Any] = {}
        history_payload: dict[str, Any] = {}
        backtest_payload: dict[str, Any] = {}

        if "load_system_state" in plan.required_tools:
            system_state_payload = await self._run_tool(
                "load_system_state",
                {"runtime_config": tool_context.get("runtime_config")},
                tool_context,
                event_handler,
            )
            loaded_keys.append("system_state")

        if "load_account_state" in plan.required_tools:
            account_state_payload = await self._run_tool(
                "load_account_state",
                {"owner_user_id": tool_context.get("owner_user_id"), "refresh": True},
                tool_context,
                event_handler,
            )
            loaded_keys.append("account_state")

        if "load_portfolio_health" in plan.required_tools:
            portfolio_health_payload = await self._run_tool(
                "load_portfolio_health",
                {"owner_user_id": tool_context.get("owner_user_id"), "refresh": True},
                tool_context,
                event_handler,
            )
            loaded_keys.append("portfolio_health")

        if "load_session_memory" in plan.required_tools:
            session_memory_payload = await self._run_tool(
                "load_session_memory",
                {},
                tool_context,
                event_handler,
            )
            loaded_keys.append("session_memory")

        if "load_user_preferences" in plan.required_tools:
            user_preferences_payload = await self._run_tool(
                "load_user_preferences",
                {
                    "owner_user_id": tool_context.get("owner_user_id"),
                    "session_overrides": plan.session_preference_overrides,
                },
                tool_context,
                event_handler,
            )
            loaded_keys.append("effective_user_preferences")

        if "load_stage_memory" in plan.required_tools:
            stage_memory_payload = await self._run_tool(
                "load_stage_memory",
                {},
                tool_context,
                event_handler,
            )
            loaded_keys.append("stage_memory")

        if "load_history" in plan.required_tools:
            history_payload = await self._run_tool(
                "load_history",
                {"owner_user_id": tool_context.get("owner_user_id"), "stock_codes": plan.stock_codes, "limit": 5},
                tool_context,
                event_handler,
            )
            loaded_keys.append("history")

        if "load_backtest" in plan.required_tools:
            backtest_payload = await self._run_tool(
                "load_backtest",
                {"owner_user_id": tool_context.get("owner_user_id"), "stock_codes": plan.stock_codes, "limit": 6},
                tool_context,
                event_handler,
            )
            loaded_keys.append("backtest")

        return ChatContextBundle(
            loaded_keys=loaded_keys,
            system_state=AgentSystemState(system_state_payload),
            account_state=AgentAccountState(account_state_payload),
            portfolio_health=dict(portfolio_health_payload),
            session_memory=dict(session_memory_payload),
            effective_user_preferences=EffectiveUserPreferences(user_preferences_payload),
            stage_memory=StageMemorySnapshot(stage_memory_payload),
            history=dict(history_payload),
            backtest=dict(backtest_payload),
        )

    async def _run_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_context: dict[str, Any],
        event_handler: ChatEventHandler | None,
    ) -> dict[str, Any]:
        handler = self.tools.get(tool_name)
        if handler is None:
            raise ValueError(f"unsupported tool: {tool_name}")
        await self._emit(
            event_handler,
            "tool_start",
            {
                "tool": tool_name,
                "summary": self._tool_summary(tool_name, args),
            },
        )
        result = await handler(args, tool_context)
        await self._emit(
            event_handler,
            "tool_done",
            {
                "tool": tool_name,
                "summary": self._tool_done_summary(tool_name, result),
            },
        )
        return result

    @staticmethod
    async def _emit(event_handler: ChatEventHandler | None, event_name: str, payload: dict[str, Any]) -> None:
        if event_handler is None:
            return
        maybe_result = event_handler(event_name, payload)
        if asyncio.iscoroutine(maybe_result):
            await maybe_result

    async def _emit_assistant_message_content(
        self,
        event_handler: ChatEventHandler | None,
        content: str,
    ) -> None:
        text = str(content or "")
        await self._emit(
            event_handler,
            "message_start",
            {
                "role": "assistant",
                "format": "markdown",
            },
        )
        if not text:
            return

        for chunk in self._chunk_assistant_message(text):
            await self._emit(event_handler, "message_delta", {"delta": chunk})
            await asyncio.sleep(0.016)

    async def _emit_outside_trading_session_warning_if_needed(
        self,
        event_handler: ChatEventHandler | None,
        execution_result: dict[str, Any] | None,
    ) -> None:
        if not self._is_outside_trading_session_execution(execution_result):
            return
        payload = self._build_outside_trading_session_warning_payload(execution_result)
        if payload:
            await self._emit(event_handler, "warning", payload)

    @staticmethod
    def _normalize_execution_status_value(status: Any) -> str:
        return str(status or "").strip().lower()

    @classmethod
    def _is_direct_outside_trading_session_result(cls, value: dict[str, Any] | None) -> bool:
        if not isinstance(value, dict):
            return False
        reason = cls._normalize_execution_status_value(value.get("reason"))
        status = cls._normalize_execution_status_value(value.get("status"))
        return reason == _OUTSIDE_TRADING_SESSION_REASON and status == "blocked"

    @staticmethod
    def _iter_nested_execution_items(execution_result: dict[str, Any] | None):
        if not isinstance(execution_result, dict):
            return
        for key in ("blocked_orders", "failed_orders", "orders"):
            items = execution_result.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    yield item

    @classmethod
    def _count_successful_execution_results(cls, execution_result: dict[str, Any] | None) -> int:
        if not isinstance(execution_result, dict):
            return 0

        mode = str(execution_result.get("mode") or "").strip().lower()
        if mode == "batch":
            orders = execution_result.get("orders")
            if isinstance(orders, list):
                return sum(
                    1
                    for item in orders
                    if isinstance(item, dict)
                    and cls._normalize_execution_status_value(item.get("status")) in _SUCCESSFUL_EXECUTION_STATUSES
                )
            return 0

        if cls._normalize_execution_status_value(execution_result.get("status")) in _SUCCESSFUL_EXECUTION_STATUSES:
            return 1
        return 0

    @classmethod
    def _is_outside_trading_session_execution(cls, execution_result: dict[str, Any] | None) -> bool:
        if not isinstance(execution_result, dict):
            return False

        if cls._count_successful_execution_results(execution_result) > 0:
            return False

        if cls._is_direct_outside_trading_session_result(execution_result):
            return True

        blocked_items = execution_result.get("blocked_orders")
        if isinstance(blocked_items, list) and blocked_items:
            normalized = [item for item in blocked_items if isinstance(item, dict)]
            return bool(normalized) and all(cls._is_direct_outside_trading_session_result(item) for item in normalized)

        failed_items = execution_result.get("failed_orders")
        if isinstance(failed_items, list) and failed_items:
            normalized = [item for item in failed_items if isinstance(item, dict)]
            direct_failed = [item for item in normalized if cls._is_direct_outside_trading_session_result(item)]
            return bool(direct_failed) and len(direct_failed) == len(normalized)

        return False

    @classmethod
    def _extract_execution_message(cls, execution_result: dict[str, Any] | None) -> str:
        if not isinstance(execution_result, dict):
            return ""

        message = str(execution_result.get("message") or "").strip()
        if message:
            return message

        for item in cls._iter_nested_execution_items(execution_result):
            nested_message = str(item.get("message") or "").strip()
            if nested_message:
                return nested_message
        return ""

    @classmethod
    def _extract_execution_session_guard(cls, execution_result: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(execution_result, dict):
            return {}

        session_guard = execution_result.get("session_guard")
        if isinstance(session_guard, dict):
            return dict(session_guard)

        for item in cls._iter_nested_execution_items(execution_result):
            nested_guard = item.get("session_guard")
            if isinstance(nested_guard, dict):
                return dict(nested_guard)
        return {}

    @classmethod
    def _extract_execution_stock_code(cls, execution_result: dict[str, Any] | None) -> str:
        if not isinstance(execution_result, dict):
            return ""

        candidate_order = execution_result.get("candidate_order")
        if isinstance(candidate_order, dict):
            code = str(candidate_order.get("code") or "").strip()
            if code:
                return code

        for item in cls._iter_nested_execution_items(execution_result):
            candidate_order = item.get("candidate_order")
            if not isinstance(candidate_order, dict):
                continue
            code = str(candidate_order.get("code") or "").strip()
            if code:
                return code
        return ""

    @classmethod
    def _build_outside_trading_session_warning_payload(
        cls,
        execution_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not cls._is_outside_trading_session_execution(execution_result):
            return {}

        session_guard = cls._extract_execution_session_guard(execution_result)
        next_open_at = str(session_guard.get("next_open_at") or "").strip()
        message = "当前非交易时段，本轮未执行模拟盘订单，候选单已保留。"
        if next_open_at:
            message = f"{message} 下次可执行时间：{next_open_at}。"

        payload: dict[str, Any] = {
            "stage": "execution",
            "message": message,
        }
        stock_code = cls._extract_execution_stock_code(execution_result)
        if stock_code:
            payload["stock_code"] = stock_code
        return payload

    @classmethod
    def _build_autonomous_execution_payload(
        cls,
        *,
        authorized: bool,
        candidate_orders: list[dict[str, Any]],
        execution_result: dict[str, Any] | None,
        default_reason: str,
        default_gate_passed: bool,
        default_gate_message: str,
    ) -> dict[str, Any]:
        payload = {
            "requested": True,
            "authorized": authorized,
            "execution_scope": "single" if len(candidate_orders) == 1 else "all",
            "candidate_order_count": len(candidate_orders),
            "executed": False,
            "executed_count": 0,
            "failed_count": 0,
            "reason": str(default_reason or "no_candidate_orders"),
            "gate_passed": bool(default_gate_passed),
            "gate_message": str(default_gate_message or ""),
        }
        if not isinstance(execution_result, dict):
            return payload

        if cls._is_outside_trading_session_execution(execution_result):
            payload.update(
                {
                    "executed": False,
                    "executed_count": 0,
                    "failed_count": 0,
                    "reason": _OUTSIDE_TRADING_SESSION_REASON,
                    "gate_passed": False,
                    "gate_message": cls._extract_execution_message(execution_result) or str(default_gate_message or ""),
                }
            )
            return payload

        success_count = cls._count_successful_execution_results(execution_result)
        failed_count = len(execution_result.get("failed_orders") or []) if isinstance(execution_result.get("failed_orders"), list) else 0
        normalized_status = cls._normalize_execution_status_value(execution_result.get("status"))
        payload.update(
            {
                "executed": success_count > 0,
                "executed_count": success_count,
                "failed_count": failed_count,
                "reason": (
                    "executed_candidate_orders"
                    if success_count > 0
                    else str(execution_result.get("reason") or execution_result.get("status") or "execution_failed")
                ),
                "gate_passed": success_count > 0 or failed_count > 0 or normalized_status in _SUCCESSFUL_EXECUTION_STATUSES,
                "gate_message": str(execution_result.get("message") or default_gate_message or ""),
            }
        )
        return payload

    @staticmethod
    def _chunk_assistant_message(content: str, chunk_size: int = 48) -> list[str]:
        text = str(content or "")
        if not text:
            return []

        chunks: list[str] = []
        for paragraph in re.split(r"(\n+)", text):
            if not paragraph:
                continue
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start:start + chunk_size])
                start += chunk_size
        return chunks

    @staticmethod
    def _tool_summary(tool_name: str, args: dict[str, Any]) -> str:
        if tool_name == "run_multi_stock_analysis":
            codes = "、".join(args.get("stock_codes") or [])
            return f"开始串行分析 {codes}"
        if tool_name == "load_system_state":
            return "读取当前 Agent 系统状态"
        if tool_name == "load_account_state":
            return "读取当前账户与持仓状态"
        if tool_name == "load_portfolio_health":
            return "执行投资组合健康检查"
        if tool_name == "load_session_memory":
            return "读取当前会话记忆"
        if tool_name == "load_user_preferences":
            return "读取当前用户偏好"
        if tool_name == "load_stage_memory":
            return "读取上一轮阶段记忆"
        if tool_name == "load_history":
            return "读取历史分析记录"
        if tool_name == "load_backtest":
            return "读取回测摘要"
        if tool_name == "run_strategy_backtest":
            code = str(args.get("code") or "").strip()
            strategy_count = len(args.get("strategies") or [])
            return f"执行 {code or '--'} 的策略回测（{strategy_count} 个策略）"
        if tool_name == "place_simulated_order":
            order = args.get("candidate_order") if isinstance(args.get("candidate_order"), dict) else {}
            return f"向模拟盘提交 {order.get('code') or ''} 候选订单"
        if tool_name == "batch_execute_candidate_orders":
            count = len(args.get("candidate_orders") or [])
            return f"根据分析结果自动执行 {count} 笔组合候选单"
        return tool_name

    @staticmethod
    def _tool_done_summary(tool_name: str, result: dict[str, Any]) -> str:
        if tool_name == "run_multi_stock_analysis":
            structured = result.get("structured_result")
            stock_count = len(structured.get("stocks") or []) if isinstance(structured, dict) else 0
            candidate_count = len(result.get("candidate_orders") or [])
            return f"已完成 {stock_count} 只股票分析，生成 {candidate_count} 笔候选订单"
        if tool_name == "load_system_state":
            execution = result.get("execution") if isinstance(result.get("execution"), dict) else {}
            mode = execution.get("runtime_execution_mode") or "--"
            return f"已读取系统状态，当前执行模式 {mode}"
        if tool_name == "load_account_state":
            account_state = result.get("account_state") if isinstance(result.get("account_state"), dict) else {}
            positions = account_state.get("positions") if isinstance(account_state.get("positions"), list) else []
            return f"已读取账户状态，当前持仓 {len(positions)} 项"
        if tool_name == "load_portfolio_health":
            payload = result.get("portfolio_health") if isinstance(result.get("portfolio_health"), dict) else {}
            diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
            score = diagnostics.get("health_score")
            level = diagnostics.get("health_level") or "--"
            return f"已完成组合健康检查，健康度 {score if score is not None else '--'}/100（{level}）"
        if tool_name == "load_session_memory":
            pending = result.get("pending_actions") if isinstance(result.get("pending_actions"), list) else []
            return f"已读取会话记忆，待确认动作 {len(pending)} 项"
        if tool_name == "load_user_preferences":
            effective = result.get("effective") if isinstance(result.get("effective"), dict) else {}
            trading = effective.get("trading") if isinstance(effective.get("trading"), dict) else {}
            return f"已读取用户偏好，当前风险偏好 {trading.get('riskProfile') or '--'}"
        if tool_name == "load_stage_memory":
            return "已读取上一轮阶段记忆"
        if tool_name == "load_history":
            items = result.get("items") if isinstance(result.get("items"), list) else []
            return f"已读取 {len(items)} 条历史分析"
        if tool_name == "load_backtest":
            items = result.get("items") if isinstance(result.get("items"), list) else []
            return f"已读取 {len(items)} 条回测摘要"
        if tool_name == "run_strategy_backtest":
            items = result.get("items") if isinstance(result.get("items"), list) else []
            code = result.get("code") or "--"
            return f"已完成 {code} 的策略回测，输出 {len(items)} 组策略结果"
        if tool_name == "place_simulated_order":
            if AgentChatService._is_outside_trading_session_execution(result):
                return AgentChatService._extract_execution_message(result) or "非交易时段，模拟盘订单未执行"
            return f"模拟盘执行结果：{result.get('status') or 'unknown'}"
        if tool_name == "batch_execute_candidate_orders":
            executed_count = int(result.get("executed_count") or 0)
            failed_count = len(result.get("failed_orders") or [])
            blocked_count = len(result.get("blocked_orders") or []) if isinstance(result.get("blocked_orders"), list) else 0
            if AgentChatService._is_outside_trading_session_execution(result):
                return f"非交易时段拦截 {blocked_count or 0} 笔候选订单，已保留待再次确认"
            return f"已完成候选订单提交，成功 {executed_count} 笔，失败 {failed_count} 笔"
        return "工具执行完成"

    async def _tool_load_system_state(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        runtime_config = args.get("runtime_config") if isinstance(args.get("runtime_config"), dict) else {}
        execution = runtime_config.get("execution") if isinstance(runtime_config.get("execution"), dict) else {}
        runtime_mode = str(execution.get("mode") or "paper").strip().lower() or "paper"
        task_stats = get_agent_task_service(
            config=self.config,
            db_manager=self.db,
            agent_service=self.agent_service,
        ).get_runtime_stats()
        market_sources = self.runtime_market_service.get_market_source_options()
        available_market_sources = [
            str(item.get("code") or "")
            for item in market_sources.get("options") or []
            if isinstance(item, dict) and bool(item.get("available"))
        ]
        llm_selection = self._resolve_chat_llm_selection(
            runtime_config,
            reason="tool_load_system_state",
            log_resolution=False,
        )
        llm_available = bool(getattr(llm_selection.analyzer, "is_available", lambda: False)())
        shared_busy = bool(task_stats.get("saturated"))
        return {
            "planner": {"available": llm_available, "busy": False, "degraded_reason": None if llm_available else "llm_unavailable"},
            "data": {
                "available": bool(available_market_sources),
                "busy": shared_busy,
                "degraded_reason": None if available_market_sources else "no_market_source_available",
            },
            "signal": {
                "available": True,
                "busy": shared_busy,
                "degraded_reason": None if llm_available else "llm_optional_unavailable",
            },
            "risk": {
                "available": True,
                "busy": shared_busy,
                "degraded_reason": None,
            },
            "execution": {
                "available": runtime_mode in {"paper", "broker"},
                "busy": shared_busy,
                "degraded_reason": None,
                "runtime_execution_mode": runtime_mode,
            },
            "task_pool": task_stats,
            "llm": {
                "available": llm_available,
            },
            "market_sources": market_sources,
        }

    async def _tool_load_account_state(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.backend_client.get_account_state(
            owner_user_id=int(args.get("owner_user_id") or 0),
            refresh=bool(args.get("refresh", True)),
        )

    async def _tool_load_portfolio_health(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.backend_client.get_portfolio_health(
            owner_user_id=int(args.get("owner_user_id") or 0),
            refresh=bool(args.get("refresh", True)),
        )

    async def _tool_load_session_memory(
        self,
        _args: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        conversation_state = tool_context.get("conversation_state") if isinstance(tool_context.get("conversation_state"), dict) else {}
        return self._build_session_memory_payload(
            conversation_state=conversation_state,
            frontend_context=tool_context.get("frontend_context") if isinstance(tool_context.get("frontend_context"), dict) else {},
            recent_assistant_messages=tool_context.get("recent_assistant_messages") if isinstance(tool_context.get("recent_assistant_messages"), list) else [],
        )

    async def _tool_load_user_preferences(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.backend_client.get_user_preferences(
            owner_user_id=int(args.get("owner_user_id") or 0),
            session_overrides=args.get("session_overrides") if isinstance(args.get("session_overrides"), dict) else {},
        )

    async def _tool_load_stage_memory(
        self,
        _args: dict[str, Any],
        tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        conversation_state = tool_context.get("conversation_state") if isinstance(tool_context.get("conversation_state"), dict) else {}
        stage_memory = conversation_state.get("last_stage_memory") if isinstance(conversation_state.get("last_stage_memory"), dict) else {}
        return dict(stage_memory)

    async def _tool_load_history(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.backend_client.get_analysis_history(
            owner_user_id=int(args.get("owner_user_id") or 0),
            stock_codes=list(args.get("stock_codes") or []),
            limit=int(args.get("limit") or 5),
        )

    async def _tool_load_backtest(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.backend_client.get_backtest_summary(
            owner_user_id=int(args.get("owner_user_id") or 0),
            stock_codes=list(args.get("stock_codes") or []),
            limit=int(args.get("limit") or 6),
        )

    async def _tool_run_strategy_backtest(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.backend_client.run_strategy_backtest(
            owner_user_id=int(args.get("owner_user_id") or 0),
            code=str(args.get("code") or "").strip(),
            start_date=str(args.get("start_date") or "").strip(),
            end_date=str(args.get("end_date") or "").strip(),
            strategies=[
                dict(item)
                for item in (args.get("strategies") or [])
                if isinstance(item, dict)
            ],
            initial_capital=self._safe_number(args.get("initial_capital")),
            commission_rate=self._safe_number(args.get("commission_rate")),
            slippage_bps=self._safe_number(args.get("slippage_bps")),
        )

    async def _tool_run_multi_stock_analysis(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        stock_codes = self._normalize_stock_codes(args.get("stock_codes"))
        if not stock_codes:
            raise ValueError("stock_codes must not be empty")
        runtime_config = self._merge_runtime_context_into_runtime_config(
            runtime_config=args.get("runtime_config") if isinstance(args.get("runtime_config"), dict) else {},
            runtime_context_payload=args.get("runtime_context_payload") if isinstance(args.get("runtime_context_payload"), dict) else {},
        )
        planning_context = args.get("planning_context") if isinstance(args.get("planning_context"), dict) else {}
        event_handler = _tool_context.get("event_handler")
        loop = asyncio.get_running_loop()
        owner_user_id = int(_tool_context.get("owner_user_id") or 0)
        session_id = str(_tool_context.get("session_id") or "").strip()

        def stage_observer(payload: dict[str, Any]) -> None:
            event_name = str(payload.get("event") or "stage_update").strip() or "stage_update"
            event_payload = {key: value for key, value in payload.items() if key != "event"}
            asyncio.run_coroutine_threadsafe(
                self._emit(event_handler, event_name, event_payload),
                loop,
            )

        def paper_order_submitter(candidate_order: dict[str, Any]) -> dict[str, Any]:
            if owner_user_id <= 0 or not session_id:
                return {}
            future = asyncio.run_coroutine_threadsafe(
                self.backend_client.place_simulated_order(
                    owner_user_id=owner_user_id,
                    session_id=session_id,
                    candidate_order=candidate_order,
                ),
                loop,
            )
            return future.result(timeout=30)

        try:
            run_result = await asyncio.to_thread(
                self.agent_service.run_once,
                stock_codes,
                runtime_config=runtime_config,
                planning_context=planning_context,
                paper_order_submitter=paper_order_submitter,
                stage_observer=stage_observer,
            )
        except TypeError as exc:
            if "planning_context" not in str(exc) and "stage_observer" not in str(exc):
                raise
            run_result = await asyncio.to_thread(
                self.agent_service.run_once,
                stock_codes,
                runtime_config=runtime_config,
            )
        return self._serialize_analysis_result(run_result)

    async def _tool_place_simulated_order(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_order = args.get("candidate_order") if isinstance(args.get("candidate_order"), dict) else {}
        return await self.backend_client.place_simulated_order(
            owner_user_id=int(args.get("owner_user_id") or 0),
            session_id=str(args.get("session_id") or "").strip(),
            candidate_order=candidate_order,
        )

    async def _tool_batch_execute_candidate_orders(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        owner_user_id = int(args.get("owner_user_id") or 0)
        session_id = str(args.get("session_id") or "").strip()
        candidate_orders = self._normalize_candidate_orders(args.get("candidate_orders"))
        orders: list[dict[str, Any]] = []
        blocked_orders: list[dict[str, Any]] = []
        failed_orders: list[dict[str, Any]] = []

        for candidate_order in candidate_orders:
            try:
                result = await self.backend_client.place_simulated_order(
                    owner_user_id=owner_user_id,
                    session_id=session_id,
                    candidate_order=candidate_order,
                )
                if self._is_outside_trading_session_execution(result):
                    blocked_orders.append(dict(result))
                    continue

                normalized_status = self._normalize_execution_status_value(result.get("status"))
                if normalized_status in _SUCCESSFUL_EXECUTION_STATUSES:
                    orders.append(dict(result))
                    continue

                failed_orders.append(
                    {
                        "candidate_order": dict(
                            result.get("candidate_order")
                            if isinstance(result.get("candidate_order"), dict)
                            else candidate_order
                        ),
                        "message": str(result.get("message") or normalized_status or "execution_failed").strip() or "execution_failed",
                    }
                )
            except Exception as exc:
                failed_orders.append(
                    {
                        "candidate_order": dict(candidate_order),
                        "message": redact_sensitive_text(str(exc)),
                    }
                )

        status = self._resolve_batch_execution_state(orders, failed_orders, blocked_orders)
        payload = {
            "mode": "batch",
            "candidate_order_count": len(candidate_orders),
            "executed_count": len(orders),
            "orders": orders,
            "blocked_orders": blocked_orders,
            "failed_orders": failed_orders,
            "status": status,
            "summary": {
                "candidate_order_count": len(candidate_orders),
                "executed_count": len(orders),
                "failed_count": len(failed_orders),
                "blocked_count": len(blocked_orders),
            },
        }
        if status == "blocked" and blocked_orders:
            first_blocked = blocked_orders[0]
            payload.update(
                {
                    "reason": _OUTSIDE_TRADING_SESSION_REASON,
                    "message": self._extract_execution_message(first_blocked),
                    "session_guard": self._extract_execution_session_guard(first_blocked),
                }
            )
        return payload

    @staticmethod
    def _resolve_batch_execution_state(
        orders: list[dict[str, Any]],
        failed_orders: list[dict[str, Any]],
        blocked_orders: list[dict[str, Any]],
    ) -> str:
        if not orders:
            if blocked_orders and not failed_orders:
                return "blocked"
            return "failed"

        normalized_statuses = {str(item.get("status") or "").strip().lower() for item in orders}
        if failed_orders or blocked_orders:
            return "submitted"
        if normalized_statuses == {"filled"}:
            return "filled"
        if normalized_statuses.issubset(_SUCCESSFUL_EXECUTION_STATUSES):
            if normalized_statuses == {"filled"}:
                return "filled"
            return "submitted"
        return "submitted"

    @staticmethod
    def _merge_runtime_context_into_runtime_config(
        *,
        runtime_config: dict[str, Any],
        runtime_context_payload: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(runtime_config or {})
        simulation_account = runtime_context_payload.get("simulation_account")
        runtime_context = runtime_context_payload.get("runtime_context")
        if isinstance(simulation_account, dict):
            broker_account_id = simulation_account.get("broker_account_id")
        else:
            broker_account_id = None
        merged_execution = dict(merged.get("execution") or {})
        merged_execution["mode"] = "paper"
        merged_execution["has_ticket"] = False
        if broker_account_id:
            merged_execution["broker_account_id"] = broker_account_id
        merged["execution"] = merged_execution

        merged_context = dict(merged.get("context") or {})
        if isinstance(runtime_context, dict):
            summary = runtime_context.get("summary") if isinstance(runtime_context.get("summary"), dict) else {}
            positions = runtime_context.get("positions") if isinstance(runtime_context.get("positions"), list) else []
            account_snapshot = {
                "broker_account_id": runtime_context.get("broker_account_id"),
                "provider_code": runtime_context.get("provider_code"),
                "provider_name": runtime_context.get("provider_name"),
                "account_uid": runtime_context.get("account_uid"),
                "account_display_name": runtime_context.get("account_display_name"),
                "snapshot_at": runtime_context.get("snapshot_at"),
                "data_source": runtime_context.get("data_source"),
                "cash": summary.get("cash"),
                "initial_cash": summary.get("initial_capital") or summary.get("initial_cash"),
                "total_market_value": summary.get("market_value") or summary.get("total_market_value"),
                "total_asset": summary.get("total_asset"),
                "positions": positions,
            }
            merged_context["account_snapshot"] = account_snapshot
            merged_context["summary"] = summary
            merged_context["positions"] = positions
        merged["context"] = merged_context
        return merged

    def _resolve_candidate_orders_for_request(
        self,
        *,
        plan: ChatPlan,
        analysis_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        candidate_orders = self._normalize_candidate_orders(analysis_payload.get("candidate_orders"))
        if not candidate_orders:
            return []

        selected_orders = [dict(item) for item in candidate_orders]
        if plan.stock_codes:
            plan_code_set = {str(code).strip() for code in plan.stock_codes if str(code).strip()}
            selected_orders = [item for item in selected_orders if str(item.get("code") or "").strip() in plan_code_set]

        if plan.requested_order_side:
            selected_orders = [
                item for item in selected_orders
                if str(item.get("action") or "").strip() == str(plan.requested_order_side).strip()
            ]

        if plan.requested_quantity and plan.requested_quantity > 0:
            selected_orders = [
                self._override_candidate_order_quantity(item, plan.requested_quantity)
                for item in selected_orders
            ]
        self._sync_analysis_payload_candidate_orders(
            analysis_payload=analysis_payload,
            candidate_orders=selected_orders,
            plan=plan,
        )
        return selected_orders

    @staticmethod
    def _override_candidate_order_quantity(candidate_order: dict[str, Any], quantity: int) -> dict[str, Any]:
        next_order = dict(candidate_order)
        next_order["quantity"] = int(quantity)
        next_order["target_qty"] = int(quantity)
        final_order = next_order.get("final_order") if isinstance(next_order.get("final_order"), dict) else None
        if isinstance(final_order, dict):
            next_final_order = dict(final_order)
            next_final_order["quantity"] = int(quantity)
            next_final_order["target_qty"] = int(quantity)
            next_order["final_order"] = next_final_order
        next_order["requested_quantity"] = int(quantity)
        return next_order

    def _evaluate_analysis_execution_gate(
        self,
        *,
        plan: ChatPlan,
        analysis_payload: dict[str, Any],
        candidate_orders: list[dict[str, Any]],
        runtime_config: dict[str, Any],
    ) -> dict[str, Any]:
        structured_result = analysis_payload.get("structured_result") if isinstance(analysis_payload.get("structured_result"), dict) else {}
        condition_evaluations = structured_result.get("condition_evaluations") if isinstance(structured_result.get("condition_evaluations"), list) else []
        for item in condition_evaluations:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("passed")) and bool(item.get("blocking", True)):
                reason = str(item.get("reason") or "policy_blocked").strip() or "policy_blocked"
                return {
                    "eligible": False,
                    "reason": reason,
                    "message": self._map_execution_gate_message(reason),
                    "candidate_orders": [],
                }

        execution_config = runtime_config.get("execution") if isinstance(runtime_config.get("execution"), dict) else {}
        runtime_mode = str(execution_config.get("mode") or "paper").strip().lower()
        if runtime_mode != "paper":
            return {
                "eligible": False,
                "reason": "unsupported_execution_mode",
                "message": "自动执行仅支持模拟盘，不会直接走真实券商下单。",
                "candidate_orders": [],
            }

        if plan.requested_quantity is not None and plan.requested_quantity <= 0:
            return {
                "eligible": False,
                "reason": "invalid_quantity",
                "message": "下单数量必须大于 0，当前条件无法执行。",
                "candidate_orders": [],
            }

        analysis_candidate_orders = self._normalize_candidate_orders(analysis_payload.get("candidate_orders"))
        if not candidate_orders:
            if analysis_candidate_orders and plan.requested_order_side:
                action_labels = {"buy": "买入", "sell": "卖出"}
                available_sides = {str(item.get("action") or "").strip() for item in analysis_candidate_orders}
                if available_sides and plan.requested_order_side not in available_sides:
                    return {
                        "eligible": False,
                        "reason": "side_mismatch",
                        "message": (
                            "条件未满足：当前分析形成的提议单方向与请求不一致，"
                            f"暂无可执行的{action_labels.get(plan.requested_order_side, plan.requested_order_side)}提议单。"
                        ),
                        "candidate_orders": [],
                    }
            reason, message = self._explain_missing_candidate_reason(plan=plan, analysis_payload=analysis_payload)
            return {
                "eligible": False,
                "reason": "no_candidate_orders",
                "message": message,
                "candidate_orders": [],
            }

        for item in candidate_orders:
            if int(item.get("quantity") or 0) <= 0:
                return {
                    "eligible": False,
                    "reason": "invalid_quantity",
                    "message": "条件未满足：提议单数量无效，当前不能提交模拟盘订单。",
                    "candidate_orders": [],
                }
            stock = self._find_analysis_stock(analysis_payload, str(item.get("code") or "").strip())
            if self._is_stock_hard_blocked(stock):
                reason = str(self._extract_stock_execution_reason(stock) or "risk_blocked")
                return {
                    "eligible": False,
                    "reason": reason,
                    "message": self._map_execution_gate_message(reason),
                    "candidate_orders": [],
                }

        return {
            "eligible": True,
            "reason": "gate_passed",
            "message": "条件门槛已满足，本轮可直接提交模拟盘订单。",
            "candidate_orders": [dict(item) for item in candidate_orders],
        }

    def _sync_analysis_payload_candidate_orders(
        self,
        *,
        analysis_payload: dict[str, Any],
        candidate_orders: list[dict[str, Any]],
        plan: ChatPlan,
    ) -> None:
        structured_result = analysis_payload.get("structured_result")
        if not isinstance(structured_result, dict):
            return
        stocks = structured_result.get("stocks")
        if not isinstance(stocks, list):
            return
        candidate_map = {
            str(item.get("code") or "").strip(): dict(item)
            for item in candidate_orders
            if isinstance(item, dict) and str(item.get("code") or "").strip()
        }
        scoped_codes = {str(code).strip() for code in plan.stock_codes if str(code).strip()}
        for stock in stocks:
            if not isinstance(stock, dict):
                continue
            code = str(stock.get("code") or "").strip()
            if code in candidate_map:
                stock["candidate_order"] = dict(candidate_map[code])
                stock["proposal_state"] = str(candidate_map[code].get("proposal_state") or "proposed")
                stock["proposal_reason"] = candidate_map[code].get("proposal_reason") or candidate_map[code].get("reason")
            elif not candidate_orders and (not scoped_codes or code in scoped_codes):
                stock["candidate_order"] = None
        portfolio_summary = structured_result.get("portfolio_summary")
        if isinstance(portfolio_summary, dict):
            portfolio_summary["candidate_order_count"] = len(candidate_orders)
        analysis_payload["candidate_orders"] = [dict(item) for item in candidate_orders]

    def _find_analysis_stock(self, analysis_payload: dict[str, Any], code: str) -> dict[str, Any]:
        structured_result = analysis_payload.get("structured_result") if isinstance(analysis_payload.get("structured_result"), dict) else {}
        stocks = structured_result.get("stocks") if isinstance(structured_result.get("stocks"), list) else []
        for item in stocks:
            if not isinstance(item, dict):
                continue
            if str(item.get("code") or "").strip() == str(code or "").strip():
                return item
        return {}

    @staticmethod
    def _extract_stock_execution_reason(stock: dict[str, Any]) -> str | None:
        raw = stock.get("raw") if isinstance(stock.get("raw"), dict) else {}
        execution = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}
        proposal_reason = str(execution.get("proposal_reason") or execution.get("reason") or "").strip()
        return proposal_reason or None

    def _explain_missing_candidate_reason(
        self,
        *,
        plan: ChatPlan,
        analysis_payload: dict[str, Any],
    ) -> tuple[str, str]:
        for code in plan.stock_codes:
            stock = self._find_analysis_stock(analysis_payload, code)
            if stock:
                reason = str(self._extract_stock_execution_reason(stock) or "no_candidate_orders")
                return reason, self._map_execution_gate_message(reason)
        return "no_candidate_orders", "条件未满足：当前没有可执行提议单，所以这轮不会自动下单。"

    @staticmethod
    def _map_execution_gate_message(reason: str) -> str:
        reason_map = {
            "insufficient_cash": "条件未满足：模拟盘现金不足，当前不能直接买入。",
            "insufficient_position": "条件未满足：当前可卖持仓不足，不能直接卖出。",
            "price_anomaly_skip": "条件未满足：执行阶段识别到异常波动，已拦截本轮自动下单。",
            "price_above_limit": "条件未满足：当前价格高于你的买入价格上限，这轮不会自动下单。",
            "price_below_limit": "条件未满足：当前价格低于你的卖出/触发价格下限，这轮不会自动下单。",
            "execution_skipped": "条件未满足：执行阶段主动放弃了这笔提议单。",
            "wait_signal": "条件未满足：当前信号仍偏观望，暂时没有形成可执行提议单。",
            "target_matched": "条件未满足：当前仓位已经匹配目标仓位，无需继续下单。",
            "side_mismatch": "条件未满足：当前提议单方向与请求不一致，无法直接执行。",
            "quantity_unavailable": "条件未满足：当前可执行数量无法满足你的精确股数要求，这轮不会自动下单。",
            "risk_not_low": "条件未满足：当前风控等级不是低风险，这轮不会自动下单。",
            "unsupported_condition": "条件未满足：当前请求里包含暂不支持自动执行的自由条件，系统只会保留分析结果。",
            "cash_below_floor": "条件未满足：执行后剩余现金会低于你的要求，这轮不会自动下单。",
            "position_pct_exceeded": "条件未满足：执行后单票仓位会超过你设定的上限，这轮不会自动下单。",
            "unsupported_execution_mode": "自动执行仅支持模拟盘，不会直接走真实券商下单。",
        }
        return reason_map.get(reason, "条件未满足：当前没有可执行提议单，所以这轮不会自动下单。")

    @staticmethod
    def _is_stock_hard_blocked(stock: dict[str, Any]) -> bool:
        if not isinstance(stock, dict):
            return False
        raw = stock.get("raw") if isinstance(stock.get("raw"), dict) else {}
        risk = raw.get("risk") if isinstance(raw.get("risk"), dict) else {}
        execution = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}
        if bool(risk.get("hard_risk_triggered")):
            return True
        warnings = {str(item).strip() for item in execution.get("warnings") or [] if str(item).strip()}
        if "price_anomaly_skip" in warnings:
            return True
        proposal_state = str(execution.get("proposal_state") or "").strip()
        return proposal_state == "blocked"

    def _serialize_analysis_result(self, run_result: AgentRunResult) -> dict[str, Any]:
        stocks = [self._serialize_stock_result(item) for item in run_result.results]
        candidate_orders = [
            item["candidate_order"]
            for item in stocks
            if isinstance(item.get("candidate_order"), dict)
        ]
        recommendations = [item for item in stocks if str(item.get("operation_advice") or "").strip()]
        positive = [item for item in recommendations if str(item.get("action") or "") == "buy"]
        negative = [item for item in recommendations if str(item.get("action") or "") == "sell"]
        portfolio_summary = {
            "run_id": run_result.run_id,
            "stock_count": len(stocks),
            "candidate_order_count": len(candidate_orders),
            "buy_count": len(positive),
            "sell_count": len(negative),
            "account_snapshot": run_result.account_snapshot,
        }
        warnings = [dict(item) for item in run_result.warnings if isinstance(item, dict)]
        stage_traces = [dict(item) for item in run_result.stage_traces if isinstance(item, dict)]
        decision_panel = [dict(item) for item in run_result.decision_panel if isinstance(item, dict)]
        planner_trace = [dict(item) for item in run_result.planner_trace if isinstance(item, dict)]
        condition_evaluations = [dict(item) for item in run_result.condition_evaluations if isinstance(item, dict)]
        return {
            "run_id": run_result.run_id,
            "structured_result": {
                "run_id": run_result.run_id,
                "trade_date": run_result.trade_date.isoformat(),
                "stocks": stocks,
                "portfolio_summary": portfolio_summary,
                "controller_plan": dict(run_result.controller_plan or {}),
                "portfolio_decision": dict(run_result.portfolio_decision or {}),
                "stage_traces": stage_traces,
                "warnings": warnings,
                "decision_panel": decision_panel,
                "planner_trace": planner_trace,
                "condition_evaluations": condition_evaluations,
                "termination_reason": run_result.termination_reason,
                "replan_count": int(run_result.replan_count or 0),
                "policy_snapshot": dict(run_result.policy_snapshot or {}),
            },
            "candidate_orders": candidate_orders,
            "execution_result": dict(run_result.execution_result or {}) if isinstance(run_result.execution_result, dict) else None,
        }

    def _extract_news_items_by_stock_for_mirror(self, analysis_result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        mapping: dict[str, list[dict[str, Any]]] = {}
        stocks = analysis_result.get("stocks") if isinstance(analysis_result.get("stocks"), list) else []

        for stock in stocks:
            if not isinstance(stock, dict):
                continue
            code = str(stock.get("code") or "").strip()
            if not code:
                continue

            raw = stock.get("raw") if isinstance(stock.get("raw"), dict) else {}
            signal = raw.get("signal") if isinstance(raw.get("signal"), dict) else {}
            ai_payload = signal.get("ai_payload") if isinstance(signal.get("ai_payload"), dict) else {}
            raw_items = ai_payload.get("news_items") if isinstance(ai_payload.get("news_items"), list) else []
            items = []
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                items.append(dict(item))

            if items:
                mapping[code] = items

        return mapping

    def _serialize_stock_result(self, item: StockAgentResult) -> dict[str, Any]:
        realtime_quote = item.data.realtime_quote if isinstance(item.data.realtime_quote, dict) else {}
        analysis_context = item.data.analysis_context if isinstance(item.data.analysis_context, dict) else {}
        today = analysis_context.get("today") if isinstance(analysis_context.get("today"), dict) else {}
        current_price = realtime_quote.get("price") or today.get("close")
        change_pct = realtime_quote.get("change_pct") or realtime_quote.get("pct_chg") or today.get("pct_chg")
        stock_name = str(
            realtime_quote.get("name")
            or analysis_context.get("stock_name")
            or item.code
        ).strip()
        candidate_order: dict[str, Any] | None = None
        proposed_order = item.execution.proposed_order if isinstance(item.execution.proposed_order, dict) else {}
        proposal_state = str(item.execution.proposal_state or "").strip()
        proposal_reason = str(item.execution.proposal_reason or item.execution.reason or "").strip()
        final_order = item.execution.final_order if isinstance(item.execution.final_order, dict) else {}
        original_order = item.execution.original_order if isinstance(item.execution.original_order, dict) else {}
        decision_source = "llm" if item.execution.llm_used else "rule"
        if proposed_order and proposal_state in {"proposed", "submitted", "executed"}:
            candidate_order = {
                "code": str(proposed_order.get("code") or item.code),
                "stock_name": stock_name,
                "action": str(proposed_order.get("action") or item.execution.action or ""),
                "quantity": int(proposed_order.get("quantity") or 0),
                "target_qty": int(proposed_order.get("target_qty") or item.execution.target_qty or 0),
                "price": float(proposed_order.get("price") or item.execution.fill_price or current_price or 0.0),
                "reason": proposal_reason,
                "proposal_state": proposal_state,
                "proposal_reason": proposal_reason,
                "current_price": float(current_price or 0.0),
                "risk_flags": list(item.risk.risk_flags or []),
                "source_run_id": item.execution.backend_task_id or "",
                "effective_market_source": item.data.data_source,
                "confidence": float(item.execution.confidence or item.risk.confidence or item.signal.confidence or item.data.confidence or 0.5),
                "adjustment_summary": item.execution.adjustment_reason,
                "decision_source": decision_source,
                "warnings": list(item.execution.warnings or []),
                "original_order": original_order or None,
                "final_order": final_order or None,
                "paper_submit_result": dict(item.execution.paper_submit_result or {}) if isinstance(item.execution.paper_submit_result, dict) else None,
            }
        return {
            "code": item.code,
            "name": stock_name,
            "current_price": float(current_price or 0.0),
            "change_pct": float(change_pct or 0.0),
            "data_state": item.data.state.value,
            "operation_advice": item.signal.operation_advice,
            "sentiment_score": int(item.signal.sentiment_score or 0),
            "trend_signal": item.signal.trend_signal,
            "stop_loss": item.risk.effective_stop_loss,
            "take_profit": item.risk.effective_take_profit,
            "target_weight": float(item.risk.target_weight or 0.0),
            "risk_flags": list(item.risk.risk_flags or []),
            "action": item.execution.action,
            "proposal_state": proposal_state,
            "proposal_reason": proposal_reason,
            "candidate_order": candidate_order,
            "warnings": list({*item.data.warnings, *item.signal.warnings, *item.risk.warnings, *item.execution.warnings}),
            "data_source": item.data.data_source,
            "stage_decisions": {
                "data": item.data.decision,
                "signal": item.signal.decision,
                "risk": item.risk.decision,
                "execution": item.execution.decision,
            },
            "stage_confidence": {
                "data": item.data.confidence,
                "signal": item.signal.confidence,
                "risk": item.risk.confidence,
                "execution": item.execution.confidence,
            },
            "planner_trace": [dict(entry) for entry in item.planner_trace if isinstance(entry, dict)],
            "condition_evaluations": [dict(entry) for entry in item.condition_evaluations if isinstance(entry, dict)],
            "termination_reason": item.termination_reason,
            "replan_count": int(item.replan_count or 0),
            "policy_snapshot": dict(item.policy_snapshot or {}),
            "execution_result": dict(item.execution_result or {}) if isinstance(item.execution_result, dict) else None,
            "raw": item.to_dict(),
        }

    async def _render_analysis_content(
        self,
        *,
        analysis_payload: dict[str, Any],
        runtime_context_payload: dict[str, Any] | None,
        history_payload: dict[str, Any] | None,
        backtest_payload: dict[str, Any] | None,
        original_message: str,
        llm_selection: ChatLlmSelection,
    ) -> str:
        structured_result = analysis_payload.get("structured_result") if isinstance(analysis_payload.get("structured_result"), dict) else {}
        candidate_orders = analysis_payload.get("candidate_orders") if isinstance(analysis_payload.get("candidate_orders"), list) else []
        fallback = self._render_analysis_template(
            structured_result=structured_result,
            candidate_orders=candidate_orders,
            runtime_context_payload=runtime_context_payload,
            history_payload=history_payload,
            backtest_payload=backtest_payload,
        )

        try:
            if not getattr(llm_selection.analyzer, "is_available", lambda: False)():
                return fallback
            prompt = self._build_analysis_summary_prompt(
                original_message=original_message,
                structured_result=structured_result,
                candidate_orders=candidate_orders,
                runtime_context_payload=runtime_context_payload,
                history_payload=history_payload,
                backtest_payload=backtest_payload,
            )
            content = await asyncio.to_thread(
                llm_selection.analyzer.generate_text,
                prompt,
                temperature=0.2,
                max_output_tokens=1800,
            )
            return self._normalize_analysis_text(
                content,
                fallback,
                has_candidate_orders=bool(candidate_orders),
                has_execution_result=False,
            )
        except Exception as exc:
            logger.warning("LLM analysis summary fallback: %s", redact_sensitive_text(str(exc)))
            return fallback

    async def _render_analysis_then_execute_content(
        self,
        *,
        analysis_payload: dict[str, Any],
        candidate_orders: list[dict[str, Any]],
        execution_result: dict[str, Any] | None,
        autonomous_execution: dict[str, Any],
        runtime_context_payload: dict[str, Any] | None,
        history_payload: dict[str, Any] | None,
        backtest_payload: dict[str, Any] | None,
        original_message: str,
        llm_selection: ChatLlmSelection,
    ) -> str:
        structured_result = analysis_payload.get("structured_result") if isinstance(analysis_payload.get("structured_result"), dict) else {}
        fallback = self._render_analysis_then_execute_template(
            structured_result=structured_result,
            candidate_orders=candidate_orders,
            execution_result=execution_result,
            autonomous_execution=autonomous_execution,
            runtime_context_payload=runtime_context_payload,
            history_payload=history_payload,
            backtest_payload=backtest_payload,
        )
        if self._is_outside_trading_session_execution(execution_result):
            return fallback

        try:
            if not getattr(llm_selection.analyzer, "is_available", lambda: False)():
                return fallback
            prompt = self._build_analysis_then_execute_prompt(
                original_message=original_message,
                structured_result=structured_result,
                candidate_orders=candidate_orders,
                execution_result=execution_result,
                autonomous_execution=autonomous_execution,
                runtime_context_payload=runtime_context_payload,
                history_payload=history_payload,
                backtest_payload=backtest_payload,
            )
            content = await asyncio.to_thread(
                llm_selection.analyzer.generate_text,
                prompt,
                temperature=0.2,
                max_output_tokens=2000,
            )
            return self._normalize_analysis_text(
                content,
                fallback,
                has_candidate_orders=bool(candidate_orders),
                has_execution_result=execution_result is not None,
            )
        except Exception as exc:
            logger.warning("LLM autonomous execution summary fallback: %s", redact_sensitive_text(str(exc)))
            return fallback

    @staticmethod
    def _build_analysis_summary_prompt(
        *,
        original_message: str,
        structured_result: dict[str, Any],
        candidate_orders: list[dict[str, Any]],
        runtime_context_payload: dict[str, Any] | None,
        history_payload: dict[str, Any] | None,
        backtest_payload: dict[str, Any] | None,
    ) -> str:
        return (
            "你是一个股票 Agent 助手，请根据以下结构化结果生成简洁中文 Markdown 回复。"
            "要求：1. 先给组合结论；2. 再逐只股票给建议；3. 如果存在候选订单，要明确说明这只是候选单，"
            "用户明确确认后才会下到模拟盘；4. 不要编造数据；5. 只输出自然语言，不要输出 JSON、键值对、代码块或伪代码。\n\n"
            f"用户问题：{original_message}\n\n"
            f"分析结果：{structured_result}\n\n"
            f"候选订单：{candidate_orders}\n\n"
            f"账户上下文：{runtime_context_payload}\n\n"
            f"历史分析：{history_payload}\n\n"
            f"回测摘要：{backtest_payload}\n"
        )

    @staticmethod
    def _build_analysis_then_execute_prompt(
        *,
        original_message: str,
        structured_result: dict[str, Any],
        candidate_orders: list[dict[str, Any]],
        execution_result: dict[str, Any] | None,
        autonomous_execution: dict[str, Any],
        runtime_context_payload: dict[str, Any] | None,
        history_payload: dict[str, Any] | None,
        backtest_payload: dict[str, Any] | None,
    ) -> str:
        return (
            "你是一个会自主决策的股票 Agent。用户已经明确授权你根据分析结果决定是否在模拟盘下单。"
            "请根据以下结构化结果输出简洁中文 Markdown。"
            "要求：1. 先给最终决策（执行/暂不执行）与理由；2. 再概括个股判断；3. 若已执行，说明执行了几笔、状态如何；"
            "4. 不要编造数据；5. 只输出自然语言，不要输出 JSON、键值对、代码块或伪代码。\n\n"
            f"用户问题：{original_message}\n\n"
            f"分析结果：{structured_result}\n\n"
            f"候选订单：{candidate_orders}\n\n"
            f"执行结果：{execution_result}\n\n"
            f"条件门槛判断：{autonomous_execution}\n\n"
            f"账户上下文：{runtime_context_payload}\n\n"
            f"历史分析：{history_payload}\n\n"
            f"回测摘要：{backtest_payload}\n"
        )

    @staticmethod
    def _normalize_analysis_text(
        content: Any,
        fallback: str,
        *,
        has_candidate_orders: bool,
        has_execution_result: bool,
    ) -> str:
        clean = str(content or "").strip()
        if not clean:
            return fallback

        if AgentChatService._looks_like_structured_output(clean):
            return fallback

        lowered = clean.lower()
        if not has_candidate_orders and any(token in clean for token in ("候选订单", "候选单", "等你确认即可执行", "已生成提议单")):
            return fallback
        if not has_execution_result and any(token in lowered for token in ("已下单", "已执行", "提交模拟盘", "执行了")):
            return fallback

        return clean

    @staticmethod
    def _looks_like_structured_output(content: str) -> bool:
        stripped = str(content or "").strip()
        if not stripped:
            return False

        if "```" in stripped:
            return True

        plain = AgentChatService._strip_code_fence(stripped)
        if plain.startswith("{") or plain.startswith("["):
            return True

        if plain.count('":') >= 3 or plain.count('\\"') >= 3:
            return True

        lines = [line.strip() for line in plain.splitlines() if line.strip()]
        structured_lines = sum(1 for line in lines if _STRUCTURED_OUTPUT_LINE_RE.match(line))
        return structured_lines >= 2

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        lines = str(content or "").strip().splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return str(content or "").strip()

    @staticmethod
    def _format_price(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "--"
        if number == 0:
            return "0.00"
        return f"{number:.2f}"

    @staticmethod
    def _format_percent(value: Any, *, multiplier: float = 1.0) -> str:
        try:
            number = float(value) * multiplier
        except (TypeError, ValueError):
            return "--"
        return f"{number:.2f}%"

    @staticmethod
    def _safe_number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number == number else None

    @staticmethod
    def _format_amount(value: Any) -> str:
        number = AgentChatService._safe_number(value)
        if number is None:
            return "--"
        return f"{number:,.2f}"

    def _extract_portfolio_health_stock_codes(
        self,
        *,
        account_state_payload: dict[str, Any],
        portfolio_health: dict[str, Any],
    ) -> list[str]:
        raw_positions = portfolio_health.get("positions") if isinstance(portfolio_health.get("positions"), list) else []
        if not raw_positions:
            account_state = account_state_payload.get("account_state") if isinstance(account_state_payload.get("account_state"), dict) else {}
            raw_positions = account_state.get("positions") if isinstance(account_state.get("positions"), list) else []
        codes: list[str] = []
        for item in raw_positions:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip()
            if code and code not in codes:
                codes.append(code)
        return codes

    @staticmethod
    def _build_portfolio_health_stage_memory(
        *,
        portfolio_health: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = portfolio_health.get("metrics") if isinstance(portfolio_health.get("metrics"), dict) else {}
        diagnostics = portfolio_health.get("diagnostics") if isinstance(portfolio_health.get("diagnostics"), dict) else {}
        alerts = diagnostics.get("alerts") if isinstance(diagnostics.get("alerts"), list) else []
        return {
            "portfolio_health": {
                "health_score": diagnostics.get("health_score"),
                "health_level": diagnostics.get("health_level"),
                "rebalancing_needed": diagnostics.get("rebalancing_needed"),
                "top_alerts": [
                    str(item.get("message") or "").strip()
                    for item in alerts[:3]
                    if isinstance(item, dict) and str(item.get("message") or "").strip()
                ],
                "metrics": {
                    "total_return_pct": metrics.get("total_return_pct"),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "top1_position_pct": metrics.get("top1_position_pct"),
                    "top3_position_pct": metrics.get("top3_position_pct"),
                },
            }
        }

    @staticmethod
    def _pick_position_industry_name(position: dict[str, Any]) -> str:
        for key in ("industry_name", "industry", "sector_name", "sector", "board_name", "board"):
            text = str(position.get(key) or "").strip()
            if text:
                return text
        return ""

    def _build_portfolio_industry_exposures(self, positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not positions:
            return []
        total_market_value = sum(self._safe_number(item.get("market_value")) or 0.0 for item in positions)
        if total_market_value <= 0:
            return []

        grouped: dict[str, dict[str, Any]] = {}
        unresolved_codes: dict[str, dict[str, Any]] = {}
        for item in positions:
            code = str(item.get("code") or "").strip()
            if not code:
                continue
            industry_name = self._pick_position_industry_name(item)
            if industry_name:
                bucket = grouped.setdefault(
                    industry_name,
                    {"industry_name": industry_name, "market_value": 0.0, "codes": [], "stock_count": 0},
                )
                bucket["market_value"] += self._safe_number(item.get("market_value")) or 0.0
                if code not in bucket["codes"]:
                    bucket["codes"].append(code)
                bucket["stock_count"] += 1
            else:
                unresolved_codes[code] = dict(item)

        if unresolved_codes:
            try:
                for board in self._load_board_catalog("industry_board"):
                    if not unresolved_codes:
                        break
                    board_name = str(board.get("board_name") or board.get("board_symbol") or "").strip()
                    board_symbol = str(board.get("board_symbol") or board_name).strip()
                    if not board_name or not board_symbol:
                        continue
                    members = self._load_board_constituents("industry_board", board_symbol)
                    member_codes = {
                        str(item.get("code") or "").strip()
                        for item in members
                        if isinstance(item, dict) and str(item.get("code") or "").strip()
                    }
                    matched_codes = [code for code in unresolved_codes if code in member_codes]
                    if not matched_codes:
                        continue
                    bucket = grouped.setdefault(
                        board_name,
                        {"industry_name": board_name, "market_value": 0.0, "codes": [], "stock_count": 0},
                    )
                    for code in matched_codes:
                        position = unresolved_codes.pop(code)
                        bucket["market_value"] += self._safe_number(position.get("market_value")) or 0.0
                        if code not in bucket["codes"]:
                            bucket["codes"].append(code)
                        bucket["stock_count"] += 1
            except Exception as exc:
                logger.warning("portfolio industry exposure fallback failed: %s", redact_sensitive_text(str(exc)))

        exposures = []
        for item in grouped.values():
            market_value = float(item.get("market_value") or 0.0)
            exposures.append(
                {
                    "industry_name": str(item.get("industry_name") or "").strip(),
                    "market_value": round(market_value, 4),
                    "weight_pct": round((market_value / total_market_value) * 100, 4) if total_market_value > 0 else None,
                    "stock_count": int(item.get("stock_count") or 0),
                    "codes": [str(code).strip() for code in item.get("codes") or [] if str(code).strip()],
                }
            )
        exposures.sort(key=lambda item: float(item.get("market_value") or 0.0), reverse=True)
        return exposures

    def _render_portfolio_health_content(
        self,
        *,
        portfolio_health: dict[str, Any],
        analysis_payload: dict[str, Any] | None,
        effective_preferences: dict[str, Any] | None,
    ) -> str:
        positions = [dict(item) for item in portfolio_health.get("positions") or [] if isinstance(item, dict)]
        metrics = portfolio_health.get("metrics") if isinstance(portfolio_health.get("metrics"), dict) else {}
        diagnostics = portfolio_health.get("diagnostics") if isinstance(portfolio_health.get("diagnostics"), dict) else {}
        alerts = [dict(item) for item in diagnostics.get("alerts") or [] if isinstance(item, dict)]
        suggestions = [str(item).strip() for item in diagnostics.get("suggestions") or [] if str(item).strip()]
        analysis_structured = analysis_payload.get("structured_result") if isinstance(analysis_payload, dict) and isinstance(analysis_payload.get("structured_result"), dict) else {}
        stocks = [dict(item) for item in analysis_structured.get("stocks") or [] if isinstance(item, dict)]
        candidate_orders = [dict(item) for item in analysis_payload.get("candidate_orders") or [] if isinstance(item, dict)] if isinstance(analysis_payload, dict) else []
        stock_map = {
            str(item.get("code") or "").strip(): item
            for item in stocks
            if str(item.get("code") or "").strip()
        }
        position_map = {
            str(item.get("code") or "").strip(): item
            for item in positions
            if str(item.get("code") or "").strip()
        }
        industry_exposures = self._build_portfolio_industry_exposures(positions)
        if not industry_exposures:
            backend_industry = portfolio_health.get("exposures") if isinstance(portfolio_health.get("exposures"), dict) else {}
            industry_exposures = [
                dict(item)
                for item in backend_industry.get("by_industry") or []
                if isinstance(item, dict)
            ]

        effective_root = effective_preferences if isinstance(effective_preferences, dict) else {}
        effective_trading = effective_root.get("effective") if isinstance(effective_root.get("effective"), dict) else {}
        trading_prefs = effective_trading.get("trading") if isinstance(effective_trading.get("trading"), dict) else {}
        position_limit_pct = self._safe_number(trading_prefs.get("positionMaxPct"))
        health_score = diagnostics.get("health_score")
        health_level = str(diagnostics.get("health_level") or "unknown").strip() or "unknown"
        health_label = {"healthy": "健康", "watch": "关注", "risky": "偏高风险"}.get(health_level, health_level)

        lines = [
            "## 组合结论",
            f"当前组合健康度约为 {health_score if health_score is not None else '--'}/100，整体处于“{health_label}”状态。"
            f" 当前总资产约 {self._format_amount(portfolio_health.get('total_asset'))}，"
            f"持仓市值约 {self._format_amount(portfolio_health.get('total_market_value'))}，"
            f"可用现金约 {self._format_amount(portfolio_health.get('available_cash'))}。",
            f"累计收益率 {self._format_percent(metrics.get('total_return_pct'))}，"
            f"最大回撤 {self._format_percent(metrics.get('max_drawdown_pct'))}，"
            f"夏普比率 {self._format_price(metrics.get('sharpe_ratio'))}。",
            f"当前共有 {int(metrics.get('position_count') or len(positions))} 项持仓，"
            f"单票最高仓位约 {self._format_percent(metrics.get('top1_position_pct'))}，"
            f"前三大持仓合计约 {self._format_percent(metrics.get('top3_position_pct'))}。",
        ]

        if not positions:
            lines.append("当前账户暂无持仓，所以这轮只能给出账户风险与现金状态，无法做逐仓位健康诊断。")
            return "\n".join(lines)

        if industry_exposures:
            lines.append("")
            lines.append("## 行业分布")
            for item in industry_exposures[:3]:
                lines.append(
                    f"- {item.get('industry_name') or '未分类'}：约占持仓 {self._format_percent(item.get('weight_pct'))}"
                    f"，覆盖 {item.get('stock_count') or 0} 只股票。"
                )

        if alerts:
            lines.append("")
            lines.append("## 主要预警")
            for item in alerts[:4]:
                lines.append(f"- {item.get('message') or '--'}")

        lines.append("")
        lines.append("## 持仓诊断")
        sorted_codes = sorted(
            position_map.keys(),
            key=lambda code: self._safe_number(position_map.get(code, {}).get("weight_pct")) or 0.0,
            reverse=True,
        )
        for code in sorted_codes:
            position = position_map.get(code, {})
            stock = stock_map.get(code, {})
            name = str(position.get("stock_name") or stock.get("name") or code).strip()
            lines.append(
                f"- {code} {name}：当前仓位约 {self._format_percent(position.get('weight_pct'))}，"
                f"浮盈亏 {self._format_amount(position.get('unrealized_pnl'))}，"
                f"浮动收益率 {self._format_percent(position.get('unrealized_return_pct'))}，"
                f"Agent 判断为 {stock.get('operation_advice') or '暂未生成新判断'}。"
            )

        rebalance_suggestions: list[str] = []
        if position_limit_pct is not None:
            overweight_positions = [
                code
                for code, position in position_map.items()
                if (self._safe_number(position.get("weight_pct")) or 0.0) > position_limit_pct
            ]
            for code in overweight_positions[:3]:
                position = position_map.get(code, {})
                stock = stock_map.get(code, {})
                rebalance_suggestions.append(
                    f"{code} {position.get('stock_name') or stock.get('name') or ''} 当前仓位已高于你设定的 {position_limit_pct:.0f}% 单票上限，建议优先评估减仓。"
                )
        if candidate_orders:
            for item in candidate_orders[:3]:
                rebalance_suggestions.append(
                    f"候选调整：{item.get('code') or '--'} {item.get('action') or '--'} "
                    f"{item.get('quantity') or '--'} 股，理由是 {item.get('reason') or item.get('proposal_reason') or '组合调仓建议'}。"
                )
        for suggestion in suggestions:
            if suggestion not in rebalance_suggestions:
                rebalance_suggestions.append(suggestion)

        if rebalance_suggestions:
            lines.append("")
            lines.append("## 调整建议")
            for item in rebalance_suggestions[:5]:
                lines.append(f"- {item}")

        return "\n".join(lines)

    @staticmethod
    def _build_strategy_backtest_clarification(error_code: str) -> str:
        normalized = str(error_code or "").strip()
        if normalized == "strategy_backtest_requires_single_stock":
            return (
                "策略回测当前一次只支持 1 只股票。请直接告诉我股票代码，"
                "或者先分析一只股票后再问“这只股票用 MACD 金叉策略过去一年收益怎样”。"
            )
        if normalized == "strategy_backtest_strategy_unrecognized":
            return (
                "我暂时没能把这条策略描述稳定解析成可执行回测模板。"
                "现在除了单一模板，也支持像“MACD 金叉且 RSI<30，跌破 5 日线止损”这样的组合规则；"
                "如果这次没识别出来，你可以把买入条件和止损/止盈条件说得更直白一些。"
            )
        return "这条策略回测请求还缺少关键信息，请补充股票代码或更明确的策略规则。"

    def _interpret_strategy_backtest_result(
        self,
        backtest_result: dict[str, Any],
        *,
        runtime_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        items = [dict(item) for item in backtest_result.get("items") or [] if isinstance(item, dict)]
        if not items:
            return {"items": []}

        runtime_llm = self._extract_runtime_llm(runtime_config if isinstance(runtime_config, dict) else {})
        payload: dict[str, Any] = {
            "language": "zh-CN",
            "items": [
                {
                    "item_key": f"strategy-run-{item.get('run_id') or index}",
                    "item_type": "strategy",
                    "label": str(item.get("strategy_name") or item.get("template_name") or item.get("strategy_code") or f"策略 {index + 1}"),
                    "code": str(backtest_result.get("code") or ""),
                    "requested_range": dict(backtest_result.get("requested_range") or {}),
                    "effective_range": dict(backtest_result.get("effective_range") or {}),
                    "metrics": dict(item.get("metrics") or {}),
                    "benchmark": dict(item.get("benchmark") or {}),
                    "context": {
                        "strategy_code": item.get("strategy_code"),
                        "template_code": item.get("template_code"),
                        "params": dict(item.get("params") or {}),
                    },
                }
                for index, item in enumerate(items)
            ],
        }
        if runtime_llm is not None:
            payload["runtime_llm"] = {
                "provider": runtime_llm.provider,
                "base_url": runtime_llm.base_url,
                "model": runtime_llm.model,
                "has_token": runtime_llm.has_token,
                **({"api_token": runtime_llm.api_token} if runtime_llm.api_token else {}),
            }
        return self.backtest_interpretation_service.interpret(payload)

    def _build_strategy_backtest_stage_memory(
        self,
        *,
        backtest_request: dict[str, Any],
        backtest_result: dict[str, Any],
    ) -> dict[str, Any]:
        items = [dict(item) for item in backtest_result.get("items") or [] if isinstance(item, dict)]
        return {
            "backtest": {
                "mode": "strategy_run",
                "code": backtest_request.get("code"),
                "window": {
                    "start_date": backtest_request.get("start_date"),
                    "end_date": backtest_request.get("end_date"),
                },
                "strategy_names": [
                    str(item.get("strategy_name") or item.get("template_name") or item.get("strategy_code") or "").strip()
                    for item in items
                    if str(item.get("strategy_name") or item.get("template_name") or item.get("strategy_code") or "").strip()
                ],
                "run_group_id": backtest_result.get("run_group_id"),
            }
        }

    def _render_strategy_backtest_content(
        self,
        *,
        backtest_request: dict[str, Any],
        backtest_result: dict[str, Any],
        interpretation_payload: dict[str, Any] | None,
    ) -> str:
        items = [dict(item) for item in backtest_result.get("items") or [] if isinstance(item, dict)]
        interpretation_items = [
            dict(item) for item in (interpretation_payload or {}).get("items") or [] if isinstance(item, dict)
        ]
        interpretation_map = {
            str(item.get("item_key") or "").strip(): item
            for item in interpretation_items
            if str(item.get("item_key") or "").strip()
        }
        requested_range = backtest_result.get("requested_range") if isinstance(backtest_result.get("requested_range"), dict) else {}
        effective_range = backtest_result.get("effective_range") if isinstance(backtest_result.get("effective_range"), dict) else {}
        code = str(backtest_result.get("code") or backtest_request.get("code") or "--").strip() or "--"
        lines = [
            "## 策略回测结论",
            f"我已对 {code} 在 {requested_range.get('start_date') or backtest_request.get('start_date') or '--'} 到 "
            f"{requested_range.get('end_date') or backtest_request.get('end_date') or '--'} 这段区间完成 "
            f"{len(items)} 个策略回测。",
        ]
        effective_start = str(effective_range.get("start_date") or "").strip()
        effective_end = str(effective_range.get("end_date") or "").strip()
        if effective_start or effective_end:
            lines.append(f"实际可用行情区间是 {effective_start or '--'} 到 {effective_end or '--'}。")

        if not items:
            lines.append("这轮没有返回有效的策略结果，建议稍后重试，或缩小回测区间再看一次。")
            return "\n".join(lines)

        lines.append("")
        lines.append("## 策略表现")
        best_return_item = max(
            items,
            key=lambda item: self._safe_number((item.get("metrics") or {}).get("total_return_pct")) or float("-inf"),
        )
        best_drawdown_item = max(
            items,
            key=lambda item: -(abs(self._safe_number((item.get("metrics") or {}).get("max_drawdown_pct")) or float("inf"))),
        )
        for index, item in enumerate(items):
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            label = str(item.get("strategy_name") or item.get("template_name") or item.get("strategy_code") or f"策略 {index + 1}").strip()
            item_key = f"strategy-run-{item.get('run_id') or index}"
            interpretation = interpretation_map.get(item_key, {})
            total_trades = int(metrics.get("total_trades") or 0)
            max_drawdown = self._safe_number(metrics.get("max_drawdown_pct"))
            lines.append(f"### {label}")
            if total_trades <= 0:
                lines.append(
                    f"这段区间没有形成完整成交，累计收益 {self._format_percent(metrics.get('total_return_pct'))}，"
                    f"最大回撤 {self._format_percent(abs(max_drawdown) if max_drawdown is not None else None)}。"
                )
            else:
                lines.append(
                    f"累计收益 {self._format_percent(metrics.get('total_return_pct'))}，"
                    f"跑赢基准 {self._format_percent(metrics.get('excess_return_pct'))}，"
                    f"最大回撤 {self._format_percent(abs(max_drawdown) if max_drawdown is not None else None)}，"
                    f"夏普比率 {self._format_price(metrics.get('sharpe_ratio'))}，"
                    f"共完成 {total_trades} 笔交易。"
                )
            if interpretation:
                verdict = str(interpretation.get("verdict") or "").strip()
                summary = str(interpretation.get("summary") or "").strip()
                if verdict:
                    lines.append(f"AI 解读：{verdict}。{summary}")
                elif summary:
                    lines.append(f"AI 解读：{summary}")
            lines.append("")

        if len(items) > 1:
            best_return_label = str(
                best_return_item.get("strategy_name") or best_return_item.get("template_name") or best_return_item.get("strategy_code") or "--"
            ).strip() or "--"
            best_drawdown_label = str(
                best_drawdown_item.get("strategy_name") or best_drawdown_item.get("template_name") or best_drawdown_item.get("strategy_code") or "--"
            ).strip() or "--"
            lines.append("## 策略对比")
            lines.append(
                f"按累计收益看，{best_return_label} 这组策略表现最好；"
                f"按回撤控制看，{best_drawdown_label} 相对更稳。"
            )

        return "\n".join(lines).strip()

    def _render_analysis_template(
        self,
        *,
        structured_result: dict[str, Any],
        candidate_orders: list[dict[str, Any]],
        runtime_context_payload: dict[str, Any] | None,
        history_payload: dict[str, Any] | None,
        backtest_payload: dict[str, Any] | None,
    ) -> str:
        stocks = structured_result.get("stocks") if isinstance(structured_result.get("stocks"), list) else []
        portfolio_summary = structured_result.get("portfolio_summary") if isinstance(structured_result.get("portfolio_summary"), dict) else {}
        stock_count = portfolio_summary.get("stock_count") or len(stocks)
        candidate_count = portfolio_summary.get("candidate_order_count") or len(candidate_orders)
        lines = [
            "## 综合判断",
            f"这次一共分析了 {stock_count} 只股票，当前形成 {candidate_count} 笔可执行提议单。",
        ]
        if candidate_orders:
            codes = "、".join(str(item.get("code") or "") for item in candidate_orders)
            lines.append(f"提议单涉及 {codes}，这些都还只是建议，只有你明确确认后才会提交到模拟盘。")
        else:
            lines.append("这轮分析暂时没有形成需要立即执行的提议单，更适合先观察。")

        if stocks:
            lead_stock = stocks[0]
            lines.append(
                f"从当前结果看，{lead_stock.get('code')} {lead_stock.get('name') or ''} "
                f"更值得优先关注，整体倾向 {lead_stock.get('operation_advice') or '观望'}。"
            )

        lines.append("")
        lines.append("## 个股说明")
        for stock in stocks:
            action = stock.get("operation_advice") or "观望"
            current_price = self._format_price(stock.get("current_price"))
            change_pct = self._format_percent(stock.get("change_pct"))
            target_weight = self._format_percent(stock.get("target_weight"), multiplier=100)
            lines.append(
                f"### {stock.get('code')} {stock.get('name')}"
            )
            lines.append(
                f"当前价格大约 {current_price}，涨跌幅 {change_pct}。"
                f"综合情绪分为 {stock.get('sentiment_score') or '--'}，当前更偏向 {action}，建议仓位大约 {target_weight}。"
            )
            if stock.get("risk_flags"):
                lines.append(f"风险提示：{'、'.join(stock.get('risk_flags') or [])}。")
            candidate = stock.get("candidate_order")
            if isinstance(candidate, dict):
                lines.append(
                    f"如果你想执行这只股票的提议单，可以考虑以 {candidate.get('price') or '--'} 的参考价"
                    f"{candidate.get('action') or '--'} {candidate.get('quantity') or '--'} 股。"
                )
            lines.append("")

        if isinstance(runtime_context_payload, dict):
            runtime_context = runtime_context_payload.get("runtime_context")
            if isinstance(runtime_context, dict):
                summary = runtime_context.get("summary") if isinstance(runtime_context.get("summary"), dict) else {}
                positions = runtime_context.get("positions") if isinstance(runtime_context.get("positions"), list) else []
                lines.append(
                    "## 账户情况"
                )
                lines.append(
                    f"当前模拟盘现金约为 {summary.get('cash') or summary.get('available_cash') or '--'}，"
                    f"总资产约为 {summary.get('total_asset') or summary.get('total_equity') or '--'}，"
                    f"现有持仓 {len(positions)} 项。"
                )

        if isinstance(history_payload, dict):
            items = history_payload.get("items") if isinstance(history_payload.get("items"), list) else []
            if items:
                lines.append("")
                lines.append("## 历史分析")
                for item in items[:3]:
                    lines.append(
                        f"{item.get('stock_code')} {item.get('stock_name') or ''} 最近一次记录显示 "
                        f"{item.get('operation_advice') or item.get('status') or '--'}，时间是 {item.get('created_at') or '--'}。"
                    )

        if isinstance(backtest_payload, dict):
            items = backtest_payload.get("items") if isinstance(backtest_payload.get("items"), list) else []
            if items:
                lines.append("")
                lines.append("## 回测摘要")
                for item in items[:3]:
                    lines.append(f"{item.get('code')} 的回测摘要是：{item.get('summary') or item.get('label') or '--'}。")

        if candidate_orders:
            lines.append("")
            lines.append("## 下一步")
            lines.append("如果你确认要执行提议单，可以直接说“确认”“下 600519 的单”或“把刚才那几笔都下了”。")
        return "\n".join(lines)

    def _render_analysis_then_execute_template(
        self,
        *,
        structured_result: dict[str, Any],
        candidate_orders: list[dict[str, Any]],
        execution_result: dict[str, Any] | None,
        autonomous_execution: dict[str, Any],
        runtime_context_payload: dict[str, Any] | None,
        history_payload: dict[str, Any] | None,
        backtest_payload: dict[str, Any] | None,
    ) -> str:
        stocks = structured_result.get("stocks") if isinstance(structured_result.get("stocks"), list) else []
        portfolio_summary = structured_result.get("portfolio_summary") if isinstance(structured_result.get("portfolio_summary"), dict) else {}
        stock_count = portfolio_summary.get("stock_count") or len(stocks)
        candidate_count = portfolio_summary.get("candidate_order_count") or len(candidate_orders)

        lines = [
            "## 自主决策结论",
            f"我先完成了 {stock_count} 只股票的分析，本轮共形成 {candidate_count} 笔可执行提议单。",
        ]
        if not candidate_orders:
            gate_message = str(autonomous_execution.get("gate_message") or "").strip()
            lines.append(gate_message or "结合这轮结果，我判断暂时不需要下模拟盘单，因为当前没有生成满足条件的可执行提议单。")
        elif execution_result is None:
            gate_message = str(autonomous_execution.get("gate_message") or "").strip()
            lines.append(gate_message or "这轮分析形成了提议单，但当前没有可用的执行结果。")
        elif self._is_outside_trading_session_execution(execution_result):
            gate_message = self._extract_execution_message(execution_result) or str(autonomous_execution.get("gate_message") or "").strip()
            lines.append(gate_message or "当前处于非交易时段，本轮未执行模拟盘订单，候选单已保留。")
        else:
            executed_count = int(execution_result.get("executed_count") or 0)
            failed_count = len(execution_result.get("failed_orders") or [])
            if executed_count > 0:
                lines.append(f"我已根据分析结果执行提议单，成功提交 {executed_count} 笔。")
            else:
                lines.append("我原本判断可以尝试执行，但提交模拟盘时未成功落单。")
            if failed_count > 0:
                lines.append(f"其中有 {failed_count} 笔执行失败，详见下方执行结果。")

        if stocks:
            lines.append("")
            lines.append("## 个股判断")
            for stock in stocks:
                current_price = self._format_price(stock.get("current_price"))
                change_pct = self._format_percent(stock.get("change_pct"))
                target_weight = self._format_percent(stock.get("target_weight"), multiplier=100)
                lines.append(f"### {stock.get('code')} {stock.get('name')}")
                lines.append(
                    f"当前价格约 {current_price}，涨跌幅 {change_pct}，情绪分 {stock.get('sentiment_score') or '--'}。"
                    f"综合结论偏向 {stock.get('operation_advice') or '观望'}，建议仓位约 {target_weight}。"
                )
                if stock.get("risk_flags"):
                    lines.append(f"风险提示：{'、'.join(stock.get('risk_flags') or [])}。")
                lines.append("")

        if execution_result is not None:
            lines.append("## 执行结果")
            lines.append(self._render_execution_content(execution_result).replace("## 模拟盘执行结果\n", "").strip())

        if isinstance(runtime_context_payload, dict):
            runtime_context = runtime_context_payload.get("runtime_context")
            if isinstance(runtime_context, dict):
                summary = runtime_context.get("summary") if isinstance(runtime_context.get("summary"), dict) else {}
                positions = runtime_context.get("positions") if isinstance(runtime_context.get("positions"), list) else []
                lines.append("")
                lines.append("## 账户情况")
                lines.append(
                    f"当前模拟盘现金约为 {summary.get('cash') or summary.get('available_cash') or '--'}，"
                    f"总资产约为 {summary.get('total_asset') or summary.get('total_equity') or '--'}，"
                    f"现有持仓 {len(positions)} 项。"
                )

        if isinstance(history_payload, dict):
            items = history_payload.get("items") if isinstance(history_payload.get("items"), list) else []
            if items:
                lines.append("")
                lines.append("## 历史分析")
                for item in items[:3]:
                    lines.append(
                        f"{item.get('stock_code')} {item.get('stock_name') or ''} 最近一次建议为 "
                        f"{item.get('operation_advice') or item.get('status') or '--'}。"
                    )

        if isinstance(backtest_payload, dict):
            items = backtest_payload.get("items") if isinstance(backtest_payload.get("items"), list) else []
            if items:
                lines.append("")
                lines.append("## 回测摘要")
                for item in items[:3]:
                    lines.append(f"{item.get('code')} 的回测摘要是：{item.get('summary') or item.get('label') or '--'}。")

        return "\n".join(lines)

    def _render_context_only_content(
        self,
        *,
        account_state_payload: dict[str, Any] | None,
        runtime_context_payload: dict[str, Any] | None,
        history_payload: dict[str, Any] | None,
        backtest_payload: dict[str, Any] | None,
    ) -> str:
        lines = ["## 当前结果"]
        account_state = account_state_payload.get("account_state") if isinstance(account_state_payload, dict) and isinstance(account_state_payload.get("account_state"), dict) else (
            account_state_payload if isinstance(account_state_payload, dict) else {}
        )
        if isinstance(account_state, dict) and account_state:
            positions = account_state.get("positions") if isinstance(account_state.get("positions"), list) else []
            lines.append(
                f"当前可用资金约为 {account_state.get('available_cash') or '--'}，"
                f"总资产约为 {account_state.get('total_asset') or '--'}，"
                f"持仓市值约为 {account_state.get('total_market_value') or '--'}。"
            )
            lines.append(f"目前一共持有 {len(positions)} 项仓位。")
            lines.append(
                f"今日委托 {account_state.get('today_order_count') or 0} 次，"
                f"今日成交 {account_state.get('today_trade_count') or 0} 次。"
            )
        elif isinstance(runtime_context_payload, dict):
            summary = runtime_context_payload.get("summary") if isinstance(runtime_context_payload.get("summary"), dict) else {}
            positions = runtime_context_payload.get("positions") if isinstance(runtime_context_payload.get("positions"), list) else []
            lines.append(
                f"当前模拟盘现金约为 {summary.get('cash') or summary.get('available_cash') or '--'}，"
                f"总资产约为 {summary.get('total_asset') or summary.get('total_equity') or '--'}。"
            )
            lines.append(f"目前一共持有 {len(positions)} 项仓位。")
        if isinstance(history_payload, dict):
            items = history_payload.get("items") if isinstance(history_payload.get("items"), list) else []
            if items:
                lines.append("")
                lines.append("## 历史分析")
                for item in items[:5]:
                    lines.append(
                        f"{item.get('stock_code')} {item.get('stock_name') or ''} 最近一次建议为 "
                        f"{item.get('operation_advice') or item.get('status') or '--'}。"
                    )
        if isinstance(backtest_payload, dict):
            items = backtest_payload.get("items") if isinstance(backtest_payload.get("items"), list) else []
            if items:
                lines.append("")
                lines.append("## 回测摘要")
                for item in items[:5]:
                    lines.append(f"{item.get('code')} 的回测结果摘要是 {item.get('summary') or item.get('label') or '--'}。")
        return "\n".join(lines)

    @staticmethod
    def _render_execution_content(execution_result: dict[str, Any]) -> str:
        if AgentChatService._is_outside_trading_session_execution(execution_result):
            message = AgentChatService._extract_execution_message(execution_result) or "当前处于非交易时段，本轮未执行模拟盘订单，候选单已保留。"
            session_guard = AgentChatService._extract_execution_session_guard(execution_result)
            session_text = "、".join(str(item).strip() for item in session_guard.get("sessions") or [] if str(item).strip())
            timezone = str(session_guard.get("timezone") or "").strip()
            lines = ["## 模拟盘执行结果", message]
            if str(execution_result.get("mode") or "").strip().lower() == "batch":
                blocked_orders = execution_result.get("blocked_orders") if isinstance(execution_result.get("blocked_orders"), list) else []
                blocked_items = blocked_orders or (
                    execution_result.get("failed_orders")
                    if isinstance(execution_result.get("failed_orders"), list)
                    else []
                )
                lines.append(f"本轮共 {len(blocked_items) or execution_result.get('candidate_order_count') or 0} 笔提议单因非交易时段未执行。")
                if blocked_items:
                    lines.append("### 已保留候选单")
                    for item in blocked_items:
                        if not isinstance(item, dict):
                            continue
                        candidate_order = item.get("candidate_order") if isinstance(item.get("candidate_order"), dict) else {}
                        code = candidate_order.get("code") or "--"
                        action = candidate_order.get("action") or "--"
                        quantity = candidate_order.get("quantity") or "--"
                        lines.append(f"- {code}：{action} {quantity} 股")
            else:
                candidate_order = execution_result.get("candidate_order") if isinstance(execution_result.get("candidate_order"), dict) else {}
                if not candidate_order:
                    for item in AgentChatService._iter_nested_execution_items(execution_result):
                        nested_candidate = item.get("candidate_order")
                        if isinstance(nested_candidate, dict):
                            candidate_order = nested_candidate
                            break
                code = candidate_order.get("code") or "--"
                action = candidate_order.get("action") or "--"
                quantity = candidate_order.get("quantity") or "--"
                lines.append(f"涉及候选单：{code} {action} {quantity} 股。")
            if session_text or timezone:
                lines.append(f"交易时段：{session_text or '--'}（{timezone or '--'}）。")
            return "\n".join(lines)

        if str(execution_result.get("mode") or "").strip().lower() == "batch":
            orders = execution_result.get("orders") if isinstance(execution_result.get("orders"), list) else []
            blocked_orders = execution_result.get("blocked_orders") if isinstance(execution_result.get("blocked_orders"), list) else []
            failed_orders = execution_result.get("failed_orders") if isinstance(execution_result.get("failed_orders"), list) else []
            lines = [
                "## 模拟盘执行结果",
                f"本轮共尝试执行 {execution_result.get('candidate_order_count') or len(orders) + len(blocked_orders) + len(failed_orders)} 笔提议单。",
            ]
            if orders:
                lines.append("### 已提交/成交")
                for item in orders:
                    candidate_order = item.get("candidate_order") if isinstance(item, dict) and isinstance(item.get("candidate_order"), dict) else {}
                    provider_order = item.get("order") if isinstance(item, dict) and isinstance(item.get("order"), dict) else {}
                    code = candidate_order.get("code") or provider_order.get("stock_code") or "--"
                    action = candidate_order.get("action") or provider_order.get("direction") or "--"
                    quantity = candidate_order.get("quantity") or provider_order.get("quantity") or "--"
                    status = item.get("status") or provider_order.get("provider_status") or provider_order.get("status") or "unknown"
                    lines.append(f"- {code}：{action} {quantity} 股，状态 {status}")
            if failed_orders:
                lines.append("### 执行失败")
                for item in failed_orders:
                    candidate_order = item.get("candidate_order") if isinstance(item, dict) and isinstance(item.get("candidate_order"), dict) else {}
                    code = candidate_order.get("code") or "--"
                    action = candidate_order.get("action") or "--"
                    quantity = candidate_order.get("quantity") or "--"
                    message = item.get("message") or "unknown"
                    lines.append(f"- {code}：{action} {quantity} 股，失败原因 {message}")
            if blocked_orders:
                lines.append("### 未执行")
                for item in blocked_orders:
                    if not isinstance(item, dict):
                        continue
                    candidate_order = item.get("candidate_order") if isinstance(item.get("candidate_order"), dict) else {}
                    code = candidate_order.get("code") or "--"
                    action = candidate_order.get("action") or "--"
                    quantity = candidate_order.get("quantity") or "--"
                    message = str(item.get("message") or "当前处于非交易时段").strip()
                    lines.append(f"- {code}：{action} {quantity} 股，未执行原因 {message}")
            return "\n".join(lines)

        candidate_order = execution_result.get("candidate_order") if isinstance(execution_result.get("candidate_order"), dict) else {}
        provider_order = execution_result.get("order") if isinstance(execution_result.get("order"), dict) else {}
        code = candidate_order.get("code") or provider_order.get("stock_code") or "--"
        action = candidate_order.get("action") or provider_order.get("direction") or "--"
        quantity = candidate_order.get("quantity") or provider_order.get("quantity") or "--"
        status = execution_result.get("status") or provider_order.get("provider_status") or provider_order.get("status") or "unknown"
        return (
            "## 模拟盘执行结果\n"
            f"已尝试对 {code} 执行 {action} 操作，数量为 {quantity} 股。\n\n"
            f"当前返回状态为 {status}。"
        )

    @staticmethod
    def _resolve_execution_status(execution_result: dict[str, Any] | None) -> str:
        if not execution_result:
            return "analysis_only"
        if AgentChatService._is_outside_trading_session_execution(execution_result):
            return "blocked"
        status = str(execution_result.get("status") or "").strip().lower()
        if status == "filled":
            return "simulation_order_filled"
        if status in {"submitted", "partial_filled", "partial"}:
            return "simulation_order_submitted"
        if AgentChatService._count_successful_execution_results(execution_result) > 0:
            return "simulation_order_submitted"
        return "blocked"


_AGENT_CHAT_SERVICE: AgentChatService | None = None
_AGENT_CHAT_SERVICE_LOCK = threading.Lock()


def get_agent_chat_service(
    *,
    config: Config | None = None,
    db_manager: DatabaseManager | None = None,
) -> AgentChatService:
    """返回聊天服务单例。"""
    global _AGENT_CHAT_SERVICE
    if _AGENT_CHAT_SERVICE is not None:
        return _AGENT_CHAT_SERVICE
    with _AGENT_CHAT_SERVICE_LOCK:
        if _AGENT_CHAT_SERVICE is None:
            _AGENT_CHAT_SERVICE = AgentChatService(config=config, db_manager=db_manager)
    return _AGENT_CHAT_SERVICE


def reset_agent_chat_service() -> None:
    """重置聊天服务单例。"""
    global _AGENT_CHAT_SERVICE
    with _AGENT_CHAT_SERVICE_LOCK:
        _AGENT_CHAT_SERVICE = None
