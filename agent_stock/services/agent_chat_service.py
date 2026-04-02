# -*- coding: utf-8 -*-
"""Agent 问股聊天服务。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from data_provider.base import canonical_stock_code

from agent_stock.agents.contracts import AgentRunResult, StockAgentResult
from agent_stock.analyzer import get_analyzer
from agent_stock.config import Config, get_config, redact_sensitive_text
from agent_stock.repositories.chat_repo import AgentChatRepository
from agent_stock.services.agent_service import AgentService
from agent_stock.services.backend_agent_chat_client import BackendAgentChatClient
from agent_stock.storage import DatabaseManager

logger = logging.getLogger(__name__)

ChatEventHandler = Callable[[str, dict[str, Any]], Awaitable[None] | None]

_A_SHARE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_HK_SHARE_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")
_US_SHARE_RE = re.compile(r"\b([A-Z]{2,5})\b")

_ANALYSIS_KEYWORDS = ("分析", "行情", "走势", "组合", "建议", "研判", "判断")
_ANALYSIS_VIEW_KEYWORDS = ("看看", "看下", "看一下")
_ORDER_KEYWORDS = ("下单", "执行", "成交", "买入", "卖出", "买进", "卖掉", "模拟盘")
_HISTORY_KEYWORDS = ("历史分析", "分析记录", "最近分析", "上次分析", "之前分析")
_BACKTEST_KEYWORDS = ("回测", "胜率", "收益", "策略表现")
_ACCOUNT_KEYWORDS = ("持仓", "账户", "仓位", "资金", "模拟盘情况", "现金")
_SAVE_KEYWORDS = ("保存本轮分析", "保存分析", "保存这轮分析")
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
        analyzer=None,
    ) -> None:
        self.config = config or get_config()
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = chat_repo or AgentChatRepository(self.db)
        self.agent_service = agent_service or AgentService(config=self.config, db_manager=self.db)
        self.backend_client = backend_client or BackendAgentChatClient(config=self.config)
        self.analyzer = analyzer or get_analyzer()
        self.tools = {
            "get_runtime_account_context": self._tool_get_runtime_account_context,
            "get_analysis_history": self._tool_get_analysis_history,
            "get_backtest_summary": self._tool_get_backtest_summary,
            "run_multi_stock_analysis": self._tool_run_multi_stock_analysis,
            "place_simulated_order": self._tool_place_simulated_order,
            "batch_execute_candidate_orders": self._tool_batch_execute_candidate_orders,
        }

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
        frontend_context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        runtime_config = payload.get("runtime_config") if isinstance(payload.get("runtime_config"), dict) else {}

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

        try:
            await self._emit(event_handler, "thinking", {"message": "正在理解你的问题"})
            recent_assistant_messages = self.repo.list_recent_assistant_messages(owner_user_id, session_id, limit=5)
            latest_assistant_message = recent_assistant_messages[0] if recent_assistant_messages else None
            latest_assistant_meta = latest_assistant_message.get("meta") if latest_assistant_message else {}
            plan = await asyncio.to_thread(
                self._build_plan,
                message=message,
                frontend_context=frontend_context,
                latest_assistant_meta=latest_assistant_meta if isinstance(latest_assistant_meta, dict) else {},
                recent_assistant_messages=recent_assistant_messages,
            )

            if plan.primary_intent == "clarify":
                content = plan.clarification or "请告诉我你想分析哪只股票，或者明确说明要执行哪一笔候选订单。"
                final_payload = {
                    "session_id": session_id,
                    "content": content,
                    "structured_result": {"intent": "clarify", "message": content},
                    "candidate_orders": [],
                    "execution_result": None,
                    "status": "blocked",
                }
                self.repo.add_message(
                    owner_user_id=owner_user_id,
                    session_id=session_id,
                    role="assistant",
                    content=content,
                    meta=final_payload,
                )
                return final_payload

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
                self.repo.add_message(
                    owner_user_id=owner_user_id,
                    session_id=session_id,
                    role="assistant",
                    content=content,
                    meta=final_payload,
                )
                return final_payload

            tool_context: dict[str, Any] = {
                "owner_user_id": owner_user_id,
                "username": username,
                "session_id": session_id,
                "runtime_config": runtime_config,
                "frontend_context": frontend_context,
            }

            runtime_context_payload: dict[str, Any] | None = None
            history_payload: dict[str, Any] | None = None
            backtest_payload: dict[str, Any] | None = None

            if plan.include_runtime_context:
                runtime_context_payload = await self._run_tool(
                    "get_runtime_account_context",
                    {"owner_user_id": owner_user_id, "refresh": True},
                    tool_context,
                    event_handler,
                )
            if plan.include_history:
                history_payload = await self._run_tool(
                    "get_analysis_history",
                    {"owner_user_id": owner_user_id, "stock_codes": plan.stock_codes, "limit": 5},
                    tool_context,
                    event_handler,
                )
            if plan.include_backtest:
                backtest_payload = await self._run_tool(
                    "get_backtest_summary",
                    {"owner_user_id": owner_user_id, "stock_codes": plan.stock_codes, "limit": 6},
                    tool_context,
                    event_handler,
                )

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
                content = self._render_execution_content(execution_result)
                final_payload = {
                    "session_id": session_id,
                    "content": content,
                    "structured_result": {
                        "intent": "order_followup_single",
                        "candidate_order": plan.target_candidate_order,
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
                content = self._render_execution_content(execution_result)
                final_payload = {
                    "session_id": session_id,
                    "content": content,
                    "structured_result": {
                        "intent": "order_followup_all",
                        "candidate_orders": plan.target_candidate_orders,
                    },
                    "candidate_orders": list(plan.target_candidate_orders),
                    "execution_result": execution_result,
                    "status": self._resolve_execution_status(execution_result),
                }
            elif plan.primary_intent in {"history", "backtest", "account"}:
                content = self._render_context_only_content(
                    runtime_context_payload=runtime_context_payload,
                    history_payload=history_payload,
                    backtest_payload=backtest_payload,
                )
                final_payload = {
                    "session_id": session_id,
                    "content": content,
                    "structured_result": {
                        "intent": plan.primary_intent,
                        "runtime_context": runtime_context_payload,
                        "history": history_payload,
                        "backtest": backtest_payload,
                    },
                    "candidate_orders": [],
                    "execution_result": None,
                    "status": "analysis_only",
                }
            else:
                analysis_payload = await self._run_tool(
                    "run_multi_stock_analysis",
                    {
                        "stock_codes": plan.stock_codes,
                        "runtime_context_payload": runtime_context_payload,
                        "runtime_config": runtime_config,
                    },
                    tool_context,
                    event_handler,
                )
                candidate_orders = list(analysis_payload.get("candidate_orders") or [])
                if plan.primary_intent == "analysis_then_execute":
                    autonomous_execution = {
                        "requested": True,
                        "authorized": plan.autonomous_execution_authorized,
                        "execution_scope": "all",
                        "candidate_order_count": len(candidate_orders),
                        "executed": False,
                        "executed_count": 0,
                        "failed_count": 0,
                        "reason": "no_candidate_orders",
                    }
                    execution_result: dict[str, Any] | None = None
                    if candidate_orders:
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
                        executed_count = int(execution_result.get("executed_count") or 0)
                        failed_count = len(execution_result.get("failed_orders") or [])
                        autonomous_execution.update(
                            {
                                "executed": executed_count > 0,
                                "executed_count": executed_count,
                                "failed_count": failed_count,
                                "reason": "executed_candidate_orders" if executed_count > 0 else "execution_failed",
                            }
                        )
                    content = await self._render_analysis_then_execute_content(
                        analysis_payload=analysis_payload,
                        execution_result=execution_result,
                        runtime_context_payload=runtime_context_payload,
                        history_payload=history_payload,
                        backtest_payload=backtest_payload,
                        original_message=message,
                    )
                    final_payload = {
                        "session_id": session_id,
                        "content": content,
                        "structured_result": {
                            "intent": "analysis_then_execute",
                            "analysis": analysis_payload.get("structured_result"),
                            "runtime_context": runtime_context_payload,
                            "history": history_payload,
                            "backtest": backtest_payload,
                            "autonomous_execution": autonomous_execution,
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
                    )
                    final_payload = {
                        "session_id": session_id,
                        "content": content,
                        "structured_result": {
                            "intent": "analysis",
                            "analysis": analysis_payload.get("structured_result"),
                            "runtime_context": runtime_context_payload,
                            "history": history_payload,
                            "backtest": backtest_payload,
                        },
                        "candidate_orders": candidate_orders,
                        "execution_result": None,
                        "status": "analysis_only",
                    }

            self.repo.add_message(
                owner_user_id=owner_user_id,
                session_id=session_id,
                role="assistant",
                content=str(final_payload.get("content") or ""),
                meta=final_payload,
            )
            return final_payload
        except Exception as exc:
            safe_message = redact_sensitive_text(str(exc))
            logger.exception("Agent chat failed: session=%s owner=%s", session_id, owner_user_id)
            await self._emit(event_handler, "error", {"message": safe_message})
            content = f"本轮 Agent 问股执行失败：{safe_message}"
            final_payload = {
                "session_id": session_id,
                "content": content,
                "structured_result": {"intent": "error", "message": safe_message},
                "candidate_orders": [],
                "execution_result": None,
                "status": "blocked",
            }
            self.repo.add_message(
                owner_user_id=owner_user_id,
                session_id=session_id,
                role="assistant",
                content=content,
                meta=final_payload,
            )
            raise

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

    def delete_session(self, owner_user_id: int, session_id: str) -> bool:
        return self.repo.delete_session(owner_user_id, session_id)

    def _build_plan(
        self,
        *,
        message: str,
        frontend_context: dict[str, Any],
        latest_assistant_meta: dict[str, Any],
        recent_assistant_messages: list[dict[str, Any]],
    ) -> ChatPlan:
        normalized = message.strip()
        stock_codes = self._extract_stock_codes(normalized)
        latest_codes = self._extract_default_stock_codes(latest_assistant_meta, frontend_context)
        resolved_codes = stock_codes or latest_codes

        contains_order = self._contains_order_intent(normalized)
        contains_history = any(keyword in normalized for keyword in _HISTORY_KEYWORDS)
        contains_backtest = any(keyword in normalized for keyword in _BACKTEST_KEYWORDS)
        contains_account = any(keyword in normalized for keyword in _ACCOUNT_KEYWORDS)
        contains_save = any(keyword in normalized for keyword in _SAVE_KEYWORDS)
        contains_analysis = self._contains_analysis_intent(normalized, resolved_codes)
        autonomous_execution_authorized = self._contains_autonomous_execution_authorization(normalized)
        candidate_snapshots = self._extract_candidate_snapshots(recent_assistant_messages)
        planner_hint = self._build_llm_planner_hint(
            message=normalized,
            resolved_codes=resolved_codes,
            candidate_snapshots=candidate_snapshots,
            contains_analysis=contains_analysis,
            contains_order=contains_order,
            contains_history=contains_history,
            contains_backtest=contains_backtest,
            contains_account=contains_account,
            autonomous_execution_authorized=autonomous_execution_authorized,
        )
        hint_intent = str(planner_hint.get("intent") or "").strip()
        hint_codes = self._normalize_stock_codes(planner_hint.get("stock_codes"))
        plan_codes = hint_codes or resolved_codes
        include_history = contains_history or bool(planner_hint.get("include_history"))
        include_backtest = contains_backtest or bool(planner_hint.get("include_backtest"))
        include_runtime_context = bool(planner_hint.get("include_runtime_context"))
        planner_source = str(planner_hint.get("planner_source") or "rule")
        planner_clarification = str(planner_hint.get("clarification") or "").strip()

        if contains_save:
            return ChatPlan(primary_intent="save_analysis", save_requested=True, planner_source=planner_source)

        if hint_intent == "clarify" and planner_clarification:
            return ChatPlan(primary_intent="clarify", clarification=planner_clarification, planner_source=planner_source)

        if hint_intent == "history" and not contains_order and not contains_analysis:
            return ChatPlan(primary_intent="history", stock_codes=plan_codes, include_history=True, planner_source=planner_source)
        if hint_intent == "backtest" and not contains_order and not contains_analysis:
            return ChatPlan(primary_intent="backtest", stock_codes=plan_codes, include_backtest=True, planner_source=planner_source)
        if hint_intent == "account" and not contains_order and not contains_analysis:
            return ChatPlan(primary_intent="account", stock_codes=plan_codes, include_runtime_context=True, planner_source=planner_source)

        if contains_order or hint_intent in {"order_followup_single", "order_followup_all", "analysis_then_execute"}:
            if stock_codes and contains_analysis:
                if autonomous_execution_authorized:
                    return ChatPlan(
                        primary_intent="analysis_then_execute",
                        stock_codes=stock_codes,
                        include_runtime_context=True,
                        include_history=include_history,
                        include_backtest=include_backtest,
                        autonomous_execution_authorized=True,
                        planner_source=planner_source,
                    )
                return ChatPlan(
                    primary_intent="analysis",
                    stock_codes=stock_codes,
                    include_runtime_context=True,
                    include_history=include_history,
                    include_backtest=include_backtest,
                    planner_source=planner_source,
                )

            selected_orders = self._resolve_followup_candidate_orders(
                message=normalized,
                stock_codes=plan_codes,
                candidate_snapshots=candidate_snapshots,
                prefer_all=hint_intent == "order_followup_all" or self._message_requests_all_orders(normalized),
                prefer_best=self._message_requests_best_order(normalized),
            )
            if selected_orders:
                if len(selected_orders) == 1 and hint_intent != "order_followup_all":
                    return ChatPlan(
                        primary_intent="order_followup_single",
                        stock_codes=[str(selected_orders[0].get("code") or "")],
                        target_candidate_order=dict(selected_orders[0]),
                        target_candidate_orders=[dict(selected_orders[0])],
                        planner_source=planner_source,
                    )
                return ChatPlan(
                    primary_intent="order_followup_all",
                    stock_codes=[str(item.get("code") or "") for item in selected_orders if str(item.get("code") or "").strip()],
                    target_candidate_orders=[dict(item) for item in selected_orders],
                    planner_source=planner_source,
                )

            if plan_codes and autonomous_execution_authorized and (hint_intent == "analysis_then_execute" or contains_analysis or contains_order):
                return ChatPlan(
                    primary_intent="analysis_then_execute",
                    stock_codes=plan_codes,
                    include_runtime_context=True,
                    include_history=include_history,
                    include_backtest=include_backtest,
                    autonomous_execution_authorized=True,
                    planner_source=planner_source,
                )

            if contains_order:
                fallback_codes = plan_codes or self._extract_default_stock_codes(latest_assistant_meta, frontend_context)
                hint_codes_text = "、".join(fallback_codes) if fallback_codes else "目标股票代码"
                return ChatPlan(
                    primary_intent="clarify",
                    clarification=(
                        "当前还不能直接执行这条下单请求。"
                        f" 请明确告诉我具体要执行哪一笔候选订单，例如“下 {hint_codes_text} 的单”，"
                        "或者明确授权我“根据结果决定是否下单”。"
                    ),
                    planner_source=planner_source,
                )

        if contains_history and (not plan_codes or not contains_analysis):
            return ChatPlan(primary_intent="history", stock_codes=plan_codes, include_history=True, planner_source=planner_source)

        if contains_backtest and (not plan_codes or not contains_analysis):
            return ChatPlan(primary_intent="backtest", stock_codes=plan_codes, include_backtest=True, planner_source=planner_source)

        if contains_account and (not plan_codes or not contains_analysis):
            return ChatPlan(primary_intent="account", stock_codes=plan_codes, include_runtime_context=True, planner_source=planner_source)

        if plan_codes:
            if autonomous_execution_authorized and hint_intent == "analysis_then_execute":
                return ChatPlan(
                    primary_intent="analysis_then_execute",
                    stock_codes=plan_codes,
                    include_runtime_context=True,
                    include_history=include_history,
                    include_backtest=include_backtest,
                    autonomous_execution_authorized=True,
                    planner_source=planner_source,
                )
            return ChatPlan(
                primary_intent="analysis",
                stock_codes=plan_codes,
                include_runtime_context=True,
                include_history=include_history,
                include_backtest=include_backtest,
                planner_source=planner_source,
            )

        if contains_analysis:
            return ChatPlan(
                primary_intent="clarify",
                clarification="请告诉我股票代码，例如“帮我分析一下今天的 600519 行情”。",
                planner_source=planner_source,
            )

        return ChatPlan(
            primary_intent="clarify",
            clarification="请直接告诉我股票代码和需求，例如“分析 600519”“把刚才那几笔都下了”，或“根据结果决定是否下单”。",
            planner_source=planner_source,
        )

    def _build_llm_planner_hint(
        self,
        *,
        message: str,
        resolved_codes: list[str],
        candidate_snapshots: list[dict[str, Any]],
        contains_analysis: bool,
        contains_order: bool,
        contains_history: bool,
        contains_backtest: bool,
        contains_account: bool,
        autonomous_execution_authorized: bool,
    ) -> dict[str, Any]:
        if not getattr(self.analyzer, "is_available", lambda: False)():
            return {}

        prompt = self._build_planner_prompt(
            message=message,
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
            raw = self.analyzer.generate_text(
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
            "2. order_followup_* 只用于引用历史候选单；3. 无法确定时返回 clarify。\n\n"
            f"用户消息：{message}\n"
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
        return bool(stock_codes) and any(keyword in message for keyword in _ANALYSIS_VIEW_KEYWORDS)

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

    @classmethod
    def _extract_candidate_snapshots(cls, recent_assistant_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        for message in recent_assistant_messages:
            meta = message.get("meta")
            if not isinstance(meta, dict):
                continue
            candidate_orders = cls._normalize_candidate_orders(meta.get("candidate_orders"))
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
    def _extract_default_stock_codes(latest_assistant_meta: dict[str, Any], frontend_context: dict[str, Any]) -> list[str]:
        codes: list[str] = []
        query_code = str(frontend_context.get("stock_code") or frontend_context.get("stockCode") or "").strip()
        if query_code:
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
        return codes

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

    @staticmethod
    def _tool_summary(tool_name: str, args: dict[str, Any]) -> str:
        if tool_name == "run_multi_stock_analysis":
            codes = "、".join(args.get("stock_codes") or [])
            return f"开始串行分析 {codes}"
        if tool_name == "get_runtime_account_context":
            return "读取当前用户模拟盘上下文"
        if tool_name == "get_analysis_history":
            return "读取历史分析记录"
        if tool_name == "get_backtest_summary":
            return "读取回测摘要"
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
        if tool_name == "get_runtime_account_context":
            summary = result.get("runtime_context") if isinstance(result.get("runtime_context"), dict) else {}
            positions = summary.get("positions") if isinstance(summary.get("positions"), list) else []
            return f"已读取账户上下文，当前持仓 {len(positions)} 项"
        if tool_name == "get_analysis_history":
            items = result.get("items") if isinstance(result.get("items"), list) else []
            return f"已读取 {len(items)} 条历史分析"
        if tool_name == "get_backtest_summary":
            items = result.get("items") if isinstance(result.get("items"), list) else []
            return f"已读取 {len(items)} 条回测摘要"
        if tool_name == "place_simulated_order":
            return f"模拟盘执行结果：{result.get('status') or 'unknown'}"
        if tool_name == "batch_execute_candidate_orders":
            executed_count = int(result.get("executed_count") or 0)
            failed_count = len(result.get("failed_orders") or [])
            return f"已完成候选订单提交，成功 {executed_count} 笔，失败 {failed_count} 笔"
        return "工具执行完成"

    async def _tool_get_runtime_account_context(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.backend_client.get_runtime_account_context(
            owner_user_id=int(args.get("owner_user_id") or 0),
            refresh=bool(args.get("refresh", True)),
        )

    async def _tool_get_analysis_history(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.backend_client.get_analysis_history(
            owner_user_id=int(args.get("owner_user_id") or 0),
            stock_codes=list(args.get("stock_codes") or []),
            limit=int(args.get("limit") or 5),
        )

    async def _tool_get_backtest_summary(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.backend_client.get_backtest_summary(
            owner_user_id=int(args.get("owner_user_id") or 0),
            stock_codes=list(args.get("stock_codes") or []),
            limit=int(args.get("limit") or 6),
        )

    async def _tool_run_multi_stock_analysis(
        self,
        args: dict[str, Any],
        _tool_context: dict[str, Any],
    ) -> dict[str, Any]:
        runtime_config = self._merge_runtime_context_into_runtime_config(
            runtime_config=args.get("runtime_config") if isinstance(args.get("runtime_config"), dict) else {},
            runtime_context_payload=args.get("runtime_context_payload") if isinstance(args.get("runtime_context_payload"), dict) else {},
        )
        run_result = await asyncio.to_thread(
            self.agent_service.run_once,
            list(args.get("stock_codes") or []),
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
        failed_orders: list[dict[str, Any]] = []

        for candidate_order in candidate_orders:
            try:
                result = await self.backend_client.place_simulated_order(
                    owner_user_id=owner_user_id,
                    session_id=session_id,
                    candidate_order=candidate_order,
                )
                orders.append(result)
            except Exception as exc:
                failed_orders.append(
                    {
                        "candidate_order": dict(candidate_order),
                        "message": redact_sensitive_text(str(exc)),
                    }
                )

        status = self._resolve_batch_execution_state(orders, failed_orders)
        return {
            "mode": "batch",
            "candidate_order_count": len(candidate_orders),
            "executed_count": len(orders),
            "orders": orders,
            "failed_orders": failed_orders,
            "status": status,
            "summary": {
                "candidate_order_count": len(candidate_orders),
                "executed_count": len(orders),
                "failed_count": len(failed_orders),
            },
        }

    @staticmethod
    def _resolve_batch_execution_state(
        orders: list[dict[str, Any]],
        failed_orders: list[dict[str, Any]],
    ) -> str:
        if not orders:
            return "failed"

        normalized_statuses = {str(item.get("status") or "").strip().lower() for item in orders}
        if failed_orders:
            return "submitted"
        if normalized_statuses == {"filled"}:
            return "filled"
        if normalized_statuses.issubset({"filled", "submitted", "partial_filled"}):
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
        return {
            "run_id": run_result.run_id,
            "structured_result": {
                "run_id": run_result.run_id,
                "trade_date": run_result.trade_date.isoformat(),
                "stocks": stocks,
                "portfolio_summary": portfolio_summary,
            },
            "candidate_orders": candidate_orders,
        }

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
        if item.execution.action in {"buy", "sell"} and int(item.execution.traded_qty or 0) > 0:
            candidate_order = {
                "code": item.code,
                "stock_name": stock_name,
                "action": item.execution.action,
                "quantity": int(item.execution.traded_qty or 0),
                "target_qty": int(item.execution.target_qty or 0),
                "price": float(item.execution.fill_price or current_price or 0.0),
                "reason": item.execution.reason,
                "current_price": float(current_price or 0.0),
                "risk_flags": list(item.risk.risk_flags or []),
                "source_run_id": item.execution.backend_task_id or "",
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
            "candidate_order": candidate_order,
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
            if not getattr(self.analyzer, "is_available", lambda: False)():
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
                self.analyzer.generate_text,
                prompt,
                temperature=0.2,
                max_output_tokens=1800,
            )
            return self._normalize_analysis_text(content, fallback)
        except Exception as exc:
            logger.warning("LLM analysis summary fallback: %s", redact_sensitive_text(str(exc)))
            return fallback

    async def _render_analysis_then_execute_content(
        self,
        *,
        analysis_payload: dict[str, Any],
        execution_result: dict[str, Any] | None,
        runtime_context_payload: dict[str, Any] | None,
        history_payload: dict[str, Any] | None,
        backtest_payload: dict[str, Any] | None,
        original_message: str,
    ) -> str:
        structured_result = analysis_payload.get("structured_result") if isinstance(analysis_payload.get("structured_result"), dict) else {}
        candidate_orders = analysis_payload.get("candidate_orders") if isinstance(analysis_payload.get("candidate_orders"), list) else []
        fallback = self._render_analysis_then_execute_template(
            structured_result=structured_result,
            candidate_orders=candidate_orders,
            execution_result=execution_result,
            runtime_context_payload=runtime_context_payload,
            history_payload=history_payload,
            backtest_payload=backtest_payload,
        )

        try:
            if not getattr(self.analyzer, "is_available", lambda: False)():
                return fallback
            prompt = self._build_analysis_then_execute_prompt(
                original_message=original_message,
                structured_result=structured_result,
                candidate_orders=candidate_orders,
                execution_result=execution_result,
                runtime_context_payload=runtime_context_payload,
                history_payload=history_payload,
                backtest_payload=backtest_payload,
            )
            content = await asyncio.to_thread(
                self.analyzer.generate_text,
                prompt,
                temperature=0.2,
                max_output_tokens=2000,
            )
            return self._normalize_analysis_text(content, fallback)
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
            f"账户上下文：{runtime_context_payload}\n\n"
            f"历史分析：{history_payload}\n\n"
            f"回测摘要：{backtest_payload}\n"
        )

    @staticmethod
    def _normalize_analysis_text(content: Any, fallback: str) -> str:
        clean = str(content or "").strip()
        if not clean:
            return fallback

        if AgentChatService._looks_like_structured_output(clean):
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
            f"这次一共分析了 {stock_count} 只股票，当前形成 {candidate_count} 笔候选订单。",
        ]
        if candidate_orders:
            codes = "、".join(str(item.get("code") or "") for item in candidate_orders)
            lines.append(f"候选订单涉及 {codes}，这些都还只是建议，只有你明确确认后才会提交到模拟盘。")
        else:
            lines.append("这轮分析暂时没有形成需要立即执行的候选订单，更适合先观察。")

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
                    f"如果你想执行这只股票的候选单，可以考虑以 {candidate.get('price') or '--'} 的参考价"
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
            lines.append("如果你确认要执行候选订单，可以直接说“去下单吧”、明确指定“下 600519 的单”，或者授权我“根据结果决定是否下单”。")
        return "\n".join(lines)

    def _render_analysis_then_execute_template(
        self,
        *,
        structured_result: dict[str, Any],
        candidate_orders: list[dict[str, Any]],
        execution_result: dict[str, Any] | None,
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
            f"我先完成了 {stock_count} 只股票的分析，本轮共形成 {candidate_count} 笔候选订单。",
        ]
        if not candidate_orders:
            lines.append("结合这轮结果，我判断暂时不需要下模拟盘单，因为当前没有生成满足条件的候选订单。")
        elif execution_result is None:
            lines.append("这轮分析形成了候选订单，但当前没有可用的执行结果。")
        else:
            executed_count = int(execution_result.get("executed_count") or 0)
            failed_count = len(execution_result.get("failed_orders") or [])
            if executed_count > 0:
                lines.append(f"我已根据分析结果执行候选订单，成功提交 {executed_count} 笔。")
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
        runtime_context_payload: dict[str, Any] | None,
        history_payload: dict[str, Any] | None,
        backtest_payload: dict[str, Any] | None,
    ) -> str:
        lines = ["## 当前结果"]
        if isinstance(runtime_context_payload, dict):
            runtime_context = runtime_context_payload.get("runtime_context")
            if isinstance(runtime_context, dict):
                summary = runtime_context.get("summary") if isinstance(runtime_context.get("summary"), dict) else {}
                positions = runtime_context.get("positions") if isinstance(runtime_context.get("positions"), list) else []
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
        if str(execution_result.get("mode") or "").strip().lower() == "batch":
            orders = execution_result.get("orders") if isinstance(execution_result.get("orders"), list) else []
            failed_orders = execution_result.get("failed_orders") if isinstance(execution_result.get("failed_orders"), list) else []
            lines = [
                "## 模拟盘执行结果",
                f"本轮共尝试执行 {execution_result.get('candidate_order_count') or len(orders) + len(failed_orders)} 笔候选订单。",
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
        status = str(execution_result.get("status") or "").strip().lower()
        if status == "filled":
            return "simulation_order_filled"
        if status in {"submitted", "partial_filled"}:
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
