# -*- coding: utf-8 -*-
"""Agent 问股聊天级主控规划器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent_stock.agents.agentic_decision import extract_json_object
from agent_stock.config import Config, get_config, redact_sensitive_text

import logging

logger = logging.getLogger(__name__)

ALLOWED_CHAT_INTENTS = {
    "analysis",
    "analysis_then_execute",
    "order_followup",
    "account",
    "history",
    "backtest",
    "clarify",
}
ALLOWED_REQUIRED_TOOLS = {
    "load_system_state",
    "load_account_state",
    "load_session_memory",
    "load_user_preferences",
    "load_stage_memory",
    "load_history",
    "load_backtest",
    "run_strategy_backtest",
    "run_multi_stock_analysis",
    "place_simulated_order",
    "batch_execute_candidate_orders",
}
ALLOWED_STOCK_SCOPE_MODES = {"explicit", "focus", "pending_actions", "none"}
ALLOWED_FOLLOWUP_TARGET_MODES = {"none", "single", "all", "best"}
ALLOWED_CONSTRAINT_TYPES = {
    "risk_gate",
    "price_gate",
    "min_remaining_cash",
    "max_single_position_pct",
    "order_side",
    "exact_quantity",
}
ALLOWED_SESSION_OVERRIDE_KEYS = {
    "riskProfile",
    "analysisStrategy",
    "maxSingleTradeAmount",
    "positionMaxPct",
    "stopLossPct",
    "takeProfitPct",
    "executionPolicy",
    "responseStyle",
}


@dataclass(frozen=True)
class AgentSystemState:
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class AgentAccountState:
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class EffectiveUserPreferences:
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class StageMemorySnapshot:
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True)
class ChatContextBundle:
    loaded_keys: list[str] = field(default_factory=list)
    system_state: AgentSystemState = field(default_factory=AgentSystemState)
    account_state: AgentAccountState = field(default_factory=AgentAccountState)
    portfolio_health: dict[str, Any] = field(default_factory=dict)
    session_memory: dict[str, Any] = field(default_factory=dict)
    effective_user_preferences: EffectiveUserPreferences = field(default_factory=EffectiveUserPreferences)
    stage_memory: StageMemorySnapshot = field(default_factory=StageMemorySnapshot)
    history: dict[str, Any] = field(default_factory=dict)
    backtest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loaded_keys": list(self.loaded_keys),
            "system_state": self.system_state.to_dict(),
            "account_state": self.account_state.to_dict(),
            "portfolio_health": dict(self.portfolio_health),
            "session_memory": dict(self.session_memory),
            "effective_user_preferences": self.effective_user_preferences.to_dict(),
            "stage_memory": self.stage_memory.to_dict(),
            "history": dict(self.history),
            "backtest": dict(self.backtest),
        }


@dataclass(frozen=True)
class ChatPlannerPlan:
    intent: str
    stock_scope: dict[str, Any] = field(default_factory=dict)
    followup_target: dict[str, Any] = field(default_factory=dict)
    execution_authorized: bool = False
    required_tools: list[str] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    session_preference_overrides: dict[str, Any] = field(default_factory=dict)
    clarification: str = ""
    planner_source: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_intent_resolution(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "stock_scope": dict(self.stock_scope),
            "followup_target": dict(self.followup_target),
            "execution_authorized": self.execution_authorized,
            "constraints": [dict(item) for item in self.constraints if isinstance(item, dict)],
            "session_preference_overrides": dict(self.session_preference_overrides),
            "source": self.planner_source,
            "confidence": 0.88 if self.intent != "clarify" else 0.45,
            "missing_slots": ["clarification"] if self.intent == "clarify" and self.clarification else [],
        }

    @classmethod
    def clarify(cls, message: str, *, planner_source: str = "llm") -> "ChatPlannerPlan":
        return cls(
            intent="clarify",
            stock_scope={"mode": "none", "stock_refs": []},
            followup_target={"mode": "none", "stock_refs": []},
            clarification=message,
            planner_source=planner_source,
        )


class ChatPlannerAgent:
    """基于 LLM 的聊天级主控规划器。"""

    def __init__(self, *, config: Config | None = None, analyzer=None) -> None:
        self.config = config or get_config()
        self.analyzer = analyzer

    def plan(
        self,
        *,
        message: str,
        session_summary: dict[str, Any],
        pending_action_summary: list[dict[str, Any]],
        tool_registry: list[str],
        analyzer=None,
    ) -> ChatPlannerPlan:
        active_analyzer = analyzer or self.analyzer
        if not getattr(active_analyzer, "is_available", lambda: False)():
            return ChatPlannerPlan.clarify(
                "当前主控规划器暂时不可用，请稍后重试，或用更明确的话描述股票代码和目标。",
                planner_source="llm_unavailable",
            )

        prompt = self._build_prompt(
            message=message,
            session_summary=session_summary,
            pending_action_summary=pending_action_summary,
            tool_registry=tool_registry,
        )
        for _attempt in range(2):
            try:
                raw = active_analyzer.generate_text(prompt, temperature=0.0, max_output_tokens=700)
            except Exception as exc:
                logger.warning("chat planner failed: %s", redact_sensitive_text(str(exc)))
                continue
            payload = extract_json_object(raw)
            plan = self._parse_payload(payload)
            if plan is not None:
                return plan
            degraded_plan = self._recover_empty_explicit_stock_scope_plan(payload)
            if degraded_plan is not None:
                return degraded_plan

        return ChatPlannerPlan.clarify(
            "我暂时无法稳定解析这条请求，请换一种更明确的说法，例如“分析 600519”或“把刚才那几笔都下了”。",
        )

    @staticmethod
    def _build_prompt(
        *,
        message: str,
        session_summary: dict[str, Any],
        pending_action_summary: list[dict[str, Any]],
        tool_registry: list[str],
    ) -> str:
        return (
            "你是 Agent问股 的聊天主控规划器，只输出严格 JSON，不要输出解释、Markdown 或代码块。\n"
            "允许的 intent 只有：analysis, analysis_then_execute, order_followup, account, history, backtest, clarify。\n"
            "stock_scope.mode 只能是：explicit, focus, pending_actions, none。\n"
            "followup_target.mode 只能是：none, single, all, best。\n"
            "required_tools 只能从以下工具中选择："
            f"{tool_registry}。\n"
            "constraints 只允许输出这些 type：risk_gate, price_gate, min_remaining_cash, max_single_position_pct, order_side, exact_quantity。\n"
            "session_preference_overrides 只允许字段：riskProfile, analysisStrategy, maxSingleTradeAmount, positionMaxPct, stopLossPct, takeProfitPct, executionPolicy, responseStyle。\n"
            "规则：\n"
            "1. 只有用户明确授权“根据结果决定是否下单/合适就下单/分析后直接下单”时，才能返回 analysis_then_execute 且 execution_authorized=true。\n"
            "2. 引用历史候选单时使用 order_followup，不要重新生成分析 intent。\n"
            "3. 如果用户想看账户/持仓/资金，使用 account。\n"
            "4. 如果用户说“再试一次/按刚才的来”，优先使用 focus 或 pending_actions，而不是虚构新股票代码。\n"
            "5. 无法确定就返回 clarify，并填写 clarification。\n"
            "6. analysis / analysis_then_execute 必须包含 run_multi_stock_analysis；history 必须包含 load_history；"
            "account 必须包含 load_account_state；如果是查看历史回测摘要，backtest 使用 load_backtest；"
            "如果是执行新的策略回测，backtest 使用 run_strategy_backtest，并在必要时补 load_account_state。\n"
            "7. 如果需要系统态、会话态、偏好态或阶段记忆，请把对应 load_* 工具加入 required_tools。\n\n"
            "8. 当 stock_scope.mode=explicit 时，stock_refs 可以包含股票代码、股票名称、行业板块名或概念板块名；如果提取不出来，就返回 clarify。\n"
            "9. “全市场/所有股票/A股全市场/沪深两市” 这类范围不是普通 stock_refs，不要把它们当成具体股票、行业或概念板块。\n\n"
            f"用户消息：{message}\n"
            f"最近会话摘要：{session_summary}\n"
            f"待确认动作摘要：{pending_action_summary}\n\n"
            "输出 JSON 字段：intent, stock_scope, followup_target, execution_authorized, required_tools, constraints, session_preference_overrides, clarification。"
        )

    def _parse_payload(self, payload: dict[str, Any]) -> ChatPlannerPlan | None:
        if not isinstance(payload, dict):
            return None

        intent = str(payload.get("intent") or "").strip()
        if intent not in ALLOWED_CHAT_INTENTS:
            return None

        stock_scope = self._normalize_stock_scope(payload.get("stock_scope"))
        followup_target = self._normalize_followup_target(payload.get("followup_target"))
        required_tools = self._normalize_required_tools(payload.get("required_tools"))
        constraints = self._normalize_constraints(payload.get("constraints"))
        session_overrides = self._normalize_session_overrides(payload.get("session_preference_overrides"))
        clarification = str(payload.get("clarification") or "").strip()
        execution_authorized = bool(payload.get("execution_authorized"))

        if intent in {"analysis", "analysis_then_execute"} and "run_multi_stock_analysis" not in required_tools:
            return None
        if intent == "history" and "load_history" not in required_tools:
            return None
        if intent == "backtest" and not any(tool in required_tools for tool in {"load_backtest", "run_strategy_backtest"}):
            return None
        if intent == "account" and "load_account_state" not in required_tools:
            return None
        if intent == "order_followup" and followup_target.get("mode") == "none":
            return None
        if intent in {"analysis", "analysis_then_execute"} and stock_scope.get("mode") == "none":
            return None
        if intent in {"analysis", "analysis_then_execute"} and stock_scope.get("mode") == "explicit" and not stock_scope.get("stock_refs"):
            return None
        if intent == "analysis_then_execute" and not execution_authorized:
            return None
        if intent == "clarify" and not clarification:
            return None

        return ChatPlannerPlan(
            intent=intent,
            stock_scope=stock_scope,
            followup_target=followup_target,
            execution_authorized=execution_authorized,
            required_tools=required_tools,
            constraints=constraints,
            session_preference_overrides=session_overrides,
            clarification=clarification,
            planner_source="llm",
        )

    def _recover_empty_explicit_stock_scope_plan(self, payload: dict[str, Any]) -> ChatPlannerPlan | None:
        if not isinstance(payload, dict):
            return None

        intent = str(payload.get("intent") or "").strip()
        if intent not in {"analysis", "analysis_then_execute"}:
            return None

        stock_scope = self._normalize_stock_scope(payload.get("stock_scope"))
        if stock_scope.get("mode") != "explicit" or stock_scope.get("stock_refs"):
            return None

        required_tools = self._normalize_required_tools(payload.get("required_tools"))
        if "run_multi_stock_analysis" not in required_tools:
            return None

        execution_authorized = bool(payload.get("execution_authorized"))
        if intent == "analysis_then_execute" and not execution_authorized:
            return None

        return ChatPlannerPlan(
            intent=intent,
            stock_scope=stock_scope,
            followup_target=self._normalize_followup_target(payload.get("followup_target")),
            execution_authorized=execution_authorized,
            required_tools=required_tools,
            constraints=self._normalize_constraints(payload.get("constraints")),
            session_preference_overrides=self._normalize_session_overrides(payload.get("session_preference_overrides")),
            clarification="",
            planner_source="llm_invalid_empty_explicit_stock_scope",
        )

    @staticmethod
    def _normalize_stock_scope(value: Any) -> dict[str, Any]:
        source = value if isinstance(value, dict) else {}
        mode = str(source.get("mode") or "none").strip().lower()
        stock_refs = source.get("stock_refs")
        if isinstance(stock_refs, str):
            stock_refs = [stock_refs]
        if not isinstance(stock_refs, list):
            stock_refs = []
        normalized_refs = []
        for item in stock_refs:
            text = str(item or "").strip()
            if text and text not in normalized_refs:
                normalized_refs.append(text)
        return {
            "mode": mode if mode in ALLOWED_STOCK_SCOPE_MODES else "none",
            "stock_refs": normalized_refs,
        }

    @staticmethod
    def _normalize_followup_target(value: Any) -> dict[str, Any]:
        source = value if isinstance(value, dict) else {}
        mode = str(source.get("mode") or "none").strip().lower()
        stock_refs = source.get("stock_refs")
        if isinstance(stock_refs, str):
            stock_refs = [stock_refs]
        if not isinstance(stock_refs, list):
            stock_refs = []
        normalized_refs = []
        for item in stock_refs:
            text = str(item or "").strip()
            if text and text not in normalized_refs:
                normalized_refs.append(text)
        return {
            "mode": mode if mode in ALLOWED_FOLLOWUP_TARGET_MODES else "none",
            "stock_refs": normalized_refs,
        }

    @staticmethod
    def _normalize_required_tools(value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        tools: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text in ALLOWED_REQUIRED_TOOLS and text not in tools:
                tools.append(text)
        return tools

    @staticmethod
    def _normalize_constraints(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        constraints: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            type_name = str(item.get("type") or "").strip()
            if type_name not in ALLOWED_CONSTRAINT_TYPES:
                continue
            constraints.append(
                {
                    "type": type_name,
                    "value": item.get("value"),
                    "operator": str(item.get("operator") or "eq").strip() or "eq",
                    "label": str(item.get("label") or type_name).strip() or type_name,
                    "supported": bool(item.get("supported", True)),
                }
            )
        return constraints

    @staticmethod
    def _normalize_session_overrides(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            clean_key = str(key or "").strip()
            if clean_key not in ALLOWED_SESSION_OVERRIDE_KEYS:
                continue
            normalized[clean_key] = item
        return normalized
