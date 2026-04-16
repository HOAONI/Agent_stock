# -*- coding: utf-8 -*-
"""Agent 问股聊天服务测试。"""

from __future__ import annotations

from calendar import monthrange
import json
import os
import re
import tempfile
import unittest
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from agent_stock.agents.contracts import (
    AgentRunResult,
    AgentState,
    DataAgentOutput,
    ExecutionAgentOutput,
    RiskAgentOutput,
    SignalAgentOutput,
    StockAgentResult,
)
from agent_stock.agents.chat_planner_agent import ChatPlannerAgent
from agent_stock.config import Config
from agent_stock.repositories.chat_repo import AgentChatRepository
from agent_stock.services.agent_chat_service import AgentChatService
from agent_stock.storage import DatabaseManager

STOCK_NAMES = {
    "600519": "贵州茅台",
    "000001": "平安银行",
    "002594": "比亚迪",
    "300750": "宁德时代",
    "601899": "紫金矿业",
    "601138": "工业富联",
    "000333": "美的集团",
}

STOCK_PRICES = {
    "600519": 1680.0,
    "000001": 12.36,
    "002594": 246.8,
    "300750": 198.52,
    "601899": 18.66,
    "601138": 21.35,
    "000333": 72.18,
}

STOCK_SCORES = {
    "600519": 78,
    "000001": 66,
    "002594": 84,
    "300750": 92,
    "601899": 73,
    "601138": 68,
    "000333": 81,
}

FAKE_INDUSTRY_BOARD_CATALOG = [
    {"board_name": "半导体", "board_symbol": "BK0917"},
]

FAKE_CONCEPT_BOARD_CATALOG = [
    {"board_name": "高股息", "board_symbol": "BK9991"},
]

FAKE_BOARD_CONSTITUENTS = {
    ("industry_board", "BK0917"): [
        {"code": "688981", "name": "中芯国际", "total_mv": 3200_000_000_000, "amount": 3_200_000_000},
        {"code": "603501", "name": "韦尔股份", "total_mv": 1800_000_000_000, "amount": 2_800_000_000},
        {"code": "603986", "name": "兆易创新", "total_mv": 1300_000_000_000, "amount": 2_000_000_000},
        {"code": "600584", "name": "长电科技", "total_mv": 1200_000_000_000, "amount": 1_900_000_000},
        {"code": "002371", "name": "北方华创", "total_mv": 1100_000_000_000, "amount": 1_800_000_000},
        {"code": "300661", "name": "圣邦股份", "total_mv": 900_000_000_000, "amount": 1_600_000_000},
        {"code": "300223", "name": "北京君正", "total_mv": 800_000_000_000, "amount": 1_500_000_000},
        {"code": "600460", "name": "士兰微", "total_mv": 700_000_000_000, "amount": 1_400_000_000},
        {"code": "300782", "name": "卓胜微", "total_mv": 650_000_000_000, "amount": 1_300_000_000},
        {"code": "688012", "name": "中微公司", "total_mv": 600_000_000_000, "amount": 1_200_000_000},
        {"code": "300346", "name": "南大光电", "total_mv": 500_000_000_000, "amount": 1_100_000_000},
        {"code": "002049", "name": "紫光国微", "total_mv": 450_000_000_000, "amount": 1_000_000_000},
    ],
    ("concept_board", "BK9991"): [
        {"code": "600519", "name": "贵州茅台", "total_mv": 2000_000_000_000, "amount": 1_100_000_000},
        {"code": "000333", "name": "美的集团", "total_mv": 900_000_000_000, "amount": 900_000_000},
    ],
}


def _subtract_calendar_months_for_test(base_date: date, months: int) -> date:
    total_months = base_date.year * 12 + (base_date.month - 1) - months
    year = total_months // 12
    month = total_months % 12 + 1
    day = min(base_date.day, monthrange(year, month)[1])
    return date(year, month, day)


def build_runtime_config() -> dict[str, Any]:
    return {
        "account": {"account_name": "u1", "initial_cash": 100000},
        "strategy": {
            "position_max_pct": 30,
            "stop_loss_pct": 8,
            "take_profit_pct": 15,
        },
    }


def build_runtime_config_with_llm(
    *,
    provider: str = "custom",
    base_url: str = "https://api.siliconflow.cn/v1",
    model: str = "Pro/zai-org/GLM-5",
    api_token: str = "silicon-token-1234567890",
) -> dict[str, Any]:
    payload = build_runtime_config()
    payload["llm"] = {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_token": api_token,
        "has_token": True,
    }
    return payload


def build_outside_trading_session_order_result(candidate_order: dict[str, Any]) -> dict[str, Any]:
    next_open_at = "2026-04-06T09:30:00+08:00"
    return {
        "status": "blocked",
        "reason": "outside_trading_session",
        "message": (
            "当前处于非交易时段（Asia/Shanghai，交易时段 09:30-11:30、13:00-15:00）。"
            f"本轮未执行模拟盘订单，候选单已保留，请在 {next_open_at} 后再次确认。"
        ),
        "candidate_order": dict(candidate_order),
        "session_guard": {
            "timezone": "Asia/Shanghai",
            "sessions": ["09:30-11:30", "13:00-15:00"],
            "evaluated_at": "2026-04-05T16:01:00+08:00",
            "next_open_at": next_open_at,
        },
    }


class FakePlannerAnalyzer:
    def is_available(self) -> bool:
        return True

    def generate_text(self, prompt: str, *_args, **_kwargs) -> str:
        if "Agent问股 的聊天主控规划器" in prompt:
            message = self._extract_user_message(prompt)
            return json.dumps(self._plan_for_message(message), ensure_ascii=False)

        # 让总结环节稳定回退到模板，避免测试依赖自然语言生成细节。
        return """```json
{"structured": true}
```"""

    @staticmethod
    def _extract_user_message(prompt: str) -> str:
        matched = re.search(r"用户消息：(.*?)\n最近会话摘要：", prompt, flags=re.S)
        return str(matched.group(1) if matched else "").strip()

    def _plan_for_message(self, message: str) -> dict[str, Any]:
        normalized = "".join(str(message or "").split())
        stock_refs = self._extract_stock_refs(message)
        required_tools: list[str] = []
        constraints: list[dict[str, Any]] = []
        overrides: dict[str, Any] = {}

        if "账户" in message or ("持仓" in message and "资金" in message):
            return self._plan(
                intent="account",
                required_tools=["load_account_state"],
                stock_scope={"mode": "none", "stock_refs": []},
            )

        if "最近分析记录" in message or ("历史" in message and "分析" in message):
            return self._plan(
                intent="history",
                required_tools=["load_history"],
                stock_scope={"mode": "none", "stock_refs": []},
            )

        if "回测" in message:
            return self._plan(
                intent="backtest",
                required_tools=["load_backtest"],
                stock_scope={"mode": "none", "stock_refs": []},
            )

        if any(token in normalized for token in ("把刚才那几笔都下了", "全部下单")):
            return self._plan(
                intent="order_followup",
                required_tools=["load_session_memory", "load_stage_memory"],
                stock_scope={"mode": "pending_actions", "stock_refs": []},
                followup_target={"mode": "all", "stock_refs": []},
            )

        if "最看好" in message:
            return self._plan(
                intent="order_followup",
                required_tools=["load_session_memory", "load_stage_memory"],
                stock_scope={"mode": "none", "stock_refs": []},
                followup_target={"mode": "best", "stock_refs": []},
            )

        if any(token in normalized for token in ("去下单吧", "确认买入", "确认")) or ("的单" in message and "下" in message):
            refs = stock_refs[:1]
            return self._plan(
                intent="order_followup",
                required_tools=["load_session_memory", "load_stage_memory"],
                stock_scope={"mode": "explicit", "stock_refs": refs} if refs else {"mode": "none", "stock_refs": []},
                followup_target={"mode": "single", "stock_refs": refs},
            )

        if any(token in normalized for token in ("再试一次", "按刚才的来")):
            return self._plan(
                intent="analysis",
                required_tools=["load_session_memory", "load_stage_memory", "run_multi_stock_analysis"],
                stock_scope={"mode": "focus", "stock_refs": []},
            )

        if "激进一点" in message:
            overrides["riskProfile"] = "aggressive"
            required_tools.append("load_user_preferences")

        is_execute_intent = any(
            token in normalized
            for token in (
                "根据结果决定是否去下单",
                "根据结果决定是否下单",
                "风险低的话帮我买",
                "如果风险低就买",
                "直接买100股",
            )
        )
        if is_execute_intent:
            required_tools.extend(["load_account_state", "load_system_state"])

        if "风险低" in message:
            constraints.append(
                {
                    "type": "risk_gate",
                    "value": "risk_low",
                    "operator": "eq",
                    "label": "risk_low",
                    "supported": True,
                }
            )
        if "买" in message:
            constraints.append(
                {
                    "type": "order_side",
                    "value": "buy",
                    "operator": "eq",
                    "label": "order_side",
                    "supported": True,
                }
            )
        qty_match = re.search(r"(\d{1,7})\s*股", message)
        if qty_match:
            constraints.append(
                {
                    "type": "exact_quantity",
                    "value": int(qty_match.group(1)),
                    "operator": "eq",
                    "label": "exact_quantity",
                    "supported": True,
                }
            )

        required_tools.append("run_multi_stock_analysis")
        required_tools = list(dict.fromkeys(required_tools))
        if not stock_refs:
            stock_scope = {"mode": "focus", "stock_refs": []}
        else:
            stock_scope = {"mode": "explicit", "stock_refs": stock_refs}
        intent = "analysis_then_execute" if is_execute_intent else "analysis"
        return self._plan(
            intent=intent,
            required_tools=required_tools,
            stock_scope=stock_scope,
            execution_authorized=is_execute_intent,
            constraints=constraints,
            session_preference_overrides=overrides,
        )

    @staticmethod
    def _extract_stock_refs(message: str) -> list[str]:
        refs: list[str] = []
        for code in re.findall(r"\b\d{6}\b", str(message or "")):
            if code not in refs:
                refs.append(code)
        for code, name in STOCK_NAMES.items():
            if name in message and name not in refs:
                refs.append(name)
        if not refs:
            matched = re.search(r"(?:再分析一下|分析一下|分析|看下|看看|那)([\u4e00-\u9fff]{2,12})(?:呢|行情|的单|和|，|。|\s|$)", message)
            if matched:
                candidate = str(matched.group(1) or "").strip("，。 ")
                if candidate and candidate not in refs:
                    refs.append(candidate)
        return refs

    @staticmethod
    def _plan(
        *,
        intent: str,
        required_tools: list[str],
        stock_scope: dict[str, Any],
        followup_target: dict[str, Any] | None = None,
        execution_authorized: bool = False,
        constraints: list[dict[str, Any]] | None = None,
        session_preference_overrides: dict[str, Any] | None = None,
        clarification: str = "",
    ) -> dict[str, Any]:
        return {
            "intent": intent,
            "stock_scope": stock_scope,
            "followup_target": followup_target or {"mode": "none", "stock_refs": []},
            "execution_authorized": execution_authorized,
            "required_tools": required_tools,
            "constraints": constraints or [],
            "session_preference_overrides": session_preference_overrides or {},
            "clarification": clarification,
        }


class FakeMappedPlannerAnalyzer(FakePlannerAnalyzer):
    def __init__(self, plan_by_message: dict[str, dict[str, Any]]) -> None:
        self.plan_by_message = {str(key): dict(value) for key, value in plan_by_message.items()}

    def _plan_for_message(self, message: str) -> dict[str, Any]:
        custom = self.plan_by_message.get(str(message or ""))
        if custom is not None:
            return dict(custom)
        return super()._plan_for_message(message)


class TrackingPlannerAnalyzer(FakePlannerAnalyzer):
    def __init__(self, *, label: str, summary_text: str) -> None:
        self.label = label
        self.summary_text = summary_text
        self.calls: list[str] = []

    def generate_text(self, prompt: str, *_args, **_kwargs) -> str:
        self.calls.append(prompt)
        if "Agent问股 的聊天主控规划器" in prompt:
            message = self._extract_user_message(prompt)
            return json.dumps(self._plan_for_message(message), ensure_ascii=False)
        return self.summary_text


class ExplodingSummaryAnalyzer(FakePlannerAnalyzer):
    def generate_text(self, prompt: str, *_args, **_kwargs) -> str:
        if "Agent问股 的聊天主控规划器" in prompt:
            message = self._extract_user_message(prompt)
            return json.dumps(self._plan_for_message(message), ensure_ascii=False)
        raise RuntimeError("summary exploded")


def fake_board_catalog_provider(board_type: str) -> list[dict[str, Any]]:
    if board_type == "industry_board":
        return [dict(item) for item in FAKE_INDUSTRY_BOARD_CATALOG]
    if board_type == "concept_board":
        return [dict(item) for item in FAKE_CONCEPT_BOARD_CATALOG]
    return []


def fake_board_constituents_provider(board_type: str, board_symbol: str) -> list[dict[str, Any]]:
    return [dict(item) for item in FAKE_BOARD_CONSTITUENTS.get((board_type, board_symbol), [])]


class FakeInvalidPlannerAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def generate_text(self, *_args, **_kwargs) -> str:
        self.calls += 1
        return "not-json"


class FakeEmptyExplicitPlannerAnalyzer:
    def is_available(self) -> bool:
        return True

    def generate_text(self, prompt: str, *_args, **_kwargs) -> str:
        if "Agent问股 的聊天主控规划器" in prompt:
            return json.dumps(
                {
                    "intent": "analysis",
                    "stock_scope": {"mode": "explicit", "stock_refs": []},
                    "followup_target": {"mode": "none", "stock_refs": []},
                    "execution_authorized": False,
                    "required_tools": ["run_multi_stock_analysis"],
                    "constraints": [],
                    "session_preference_overrides": {},
                    "clarification": "",
                },
                ensure_ascii=False,
            )
        return """```json
{"structured": true}
```"""


class FakeMarketWidePlannerAnalyzer:
    def is_available(self) -> bool:
        return True

    def generate_text(self, prompt: str, *_args, **_kwargs) -> str:
        if "Agent问股 的聊天主控规划器" in prompt:
            return json.dumps(
                {
                    "intent": "analysis",
                    "stock_scope": {"mode": "explicit", "stock_refs": ["A股全市场"]},
                    "followup_target": {"mode": "best", "stock_refs": []},
                    "execution_authorized": False,
                    "required_tools": ["run_multi_stock_analysis"],
                    "constraints": [],
                    "session_preference_overrides": {},
                    "clarification": "",
                },
                ensure_ascii=False,
            )
        return """```json
{"structured": true}
```"""


class FakeStockNameLookupManager:
    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        errors: list[Exception | None] | None = None,
    ) -> None:
        self.rows = [dict(item) for item in (rows or [])]
        self.errors = list(errors or [])
        self.calls = 0

    def get_stock_list(self) -> pd.DataFrame:
        self.calls += 1
        if self.errors and self.calls <= len(self.errors):
            error = self.errors[self.calls - 1]
            if error is not None:
                raise error
        return pd.DataFrame(self.rows)


class FakeBackendClient:
    def __init__(self) -> None:
        self.placed_orders: list[dict[str, Any]] = []
        self.preference_calls: list[dict[str, Any]] = []
        self.saved_analysis_calls: list[dict[str, Any]] = []
        self.saved_strategy_backtest_interpretation_calls: list[dict[str, Any]] = []
        self.portfolio_health_calls: list[dict[str, Any]] = []
        self.strategy_backtest_calls: list[dict[str, Any]] = []
        self.blocked_order_codes: set[str] = set()
        self.order_result_overrides: dict[str, dict[str, Any]] = {}

    async def get_runtime_account_context(self, *, owner_user_id: int, refresh: bool = True):
        return await self.get_account_state(owner_user_id=owner_user_id, refresh=refresh)

    async def get_account_state(self, *, owner_user_id: int, refresh: bool = True):
        return {
            "simulation_account": {
                "is_bound": True,
                "is_verified": True,
                "broker_account_id": 7,
            },
            "account_state": {
                "broker_account_id": 7,
                "provider_code": "backtrader_local",
                "provider_name": "Backtrader Local Sim",
                "account_uid": "bt-u1",
                "account_display_name": "u1",
                "snapshot_at": "2026-04-02T09:30:00",
                "data_source": "cache",
                "positions": [
                    {
                        "code": "600519",
                        "stock_name": "贵州茅台",
                        "quantity": 200,
                        "available_qty": 100,
                        "avg_cost": 1600.0,
                        "last_price": 1680.0,
                        "market_value": 336000.0,
                    },
                ],
                "available_cash": 100000.0,
                "total_market_value": 336000.0,
                "total_asset": 436000.0,
                "order_count": 2,
                "trade_count": 1,
                "today_order_count": 2,
                "today_trade_count": 1,
            },
            "runtime_context": {
                "broker_account_id": 7,
                "provider_code": "backtrader_local",
                "provider_name": "Backtrader Local Sim",
                "account_uid": "bt-u1",
                "account_display_name": "u1",
                "snapshot_at": "2026-04-02T09:30:00",
                "data_source": "cache",
                "summary": {
                    "cash": 100000.0,
                    "initial_capital": 100000.0,
                    "total_market_value": 336000.0,
                    "total_asset": 436000.0,
                },
                "positions": [
                    {
                        "code": "600519",
                        "quantity": 200,
                        "available_qty": 100,
                        "market_value": 336000.0,
                    },
                ],
            },
        }

    async def get_portfolio_health(self, *, owner_user_id: int, refresh: bool = True):
        self.portfolio_health_calls.append(
            {
                "owner_user_id": owner_user_id,
                "refresh": refresh,
            }
        )
        return {
            "portfolio_health": {
                "broker_account_id": 7,
                "provider_code": "backtrader_local",
                "provider_name": "Backtrader Local Sim",
                "account_uid": "bt-u1",
                "account_display_name": "u1",
                "snapshot_at": "2026-04-02T09:30:00",
                "data_source": "cache",
                "positions": [
                    {
                        "code": "600519",
                        "stock_name": "贵州茅台",
                        "industry_name": "白酒",
                        "quantity": 200,
                        "available_qty": 100,
                        "avg_cost": 1600.0,
                        "last_price": 1680.0,
                        "market_value": 336000.0,
                        "weight_pct": 62.77,
                        "unrealized_pnl": 16000.0,
                        "unrealized_return_pct": 5.0,
                    },
                    {
                        "code": "300750",
                        "stock_name": "宁德时代",
                        "industry_name": "新能源电池",
                        "quantity": 500,
                        "available_qty": 500,
                        "avg_cost": 180.0,
                        "last_price": 198.52,
                        "market_value": 99260.0,
                        "weight_pct": 18.55,
                        "unrealized_pnl": 9260.0,
                        "unrealized_return_pct": 10.29,
                    },
                ],
                "available_cash": 100000.0,
                "total_market_value": 435260.0,
                "total_asset": 535260.0,
                "today_order_count": 2,
                "today_trade_count": 1,
                "metrics": {
                    "total_return_pct": 9.42,
                    "total_pnl": 37260.0,
                    "daily_pnl": 6520.0,
                    "realized_pnl": 12000.0,
                    "unrealized_pnl": 25260.0,
                    "max_drawdown_pct": 12.6,
                    "sharpe_ratio": 1.18,
                    "win_rate_pct": 66.67,
                    "cash_ratio_pct": 18.68,
                    "invested_ratio_pct": 81.32,
                    "top1_position_pct": 62.77,
                    "top3_position_pct": 81.32,
                    "position_count": 2,
                },
                "exposures": {
                    "by_industry": [
                        {
                            "industry_name": "白酒",
                            "market_value": 336000.0,
                            "weight_pct": 62.77,
                            "invested_weight_pct": 77.19,
                            "stock_count": 1,
                        },
                        {
                            "industry_name": "新能源电池",
                            "market_value": 99260.0,
                            "weight_pct": 18.55,
                            "invested_weight_pct": 22.81,
                            "stock_count": 1,
                        },
                    ],
                    "by_stock": [
                        {"code": "600519", "stock_name": "贵州茅台", "market_value": 336000.0, "weight_pct": 62.77},
                        {"code": "300750", "stock_name": "宁德时代", "market_value": 99260.0, "weight_pct": 18.55},
                    ],
                },
                "diagnostics": {
                    "health_score": 74,
                    "health_level": "watch",
                    "rebalancing_needed": True,
                    "alerts": [
                        {
                            "code": "single_stock_concentration",
                            "severity": "high",
                            "message": "贵州茅台当前仓位约 62.77%，单票集中度偏高。",
                        },
                        {
                            "code": "industry_overweight",
                            "severity": "medium",
                            "message": "白酒行业权重约 62.77%，行业集中度偏高。",
                        },
                    ],
                    "suggestions": [
                        "白酒行业仓位明显偏重，可考虑逐步降到 40% 以下。",
                        "适当保留现金或分散到低相关板块，降低组合回撤波动。",
                    ],
                },
            }
        }

    async def get_user_preferences(self, *, owner_user_id: int, session_overrides: dict[str, Any] | None = None):
        overrides = dict(session_overrides or {})
        self.preference_calls.append(overrides)
        risk_profile = str(overrides.get("riskProfile") or "conservative")
        response_style = str(overrides.get("responseStyle") or "concise_factual")
        return {
            "persistent": {
                "trading": {
                    "riskProfile": "conservative",
                    "analysisStrategy": "auto",
                    "maxSingleTradeAmount": None,
                    "positionMaxPct": 30,
                    "stopLossPct": 8,
                    "takeProfitPct": 15,
                },
                "chat": {
                    "executionPolicy": "auto_execute_if_condition_met",
                    "confirmationShortcutsEnabled": True,
                    "followupFocusResolutionEnabled": True,
                    "responseStyle": "concise_factual",
                },
            },
            "session_overrides": overrides,
            "effective": {
                "trading": {
                    "riskProfile": risk_profile,
                    "analysisStrategy": str(overrides.get("analysisStrategy") or "auto"),
                    "maxSingleTradeAmount": overrides.get("maxSingleTradeAmount"),
                    "positionMaxPct": 30,
                    "stopLossPct": 8,
                    "takeProfitPct": 15,
                },
                "chat": {
                    "executionPolicy": "auto_execute_if_condition_met",
                    "confirmationShortcutsEnabled": True,
                    "followupFocusResolutionEnabled": True,
                    "responseStyle": response_style,
                },
            },
            "source": {
                "trading": {
                    "riskProfile": "session" if "riskProfile" in overrides else "profile",
                    "analysisStrategy": "session" if "analysisStrategy" in overrides else "profile",
                    "maxSingleTradeAmount": "session" if "maxSingleTradeAmount" in overrides else "profile",
                    "positionMaxPct": "profile",
                    "stopLossPct": "profile",
                    "takeProfitPct": "profile",
                },
                "chat": {
                    "executionPolicy": "profile",
                    "confirmationShortcutsEnabled": "profile",
                    "followupFocusResolutionEnabled": "profile",
                    "responseStyle": "profile",
                },
            },
        }

    async def get_analysis_history(self, *, owner_user_id: int, stock_codes: list[str] | None = None, limit: int = 5):
        return {"total": 0, "items": []}

    async def get_backtest_summary(self, *, owner_user_id: int, stock_codes: list[str] | None = None, limit: int = 6):
        return {"total": 0, "items": []}

    async def run_strategy_backtest(
        self,
        *,
        owner_user_id: int,
        code: str,
        start_date: str,
        end_date: str,
        strategies: list[dict[str, Any]],
        initial_capital: float | None = None,
        commission_rate: float | None = None,
        slippage_bps: float | None = None,
    ):
        self.strategy_backtest_calls.append(
            {
                "owner_user_id": owner_user_id,
                "code": code,
                "start_date": start_date,
                "end_date": end_date,
                "strategies": [dict(item) for item in strategies],
                "initial_capital": initial_capital,
                "commission_rate": commission_rate,
                "slippage_bps": slippage_bps,
            }
        )
        items: list[dict[str, Any]] = []
        for index, strategy in enumerate(strategies):
            template_code = str(strategy.get("template_code") or "")
            strategy_name = str(strategy.get("strategy_name") or template_code or f"策略{index + 1}")
            total_return = {
                "macd_cross": 18.6,
                "rsi_threshold": 12.4,
                "ma_cross": 9.8,
            }.get(template_code, 8.0 + index)
            max_drawdown = {
                "macd_cross": -9.4,
                "rsi_threshold": -12.1,
                "ma_cross": -8.2,
            }.get(template_code, -10.0)
            sharpe_ratio = {
                "macd_cross": 1.34,
                "rsi_threshold": 0.96,
                "ma_cross": 0.88,
            }.get(template_code, 0.7)
            items.append(
                {
                    "strategy_id": strategy.get("strategy_id"),
                    "run_id": 100 + index,
                    "strategy_code": template_code,
                    "strategy_name": strategy_name,
                    "template_code": template_code,
                    "template_name": strategy_name,
                    "strategy_version": "v1",
                    "params": dict(strategy.get("params") or {}),
                    "metrics": {
                        "total_return_pct": total_return,
                        "benchmark_return_pct": 10.5,
                        "excess_return_pct": round(total_return - 10.5, 2),
                        "max_drawdown_pct": max_drawdown,
                        "sharpe_ratio": sharpe_ratio,
                        "total_trades": 6 + index,
                        "win_rate_pct": 58.3,
                    },
                    "benchmark": {
                        "initial_equity": float(initial_capital or 100000.0),
                        "final_equity": 110500.0,
                        "total_return_pct": 10.5,
                    },
                    "trades": [],
                    "equity": [
                        {"trade_date": start_date, "equity": float(initial_capital or 100000.0), "drawdown_pct": 0.0, "benchmark_equity": float(initial_capital or 100000.0)},
                        {"trade_date": end_date, "equity": float(initial_capital or 100000.0) * (1 + total_return / 100), "drawdown_pct": max_drawdown, "benchmark_equity": 110500.0},
                    ],
                }
            )
        return {
            "run_group_id": 901,
            "code": code,
            "requested_range": {"start_date": start_date, "end_date": end_date},
            "effective_range": {"start_date": start_date, "end_date": end_date},
            "created_at": "2026-04-05T10:00:00+08:00",
            "items": items,
        }

    async def save_analysis_records(
        self,
        *,
        owner_user_id: int,
        session_id: str,
        assistant_message_id: int,
        analysis_result: dict[str, Any],
        news_items_by_stock: dict[str, list[dict[str, Any]]] | None = None,
    ):
        payload = {
            "owner_user_id": owner_user_id,
            "session_id": session_id,
            "assistant_message_id": assistant_message_id,
            "analysis_result": dict(analysis_result),
            "news_items_by_stock": dict(news_items_by_stock or {}),
        }
        self.saved_analysis_calls.append(payload)
        return {"saved_count": len(analysis_result.get("stocks") or []), "skipped_count": 0, "items": []}

    async def save_strategy_backtest_interpretation(
        self,
        *,
        owner_user_id: int,
        run_group_id: int,
        items: list[dict[str, Any]],
    ):
        self.saved_strategy_backtest_interpretation_calls.append(
            {
                "owner_user_id": owner_user_id,
                "run_group_id": run_group_id,
                "items": [dict(item) for item in items],
            }
        )
        return {"run_group_id": run_group_id, "saved_count": len(items), "ai_interpretation_status": "completed"}

    async def place_simulated_order(self, *, owner_user_id: int, session_id: str, candidate_order: dict[str, Any]):
        self.placed_orders.append(dict(candidate_order))
        code = str(candidate_order.get("code") or "").strip()
        if code in self.blocked_order_codes:
            return build_outside_trading_session_order_result(candidate_order)
        if code in self.order_result_overrides:
            payload = dict(self.order_result_overrides[code])
            payload.setdefault("candidate_order", dict(candidate_order))
            return payload
        return {
            "status": "filled",
            "candidate_order": candidate_order,
            "order": {
                "provider_status": "filled",
                "stock_code": candidate_order.get("code"),
                "direction": candidate_order.get("action"),
                "quantity": candidate_order.get("quantity"),
            },
        }


class FakeAgentService:
    def __init__(
        self,
        *,
        no_candidate_codes: set[str] | None = None,
        sentiment_scores: dict[str, int] | None = None,
    ) -> None:
        self.no_candidate_codes = no_candidate_codes or set()
        self.sentiment_scores = sentiment_scores or STOCK_SCORES

    def run_once(self, stock_codes, **_kwargs):
        today = date(2026, 4, 2)
        results: list[StockAgentResult] = []
        positions: list[dict[str, Any]] = []
        for index, code in enumerate(stock_codes):
            price = STOCK_PRICES.get(code, 10.0 + index)
            name = STOCK_NAMES.get(code, code)
            sentiment = int(self.sentiment_scores.get(code, 60 - index))
            has_candidate = code not in self.no_candidate_codes
            action = "buy" if has_candidate else "hold"
            traded_qty = 100 * (index + 1) if has_candidate else 0
            target_weight = 0.2 + 0.1 * index if has_candidate else 0.0
            result = StockAgentResult(
                code=code,
                data=DataAgentOutput(
                    code=code,
                    trade_date=today,
                    state=AgentState.READY,
                    analysis_context={"today": {"close": price, "pct_chg": 1.23 + index}, "stock_name": name},
                    realtime_quote={"name": name, "price": price, "change_pct": 1.23 + index},
                ),
                signal=SignalAgentOutput(
                    code=code,
                    trade_date=today,
                    state=AgentState.READY,
                    operation_advice="买入" if has_candidate else "观望",
                    sentiment_score=sentiment,
                    trend_signal="BUY" if has_candidate else "HOLD",
                    ai_payload={
                        "analysis_summary": f"{name} 当前维持 {'买入' if has_candidate else '观望'}判断。",
                        "news_summary": f"{name} 近期有 2 条相关情报。",
                        "news_items": [
                            {
                                "title": f"{name} 新闻 1",
                                "snippet": f"{name} 相关新闻摘要 1",
                                "url": f"https://example.com/{code}/news-1",
                                "source": "example.com",
                                "published_date": "2026-04-02T09:00:00+08:00",
                                "provider": "mock_search",
                                "dimension": "news",
                                "query": f"{name} 最新新闻",
                            },
                            {
                                "title": f"{name} 新闻 2",
                                "snippet": f"{name} 相关新闻摘要 2",
                                "url": f"https://example.com/{code}/news-2",
                                "source": "example.com",
                                "published_date": "2026-04-01T21:00:00+08:00",
                                "provider": "mock_search",
                                "dimension": "announcement",
                                "query": f"{name} 公告",
                            },
                        ],
                    },
                ),
                risk=RiskAgentOutput(
                    code=code,
                    trade_date=today,
                    state=AgentState.READY,
                    target_weight=target_weight,
                    target_notional=30000 + index * 5000 if has_candidate else 0,
                    current_price=price,
                    risk_level="low" if has_candidate else "medium",
                    execution_allowed=has_candidate,
                    risk_flags=["position_cap_ok"] if has_candidate else ["wait_for_signal"],
                ),
                execution=ExecutionAgentOutput(
                    code=code,
                    trade_date=today,
                    state=AgentState.READY,
                    action=action,
                    reason="intent_generated" if has_candidate else "wait_signal",
                    traded_qty=traded_qty,
                    target_qty=traded_qty,
                    fill_price=round(price * 1.0005, 2) if has_candidate else None,
                    final_order={
                        "code": code,
                        "action": "buy",
                        "quantity": traded_qty,
                        "target_qty": traded_qty,
                        "price": round(price * 1.0005, 2),
                    } if has_candidate else None,
                    proposed_order={
                        "code": code,
                        "action": "buy",
                        "quantity": traded_qty,
                        "target_qty": traded_qty,
                        "price": round(price * 1.0005, 2),
                    } if has_candidate else None,
                    proposal_state="proposed" if has_candidate else "blocked",
                    proposal_reason="intent_generated" if has_candidate else "wait_signal",
                ),
            )
            results.append(result)
            if has_candidate:
                positions.append({"code": code, "quantity": traded_qty})

        return AgentRunResult(
            run_id="run-chat-1",
            mode="once",
            started_at=datetime(2026, 4, 2, 9, 30, 0),
            ended_at=datetime(2026, 4, 2, 9, 30, 2),
            trade_date=today,
            results=results,
            account_snapshot={"cash": 70000, "positions": positions},
        )


class ExtraCandidateAgentService(FakeAgentService):
    def __init__(self, *, extra_code: str = "000001") -> None:
        super().__init__()
        self.extra_code = extra_code

    def run_once(self, stock_codes, **kwargs):
        expanded_codes = list(stock_codes)
        if self.extra_code not in expanded_codes:
            expanded_codes.append(self.extra_code)
        return super().run_once(expanded_codes, **kwargs)


class FakeBacktestInterpretationService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def interpret(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(payload))
        items: list[dict[str, Any]] = []
        for row in payload.get("items") or []:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or "策略").strip()
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            total_return = metrics.get("total_return_pct")
            items.append(
                {
                    "item_key": row.get("item_key"),
                    "status": "ready",
                    "verdict": "表现较强" if float(total_return or 0) >= 15 else "表现中等",
                    "summary": f"{label} 在这段历史里收益和回撤匹配度尚可，适合继续观察其稳定性。",
                }
            )
        return {"items": items}


class AgentChatServiceTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["DATABASE_PATH"] = os.path.join(self.temp_dir.name, "agent_chat_service.db")
        os.environ["DATABASE_URL"] = ""
        os.environ["AGENT_SERVICE_AUTH_TOKEN"] = "test-token"
        Config.reset_instance()
        DatabaseManager.reset_instance()
        self.db = DatabaseManager.get_instance()
        self.repo = AgentChatRepository(self.db)
        self.backend_client = FakeBackendClient()
        self.agent_service = FakeAgentService()
        self.backtest_interpretation_service = FakeBacktestInterpretationService()
        self.service = AgentChatService(
            config=Config.get_instance(),
            db_manager=self.db,
            chat_repo=self.repo,
            agent_service=self.agent_service,
            backend_client=self.backend_client,
            backtest_interpretation_service=self.backtest_interpretation_service,
            analyzer=FakePlannerAnalyzer(),
        )

    def build_service(self, analyzer, *, analyzer_factory=None) -> AgentChatService:
        return AgentChatService(
            config=Config.get_instance(),
            db_manager=self.db,
            chat_repo=self.repo,
            agent_service=self.agent_service,
            backend_client=self.backend_client,
            backtest_interpretation_service=self.backtest_interpretation_service,
            analyzer=analyzer,
            analyzer_factory=analyzer_factory,
            board_catalog_provider=fake_board_catalog_provider,
            board_constituents_provider=fake_board_constituents_provider,
        )

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        os.environ.pop("DATABASE_URL", None)
        self.temp_dir.cleanup()

    async def test_analysis_chat_creates_candidate_order(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "analysis")
        self.assertEqual(payload["structured_result"]["intent_source"], "llm")
        self.assertEqual(payload["candidate_orders"][0]["code"], "600519")
        self.assertIn("综合判断", payload["content"])
        self.assertEqual(len(self.backend_client.saved_analysis_calls), 1)
        self.assertEqual(
            self.backend_client.saved_analysis_calls[0]["analysis_result"]["stocks"][0]["code"],
            "600519",
        )
        self.assertEqual(
            [item["title"] for item in self.backend_client.saved_analysis_calls[0]["news_items_by_stock"]["600519"]],
            ["贵州茅台 新闻 1", "贵州茅台 新闻 2"],
        )
        self.assertEqual(
            self.backend_client.saved_analysis_calls[0]["analysis_result"]["stocks"][0]["raw"]["signal"]["ai_payload"]["news_items"][0]["url"],
            "https://example.com/600519/news-1",
        )

        detail = self.service.get_session_detail(1, payload["session_id"])
        assert detail is not None
        self.assertEqual(detail["context"]["conversation_state"]["focus_stocks"], ["600519"])
        self.assertEqual(detail["context"]["conversation_state"]["current_task"]["intent"], "analysis")

    async def test_account_request_only_loads_account_state(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "看看我当前持仓和资金",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "account")
        self.assertEqual(payload["structured_result"]["loaded_context_keys"], ["account_state"])
        self.assertEqual(payload["candidate_orders"], [])
        self.assertIn("当前可用资金约为", payload["content"])
        self.assertIn("今日委托 2 次", payload["content"])
        self.assertEqual(self.backend_client.saved_analysis_calls, [])

    async def test_portfolio_health_request_builds_report_and_candidate_orders(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "我的仓位健康吗？",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "portfolio_health")
        self.assertEqual(payload["structured_result"]["intent_source"], "rule")
        self.assertIn("account_state", payload["structured_result"]["loaded_context_keys"])
        self.assertIn("portfolio_health", payload["structured_result"]["loaded_context_keys"])
        self.assertIn("effective_user_preferences", payload["structured_result"]["loaded_context_keys"])
        self.assertEqual(len(self.backend_client.portfolio_health_calls), 1)
        self.assertIn("组合结论", payload["content"])
        self.assertIn("行业分布", payload["content"])
        self.assertIn("最大回撤 12.60%", payload["content"])
        self.assertIn("夏普比率 1.18", payload["content"])
        self.assertIn("白酒行业权重约 62.77%", payload["content"])
        self.assertEqual([item["code"] for item in payload["candidate_orders"]], ["600519", "300750"])
        self.assertIsNotNone(payload["structured_result"]["analysis"])
        self.assertEqual(len(self.backend_client.saved_analysis_calls), 1)
        self.assertEqual(
            [item["code"] for item in self.backend_client.saved_analysis_calls[0]["analysis_result"]["stocks"]],
            ["600519", "300750"],
        )
        self.assertEqual(
            sorted(self.backend_client.saved_analysis_calls[0]["news_items_by_stock"].keys()),
            ["300750", "600519"],
        )
        self.assertEqual(self.backend_client.placed_orders, [])

        detail = self.service.get_session_detail(1, payload["session_id"])
        assert detail is not None
        self.assertEqual(detail["context"]["conversation_state"]["current_task"]["intent"], "portfolio_health")
        self.assertEqual(
            [item["code"] for item in detail["context"]["conversation_state"]["pending_actions"]],
            ["600519", "300750"],
        )

    async def test_portfolio_rebalance_request_auto_executes_current_holdings(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "分析一下持仓，进行调整",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "simulation_order_filled")
        self.assertEqual(payload["structured_result"]["intent"], "portfolio_health")
        self.assertIn("system_state", payload["structured_result"]["loaded_context_keys"])
        self.assertIn("自动执行结果", payload["content"])
        self.assertEqual([item["code"] for item in payload["candidate_orders"]], ["600519", "300750"])
        autonomous_execution = payload["structured_result"]["autonomous_execution"]
        self.assertTrue(autonomous_execution["authorized"])
        self.assertTrue(autonomous_execution["executed"])
        self.assertEqual(autonomous_execution["execution_scope"], "all")
        self.assertEqual(autonomous_execution["executed_count"], 2)
        self.assertEqual([item["code"] for item in self.backend_client.placed_orders], ["600519", "300750"])

        detail = self.service.get_session_detail(1, payload["session_id"])
        assert detail is not None
        self.assertEqual(detail["context"]["conversation_state"]["pending_actions"], [])

    async def test_portfolio_rebalance_request_blocked_outside_trading_session_keeps_pending_actions(self):
        self.backend_client.blocked_order_codes.update({"600519", "300750"})

        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我调仓",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["structured_result"]["intent"], "portfolio_health")
        self.assertEqual(payload["execution_result"]["reason"], "outside_trading_session")
        self.assertIn("当前处于非交易时段", payload["content"])
        autonomous_execution = payload["structured_result"]["autonomous_execution"]
        self.assertFalse(autonomous_execution["executed"])
        self.assertEqual(autonomous_execution["reason"], "outside_trading_session")
        self.assertIn("当前处于非交易时段", autonomous_execution["gate_message"])

        detail = self.service.get_session_detail(1, payload["session_id"])
        assert detail is not None
        self.assertEqual(
            [item["code"] for item in detail["context"]["conversation_state"]["pending_actions"]],
            ["600519", "300750"],
        )

    async def test_portfolio_rebalance_request_without_candidate_orders_stays_analysis_only(self):
        backend_client = FakeBackendClient()
        service = AgentChatService(
            config=Config.get_instance(),
            db_manager=self.db,
            chat_repo=self.repo,
            agent_service=FakeAgentService(no_candidate_codes={"600519", "300750"}),
            backend_client=backend_client,
            backtest_interpretation_service=self.backtest_interpretation_service,
            analyzer=FakePlannerAnalyzer(),
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "根据持仓自己决定买卖",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "portfolio_health")
        self.assertEqual(payload["candidate_orders"], [])
        self.assertIsNone(payload["execution_result"])
        autonomous_execution = payload["structured_result"]["autonomous_execution"]
        self.assertFalse(autonomous_execution["executed"])
        self.assertEqual(autonomous_execution["reason"], "no_candidate_orders")
        self.assertEqual(backend_client.placed_orders, [])

    async def test_portfolio_rebalance_request_filters_out_non_holding_candidate_orders(self):
        backend_client = FakeBackendClient()
        service = AgentChatService(
            config=Config.get_instance(),
            db_manager=self.db,
            chat_repo=self.repo,
            agent_service=ExtraCandidateAgentService(),
            backend_client=backend_client,
            backtest_interpretation_service=self.backtest_interpretation_service,
            analyzer=FakePlannerAnalyzer(),
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "分析一下持仓，进行调整",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "simulation_order_filled")
        self.assertEqual([item["code"] for item in payload["candidate_orders"]], ["600519", "300750"])
        self.assertEqual([item["code"] for item in backend_client.placed_orders], ["600519", "300750"])
        self.assertEqual(len(backend_client.saved_analysis_calls), 1)
        saved_stocks = {
            item["code"]: item
            for item in backend_client.saved_analysis_calls[0]["analysis_result"]["stocks"]
        }
        self.assertIn("000001", saved_stocks)
        self.assertIsNone(saved_stocks["000001"]["candidate_order"])

    async def test_strategy_backtest_request_reuses_focus_stock_and_interprets_result(self):
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "runtime_config": build_runtime_config(),
            }
        )

        second = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "如果我在每次MACD金叉时买入，过去一年收益怎样",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(second["status"], "analysis_only")
        self.assertEqual(second["structured_result"]["intent"], "backtest")
        self.assertEqual(second["structured_result"]["backtest_mode"], "strategy_run")
        self.assertIn("account_state", second["structured_result"]["loaded_context_keys"])
        self.assertEqual(len(self.backend_client.strategy_backtest_calls), 1)
        backtest_call = self.backend_client.strategy_backtest_calls[0]
        self.assertEqual(backtest_call["code"], "600519")
        self.assertEqual(backtest_call["strategies"][0]["template_code"], "macd_cross")
        self.assertEqual(backtest_call["strategies"][0]["params"], {"macdFast": 12, "macdSlow": 26, "macdSignal": 9})
        self.assertEqual(backtest_call["end_date"], date.today().isoformat())
        self.assertIn("策略回测结论", second["content"])
        self.assertIn("MACD 金叉", second["content"])
        self.assertIn("AI 解读", second["content"])
        self.assertEqual(len(self.backtest_interpretation_service.calls), 1)
        self.assertEqual(len(self.backend_client.saved_strategy_backtest_interpretation_calls), 1)
        persisted_call = self.backend_client.saved_strategy_backtest_interpretation_calls[0]
        self.assertEqual(persisted_call["owner_user_id"], 1)
        self.assertEqual(persisted_call["run_group_id"], 901)
        self.assertEqual([item["item_key"] for item in persisted_call["items"]], ["strategy-run-100"])
        self.assertEqual(persisted_call["items"][0]["status"], "ready")

    async def test_strategy_backtest_supports_generic_day_span(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "如果我在比亚迪每次MACD金叉买入，过去300天收益怎样",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "backtest")
        self.assertEqual(len(self.backend_client.strategy_backtest_calls), 1)
        backtest_call = self.backend_client.strategy_backtest_calls[0]
        self.assertEqual(backtest_call["code"], "002594")
        self.assertEqual(backtest_call["strategies"][0]["template_code"], "macd_cross")
        self.assertEqual(backtest_call["start_date"], (date.today() - timedelta(days=300)).isoformat())
        self.assertEqual(backtest_call["end_date"], date.today().isoformat())
        self.assertIn("策略回测结论", payload["content"])

    async def test_strategy_backtest_compare_request_runs_multiple_templates(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "用 600519 对比一下 MACD 金叉 和 RSI 超卖策略，看看过去一年谁更好",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "backtest")
        self.assertEqual(payload["structured_result"]["backtest_mode"], "strategy_run")
        self.assertEqual(len(self.backend_client.strategy_backtest_calls), 1)
        strategy_codes = [item["template_code"] for item in self.backend_client.strategy_backtest_calls[0]["strategies"]]
        self.assertEqual(strategy_codes, ["macd_cross", "rsi_threshold"])
        self.assertIn("策略对比", payload["content"])

    async def test_strategy_backtest_supports_combined_rule_dsl(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "用 600519 看下 MACD 金叉且 RSI<30 时买入，跌破 5 日线止损，过去一年收益怎样",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "backtest")
        self.assertEqual(len(self.backend_client.strategy_backtest_calls), 1)
        strategy = self.backend_client.strategy_backtest_calls[0]["strategies"][0]
        self.assertEqual(strategy["template_code"], "rule_dsl")
        self.assertEqual(strategy["params"]["entry"]["operator"], "and")
        self.assertEqual(
            [item["kind"] for item in strategy["params"]["entry"]["conditions"]],
            ["macd_cross", "rsi_threshold"],
        )
        self.assertEqual(strategy["params"]["exit"]["conditions"][0]["kind"], "price_ma_relation")
        self.assertEqual(strategy["params"]["exit"]["conditions"][0]["maWindow"], 5)
        self.assertIn("策略回测结论", payload["content"])

    async def test_strategy_backtest_supports_embedded_stock_name_with_combined_rule_dsl(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "如果我在比亚迪每次MACD 金叉且 RSI<30 且跌破 5 日线止损，过去一年收益怎样",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "backtest")
        self.assertEqual(len(self.backend_client.strategy_backtest_calls), 1)
        backtest_call = self.backend_client.strategy_backtest_calls[0]
        self.assertEqual(backtest_call["code"], "002594")
        strategy = backtest_call["strategies"][0]
        self.assertEqual(strategy["template_code"], "rule_dsl")
        self.assertEqual(strategy["params"]["entry"]["operator"], "and")
        self.assertEqual(
            [item["kind"] for item in strategy["params"]["entry"]["conditions"]],
            ["macd_cross", "rsi_threshold"],
        )
        self.assertEqual(strategy["params"]["exit"]["conditions"][0]["kind"], "price_ma_relation")
        self.assertEqual(strategy["params"]["exit"]["conditions"][0]["maWindow"], 5)
        self.assertIn("策略回测结论", payload["content"])

    def test_strategy_backtest_stock_ref_extractor_finds_embedded_name_and_ignores_indicator_tokens(self):
        refs = self.service._extract_strategy_backtest_stock_refs(
            "如果我在比亚迪每次MACD 金叉且 RSI<30 且跌破 5 日线止损，过去一年收益怎样",
        )

        self.assertEqual(refs, ["比亚迪"])
        self.assertNotIn("RSI", refs)
        self.assertNotIn("MACD", refs)

    def test_strategy_backtest_window_supports_generic_relative_spans(self):
        today = date.today()

        day_window = self.service._extract_strategy_backtest_window("如果我在比亚迪每次MACD金叉买入，过去300天收益怎样")
        self.assertEqual(day_window["start_date"], (today - timedelta(days=300)).isoformat())
        self.assertEqual(day_window["end_date"], today.isoformat())
        self.assertEqual(day_window["window_label"], "过去300天")

        week_window = self.service._extract_strategy_backtest_window("最近8周回测收益如何")
        self.assertEqual(week_window["start_date"], (today - timedelta(days=56)).isoformat())
        self.assertEqual(week_window["end_date"], today.isoformat())
        self.assertEqual(week_window["window_label"], "过去8周")

        month_window = self.service._extract_strategy_backtest_window("近15个月表现如何")
        self.assertEqual(month_window["start_date"], _subtract_calendar_months_for_test(today, 15).isoformat())
        self.assertEqual(month_window["end_date"], today.isoformat())
        self.assertEqual(month_window["window_label"], "过去15个月")

        year_window = self.service._extract_strategy_backtest_window("过去两年回测收益怎样")
        self.assertEqual(year_window["start_date"], _subtract_calendar_months_for_test(today, 24).isoformat())
        self.assertEqual(year_window["end_date"], today.isoformat())
        self.assertEqual(year_window["window_label"], "过去两年")

    async def test_follow_up_order_executes_single_candidate(self):
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "runtime_config": build_runtime_config(),
            }
        )

        second = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "去下单吧",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(second["status"], "simulation_order_filled")
        self.assertEqual(second["structured_result"]["intent"], "order_followup_single")
        self.assertEqual(second["execution_result"]["candidate_order"]["code"], "600519")
        self.assertEqual(second["structured_result"]["stage_memory"]["execution"]["result"]["status"], "filled")

    async def test_composite_request_analyzes_then_executes(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "分析 600519，如果风险低就买100股",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["structured_result"]["intent"], "analysis_then_execute")
        self.assertEqual(payload["status"], "simulation_order_filled")
        self.assertEqual(payload["candidate_orders"][0]["quantity"], 100)
        self.assertTrue(payload["structured_result"]["autonomous_execution"]["gate_passed"])
        self.assertIn("system_state", payload["structured_result"]["loaded_context_keys"])
        self.assertIn("account_state", payload["structured_result"]["loaded_context_keys"])

    async def test_ambiguous_confirmation_with_multiple_pending_actions_clarifies(self):
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 和 000001",
                "runtime_config": build_runtime_config(),
            }
        )

        second = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "确认",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(second["status"], "blocked")
        self.assertEqual(second["structured_result"]["intent"], "clarify")
        self.assertIn("多笔待确认动作", second["content"])

    async def test_follow_up_specific_code_executes_single_candidate(self):
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 和 300750",
                "runtime_config": build_runtime_config(),
            }
        )

        second = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "下 300750 的单",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(second["status"], "simulation_order_filled")
        self.assertEqual(second["execution_result"]["candidate_order"]["code"], "300750")

    async def test_retry_uses_stage_memory_and_session_memory(self):
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "runtime_config": build_runtime_config(),
            }
        )

        second = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "再试一次",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(second["structured_result"]["intent"], "analysis")
        self.assertIn("session_memory", second["structured_result"]["loaded_context_keys"])
        self.assertIn("stage_memory", second["structured_result"]["loaded_context_keys"])
        self.assertEqual(second["structured_result"]["conversation_state"]["focus_stocks"], ["600519"])

    async def test_session_preference_overrides_persist_inside_conversation_state(self):
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "这轮激进一点，再分析 600519",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(first["structured_result"]["effective_preferences"]["effective"]["trading"]["riskProfile"], "aggressive")
        self.assertEqual(first["structured_result"]["effective_preferences"]["source"]["trading"]["riskProfile"], "session")
        self.assertEqual(self.backend_client.preference_calls[-1]["riskProfile"], "aggressive")

        detail = self.service.get_session_detail(1, first["session_id"])
        assert detail is not None
        self.assertEqual(
            detail["context"]["conversation_state"]["session_preference_overrides"]["riskProfile"],
            "aggressive",
        )

    async def test_invalid_planner_output_blocks_without_rule_fallback(self):
        analyzer = FakeInvalidPlannerAnalyzer()
        service = AgentChatService(
            config=Config.get_instance(),
            db_manager=self.db,
            chat_repo=self.repo,
            agent_service=self.agent_service,
            backend_client=self.backend_client,
            analyzer=analyzer,
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "分析 600519，如果风险低就买100股",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["structured_result"]["intent"], "clarify")
        self.assertEqual(payload["structured_result"]["intent_source"], "llm")
        self.assertGreaterEqual(analyzer.calls, 2)
        self.assertEqual(payload["candidate_orders"], [])

    async def test_structured_summary_output_falls_back_to_template(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "context": {"agent_chat_preferences": {"responseStyle": "balanced"}},
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertNotIn("```", payload["content"])
        self.assertIn("综合判断", payload["content"])
        self.assertIn("600519", payload["content"])
        self.assertEqual(payload["render_source"], "fallback_template")
        self.assertEqual(payload["render_fallback_reason"], "summary_guard_rejected")

    async def test_runtime_llm_overrides_chat_planner_and_analysis_summary(self):
        default_analyzer = TrackingPlannerAnalyzer(label="default", summary_text="default analysis summary")
        runtime_analyzer = TrackingPlannerAnalyzer(label="runtime", summary_text="runtime analysis summary")
        runtime_factory_calls: list[Any] = []

        def analyzer_factory(runtime_llm):
            runtime_factory_calls.append(runtime_llm)
            return runtime_analyzer

        service = self.build_service(default_analyzer, analyzer_factory=analyzer_factory)

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "context": {"agent_chat_preferences": {"responseStyle": "balanced"}},
                "runtime_config": build_runtime_config_with_llm(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["content"], "runtime analysis summary")
        self.assertEqual(payload["render_source"], "llm_summary")
        self.assertEqual(payload["render_model"], "Pro/zai-org/GLM-5")
        self.assertEqual(len(runtime_factory_calls), 1)
        self.assertEqual(runtime_factory_calls[0].base_url, "https://api.siliconflow.cn/v1")
        self.assertEqual(runtime_factory_calls[0].model, "Pro/zai-org/GLM-5")
        self.assertTrue(any("Agent问股 的聊天主控规划器" in prompt for prompt in runtime_analyzer.calls))
        self.assertTrue(any("回复风格：balanced" in prompt for prompt in runtime_analyzer.calls))
        self.assertFalse(default_analyzer.calls)

    async def test_runtime_llm_overrides_analysis_then_execute_summary(self):
        default_analyzer = TrackingPlannerAnalyzer(label="default", summary_text="default execute summary")
        runtime_analyzer = TrackingPlannerAnalyzer(label="runtime", summary_text="runtime execute summary")

        service = self.build_service(
            default_analyzer,
            analyzer_factory=lambda runtime_llm: runtime_analyzer,
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "分析 600519，如果风险低就买100股",
                "context": {"agent_chat_preferences": {"responseStyle": "balanced"}},
                "runtime_config": build_runtime_config_with_llm(),
            }
        )

        self.assertEqual(payload["status"], "simulation_order_filled")
        self.assertEqual(payload["content"], "runtime execute summary")
        self.assertEqual(payload["render_source"], "llm_summary")
        self.assertGreaterEqual(len(runtime_analyzer.calls), 2)
        self.assertFalse(default_analyzer.calls)

    async def test_chat_without_runtime_llm_falls_back_to_default_analyzer(self):
        default_analyzer = TrackingPlannerAnalyzer(label="default", summary_text="default analysis summary")
        runtime_factory_calls: list[Any] = []

        def analyzer_factory(runtime_llm):
            runtime_factory_calls.append(runtime_llm)
            return TrackingPlannerAnalyzer(label="runtime", summary_text="runtime analysis summary")

        service = self.build_service(default_analyzer, analyzer_factory=analyzer_factory)

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "context": {"agent_chat_preferences": {"responseStyle": "balanced"}},
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["content"], "default analysis summary")
        self.assertEqual(payload["render_source"], "llm_summary")
        self.assertFalse(runtime_factory_calls)
        self.assertTrue(any("Agent问股 的聊天主控规划器" in prompt for prompt in default_analyzer.calls))

    async def test_concise_response_style_stays_deterministic_across_repeated_runs(self):
        analyzer = TrackingPlannerAnalyzer(
            label="default",
            summary_text="# 寒武纪(688256)决策仪表盘\n\n## 组合结论\n\n**🟡 持有观望 — 暂不开新仓**",
        )
        service = self.build_service(analyzer)
        contents: list[str] = []
        for index in range(10):
            payload = await service.handle_chat(
                {
                    "owner_user_id": 1,
                    "username": "tester",
                    "session_id": f"session-concise-{index}",
                    "message": "帮我分析一下今天的 600519 行情",
                    "runtime_config": build_runtime_config(),
                }
            )
            contents.append(payload["content"])
            self.assertEqual(payload["render_source"], "llm_summary")
            self.assertIn("决策仪表盘", payload["content"])
        self.assertEqual(len(set(contents)), 1)

    async def test_detailed_response_style_uses_summary_prompt_and_model_output(self):
        analyzer = TrackingPlannerAnalyzer(
            label="default",
            summary_text="# 寒武纪(688256)决策仪表盘\n\n## 组合结论\n\n详细分析正文",
        )
        service = self.build_service(analyzer)

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "context": {"agent_chat_preferences": {"responseStyle": "detailed"}},
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertIn("决策仪表盘", payload["content"])
        self.assertEqual(payload["render_source"], "llm_summary")
        self.assertTrue(any("回复风格：detailed" in prompt for prompt in analyzer.calls))

    async def test_summary_guard_rejects_false_execution_claims(self):
        analyzer = TrackingPlannerAnalyzer(label="default", summary_text="本轮已执行 1 笔模拟盘下单。")
        service = self.build_service(analyzer)

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "context": {"agent_chat_preferences": {"responseStyle": "balanced"}},
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertIn("综合判断", payload["content"])
        self.assertEqual(payload["render_source"], "fallback_template")
        self.assertEqual(payload["render_fallback_reason"], "summary_guard_rejected")

    async def test_summary_exception_falls_back_with_reason(self):
        service = self.build_service(ExplodingSummaryAnalyzer())

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "context": {"agent_chat_preferences": {"responseStyle": "balanced"}},
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertIn("综合判断", payload["content"])
        self.assertEqual(payload["render_source"], "fallback_template")
        self.assertEqual(payload["render_fallback_reason"], "summary_exception")

    async def test_unresolved_chinese_stock_name_clarifies(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "再分析一下不存在科技",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["structured_result"]["intent"], "clarify")
        self.assertIn("不存在科技", payload["content"])

    async def test_dynamic_stock_name_resolution_supports_china_bank(self):
        service = self.build_service(
            FakeMappedPlannerAnalyzer(
                {
                    "分析一下中国银行": FakePlannerAnalyzer._plan(
                        intent="analysis",
                        required_tools=["run_multi_stock_analysis"],
                        stock_scope={"mode": "explicit", "stock_refs": ["中国银行"]},
                    ),
                }
            )
        )
        service._stock_name_lookup_manager = FakeStockNameLookupManager(
            [
                {"code": "601988", "name": "中国银行"},
                {"code": "600036", "name": "招商银行"},
            ]
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "分析一下中国银行",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "analysis")
        self.assertEqual([item["code"] for item in payload["candidate_orders"]], ["601988"])
        self.assertEqual(payload["structured_result"]["conversation_state"]["focus_stocks"], ["601988"])

    async def test_dynamic_multi_stock_names_run_analysis(self):
        service = self.build_service(
            FakeMappedPlannerAnalyzer(
                {
                    "分析一下中国银行和招商银行": FakePlannerAnalyzer._plan(
                        intent="analysis",
                        required_tools=["run_multi_stock_analysis"],
                        stock_scope={"mode": "explicit", "stock_refs": ["中国银行", "招商银行"]},
                    ),
                }
            )
        )
        service._stock_name_lookup_manager = FakeStockNameLookupManager(
            [
                {"code": "601988", "name": "中国银行"},
                {"code": "600036", "name": "招商银行"},
            ]
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "分析一下中国银行和招商银行",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "analysis")
        self.assertEqual([item["code"] for item in payload["candidate_orders"]], ["601988", "600036"])

    async def test_name_only_query_runs_analysis(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "贵州茅台",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "analysis")
        self.assertEqual([item["code"] for item in payload["candidate_orders"]], ["600519"])
        self.assertEqual(payload["structured_result"]["conversation_state"]["focus_stocks"], ["600519"])

    async def test_alias_only_query_runs_analysis(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "茅台",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "analysis")
        self.assertEqual([item["code"] for item in payload["candidate_orders"]], ["600519"])
        self.assertEqual(payload["structured_result"]["conversation_state"]["focus_stocks"], ["600519"])

    async def test_multi_stock_names_run_analysis(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "平安银行和宁德时代",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "analysis")
        self.assertEqual([item["code"] for item in payload["candidate_orders"]], ["000001", "300750"])

    async def test_ambiguous_stock_alias_clarifies_with_candidates(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "平安",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["structured_result"]["intent"], "clarify")
        self.assertIn("000001 平安银行", payload["content"])
        self.assertIn("601318 中国平安", payload["content"])

    def test_dynamic_stock_name_lookup_retries_after_cooldown(self):
        service = self.build_service(FakePlannerAnalyzer())
        lookup_manager = FakeStockNameLookupManager(
            [{"code": "601988", "name": "中国银行"}],
            errors=[RuntimeError("temporary stock list failure")],
        )
        service._stock_name_lookup_manager = lookup_manager

        first = service._resolve_stock_name_reference("中国银行")
        second = service._resolve_stock_name_reference("中国银行")

        self.assertEqual(first["status"], "unknown")
        self.assertEqual(second["status"], "unknown")
        self.assertEqual(lookup_manager.calls, 1)

        service._dynamic_stock_name_retry_after = 0.0
        third = service._resolve_stock_name_reference("中国银行")

        self.assertEqual(third["status"], "resolved")
        self.assertEqual(third["stock_codes"], ["601988"])
        self.assertEqual(lookup_manager.calls, 2)

    async def test_llm_clarify_falls_back_to_local_name_resolution(self):
        service = self.build_service(
            FakeMappedPlannerAnalyzer(
                {
                    "茅台": FakePlannerAnalyzer._plan(
                        intent="clarify",
                        required_tools=[],
                        stock_scope={"mode": "none", "stock_refs": []},
                        clarification="请补充股票代码。",
                    ),
                }
            )
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "茅台",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "analysis")
        self.assertEqual([item["code"] for item in payload["candidate_orders"]], ["600519"])

    async def test_route_stock_name_context_persists_focus_as_code(self):
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 贵州茅台 行情",
                "context": {"stock_code": "贵州茅台", "source_path": "/analysis/agent-chat?stockCode=贵州茅台"},
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(first["structured_result"]["conversation_state"]["focus_stocks"], ["600519"])

        second = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "再试一次",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(second["status"], "analysis_only")
        self.assertEqual(second["structured_result"]["intent"], "analysis")
        self.assertEqual(second["structured_result"]["conversation_state"]["focus_stocks"], ["600519"])

    async def test_industry_board_ref_expands_to_top_10_components(self):
        service = self.build_service(
            FakeMappedPlannerAnalyzer(
                {
                    "半导体": FakePlannerAnalyzer._plan(
                        intent="analysis",
                        required_tools=["run_multi_stock_analysis"],
                        stock_scope={"mode": "explicit", "stock_refs": ["半导体"]},
                    ),
                }
            )
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "半导体",
                "runtime_config": build_runtime_config(),
            }
        )

        expected_codes = [item["code"] for item in FAKE_BOARD_CONSTITUENTS[("industry_board", "BK0917")][:10]]
        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["intent"], "analysis")
        self.assertEqual(
            [item["code"] for item in payload["structured_result"]["analysis"]["stocks"]],
            expected_codes,
        )
        self.assertEqual([item["code"] for item in payload["candidate_orders"]], expected_codes)
        self.assertEqual(
            payload["structured_result"]["intent_resolution"]["scope_resolution"]["resolved_stock_codes"],
            expected_codes,
        )
        self.assertEqual(
            payload["structured_result"]["intent_resolution"]["resolved_scope_entities"][0]["entity_type"],
            "industry_board",
        )

    async def test_follow_up_industry_keyword_reuses_session_context(self):
        service = self.build_service(
            FakeMappedPlannerAnalyzer(
                {
                    "推荐一只股票": FakePlannerAnalyzer._plan(
                        intent="clarify",
                        required_tools=["load_session_memory"],
                        stock_scope={"mode": "none", "stock_refs": []},
                        clarification="请告诉我行业、概念板块或股票代码。",
                    ),
                    "半导体": FakePlannerAnalyzer._plan(
                        intent="analysis",
                        required_tools=["load_session_memory", "run_multi_stock_analysis"],
                        stock_scope={"mode": "explicit", "stock_refs": ["半导体"]},
                    ),
                }
            )
        )

        first = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "推荐一只股票",
                "runtime_config": build_runtime_config(),
            }
        )
        second = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "半导体",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(first["structured_result"]["intent"], "clarify")
        self.assertEqual(second["status"], "analysis_only")
        self.assertEqual(second["structured_result"]["intent"], "analysis")
        self.assertIn("session_memory", second["structured_result"]["loaded_context_keys"])
        self.assertEqual(
            second["structured_result"]["intent_resolution"]["resolved_scope_entities"][0]["entity_type"],
            "industry_board",
        )
        self.assertNotIn("还没法准确映射到股票代码", second["content"])

    async def test_unknown_board_keyword_clarifies_with_original_ref(self):
        service = self.build_service(
            FakeMappedPlannerAnalyzer(
                {
                    "不存在赛道": FakePlannerAnalyzer._plan(
                        intent="analysis",
                        required_tools=["run_multi_stock_analysis"],
                        stock_scope={"mode": "explicit", "stock_refs": ["不存在赛道"]},
                    ),
                }
            )
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "不存在赛道",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["structured_result"]["intent"], "clarify")
        self.assertIn("不存在赛道", payload["content"])

    async def test_board_constituents_unavailable_clarifies(self):
        analyzer = FakeMappedPlannerAnalyzer(
            {
                "半导体": FakePlannerAnalyzer._plan(
                    intent="analysis",
                    required_tools=["run_multi_stock_analysis"],
                    stock_scope={"mode": "explicit", "stock_refs": ["半导体"]},
                ),
            }
        )
        service = AgentChatService(
            config=Config.get_instance(),
            db_manager=self.db,
            chat_repo=self.repo,
            agent_service=self.agent_service,
            backend_client=self.backend_client,
            analyzer=analyzer,
            board_catalog_provider=fake_board_catalog_provider,
            board_constituents_provider=lambda *_args: [],
        )

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "半导体",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["structured_result"]["intent"], "clarify")
        self.assertIn("半导体", payload["content"])
        self.assertIn("无法拉取它的成分股", payload["content"])

    async def test_run_multi_stock_analysis_tool_rejects_empty_stock_codes(self):
        with self.assertRaises(ValueError):
            await self.service._tool_run_multi_stock_analysis(
                {"stock_codes": []},
                {"event_handler": None, "owner_user_id": 1, "session_id": "s-empty"},
            )

    async def test_event_stream_uses_new_tool_names(self):
        events: list[tuple[str, str | None]] = []

        async def handler(event_name: str, payload: dict[str, Any]):
            events.append((event_name, payload.get("tool")))

        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "分析 300750，如果风险低就买100股",
                "runtime_config": build_runtime_config(),
            },
            event_handler=handler,
        )

        self.assertEqual(payload["status"], "simulation_order_filled")
        self.assertEqual(
            [item[1] for item in events if item[0] in {"tool_start", "tool_done"}][::2],
            ["load_system_state", "load_account_state", "run_multi_stock_analysis", "batch_execute_candidate_orders"],
        )

    async def test_event_stream_emits_incremental_assistant_message_content(self):
        events: list[tuple[str, dict[str, Any]]] = []

        async def handler(event_name: str, payload: dict[str, Any]):
            events.append((event_name, dict(payload)))

        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "runtime_config": build_runtime_config(),
            },
            event_handler=handler,
        )

        event_names = [item[0] for item in events]
        self.assertIn("message_start", event_names)
        self.assertIn("message_delta", event_names)
        self.assertGreater(event_names.index("message_start"), event_names.index("tool_done"))

        streamed_content = "".join(
            str(item[1].get("delta") or "")
            for item in events
            if item[0] == "message_delta"
        )
        self.assertEqual(streamed_content, payload["content"])

    async def test_empty_explicit_planner_refs_fall_back_to_deterministic_stock_names(self):
        service = self.build_service(FakeEmptyExplicitPlannerAnalyzer())

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "分析一下紫金矿业、工业富联和美的集团",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(
            [item["code"] for item in payload["structured_result"]["analysis"]["stocks"]],
            ["601899", "601138", "000333"],
        )
        self.assertEqual(
            [item["code"] for item in payload["candidate_orders"]],
            ["601899", "601138", "000333"],
        )
        self.assertEqual(len(self.backend_client.saved_analysis_calls), 1)
        self.assertEqual(
            [item["code"] for item in self.backend_client.saved_analysis_calls[0]["analysis_result"]["stocks"]],
            ["601899", "601138", "000333"],
        )

    async def test_empty_explicit_planner_refs_support_price_suffix_resolution(self):
        service = self.build_service(FakeEmptyExplicitPlannerAnalyzer())

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "分析一下比亚迪股价",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "analysis_only")
        self.assertEqual(payload["structured_result"]["analysis"]["stocks"][0]["code"], "002594")
        self.assertEqual(payload["candidate_orders"][0]["code"], "002594")

    async def test_empty_explicit_planner_refs_clarify_instead_of_zero_stock_success(self):
        service = self.build_service(FakeEmptyExplicitPlannerAnalyzer())

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "分析一下不存在科技",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["structured_result"]["intent"], "clarify")
        self.assertIn("不存在科技", payload["content"])
        self.assertEqual(payload["candidate_orders"], [])

    async def test_market_wide_stock_selection_request_is_explicitly_blocked(self):
        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "我想知道 A 股今天所有股票中哪只最值得买",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["structured_result"]["intent"], "unsupported")
        self.assertEqual(
            payload["structured_result"]["blocked_code"],
            "market_wide_stock_selection_unsupported",
        )
        self.assertFalse(payload["structured_result"]["supported"])
        self.assertIn("这个问题我目前做不到", payload["content"])
        self.assertIn("暂不支持直接对 A 股全市场做实时扫描", payload["content"])
        self.assertEqual(payload["candidate_orders"], [])

    async def test_follow_up_order_blocked_outside_trading_session_keeps_pending_actions_and_emits_warning(self):
        self.backend_client.blocked_order_codes.add("600519")
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 行情",
                "runtime_config": build_runtime_config(),
            }
        )

        events: list[tuple[str, dict[str, Any]]] = []

        async def collect_event(event_name: str, payload: dict[str, Any]) -> None:
            events.append((event_name, dict(payload)))

        second = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "去下单吧",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            },
            event_handler=collect_event,
        )

        self.assertEqual(second["status"], "blocked")
        self.assertEqual(second["structured_result"]["intent"], "order_followup_single")
        self.assertEqual(second["execution_result"]["reason"], "outside_trading_session")
        self.assertIn("当前处于非交易时段", second["execution_result"]["message"])
        self.assertEqual(second["execution_result"]["session_guard"]["next_open_at"], "2026-04-06T09:30:00+08:00")
        self.assertIn("候选单已保留", second["content"])

        event_names = [name for name, _payload in events]
        self.assertIn("warning", event_names)
        self.assertIn("message_start", event_names)
        self.assertLess(event_names.index("warning"), event_names.index("message_start"))
        warning_payload = next(payload for name, payload in events if name == "warning")
        self.assertEqual(warning_payload["stage"], "execution")
        self.assertIn("本轮未执行模拟盘订单", warning_payload["message"])

        detail = self.service.get_session_detail(1, first["session_id"])
        assert detail is not None
        pending_actions = detail["context"]["conversation_state"]["pending_actions"]
        self.assertEqual(len(pending_actions), 1)
        self.assertEqual(pending_actions[0]["code"], "600519")

    async def test_analysis_then_execute_blocked_outside_trading_session_keeps_candidate_orders(self):
        self.backend_client.blocked_order_codes.add("600519")

        payload = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "分析 600519，如果风险低就买100股",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["structured_result"]["intent"], "analysis_then_execute")
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["execution_result"]["reason"], "outside_trading_session")
        self.assertIn("当前处于非交易时段", payload["execution_result"]["message"])
        self.assertEqual(payload["execution_result"]["session_guard"]["next_open_at"], "2026-04-06T09:30:00+08:00")
        self.assertIn("候选单已保留", payload["content"])

        autonomous_execution = payload["structured_result"]["autonomous_execution"]
        self.assertFalse(autonomous_execution["executed"])
        self.assertEqual(autonomous_execution["executed_count"], 0)
        self.assertEqual(autonomous_execution["failed_count"], 0)
        self.assertFalse(autonomous_execution["gate_passed"])
        self.assertEqual(autonomous_execution["reason"], "outside_trading_session")
        self.assertIn("当前处于非交易时段", autonomous_execution["gate_message"])

        detail = self.service.get_session_detail(1, payload["session_id"])
        assert detail is not None
        pending_actions = detail["context"]["conversation_state"]["pending_actions"]
        self.assertEqual(len(pending_actions), 1)
        self.assertEqual(pending_actions[0]["code"], "600519")
        self.assertEqual(pending_actions[0]["quantity"], 100)

    async def test_follow_up_all_blocked_outside_trading_session_returns_blocked(self):
        self.backend_client.blocked_order_codes.update({"600519", "000001"})
        first = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我分析一下今天的 600519 和 000001",
                "runtime_config": build_runtime_config(),
            }
        )

        second = await self.service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "把刚才那几笔都下了",
                "session_id": first["session_id"],
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(second["status"], "blocked")
        self.assertEqual(second["structured_result"]["intent"], "order_followup_all")
        self.assertEqual(second["execution_result"]["status"], "blocked")
        self.assertEqual(second["execution_result"]["reason"], "outside_trading_session")
        self.assertEqual(len(second["execution_result"]["blocked_orders"]), 2)
        self.assertIn("因非交易时段未执行", second["content"])
        self.assertNotIn("### 已提交/成交", second["content"])

        detail = self.service.get_session_detail(1, first["session_id"])
        assert detail is not None
        pending_actions = detail["context"]["conversation_state"]["pending_actions"]
        self.assertEqual(len(pending_actions), 2)

    async def test_market_wide_scope_refs_block_instead_of_zero_stock_success(self):
        service = self.build_service(FakeMarketWidePlannerAnalyzer())

        payload = await service.handle_chat(
            {
                "owner_user_id": 1,
                "username": "tester",
                "message": "帮我从 A 股全市场里选一只最值得买的",
                "runtime_config": build_runtime_config(),
            }
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["structured_result"]["intent"], "unsupported")
        self.assertEqual(
            payload["structured_result"]["blocked_code"],
            "market_wide_stock_selection_unsupported",
        )
        self.assertFalse(payload["structured_result"]["supported"])
        self.assertIn("这个问题我目前做不到", payload["content"])
        self.assertIn("暂不支持直接对 A 股全市场做实时扫描", payload["content"])
        self.assertEqual(payload["candidate_orders"], [])


class ChatPlannerAgentTestCase(unittest.TestCase):
    def test_parse_payload_rejects_analysis_with_empty_explicit_stock_refs(self):
        planner = ChatPlannerAgent(config=Config(), analyzer=None)

        payload = {
            "intent": "analysis",
            "stock_scope": {"mode": "explicit", "stock_refs": []},
            "followup_target": {"mode": "none", "stock_refs": []},
            "execution_authorized": False,
            "required_tools": ["run_multi_stock_analysis"],
            "constraints": [],
            "session_preference_overrides": {},
            "clarification": "",
        }

        self.assertIsNone(planner._parse_payload(payload))


if __name__ == "__main__":
    unittest.main()
