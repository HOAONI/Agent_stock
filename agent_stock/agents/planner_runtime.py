# -*- coding: utf-8 -*-
"""Planner kernel shared models and condition helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IntentCondition:
    """A normalized, planner-consumable request condition."""

    type: str
    value: Any = None
    operator: str = "eq"
    label: str = ""
    source: str = "rule"
    supported: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "value": self.value,
            "operator": self.operator,
            "label": self.label,
            "source": self.source,
            "supported": self.supported,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class RequestIntent:
    """Unified request intent passed into the planner kernel."""

    goal: str
    stock_codes: list[str]
    primary_intent: str = "analysis"
    user_message: str = ""
    include_runtime_context: bool = True
    autonomous_execution_authorized: bool = False
    requested_order_side: str | None = None
    requested_quantity: int | None = None
    conditions: list[IntentCondition] = field(default_factory=list)
    unsupported_conditions: list[IntentCondition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "stock_codes": list(self.stock_codes),
            "primary_intent": self.primary_intent,
            "user_message": self.user_message,
            "include_runtime_context": self.include_runtime_context,
            "autonomous_execution_authorized": self.autonomous_execution_authorized,
            "requested_order_side": self.requested_order_side,
            "requested_quantity": self.requested_quantity,
            "conditions": [item.to_dict() for item in self.conditions],
            "unsupported_conditions": [item.to_dict() for item in self.unsupported_conditions],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PlannerAction:
    """A single structured action emitted by the planner."""

    kind: str
    tool: str | None = None
    reason: str = ""
    summary: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tool": self.tool,
            "reason": self.reason,
            "summary": self.summary,
            "inputs": dict(self.inputs),
        }


def normalize_intent_conditions(value: Any) -> tuple[list[IntentCondition], list[IntentCondition]]:
    """Normalize any request condition payload into supported/unsupported clauses."""
    if not isinstance(value, list):
        return [], []

    supported: list[IntentCondition] = []
    unsupported: list[IntentCondition] = []
    for item in value:
        if isinstance(item, IntentCondition):
            target = supported if item.supported else unsupported
            target.append(item)
            continue
        if not isinstance(item, dict):
            continue
        condition = IntentCondition(
            type=str(item.get("type") or "").strip() or "unknown",
            value=item.get("value"),
            operator=str(item.get("operator") or "eq").strip() or "eq",
            label=str(item.get("label") or "").strip(),
            source=str(item.get("source") or "rule").strip() or "rule",
            supported=bool(item.get("supported", True)),
            meta=dict(item.get("meta") or {}) if isinstance(item.get("meta"), dict) else {},
        )
        target = supported if condition.supported else unsupported
        target.append(condition)
    return supported, unsupported


def build_request_intent(
    *,
    stock_codes: list[str],
    planning_context: dict[str, Any] | None,
) -> RequestIntent:
    """Build a request intent from planning context with safe defaults."""
    context = planning_context if isinstance(planning_context, dict) else {}
    message = str(context.get("user_message") or context.get("message") or "").strip()
    supported, unsupported = normalize_intent_conditions(context.get("constraints"))
    if not supported and not unsupported:
        supported, unsupported = normalize_intent_conditions(context.get("conditions"))
    loaded_context = dict(context.get("loaded_context") or {}) if isinstance(context.get("loaded_context"), dict) else {}
    session_overrides = dict(context.get("session_overrides") or {}) if isinstance(context.get("session_overrides"), dict) else {}
    stage_memory = dict(context.get("stage_memory") or {}) if isinstance(context.get("stage_memory"), dict) else {}

    return RequestIntent(
        goal=message or "分析股票并生成保守执行计划",
        stock_codes=list(stock_codes or []),
        primary_intent=str(context.get("primary_intent") or context.get("intent") or "analysis").strip() or "analysis",
        user_message=message,
        include_runtime_context=bool(context.get("include_runtime_context", True)),
        autonomous_execution_authorized=bool(context.get("autonomous_execution_authorized")),
        requested_order_side=str(context.get("requested_order_side") or "").strip() or None,
        requested_quantity=_as_positive_int(context.get("requested_quantity")),
        conditions=supported,
        unsupported_conditions=unsupported,
        metadata={
            "intent_resolution": dict(context.get("intent_resolution") or {})
            if isinstance(context.get("intent_resolution"), dict)
            else {},
            "loaded_context": loaded_context,
            "session_overrides": session_overrides,
            "stage_memory": stage_memory,
        },
    )


def compile_message_conditions(message: str) -> tuple[list[IntentCondition], list[IntentCondition]]:
    """Compile common trading constraints from natural language."""
    normalized = str(message or "").strip()
    if not normalized:
        return [], []

    supported: list[IntentCondition] = []
    unsupported: list[IntentCondition] = []

    risk_tokens = (
        "风险低的话",
        "风险不大的话",
        "风险可控的话",
        "如果风险低",
        "如果风险不大",
    )
    if any(token in normalized for token in risk_tokens):
        supported.append(
            IntentCondition(
                type="risk_gate",
                value="risk_low",
                operator="eq",
                label="risk_low",
            )
        )

    for pattern, operator, label in (
        (r"(?:低于|跌到|跌破|小于)\s*(\d+(?:\.\d+)?)\s*(?:元|块|股价)", "lte", "price_upper_bound"),
        (r"(?:高于|涨到|突破|大于)\s*(\d+(?:\.\d+)?)\s*(?:元|块|股价)", "gte", "price_lower_bound"),
    ):
        matched = re.search(pattern, normalized)
        if matched:
            supported.append(
                IntentCondition(
                    type="price_gate",
                    value=float(matched.group(1)),
                    operator=operator,
                    label=label,
                )
            )

    cash_matched = re.search(r"(?:至少保留|保留至少|剩余现金不少于)\s*(\d+(?:\.\d+)?)", normalized)
    if cash_matched:
        supported.append(
            IntentCondition(
                type="min_remaining_cash",
                value=float(cash_matched.group(1)),
                operator="gte",
                label="min_remaining_cash",
            )
        )

    position_pct_matched = re.search(r"(?:单票仓位|单只仓位|仓位)\s*(?:不超过|最多|上限)\s*(\d+(?:\.\d+)?)\s*%", normalized)
    if position_pct_matched:
        supported.append(
            IntentCondition(
                type="max_single_position_pct",
                value=float(position_pct_matched.group(1)),
                operator="lte",
                label="max_single_position_pct",
            )
        )

    # Surface common but currently unsupported "free-form" triggers explicitly.
    unsupported_patterns = (
        r"(?:回撤|波动率|夏普|beta|贝塔)",
        r"(?:财报|业绩|公告|研报)满足",
        r"(?:如果合适就买|看情况买|差不多就买)",
    )
    for pattern in unsupported_patterns:
        matched = re.search(pattern, normalized, flags=re.IGNORECASE)
        if matched:
            unsupported.append(
                IntentCondition(
                    type="unsupported_condition",
                    value=matched.group(0),
                    operator="raw",
                    label=matched.group(0),
                    supported=False,
                )
            )

    return supported, unsupported


def evaluate_conditions(
    *,
    conditions: list[IntentCondition],
    unsupported_conditions: list[IntentCondition],
    stock_code: str,
    current_price: float,
    risk_output: Any | None,
    execution_output: Any | None,
    account_snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate normalized conditions against stock/risk/execution state."""
    evaluations: list[dict[str, Any]] = []

    for item in unsupported_conditions:
        evaluations.append(
            {
                "stock_code": stock_code,
                "condition": item.to_dict(),
                "passed": False,
                "blocking": True,
                "reason": "unsupported_condition",
                "expected": item.value,
                "actual": None,
            }
        )

    for item in conditions:
        evaluation = _evaluate_single_condition(
            condition=item,
            stock_code=stock_code,
            current_price=current_price,
            risk_output=risk_output,
            execution_output=execution_output,
            account_snapshot=account_snapshot,
        )
        evaluations.append(evaluation)
    return evaluations


def _evaluate_single_condition(
    *,
    condition: IntentCondition,
    stock_code: str,
    current_price: float,
    risk_output: Any | None,
    execution_output: Any | None,
    account_snapshot: dict[str, Any],
) -> dict[str, Any]:
    expected = condition.value
    actual: Any = None
    passed = False
    blocking = True
    reason = "condition_failed"

    risk_level = str(getattr(risk_output, "risk_level", "") or "").strip().lower()
    execution_allowed = bool(getattr(risk_output, "execution_allowed", False))
    hard_blocks = list(getattr(risk_output, "hard_blocks", []) or [])
    account_total_asset = _as_float(account_snapshot.get("total_asset"))
    account_cash = _as_float(account_snapshot.get("cash"))
    execution_cash_after = _as_float(getattr(execution_output, "cash_after", 0.0)) if execution_output is not None else 0.0
    execution_position_after = _as_float(getattr(execution_output, "position_after", 0.0)) if execution_output is not None else 0.0

    if condition.type == "risk_gate" and str(condition.value or "") == "risk_low":
        actual = {
            "risk_level": risk_level or "unknown",
            "execution_allowed": execution_allowed,
            "hard_blocks": hard_blocks,
        }
        passed = risk_level == "low" and execution_allowed and not hard_blocks
        reason = "risk_low" if passed else "risk_not_low"
    elif condition.type == "price_gate":
        actual = current_price
        threshold = _as_float(expected)
        if condition.operator == "lte":
            passed = current_price > 0 and threshold > 0 and current_price <= threshold
            reason = "price_lte" if passed else "price_above_limit"
        elif condition.operator == "gte":
            passed = current_price > 0 and threshold > 0 and current_price >= threshold
            reason = "price_gte" if passed else "price_below_limit"
    elif condition.type == "min_remaining_cash":
        actual = execution_cash_after or account_cash
        threshold = _as_float(expected)
        passed = actual >= threshold > 0
        reason = "cash_sufficient" if passed else "cash_below_floor"
    elif condition.type == "max_single_position_pct":
        position_value = execution_position_after * current_price if execution_position_after > 0 and current_price > 0 else 0.0
        actual = (position_value / account_total_asset * 100.0) if account_total_asset > 0 else 0.0
        threshold = _as_float(expected)
        passed = actual <= threshold if threshold > 0 else False
        reason = "position_pct_ok" if passed else "position_pct_exceeded"
    else:
        actual = None
        passed = False
        reason = "unsupported_condition"

    return {
        "stock_code": stock_code,
        "condition": condition.to_dict(),
        "passed": passed,
        "blocking": blocking,
        "reason": reason,
        "expected": expected,
        "actual": actual,
    }


def _as_positive_int(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _as_float(value: Any) -> float:
    try:
        number = float(value)
        if number != number:
            return 0.0
        return number
    except (TypeError, ValueError):
        return 0.0
