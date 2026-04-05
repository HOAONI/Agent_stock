# -*- coding: utf-8 -*-
"""组合策略 DSL 的解析、归一化与描述工具。"""

from __future__ import annotations

import json
import math
import re
from typing import Any

RULE_DSL_TEMPLATE_CODE = "rule_dsl"
RULE_DSL_TEMPLATE_NAME = "组合规则 DSL"
RULE_DSL_VERSION = "rule_v1"

_COMPARE_KEYWORDS = ("对比", "比较", "哪个好", "哪个更好", "优劣", "胜出")
_OR_CONNECTORS = ("或者", "或是", "任一", "任意一个", "or")
_STOP_WORDS = ("止损", "卖出", "离场", "平仓")

_MACD_PARAM_RE = re.compile(
    r"macd[\(\[（]?\s*(\d{1,2})\s*[,/，]\s*(\d{1,3})\s*[,/，]\s*(\d{1,2})",
    flags=re.I,
)
_RSI_COMPARE_RE = re.compile(
    r"rsi(?:[\(\（]?\s*(\d{1,2})[\)\）]?)?\s*(<=|=<|>=|=>|<|>|≤|≥|低于|小于|高于|大于)\s*(\d{1,2})",
    flags=re.I,
)
_PRICE_MA_RE = re.compile(
    r"(跌破|下穿|失守|站上|上穿|突破)\s*(\d{1,3})\s*(?:日|天)?(?:均线|日均线|日线|线|ma)?",
    flags=re.I,
)
_STOP_LOSS_PCT_RE = re.compile(
    r"(?:止损|亏损(?:达到)?|回撤(?:达到)?)\s*(\d{1,2}(?:\.\d+)?)\s*%",
    flags=re.I,
)
_TAKE_PROFIT_PCT_RE = re.compile(
    r"(?:止盈|盈利(?:达到)?|收益(?:达到)?|涨幅(?:达到)?)\s*(\d{1,3}(?:\.\d+)?)\s*%",
    flags=re.I,
)


def _compact_message_text(message: str) -> str:
    return re.sub(r"\s+", "", str(message or "")).lower()


def _to_float(value: Any, fallback: float | None = None) -> float | None:
    if value is None:
        return fallback
    if isinstance(value, str) and not value.strip():
        return fallback
    try:
        number = float(value)
    except Exception:
        return fallback
    if not math.isfinite(number):
        return fallback
    return number


def _to_int(value: Any, fallback: int) -> int:
    number = _to_float(value, float(fallback))
    if number is None:
        return fallback
    return int(number)


def _normalize_operator(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"<", "lt", "小于", "低于"}:
        return "lt"
    if normalized in {"<=", "=<", "lte", "≤"}:
        return "lte"
    if normalized in {">", "gt", "大于", "高于"}:
        return "gt"
    if normalized in {">=", "=>", "gte", "≥"}:
        return "gte"
    raise ValueError(f"validation_error: unsupported rule operator={normalized or '--'}")


def _stable_signature(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _truncate_text(value: Any, *, max_length: int) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:max_length]


def _dedupe_conditions(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in conditions:
        signature = _stable_signature(item)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    return deduped


def _normalize_condition(raw_condition: Any) -> dict[str, Any]:
    source = raw_condition if isinstance(raw_condition, dict) else {}
    kind = str(source.get("kind") or "").strip()
    if kind == "macd_cross":
        direction = str(source.get("direction") or "bullish").strip().lower()
        if direction not in {"bullish", "bearish"}:
            raise ValueError("validation_error: rule_dsl macd_cross.direction must be bullish or bearish")
        fast = _to_int(source.get("fast"), 12)
        slow = _to_int(source.get("slow"), 26)
        signal = _to_int(source.get("signal"), 9)
        if fast < 5 or fast > 60:
            raise ValueError("validation_error: rule_dsl macd_cross.fast must be between 5 and 60")
        if slow < 10 or slow > 120:
            raise ValueError("validation_error: rule_dsl macd_cross.slow must be between 10 and 120")
        if signal < 3 or signal > 30:
            raise ValueError("validation_error: rule_dsl macd_cross.signal must be between 3 and 30")
        if fast >= slow:
            raise ValueError("validation_error: rule_dsl macd_cross.fast must be less than slow")
        return {
            "kind": kind,
            "direction": direction,
            "fast": fast,
            "slow": slow,
            "signal": signal,
        }

    if kind == "rsi_threshold":
        period = _to_int(source.get("period"), 14)
        threshold = _to_int(source.get("threshold"), 30)
        operator = _normalize_operator(source.get("operator") or "lt")
        if period < 5 or period > 60:
            raise ValueError("validation_error: rule_dsl rsi_threshold.period must be between 5 and 60")
        if threshold < 1 or threshold > 99:
            raise ValueError("validation_error: rule_dsl rsi_threshold.threshold must be between 1 and 99")
        return {
            "kind": kind,
            "period": period,
            "operator": operator,
            "threshold": threshold,
        }

    if kind == "price_ma_relation":
        ma_window = _to_int(source.get("maWindow"), 5)
        relation = str(source.get("relation") or "cross_below").strip().lower()
        if ma_window < 3 or ma_window > 250:
            raise ValueError("validation_error: rule_dsl price_ma_relation.maWindow must be between 3 and 250")
        if relation not in {"above", "below", "cross_above", "cross_below"}:
            raise ValueError("validation_error: rule_dsl price_ma_relation.relation is invalid")
        return {
            "kind": kind,
            "maWindow": ma_window,
            "relation": relation,
        }

    if kind == "stop_loss_pct":
        pct = float(_to_float(source.get("pct"), 0.0) or 0.0)
        if pct <= 0 or pct > 50:
            raise ValueError("validation_error: rule_dsl stop_loss_pct.pct must be between 0 and 50")
        return {
            "kind": kind,
            "pct": round(pct, 4),
        }

    if kind == "take_profit_pct":
        pct = float(_to_float(source.get("pct"), 0.0) or 0.0)
        if pct <= 0 or pct > 500:
            raise ValueError("validation_error: rule_dsl take_profit_pct.pct must be between 0 and 500")
        return {
            "kind": kind,
            "pct": round(pct, 4),
        }

    raise ValueError(f"validation_error: unsupported rule_dsl condition kind={kind or '--'}")


def _normalize_rule_group(
    raw_group: Any,
    *,
    group_name: str,
    default_operator: str,
    required: bool,
) -> dict[str, Any] | None:
    source = raw_group if isinstance(raw_group, dict) else {}
    raw_conditions = source.get("conditions")
    if raw_conditions is None:
        raw_conditions = []
    if not isinstance(raw_conditions, list):
        raise ValueError(f"validation_error: rule_dsl.{group_name}.conditions must be a list")

    conditions = _dedupe_conditions([
        _normalize_condition(item)
        for item in raw_conditions
    ])
    if required and not conditions:
        raise ValueError(f"validation_error: rule_dsl.{group_name}.conditions is required")
    if not conditions:
        return None

    operator = str(source.get("operator") or default_operator).strip().lower()
    if operator not in {"and", "or"}:
        raise ValueError(f"validation_error: rule_dsl.{group_name}.operator must be and/or")
    return {
        "operator": operator,
        "conditions": conditions,
    }


def normalize_rule_dsl_params(raw_params: Any) -> dict[str, Any]:
    """校验并标准化组合规则 DSL。"""
    source = raw_params if isinstance(raw_params, dict) else {}
    entry = _normalize_rule_group(
        source.get("entry"),
        group_name="entry",
        default_operator="and",
        required=True,
    )
    exit_group = _normalize_rule_group(
        source.get("exit"),
        group_name="exit",
        default_operator="or",
        required=False,
    )

    payload: dict[str, Any] = {
        "dslVersion": RULE_DSL_VERSION,
        "entry": entry,
    }
    source_text = _truncate_text(source.get("sourceText") or source.get("source_text"), max_length=160)
    if source_text:
        payload["sourceText"] = source_text
    if exit_group is not None:
        payload["exit"] = exit_group
    return payload


def summarize_rule_condition(condition: dict[str, Any]) -> str:
    """把单个条件转换成简短中文描述。"""
    kind = str(condition.get("kind") or "").strip()
    if kind == "macd_cross":
        label = "金叉" if str(condition.get("direction")) == "bullish" else "死叉"
        return (
            f"MACD({int(condition.get('fast') or 12)},{int(condition.get('slow') or 26)},{int(condition.get('signal') or 9)})"
            f"{label}"
        )
    if kind == "rsi_threshold":
        op_map = {
            "lt": "<",
            "lte": "<=",
            "gt": ">",
            "gte": ">=",
        }
        operator = op_map.get(str(condition.get("operator") or ""), str(condition.get("operator") or ""))
        return f"RSI{int(condition.get('period') or 14)} {operator} {int(condition.get('threshold') or 0)}"
    if kind == "price_ma_relation":
        ma_window = int(condition.get("maWindow") or 5)
        relation = str(condition.get("relation") or "")
        relation_map = {
            "above": f"收盘站上 MA{ma_window}",
            "below": f"收盘跌破 MA{ma_window}",
            "cross_above": f"上穿 MA{ma_window}",
            "cross_below": f"跌破 MA{ma_window}",
        }
        return relation_map.get(relation, f"MA{ma_window} 条件")
    if kind == "stop_loss_pct":
        return f"回撤 {float(condition.get('pct') or 0.0):g}% 止损"
    if kind == "take_profit_pct":
        return f"盈利 {float(condition.get('pct') or 0.0):g}% 止盈"
    return kind or "条件"


def summarize_rule_group(group: dict[str, Any] | None) -> str:
    """把规则组转换成中文描述。"""
    if not isinstance(group, dict):
        return ""
    conditions = [
        summarize_rule_condition(item)
        for item in group.get("conditions") or []
        if isinstance(item, dict)
    ]
    if not conditions:
        return ""
    joiner = " 或 " if str(group.get("operator") or "and").lower() == "or" else " 且 "
    return joiner.join(conditions)


def summarize_rule_dsl(params: Any) -> str:
    """输出完整 DSL 的中文说明。"""
    normalized = normalize_rule_dsl_params(params)
    entry_text = summarize_rule_group(normalized.get("entry"))
    exit_text = summarize_rule_group(normalized.get("exit"))
    if exit_text:
        return f"入场：{entry_text}；离场：{exit_text}"
    return f"入场：{entry_text}；离场：窗口结束时平仓"


def build_rule_dsl_strategy_name(params: Any, *, max_length: int = 64) -> str:
    """为 DSL 生成可展示的策略名称。"""
    normalized = normalize_rule_dsl_params(params)
    entry_text = summarize_rule_group(normalized.get("entry")) or RULE_DSL_TEMPLATE_NAME
    exit_text = summarize_rule_group(normalized.get("exit"))
    name = f"{entry_text} / {exit_text}" if exit_text else entry_text
    if len(name) <= max_length:
        return name
    return f"{name[:max_length - 1]}…"


def _extract_macd_params(message: str) -> tuple[int, int, int]:
    matched = _MACD_PARAM_RE.search(message)
    if matched:
        return int(matched.group(1)), int(matched.group(2)), int(matched.group(3))
    return 12, 26, 9


def _build_rsi_condition(period: int, operator: str, threshold: int) -> dict[str, Any]:
    return {
        "kind": "rsi_threshold",
        "period": period,
        "operator": operator,
        "threshold": threshold,
    }


def build_rule_dsl_from_text(message: str) -> dict[str, Any] | None:
    """把自然语言中的组合规则解析成 DSL；解析失败时返回 None。"""
    raw_message = str(message or "").strip()
    compact = _compact_message_text(raw_message)
    if not raw_message:
        return None
    if any(keyword in compact for keyword in _COMPARE_KEYWORDS):
        return None

    entry_conditions: list[dict[str, Any]] = []
    exit_conditions: list[dict[str, Any]] = []
    macd_fast, macd_slow, macd_signal = _extract_macd_params(raw_message)

    if "macd" in compact and any(token in compact for token in ("金叉", "上穿")):
        entry_conditions.append(
            {
                "kind": "macd_cross",
                "direction": "bullish",
                "fast": macd_fast,
                "slow": macd_slow,
                "signal": macd_signal,
            }
        )
    if "macd" in compact and any(token in compact for token in ("死叉卖出", "死叉止损", "死叉离场", "死叉平仓")):
        exit_conditions.append(
            {
                "kind": "macd_cross",
                "direction": "bearish",
                "fast": macd_fast,
                "slow": macd_slow,
                "signal": macd_signal,
            }
        )

    for matched in _RSI_COMPARE_RE.finditer(raw_message):
        period = int(matched.group(1) or 14)
        operator = _normalize_operator(matched.group(2))
        threshold = int(matched.group(3))
        condition = _build_rsi_condition(period, operator, threshold)
        if operator in {"gt", "gte"}:
            exit_conditions.append(condition)
        else:
            entry_conditions.append(condition)

    if "超卖" in raw_message and not any(item.get("kind") == "rsi_threshold" and str(item.get("operator")) in {"lt", "lte"} for item in entry_conditions):
        entry_conditions.append(_build_rsi_condition(14, "lt", 30))
    if "超买" in raw_message and not any(item.get("kind") == "rsi_threshold" and str(item.get("operator")) in {"gt", "gte"} for item in exit_conditions):
        exit_conditions.append(_build_rsi_condition(14, "gt", 70))

    for matched in _PRICE_MA_RE.finditer(raw_message):
        action = str(matched.group(1) or "")
        ma_window = int(matched.group(2))
        context = raw_message[max(0, matched.start() - 6): min(len(raw_message), matched.end() + 6)]
        if action in {"跌破", "下穿", "失守"}:
            exit_conditions.append(
                {
                    "kind": "price_ma_relation",
                    "maWindow": ma_window,
                    "relation": "cross_below",
                }
            )
            continue
        if any(token in context for token in _STOP_WORDS):
            exit_conditions.append(
                {
                    "kind": "price_ma_relation",
                    "maWindow": ma_window,
                    "relation": "cross_above",
                }
            )
            continue
        entry_conditions.append(
            {
                "kind": "price_ma_relation",
                "maWindow": ma_window,
                "relation": "above" if action == "站上" else "cross_above",
            }
        )

    for matched in _STOP_LOSS_PCT_RE.finditer(raw_message):
        pct = float(matched.group(1))
        exit_conditions.append(
            {
                "kind": "stop_loss_pct",
                "pct": pct,
            }
        )

    for matched in _TAKE_PROFIT_PCT_RE.finditer(raw_message):
        pct = float(matched.group(1))
        exit_conditions.append(
            {
                "kind": "take_profit_pct",
                "pct": pct,
            }
        )

    entry_conditions = _dedupe_conditions(entry_conditions)
    exit_conditions = _dedupe_conditions(exit_conditions)
    if not entry_conditions:
        return None
    if len(entry_conditions) <= 1 and not exit_conditions:
        return None

    entry_operator = "or" if len(entry_conditions) > 1 and any(token in compact for token in _OR_CONNECTORS) else "and"
    payload: dict[str, Any] = {
        "dslVersion": RULE_DSL_VERSION,
        "sourceText": raw_message[:160],
        "entry": {
            "operator": entry_operator,
            "conditions": entry_conditions,
        },
    }
    if exit_conditions:
        payload["exit"] = {
            "operator": "or",
            "conditions": exit_conditions,
        }
    return normalize_rule_dsl_params(payload)
