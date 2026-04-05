# -*- coding: utf-8 -*-
"""五层 Agent 的结构化阶段决策辅助工具。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent_stock.config import redact_sensitive_text

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def strip_code_fence(content: Any) -> str:
    """移除 Markdown 代码块包装。"""
    text = str(content or "").strip()
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def extract_json_object(content: Any) -> dict[str, Any]:
    """从 LLM 返回中尽量提取 JSON 对象。"""
    raw = strip_code_fence(content)
    if not raw:
        return {}

    candidates = [raw]
    matched = _JSON_BLOCK_RE.search(raw)
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


def normalize_warning_list(value: Any) -> list[str]:
    """将任意警告字段清洗为字符串列表。"""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    warnings: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in warnings:
            warnings.append(text)
    return warnings


def clamp_confidence(value: Any, default: float = 0.5) -> float:
    """将置信度规整到 0..1。"""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(confidence, 1.0))


def generate_structured_decision(
    *,
    analyzer: Any,
    stage: str,
    prompt: str,
    allowed_actions: set[str],
    default_decision: dict[str, Any],
    max_output_tokens: int = 500,
) -> tuple[dict[str, Any], bool]:
    """调用 LLM 生成结构化决策，并做最小确定性校验。"""
    if not getattr(analyzer, "is_available", lambda: False)():
        return dict(default_decision), False

    try:
        raw = analyzer.generate_text(
            prompt,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        )
    except Exception as exc:
        logger.warning("[%s] structured decision fallback: %s", stage, redact_sensitive_text(str(exc)))
        return dict(default_decision), False

    payload = extract_json_object(raw)
    if not payload:
        return dict(default_decision), False

    action = str(payload.get("action") or "").strip()
    if allowed_actions and action not in allowed_actions:
        return dict(default_decision), False

    decision = dict(default_decision)
    for key in (
        "action",
        "summary",
        "reason",
        "next_action",
        "requested_market_source",
        "adjustment_mode",
        "adjustment_reason",
        "retry_reason",
    ):
        if payload.get(key) is None:
            continue
        decision[key] = str(payload.get(key) or "").strip()

    if payload.get("requested_target_weight_pct") is not None:
        try:
            decision["requested_target_weight_pct"] = float(payload.get("requested_target_weight_pct"))
        except (TypeError, ValueError):
            pass

    if payload.get("requested_notional_factor") is not None:
        try:
            decision["requested_notional_factor"] = float(payload.get("requested_notional_factor"))
        except (TypeError, ValueError):
            pass

    if payload.get("confidence") is not None:
        decision["confidence"] = clamp_confidence(payload.get("confidence"), clamp_confidence(default_decision.get("confidence"), 0.5))

    llm_warnings = normalize_warning_list(payload.get("warnings"))
    if llm_warnings:
        existing = normalize_warning_list(decision.get("warnings"))
        decision["warnings"] = [*existing, *[item for item in llm_warnings if item not in existing]]

    return decision, True
