# -*- coding: utf-8 -*-
"""Agent 服务流程共享的运行时配置解析工具。"""

from __future__ import annotations

from typing import Any


from agent_stock.config import (
    ALLOWED_MARKET_SOURCES,
    AgentRuntimeConfig,
    RuntimeAccountConfig,
    RuntimeContextConfig,
    RuntimeDataSourceConfig,
    RuntimeExecutionConfig,
    RuntimeLlmConfig,
    RuntimeStrategyConfig,
)

ALLOWED_LLM_PROVIDERS = ("gemini", "anthropic", "openai", "deepseek", "custom")
ALLOWED_EXECUTION_MODES = ("paper", "broker")
ALLOWED_EXECUTION_FIELDS = frozenset({"mode", "has_ticket", "broker_account_id"})
ALLOWED_DATA_SOURCE_FIELDS = frozenset({"market_source"})
ALLOWED_CONTEXT_FIELDS = frozenset({"account_snapshot", "summary", "positions"})


def parse_runtime_config(runtime_config: dict[str, Any] | None) -> AgentRuntimeConfig | None:
    """将请求中的运行时配置字典解析为强类型对象。"""
    if not runtime_config:
        return None
    if not isinstance(runtime_config, dict):
        raise ValueError("runtime_config must be an object")

    return AgentRuntimeConfig(
        account=_parse_account_config(runtime_config.get("account")),
        llm=_parse_llm_config(runtime_config.get("llm")),
        strategy=_parse_strategy_config(runtime_config.get("strategy")),
        execution=_parse_execution_config(runtime_config.get("execution")),
        data_source=_parse_data_source_config(runtime_config.get("data_source")),
        context=_parse_context_config(runtime_config.get("context")),
    )


def _parse_account_config(account_raw: Any) -> RuntimeAccountConfig | None:
    """解析运行时账户配置并校验必填字段。"""
    account_raw = _ensure_object("runtime_config.account", account_raw)
    if account_raw is None:
        return None

    account_name = str(account_raw.get("account_name") or "").strip()
    if not account_name:
        raise ValueError("runtime_config.account.account_name is required")
    if len(account_name) > 128:
        raise ValueError("runtime_config.account.account_name length must be <= 128")

    initial_cash_raw = account_raw.get("initial_cash")
    if initial_cash_raw is None:
        raise ValueError("runtime_config.account.initial_cash is required")

    initial_cash = float(initial_cash_raw)
    if initial_cash <= 0:
        raise ValueError("runtime_config.account.initial_cash must be > 0")

    display_name = account_raw.get("account_display_name")
    return RuntimeAccountConfig(
        account_name=account_name,
        initial_cash=initial_cash,
        account_display_name=str(display_name).strip() if display_name else None,
    )


def _parse_llm_config(llm_raw: Any) -> RuntimeLlmConfig | None:
    """解析运行时 LLM 配置。"""
    llm_raw = _ensure_object("runtime_config.llm", llm_raw)
    if llm_raw is None:
        return None

    provider = str(llm_raw.get("provider") or "").strip().lower()
    if provider not in ALLOWED_LLM_PROVIDERS:
        raise ValueError("runtime_config.llm.provider must be one of gemini|anthropic|openai|deepseek|custom")

    base_url = str(llm_raw.get("base_url") or "").strip()
    model = str(llm_raw.get("model") or "").strip()
    if not base_url:
        raise ValueError("runtime_config.llm.base_url is required")
    if not model:
        raise ValueError("runtime_config.llm.model is required")

    api_token = llm_raw.get("api_token")
    api_token_text = str(api_token).strip() if api_token else None
    return RuntimeLlmConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_token=api_token_text,
        has_token=bool(llm_raw.get("has_token") or api_token_text),
    )


def _parse_strategy_config(strategy_raw: Any) -> RuntimeStrategyConfig | None:
    """解析运行时策略覆盖项。"""
    strategy_raw = _ensure_object("runtime_config.strategy", strategy_raw)
    if strategy_raw is None:
        return None

    try:
        position_max_pct = float(strategy_raw.get("position_max_pct"))
        stop_loss_pct = float(strategy_raw.get("stop_loss_pct"))
        take_profit_pct = float(strategy_raw.get("take_profit_pct"))
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime_config.strategy fields must be numbers") from exc

    if position_max_pct < 0 or position_max_pct > 100:
        raise ValueError("runtime_config.strategy.position_max_pct must be in [0, 100]")
    if stop_loss_pct < 0 or stop_loss_pct > 100:
        raise ValueError("runtime_config.strategy.stop_loss_pct must be in [0, 100]")
    if take_profit_pct < 0 or take_profit_pct > 100:
        raise ValueError("runtime_config.strategy.take_profit_pct must be in [0, 100]")

    return RuntimeStrategyConfig(
        position_max_pct=position_max_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )


def _parse_execution_config(execution_raw: Any) -> RuntimeExecutionConfig | None:
    """解析运行时执行模式配置。"""
    execution_raw = _ensure_object("runtime_config.execution", execution_raw)
    if execution_raw is None:
        return None

    if "credential_ticket" in execution_raw or "ticket_id" in execution_raw:
        raise ValueError("runtime_config.execution.credential_ticket/ticket_id are no longer supported")

    unknown_execution_fields = set(execution_raw.keys()) - ALLOWED_EXECUTION_FIELDS
    if unknown_execution_fields:
        field_list = ", ".join(sorted(unknown_execution_fields))
        raise ValueError(f"runtime_config.execution contains unsupported fields: {field_list}")

    mode = str(execution_raw.get("mode") or "").strip().lower()
    if mode not in ALLOWED_EXECUTION_MODES:
        raise ValueError("runtime_config.execution.mode must be one of paper|broker")

    broker_account_id_raw = execution_raw.get("broker_account_id")
    broker_account_id: int | None = None
    if broker_account_id_raw is not None:
        try:
            broker_account_id = int(broker_account_id_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("runtime_config.execution.broker_account_id must be an integer") from exc
        if broker_account_id <= 0:
            raise ValueError("runtime_config.execution.broker_account_id must be >= 1")
    if mode == "broker" and broker_account_id is None:
        raise ValueError("runtime_config.execution.broker_account_id is required when mode=broker")

    return RuntimeExecutionConfig(
        mode=mode,
        has_ticket=bool(execution_raw.get("has_ticket")),
        broker_account_id=broker_account_id,
    )


def _parse_data_source_config(data_source_raw: Any) -> RuntimeDataSourceConfig | None:
    """解析请求级行情源固定选择。"""
    data_source_raw = _ensure_object("runtime_config.data_source", data_source_raw)
    if data_source_raw is None:
        return None

    unknown_fields = set(data_source_raw.keys()) - ALLOWED_DATA_SOURCE_FIELDS
    if unknown_fields:
        field_list = ", ".join(sorted(unknown_fields))
        raise ValueError(f"runtime_config.data_source contains unsupported fields: {field_list}")

    market_source = str(data_source_raw.get("market_source") or "").strip().lower()
    if market_source not in ALLOWED_MARKET_SOURCES:
        allowed = "|".join(ALLOWED_MARKET_SOURCES)
        raise ValueError(f"runtime_config.data_source.market_source must be one of {allowed}")

    return RuntimeDataSourceConfig(market_source=market_source)


def _parse_context_config(context_raw: Any) -> RuntimeContextConfig | None:
    """解析 Backend 透传的运行时账户上下文。"""
    context_raw = _ensure_object("runtime_config.context", context_raw)
    if context_raw is None:
        return None

    unknown_context_fields = set(context_raw.keys()) - ALLOWED_CONTEXT_FIELDS
    if unknown_context_fields:
        field_list = ", ".join(sorted(unknown_context_fields))
        raise ValueError(f"runtime_config.context contains unsupported fields: {field_list}")

    account_snapshot = context_raw.get("account_snapshot")
    summary = context_raw.get("summary")
    positions = context_raw.get("positions")

    if account_snapshot is not None and not isinstance(account_snapshot, dict):
        raise ValueError("runtime_config.context.account_snapshot must be an object")
    if summary is not None and not isinstance(summary, dict):
        raise ValueError("runtime_config.context.summary must be an object")
    if positions is not None:
        if not isinstance(positions, list):
            raise ValueError("runtime_config.context.positions must be a list")
        if any(not isinstance(item, dict) for item in positions):
            raise ValueError("runtime_config.context.positions items must be objects")

    return RuntimeContextConfig(
        account_snapshot=dict(account_snapshot) if isinstance(account_snapshot, dict) else None,
        summary=dict(summary) if isinstance(summary, dict) else None,
        positions=[dict(item) for item in positions] if isinstance(positions, list) else None,
    )


def _ensure_object(section_name: str, raw: Any) -> dict[str, Any] | None:
    """确保某个配置段要么为空，要么是对象。"""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{section_name} must be an object")
    return raw
