# -*- coding: utf-8 -*-
"""提供给 Backend_stock 的内部运行时元数据接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from agent_api.deps import get_runtime_market_service_dep
from agent_api.v1.schemas.common import RuntimeLlmDefaultResponse, RuntimeMarketSourcesResponse
from agent_stock.config import get_config
from agent_stock.services.runtime_market_service import RuntimeMarketService

router = APIRouter()


@router.get("/llm-default", response_model=RuntimeLlmDefaultResponse)
def get_runtime_llm_default() -> RuntimeLlmDefaultResponse:
    """返回当前生效的内置默认 LLM 元数据。"""
    resolved = get_config().resolve_default_runtime_llm()
    if resolved is None:
        return RuntimeLlmDefaultResponse(available=False, has_token=False)

    return RuntimeLlmDefaultResponse(
        available=True,
        provider=resolved.provider,
        model=resolved.model,
        base_url=resolved.base_url,
        has_token=resolved.has_token,
    )


@router.get("/market-sources", response_model=RuntimeMarketSourcesResponse)
def get_runtime_market_sources(
    service: RuntimeMarketService = Depends(get_runtime_market_service_dep),
) -> RuntimeMarketSourcesResponse:
    """返回 Agent 当前支持的市场源候选列表与可用性。"""
    return RuntimeMarketSourcesResponse(**service.get_market_source_options())
