# -*- coding: utf-8 -*-
"""提供给 Backend_stock 的内部运行时元数据接口。"""

from __future__ import annotations

from fastapi import APIRouter

from agent_api.v1.schemas.common import RuntimeLlmDefaultResponse
from agent_stock.config import get_config

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
