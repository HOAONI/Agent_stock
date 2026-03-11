# -*- coding: utf-8 -*-
"""Internal runtime metadata endpoints for Backend_stock."""

from __future__ import annotations

from fastapi import APIRouter

from agent_api.v1.schemas.common import RuntimeLlmDefaultResponse
from src.config import get_config

router = APIRouter()


@router.get("/llm-default", response_model=RuntimeLlmDefaultResponse)
def get_runtime_llm_default() -> RuntimeLlmDefaultResponse:
    """Expose the currently effective built-in default LLM metadata."""
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
