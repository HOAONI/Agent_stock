# -*- coding: utf-8 -*-
"""Agent 问股内部聊天接口。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from agent_api.deps import get_agent_chat_service_dep
from agent_api.v1.schemas.chat import (
    ChatDeleteResponse,
    ChatDoneResponse,
    ChatSessionDetailResponse,
    ChatSessionListResponse,
    InternalChatRequest,
)
from agent_api.v1.schemas.common import ErrorResponse
from agent_stock.services.agent_chat_service import AgentChatHandledError, AgentChatService
from agent_stock.config import redact_sensitive_text

router = APIRouter()
logger = logging.getLogger(__name__)
_INTERNAL_CHAT_ERROR_MESSAGE = "Agent 问股服务异常，请稍后重试。"


def _to_sse(event_name: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


@router.post(
    "/",
    response_model=ChatDoneResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Run one internal agent chat request",
)
async def create_chat(
    request: InternalChatRequest,
    chat_service: AgentChatService = Depends(get_agent_chat_service_dep),
) -> ChatDoneResponse:
    """执行一轮 Agent 问股并同步返回结果。"""
    try:
        payload = await chat_service.handle_chat(request.model_dump(exclude_none=True))
        return ChatDoneResponse.model_validate(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "validation_error", "message": redact_sensitive_text(str(exc))},
        ) from exc
    except AgentChatHandledError as exc:
        return ChatDoneResponse.model_validate(exc.final_payload)
    except Exception as exc:
        logger.exception("Internal agent chat request failed")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": _INTERNAL_CHAT_ERROR_MESSAGE},
        ) from exc


@router.post(
    "/stream",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Run one internal agent chat request with SSE streaming",
)
async def create_chat_stream(
    request: InternalChatRequest,
    chat_service: AgentChatService = Depends(get_agent_chat_service_dep),
) -> StreamingResponse:
    """执行一轮 Agent 问股并按 SSE 推送 thinking/tool/done 事件。"""
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    request_payload = request.model_dump(exclude_none=True)

    async def _emit(event_name: str, payload: dict[str, Any]) -> None:
        await queue.put(_to_sse(event_name, payload))

    async def _worker() -> None:
        try:
            result = await chat_service.handle_chat(request_payload, event_handler=_emit)
            await queue.put(_to_sse("done", result))
        except ValueError as exc:
            await queue.put(
                _to_sse(
                    "error",
                    {"message": redact_sensitive_text(str(exc)), "error": "validation_error"},
                )
            )
        except AgentChatHandledError as exc:
            await queue.put(_to_sse("done", ChatDoneResponse.model_validate(exc.final_payload).model_dump()))
        except Exception as exc:
            logger.exception("Internal agent chat stream failed")
            await queue.put(
                _to_sse(
                    "error",
                    {"message": _INTERNAL_CHAT_ERROR_MESSAGE, "error": "internal_error"},
                )
            )
        finally:
            await queue.put(None)

    asyncio.create_task(_worker())

    async def _stream():
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get(
    "/sessions",
    response_model=ChatSessionListResponse,
    responses={400: {"model": ErrorResponse}},
    summary="List chat sessions by owner",
)
def list_chat_sessions(
    owner_user_id: int = Query(..., ge=1),
    limit: int = Query(50, ge=1, le=100),
    chat_service: AgentChatService = Depends(get_agent_chat_service_dep),
) -> ChatSessionListResponse:
    """按 owner_user_id 列出聊天会话。"""
    payload = chat_service.list_sessions(owner_user_id, limit=limit)
    return ChatSessionListResponse.model_validate(payload)


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionDetailResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get one chat session detail",
)
def get_chat_session(
    session_id: str,
    owner_user_id: int = Query(..., ge=1),
    chat_service: AgentChatService = Depends(get_agent_chat_service_dep),
) -> ChatSessionDetailResponse:
    """查询单个会话与消息明细。"""
    payload = chat_service.get_session_detail(owner_user_id, session_id)
    if not payload:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": f"session_id {session_id} not found"},
        )
    return ChatSessionDetailResponse.model_validate(payload)


@router.delete(
    "/sessions/{session_id}",
    response_model=ChatDeleteResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Delete one chat session",
)
def delete_chat_session(
    session_id: str,
    owner_user_id: int = Query(..., ge=1),
    chat_service: AgentChatService = Depends(get_agent_chat_service_dep),
) -> ChatDeleteResponse:
    """删除单个会话及其消息。"""
    success = chat_service.delete_session(owner_user_id, session_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": f"session_id {session_id} not found"},
        )
    return ChatDeleteResponse(success=True)
