# -*- coding: utf-8 -*-
"""Agent 聊天接口数据模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_api.v1.schemas.runs import RuntimeConfigRequest


class InternalChatRequest(BaseModel):
    """内部聊天请求。"""

    model_config = ConfigDict(extra="forbid")

    owner_user_id: int = Field(ge=1)
    username: str | None = Field(default=None, max_length=128)
    message: str = Field(min_length=1, max_length=10000)
    session_id: str | None = Field(default=None, min_length=1, max_length=64)
    context: dict[str, Any] | None = None
    runtime_config: RuntimeConfigRequest | None = None


class ChatDoneResponse(BaseModel):
    """统一 done 响应。"""

    session_id: str
    content: str
    structured_result: dict[str, Any] | None = None
    candidate_orders: list[dict[str, Any]] = Field(default_factory=list)
    execution_result: dict[str, Any] | None = None
    status: str


class ChatSessionItemResponse(BaseModel):
    """会话列表项。"""

    session_id: str
    title: str | None = None
    latest_message_preview: str | None = None
    message_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class ChatSessionListResponse(BaseModel):
    """会话列表响应。"""

    total: int
    items: list[ChatSessionItemResponse]


class ChatMessageResponse(BaseModel):
    """单条会话消息。"""

    id: int
    session_id: str
    role: str
    content: str
    meta: dict[str, Any] | None = None
    created_at: str | None = None


class ChatSessionDetailResponse(BaseModel):
    """会话详情响应。"""

    session_id: str
    title: str | None = None
    latest_message_preview: str | None = None
    message_count: int = 0
    context: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    messages: list[ChatMessageResponse]


class ChatDeleteResponse(BaseModel):
    """删除会话响应。"""

    success: bool
