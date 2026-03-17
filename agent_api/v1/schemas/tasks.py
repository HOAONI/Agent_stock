# -*- coding: utf-8 -*-
"""Agent API 的任务状态数据模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskStatusResponse(BaseModel):
    """任务状态响应载荷。"""

    task_id: str
    status: str
    request_id: str | None = None
    stock_codes: list[str] = Field(default_factory=list)
    account_name: str | None = None
    run_id: str | None = None
    error_message: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str | None = None
