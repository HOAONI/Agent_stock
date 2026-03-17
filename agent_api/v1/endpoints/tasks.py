# -*- coding: utf-8 -*-
"""Agent API 的异步任务接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent_api.deps import get_task_service_dep
from agent_api.v1.schemas.common import ErrorResponse
from agent_api.v1.schemas.tasks import TaskStatusResponse
from agent_stock.services.agent_task_service import AgentTaskService

router = APIRouter()


@router.get("/{task_id}", response_model=TaskStatusResponse, responses={404: {"model": ErrorResponse}}, summary="Get task status")
def get_task(task_id: str, task_service: AgentTaskService = Depends(get_task_service_dep)) -> TaskStatusResponse:
    """获取单个异步任务状态。"""
    payload = task_service.get_task(task_id)
    if not payload:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": f"task_id {task_id} not found"},
        )
    return TaskStatusResponse.model_validate(payload)
