# -*- coding: utf-8 -*-
"""Agent API 的健康检查接口。"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from agent_api.deps import get_task_service_dep
from agent_api.v1.schemas.common import HealthResponse
from agent_stock.services.agent_task_service import AgentTaskService

router = APIRouter()


@router.get("/live", response_model=HealthResponse, summary="Liveness probe")
def liveness() -> HealthResponse:
    """供容器编排器使用的存活探针。"""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        detail="alive",
    )


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
def readiness(task_service: AgentTaskService = Depends(get_task_service_dep)) -> HealthResponse:
    """用于校验数据库连接的就绪探针。"""
    if not task_service.db.ping():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "not_ready",
                "message": "Database connection is not ready",
            },
        )
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        detail="ready",
    )
