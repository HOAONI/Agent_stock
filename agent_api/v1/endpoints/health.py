# -*- coding: utf-8 -*-
"""Health endpoints for Agent API."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from agent_api.deps import get_task_service_dep
from agent_api.v1.schemas.common import HealthResponse
from agent_stock.services.agent_task_service import AgentTaskService

router = APIRouter()


@router.get("/live", response_model=HealthResponse, summary="Liveness probe")
def liveness() -> HealthResponse:
    """Liveness probe for container orchestrators."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        detail="alive",
    )


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
def readiness(task_service: AgentTaskService = Depends(get_task_service_dep)) -> HealthResponse:
    """Readiness probe validating DB connectivity."""
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
