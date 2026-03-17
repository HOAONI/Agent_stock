# -*- coding: utf-8 -*-
"""Agent API 的账户查询接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent_api.deps import get_task_service_dep
from agent_api.v1.schemas.accounts import AccountSnapshotResponse
from agent_api.v1.schemas.common import ErrorResponse
from agent_stock.services.agent_task_service import AgentTaskService

router = APIRouter()


@router.get(
    "/{account_name}/snapshot",
    response_model=AccountSnapshotResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get account snapshot",
)
def get_account_snapshot(
    account_name: str,
    task_service: AgentTaskService = Depends(get_task_service_dep),
) -> AccountSnapshotResponse:
    """获取账户资金与持仓快照。"""
    payload = task_service.get_account_snapshot(account_name)
    if not payload:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": f"account {account_name} not found"},
        )
    return AccountSnapshotResponse.model_validate(payload)
