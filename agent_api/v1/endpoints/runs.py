# -*- coding: utf-8 -*-
"""Agent API 的运行任务接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from agent_api.deps import get_task_service_dep
from agent_api.v1.schemas.common import ErrorResponse
from agent_api.v1.schemas.runs import RunCreateRequest, RunListResponse, RunPayload
from agent_api.v1.schemas.tasks import TaskStatusResponse
from agent_stock.services.agent_task_service import AgentTaskService
from agent_stock.config import redact_sensitive_text

router = APIRouter()


@router.post(
    "",
    response_model=RunPayload,
    responses={
        202: {"model": TaskStatusResponse, "description": "Task accepted"},
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Create one agent run",
)
def create_run(
    request: RunCreateRequest,
    task_service: AgentTaskService = Depends(get_task_service_dep),
):
    """按同步或异步模式创建一次 Agent 运行。"""
    if not request.stock_codes:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": "stock_codes must not be empty",
            },
        )

    runtime_config_payload = request.runtime_config.model_dump(exclude_none=True) if request.runtime_config else None
    runtime_account_name = None
    if runtime_config_payload:
        runtime_account_name = (
            runtime_config_payload.get("account", {}).get("account_name")
            if isinstance(runtime_config_payload.get("account"), dict)
            else None
        )

    if request.account_name and runtime_account_name and request.account_name != runtime_account_name:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_error",
                "message": "account_name conflicts with runtime_config.account.account_name",
            },
        )

    resolved_account_name = runtime_account_name or request.account_name

    if request.async_mode:
        try:
            task_payload = task_service.submit_task(
                stock_codes=request.stock_codes,
                request_id=request.request_id,
                account_name=resolved_account_name,
                runtime_config=runtime_config_payload,
            )
            return JSONResponse(
                status_code=202,
                content=TaskStatusResponse.model_validate(task_payload).model_dump(),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": "validation_error", "message": redact_sensitive_text(str(exc))},
            ) from exc

    try:
        run_payload = task_service.run_sync(
            stock_codes=request.stock_codes,
            request_id=request.request_id,
            account_name=resolved_account_name,
            runtime_config=runtime_config_payload,
        )
        return RunPayload.model_validate(run_payload)
    except ValueError as exc:
        safe_message = redact_sensitive_text(str(exc))
        status_code = 409 if "already in progress" in safe_message else 400
        error_code = "conflict" if status_code == 409 else "validation_error"
        raise HTTPException(
            status_code=status_code,
            detail={"error": error_code, "message": safe_message},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": redact_sensitive_text(str(exc))},
        ) from exc


@router.get("/{run_id}", response_model=RunPayload, responses={404: {"model": ErrorResponse}}, summary="Get one run")
def get_run(run_id: str, task_service: AgentTaskService = Depends(get_task_service_dep)) -> RunPayload:
    """按 `run_id` 查询一次运行结果。"""
    payload = task_service.get_run(run_id)
    if not payload:
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": f"run_id {run_id} not found"},
        )
    return RunPayload.model_validate(payload)


@router.get("", response_model=RunListResponse, summary="List runs")
def list_runs(
    limit: int = Query(20, ge=1, le=200),
    status: str | None = Query(default=None),
    trade_date: str | None = Query(default=None),
    task_service: AgentTaskService = Depends(get_task_service_dep),
) -> RunListResponse:
    """按条件列出最近的运行记录。"""
    try:
        rows = task_service.list_runs(limit=limit, status=status, trade_date_value=trade_date)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "validation_error", "message": redact_sensitive_text(str(exc))},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": redact_sensitive_text(str(exc))},
        ) from exc

    return RunListResponse(
        total=len(rows),
        runs=[RunPayload.model_validate(item) for item in rows],
    )
