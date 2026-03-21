# -*- coding: utf-8 -*-
"""FastAPI 应用工厂。

这里统一装配中间件、公开路由和内部路由，并在生命周期钩子中恢复异常中断的
异步任务。阅读 API 链路时，一般从本文件进入，再看 `agent_api/v1/router.py`
和各个 endpoint 模块。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent_api.middlewares.auth import AgentAuthMiddleware
from agent_api.v1.router import (
    api_v1_router,
    health_router,
    internal_backtest_router,
    internal_backtrader_router,
    internal_runtime_router,
    internal_stocks_router,
)
from agent_stock.services.agent_task_service import get_agent_task_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """在应用启动时恢复任务状态并初始化共享资源。"""
    service = get_agent_task_service()
    # 如果上次进程异常退出，任务表里可能残留 processing 状态，这里统一补偿。
    recovered = service.recover_inflight_tasks()
    if recovered > 0:
        logger.warning("Recovered %s stale task(s) on startup", recovered)
    yield


def create_app() -> FastAPI:
    """创建并返回配置完成的 FastAPI 应用。"""
    app = FastAPI(
        title="Agent_stock Service API",
        description="Microservice API for multi-agent paper trading execution",
        version="1.0.0",
        lifespan=app_lifespan,
    )

    app.add_middleware(AgentAuthMiddleware)
    app.include_router(health_router)
    app.include_router(api_v1_router)
    app.include_router(internal_backtrader_router)
    app.include_router(internal_backtest_router)
    app.include_router(internal_runtime_router)
    app.include_router(internal_stocks_router)
    return app


app = create_app()
