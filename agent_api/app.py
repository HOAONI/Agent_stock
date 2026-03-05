# -*- coding: utf-8 -*-
"""FastAPI app factory for Agent microservice."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent_api.middlewares.auth import AgentAuthMiddleware
from agent_api.v1.router import api_v1_router, health_router, internal_backtest_router, internal_backtrader_router
from agent_stock.services.agent_task_service import get_agent_task_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Initialize shared state on startup."""
    service = get_agent_task_service()
    recovered = service.recover_inflight_tasks()
    if recovered > 0:
        logger.warning("Recovered %s stale task(s) on startup", recovered)
    yield


def create_app() -> FastAPI:
    """Create configured FastAPI app."""
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
    return app


app = create_app()
