# -*- coding: utf-8 -*-
"""API router composition."""

from __future__ import annotations

from fastapi import APIRouter

from agent_api.v1.endpoints import accounts, health, runs, tasks

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(runs.router, prefix="/runs", tags=["Runs"])
api_v1_router.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])
api_v1_router.include_router(accounts.router, prefix="/accounts", tags=["Accounts"])

health_router = APIRouter(prefix="/api/health")
health_router.include_router(health.router, tags=["Health"])
