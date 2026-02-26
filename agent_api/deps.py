# -*- coding: utf-8 -*-
"""Dependency providers for Agent API."""

from __future__ import annotations

from fastapi import Depends

from agent_stock.services.agent_task_service import AgentTaskService, get_agent_task_service
from src.config import Config, get_config


def get_config_dep() -> Config:
    """Get config singleton."""
    return get_config()


def get_task_service_dep(config: Config = Depends(get_config_dep)) -> AgentTaskService:
    """Get task service singleton."""
    return get_agent_task_service(config=config)

