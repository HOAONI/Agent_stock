# -*- coding: utf-8 -*-
"""Compatibility re-export for AgentTaskService from agent_stock."""

from agent_stock.services.agent_task_service import AgentTaskService, get_agent_task_service, reset_agent_task_service

__all__ = [
    "AgentTaskService",
    "get_agent_task_service",
    "reset_agent_task_service",
]
