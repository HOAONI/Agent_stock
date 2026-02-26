# -*- coding: utf-8 -*-
"""Entry point for Agent_stock microservice."""

from __future__ import annotations

import logging

from src.config import get_config, setup_env
from src.logging_config import setup_logging

setup_env()

config = get_config()
config.validate_service_requirements()

level_name = (config.log_level or "INFO").upper()
level = getattr(logging, level_name, logging.INFO)
setup_logging(
    log_prefix="agent_service",
    console_level=level,
    extra_quiet_loggers=["uvicorn", "fastapi"],
)

from agent_api.app import app  # noqa: E402

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agent_server:app",
        host=config.agent_service_host,
        port=int(config.agent_service_port),
        reload=False,
    )
