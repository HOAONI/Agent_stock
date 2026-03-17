# -*- coding: utf-8 -*-
"""微服务入口。

本文件在导入 FastAPI 应用前完成环境变量加载、服务模式校验和日志初始化，
确保 `agent_api.app` 在导入时拿到的是最终生效的配置。
"""

from __future__ import annotations

import logging

from agent_stock.config import get_config, setup_env
from agent_stock.logging_config import setup_logging

setup_env()

config = get_config()
# 服务模式缺少关键配置时在启动阶段尽早失败，避免应用启动后才在首个请求时报错。
config.validate_service_requirements()

level_name = (config.log_level or "INFO").upper()
level = getattr(logging, level_name, logging.INFO)
setup_logging(
    log_prefix="agent_service",
    log_dir=None,
    console_level=level,
    extra_quiet_loggers=["uvicorn", "fastapi"],
    write_files=False,
)

# 需要在环境与日志准备完成后再导入 FastAPI 应用，避免模块导入顺序带来副作用。
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
