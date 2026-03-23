# -*- coding: utf-8 -*-
"""
===================================
日志配置模块 - 统一的日志系统初始化
===================================

职责：
1. 提供统一的日志格式和配置常量
2. 支持控制台 + 文件（常规/调试）三层日志输出
3. 自动降低第三方库日志级别
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from agent_stock.time_utils import local_now

# ============================================================
# 日志格式常量
# ============================================================

LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 默认需要降低日志级别的第三方库
DEFAULT_QUIET_LOGGERS = [
    'urllib3',
    'sqlalchemy',
    'google',
    'httpx',
]


def setup_logging(
    log_prefix: str = "app",
    log_dir: str | None = "./logs",
    console_level: int | None = None,
    debug: bool = False,
    extra_quiet_loggers: list[str] | None = None,
    write_files: bool = True,
) -> None:
    """
    统一的日志系统初始化

    配置三层日志输出：
    1. 控制台：根据 debug 参数或 console_level 设置级别
    2. 常规日志文件：INFO 级别，10MB 轮转，保留 5 个备份
    3. 调试日志文件：DEBUG 级别，50MB 轮转，保留 3 个备份

    参数：
        log_prefix: 日志文件名前缀（如 "api_server" -> api_server_20240101.log）
        log_dir: 日志文件目录，默认。/logs
        console_level: 控制台日志级别（可选，优先于 debug 参数）
        debug: 是否启用调试模式（控制台输出 DEBUG 级别）
        extra_quiet_loggers: 额外需要降低日志级别的第三方库列表
        write_files: 是否写入本地日志文件
    """
    # 确定控制台日志级别
    if console_level is not None:
        level = console_level
    else:
        level = logging.DEBUG if debug else logging.INFO

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 根日志记录器设为 DEBUG，由各 handler 控制输出级别

    # 清除已有 handler，避免重复添加
    if root_logger.handlers:
        root_logger.handlers.clear()

    # 处理器 1：控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    if write_files and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        today_str = local_now().strftime('%Y%m%d')
        log_file = log_path / f"{log_prefix}_{today_str}.log"
        debug_log_file = log_path / f"{log_prefix}_debug_{today_str}.log"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        root_logger.addHandler(file_handler)

        debug_handler = RotatingFileHandler(
            debug_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=3,
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        root_logger.addHandler(debug_handler)

    # 降低第三方库的日志级别
    quiet_loggers = DEFAULT_QUIET_LOGGERS.copy()
    if extra_quiet_loggers:
        quiet_loggers.extend(extra_quiet_loggers)

    for logger_name in quiet_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if write_files and log_dir:
        logging.info(f"日志系统初始化完成，日志目录: {Path(log_dir).absolute()}")
    else:
        logging.info("日志系统初始化完成，仅输出到控制台")
