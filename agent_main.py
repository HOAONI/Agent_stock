# -*- coding: utf-8 -*-
"""命令行入口。

建议新同学从本文件开始阅读，再顺着 `AgentService -> AgentOrchestrator`
进入核心链路。CLI 本身只负责解析参数、补齐配置并调用应用服务。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# 确保以 `python agent_main.py` 运行时可以导入项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.base import canonical_stock_code
from agent_stock.config import Config, get_config, setup_env
from agent_stock.logging_config import setup_logging
from agent_stock.services.agent_service import AgentService

# 先加载 `.env`，再导入/实例化配置对象，避免入口参数和环境变量脱节。
setup_env()

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """解析 Agent 运行器的命令行参数。"""
    parser = argparse.ArgumentParser(description="Multi-agent stock paper trading runner")
    parser.add_argument("--mode", choices=["once", "realtime"], default=None, help="Run once or realtime loop")
    parser.add_argument("--stocks", type=str, default=None, help="Comma-separated stock list override")
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=None,
        help="Realtime polling interval in minutes (default from config)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional max cycles for realtime mode (useful for tests)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def resolve_stock_codes(config: Config, stocks_arg: Optional[str]) -> List[str]:
    """从参数或配置中解析股票列表。"""
    if stocks_arg:
        return [canonical_stock_code(item) for item in stocks_arg.split(",") if item.strip()]

    config.refresh_stock_list()
    return [canonical_stock_code(item) for item in config.stock_list if item]


def run_once(service: AgentService, stock_codes: List[str]) -> int:
    """执行一次 Agent 运行周期。"""
    result = service.run_once(stock_codes)
    logger.info(
        "Agent once cycle finished: run_id=%s stocks=%s total_asset=%.2f",
        result.run_id,
        len(result.results),
        float(result.account_snapshot.get("total_asset") or 0.0),
    )
    return 0


def run_realtime(
    service: AgentService,
    stock_codes: List[str],
    interval_minutes: int,
    max_cycles: Optional[int],
) -> int:
    """在交易时段守卫控制下运行实时循环。"""
    results = service.run_realtime(
        stock_codes,
        interval_minutes=interval_minutes,
        max_cycles=max_cycles,
    )
    logger.info("Agent realtime finished: cycles=%s", len(results))
    return 0


def main() -> int:
    """主函数。"""
    args = parse_arguments()
    config = get_config()

    setup_logging(log_prefix="agent_runner", debug=args.debug, log_dir=config.log_dir)

    mode = args.mode or str(getattr(config, "agent_run_mode", "once") or "once").lower()
    interval_minutes = int(args.interval_minutes or getattr(config, "agent_poll_interval_minutes", 5))

    # CLI 只解析股票列表，不直接参与交易决策，后续状态都交给应用服务统一处理。
    stock_codes = resolve_stock_codes(config, args.stocks)
    if not stock_codes:
        logger.error("No stock codes configured. Set STOCK_LIST or pass --stocks.")
        return 1

    service = AgentService(config=config)

    try:
        if mode == "realtime":
            return run_realtime(
                service,
                stock_codes,
                interval_minutes=interval_minutes,
                max_cycles=args.max_cycles,
            )
        return run_once(service, stock_codes)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Agent runner failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
