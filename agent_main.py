# -*- coding: utf-8 -*-
"""CLI entrypoint for multi-agent paper trading workflow."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is importable when running as `python agent_main.py`.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.base import canonical_stock_code
from src.config import Config, get_config, setup_env
from src.logging_config import setup_logging
from agent_stock.services.agent_service import AgentService

setup_env()

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for agent runner."""
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
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Enable legacy notification sending for CLI runs",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def resolve_stock_codes(config: Config, stocks_arg: Optional[str]) -> List[str]:
    """Resolve stock list from args or config."""
    if stocks_arg:
        return [canonical_stock_code(item) for item in stocks_arg.split(",") if item.strip()]

    config.refresh_stock_list()
    return [canonical_stock_code(item) for item in config.stock_list if item]


def run_once(service: AgentService, stock_codes: List[str], *, notify_enabled: bool) -> int:
    """Run one agent cycle."""
    result = service.run_once(
        stock_codes,
        notify_enabled=notify_enabled,
    )
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
    notify_enabled: bool,
) -> int:
    """Run realtime loop with market-session guard."""
    results = service.run_realtime(
        stock_codes,
        interval_minutes=interval_minutes,
        max_cycles=max_cycles,
        notify_enabled=notify_enabled,
    )
    logger.info("Agent realtime finished: cycles=%s", len(results))
    return 0


def main() -> int:
    """Main function."""
    args = parse_arguments()
    config = get_config()

    setup_logging(log_prefix="agent_runner", debug=args.debug, log_dir=config.log_dir)

    mode = args.mode or str(getattr(config, "agent_run_mode", "once") or "once").lower()
    interval_minutes = int(args.interval_minutes or getattr(config, "agent_poll_interval_minutes", 5))

    stock_codes = resolve_stock_codes(config, args.stocks)
    if not stock_codes:
        logger.error("No stock codes configured. Set STOCK_LIST or pass --stocks.")
        return 1

    if not getattr(config, "agent_enabled", False):
        logger.warning("AGENT_ENABLED is false; running agent CLI anyway because explicit entrypoint was used.")

    service = AgentService(config=config)

    try:
        if mode == "realtime":
            return run_realtime(
                service,
                stock_codes,
                interval_minutes=interval_minutes,
                max_cycles=args.max_cycles,
                notify_enabled=args.notify,
            )
        return run_once(service, stock_codes, notify_enabled=args.notify)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Agent runner failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
