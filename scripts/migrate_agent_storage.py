#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create or upgrade Agent_stock storage tables."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_stock.storage import Base
from src.config import get_config, setup_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    setup_env()
    config = get_config()
    db_url = config.get_db_url()

    engine = create_engine(db_url, pool_pre_ping=True, future=True)
    Base.metadata.create_all(engine)
    logger.info("Base metadata ensured")

    if engine.dialect.name == "postgresql":
        stmts = [
            "ALTER TABLE IF EXISTS paper_accounts ALTER COLUMN name TYPE VARCHAR(128)",
            "ALTER TABLE IF EXISTS agent_runs ADD COLUMN IF NOT EXISTS account_name VARCHAR(128) DEFAULT 'paper-default'",
            "ALTER TABLE IF EXISTS agent_runs ALTER COLUMN account_name TYPE VARCHAR(128)",
            "ALTER TABLE IF EXISTS agent_runs ADD COLUMN IF NOT EXISTS started_at TIMESTAMP",
            "ALTER TABLE IF EXISTS agent_runs ADD COLUMN IF NOT EXISTS ended_at TIMESTAMP",
            "CREATE INDEX IF NOT EXISTS ix_agent_runs_account_name ON agent_runs (account_name)",
            "CREATE TABLE IF NOT EXISTS agent_tasks ("
            "id SERIAL PRIMARY KEY,"
            "task_id VARCHAR(64) UNIQUE NOT NULL,"
            "request_id VARCHAR(128) UNIQUE NULL,"
            "status VARCHAR(16) NOT NULL DEFAULT 'pending',"
            "stock_codes VARCHAR(1000) NOT NULL,"
            "account_name VARCHAR(128) NOT NULL DEFAULT 'paper-default',"
            "run_id VARCHAR(64) NULL,"
            "error_message TEXT NULL,"
            "created_at TIMESTAMP NOT NULL DEFAULT NOW(),"
            "started_at TIMESTAMP NULL,"
            "completed_at TIMESTAMP NULL,"
            "updated_at TIMESTAMP NOT NULL DEFAULT NOW()"
            ")",
            "ALTER TABLE IF EXISTS agent_tasks ALTER COLUMN account_name TYPE VARCHAR(128)",
            "CREATE INDEX IF NOT EXISTS ix_agent_tasks_task_id ON agent_tasks (task_id)",
            "CREATE INDEX IF NOT EXISTS ix_agent_tasks_request_id ON agent_tasks (request_id)",
            "CREATE INDEX IF NOT EXISTS ix_agent_tasks_status ON agent_tasks (status)",
            "CREATE INDEX IF NOT EXISTS ix_agent_tasks_run_id ON agent_tasks (run_id)",
        ]
        with engine.begin() as conn:
            for stmt in stmts:
                conn.execute(text(stmt))
        logger.info("PostgreSQL migrations applied")

    logger.info("Migration completed: %s", db_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
