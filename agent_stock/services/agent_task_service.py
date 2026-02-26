# -*- coding: utf-8 -*-
"""Asynchronous task service for Agent runs."""

from __future__ import annotations

import logging
import threading
import uuid
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from data_provider.base import canonical_stock_code
from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.storage import DatabaseManager
from src.config import Config, get_config, redact_sensitive_text

logger = logging.getLogger(__name__)


class AgentTaskService:
    """Task lifecycle manager backed by DB + in-process thread pool."""

    def __init__(
        self,
        *,
        config: Optional[Config] = None,
        db_manager: Optional[DatabaseManager] = None,
        execution_repo: Optional[ExecutionRepository] = None,
        agent_service=None,
    ) -> None:
        self.config = config or get_config()
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = execution_repo or ExecutionRepository(self.db)
        self._agent_service = agent_service
        self._max_workers = max(1, int(getattr(self.config, "agent_task_max_workers", 3)))
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="agent_task")
        self._futures: Dict[str, object] = {}
        self._lock = threading.RLock()

    @property
    def agent_service(self):
        if self._agent_service is None:
            from agent_stock.services.agent_service import AgentService

            self._agent_service = AgentService(
                config=self.config,
                db_manager=self.db,
                execution_repo=self.repo,
            )
        return self._agent_service

    def recover_inflight_tasks(self) -> int:
        """Mark stale in-flight tasks as failed."""
        return self.repo.mark_inflight_tasks_failed(reason="service_restarted")

    def run_sync(
        self,
        *,
        stock_codes: List[str],
        request_id: Optional[str] = None,
        account_name: Optional[str] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Run one cycle synchronously and return persisted run payload."""
        normalized_codes = self._normalize_codes(stock_codes)
        resolved_account = self._resolve_account_name(account_name=account_name, runtime_config=runtime_config)

        if request_id:
            existing = self.repo.get_agent_task_by_request_id(request_id)
            if existing:
                if existing.get("status") == "completed" and existing.get("run_id"):
                    run_payload = self.repo.get_agent_run(existing["run_id"])
                    if run_payload:
                        return run_payload
                if existing.get("status") in ("pending", "processing"):
                    raise ValueError(f"request_id {request_id} is already in progress")

        run_result = self.agent_service.run_once(
            normalized_codes,
            account_name=resolved_account,
            request_id=request_id,
            notify_enabled=False,
            write_reports=bool(getattr(self.config, "agent_write_local_reports", False)),
            runtime_config=runtime_config,
        )
        run_payload = self.repo.get_agent_run(run_result.run_id) or run_result.to_dict()

        if request_id:
            task_id = uuid.uuid4().hex
            self.repo.create_agent_task(
                task_id=task_id,
                request_id=request_id,
                stock_codes=normalized_codes,
                account_name=resolved_account,
                status="completed",
            )
            self.repo.update_agent_task(
                task_id,
                status="completed",
                run_id=run_result.run_id,
                started_at=run_result.started_at,
                completed_at=run_result.ended_at,
            )

        return run_payload

    def submit_task(
        self,
        *,
        stock_codes: List[str],
        request_id: Optional[str] = None,
        account_name: Optional[str] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Submit one asynchronous run request."""
        normalized_codes = self._normalize_codes(stock_codes)
        resolved_account = self._resolve_account_name(account_name=account_name, runtime_config=runtime_config)

        if request_id:
            existing = self.repo.get_agent_task_by_request_id(request_id)
            if existing:
                return existing

        task_id = uuid.uuid4().hex
        self.repo.create_agent_task(
            task_id=task_id,
            request_id=request_id,
            stock_codes=normalized_codes,
            account_name=resolved_account,
            status="pending",
        )

        with self._lock:
            safe_runtime_config = copy.deepcopy(runtime_config or {})
            future = self._executor.submit(
                self._execute_task,
                task_id,
                request_id,
                normalized_codes,
                resolved_account,
                safe_runtime_config,
            )
            self._futures[task_id] = future

        return self.repo.get_agent_task(task_id)

    def _execute_task(
        self,
        task_id: str,
        request_id: Optional[str],
        stock_codes: List[str],
        account_name: str,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Worker body for asynchronous task execution."""
        started_at = datetime.now()
        self.repo.update_agent_task(task_id, status="processing", started_at=started_at)

        try:
            result = self.agent_service.run_once(
                stock_codes,
                account_name=account_name,
                request_id=request_id,
                notify_enabled=False,
                write_reports=bool(getattr(self.config, "agent_write_local_reports", False)),
                runtime_config=runtime_config,
            )
            self.repo.update_agent_task(
                task_id,
                status="completed",
                run_id=result.run_id,
                completed_at=result.ended_at,
            )
        except Exception as exc:
            safe_error = redact_sensitive_text(str(exc))
            logger.error("Async task failed: task_id=%s error=%s", task_id, safe_error)
            self.repo.update_agent_task(
                task_id,
                status="failed",
                error_message=safe_error,
                completed_at=datetime.now(),
            )
        finally:
            with self._lock:
                self._futures.pop(task_id, None)

    def get_task(self, task_id: str) -> Dict:
        """Fetch task status by task_id."""
        return self.repo.get_agent_task(task_id)

    def get_run(self, run_id: str) -> Dict:
        """Fetch run payload by run_id."""
        return self.repo.get_agent_run(run_id)

    def list_runs(
        self,
        *,
        limit: int = 20,
        status: Optional[str] = None,
        trade_date_value: Optional[str] = None,
    ) -> List[Dict]:
        """List run payloads with optional filters."""
        parsed_trade_date: Optional[date] = None
        if trade_date_value:
            parsed_trade_date = date.fromisoformat(trade_date_value)
        return self.repo.list_agent_runs(limit=limit, status=status, trade_date=parsed_trade_date)

    def get_account_snapshot(self, account_name: str) -> Dict:
        """Fetch account snapshot by account name."""
        return self.repo.get_account_snapshot(account_name)

    @staticmethod
    def _normalize_codes(stock_codes: List[str]) -> List[str]:
        normalized = [canonical_stock_code(item) for item in stock_codes if item and str(item).strip()]
        unique = list(dict.fromkeys(normalized))
        if not unique:
            raise ValueError("stock_codes must not be empty")
        return unique

    def _resolve_account_name(
        self,
        *,
        account_name: Optional[str],
        runtime_config: Optional[Dict[str, Any]],
    ) -> str:
        """Resolve effective account name for sync/async task entrypoints."""
        top_level_account = str(account_name or "").strip() or None
        runtime_account = self._extract_runtime_account_name(runtime_config)

        if top_level_account and runtime_account and top_level_account != runtime_account:
            raise ValueError("account_name conflicts with runtime_config.account.account_name")

        resolved = (
            runtime_account
            or top_level_account
            or str(getattr(self.config, "agent_account_name", "paper-default") or "paper-default")
        )
        if len(resolved) > 128:
            raise ValueError("account_name length must be <= 128")
        return resolved

    @staticmethod
    def _extract_runtime_account_name(runtime_config: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(runtime_config, dict):
            return None
        account = runtime_config.get("account")
        if not isinstance(account, dict):
            return None
        account_name = str(account.get("account_name") or "").strip()
        return account_name or None


_TASK_SERVICE: Optional[AgentTaskService] = None
_TASK_SERVICE_LOCK = threading.Lock()


def get_agent_task_service(
    *,
    config: Optional[Config] = None,
    db_manager: Optional[DatabaseManager] = None,
    execution_repo: Optional[ExecutionRepository] = None,
    agent_service=None,
) -> AgentTaskService:
    """Get singleton task service."""
    global _TASK_SERVICE
    if _TASK_SERVICE is None:
        with _TASK_SERVICE_LOCK:
            if _TASK_SERVICE is None:
                _TASK_SERVICE = AgentTaskService(
                    config=config,
                    db_manager=db_manager,
                    execution_repo=execution_repo,
                    agent_service=agent_service,
                )
    return _TASK_SERVICE


def reset_agent_task_service() -> None:
    """Reset singleton task service for tests."""
    global _TASK_SERVICE
    with _TASK_SERVICE_LOCK:
        if _TASK_SERVICE is not None:
            _TASK_SERVICE._executor.shutdown(wait=False, cancel_futures=True)
        _TASK_SERVICE = None
