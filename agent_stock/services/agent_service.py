# -*- coding: utf-8 -*-
"""Application service for running multi-agent paper trading workflows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_stock.agents.contracts import AgentRunResult
from agent_stock.agents.orchestrator import AgentOrchestrator
from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.storage import DatabaseManager
from src.config import (
    AgentRuntimeConfig,
    Config,
    RuntimeAccountConfig,
    RuntimeContextConfig,
    RuntimeExecutionConfig,
    RuntimeLlmConfig,
    RuntimeStrategyConfig,
    get_config,
)
from src.notification import NotificationService


@dataclass(frozen=True)
class _ResolvedRuntimeContext:
    """Resolved request-scoped context for one run invocation."""

    account_name: str
    initial_cash_override: Optional[float]
    runtime_config: Optional[AgentRuntimeConfig]


class AgentService:
    """High-level use-case layer for agent orchestration."""

    def __init__(
        self,
        config: Optional[Config] = None,
        db_manager: Optional[DatabaseManager] = None,
        orchestrator: Optional[AgentOrchestrator] = None,
        execution_repo: Optional[ExecutionRepository] = None,
        notifier: Optional[NotificationService] = None,
    ) -> None:
        self.config = config or get_config()
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = execution_repo or ExecutionRepository(self.db)
        self.orchestrator = orchestrator or AgentOrchestrator(config=self.config, db_manager=self.db, execution_repo=self.repo)
        self.notifier = notifier

    def run_once(
        self,
        stock_codes: List[str],
        *,
        account_name: Optional[str] = None,
        request_id: Optional[str] = None,
        notify_enabled: Optional[bool] = None,
        write_reports: Optional[bool] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> AgentRunResult:
        """Run one cycle and persist snapshots."""
        runtime_ctx = self._resolve_runtime_context(
            account_name=account_name,
            runtime_config=runtime_config,
        )
        result = self.orchestrator.run_once(
            stock_codes,
            account_name=runtime_ctx.account_name,
            request_id=request_id,
            initial_cash_override=runtime_ctx.initial_cash_override,
            runtime_config=runtime_ctx.runtime_config,
        )
        self._finalize_run(
            run_result=result,
            account_name=runtime_ctx.account_name,
            notify_enabled=self._resolve_notify_enabled(notify_enabled),
            write_reports=self._resolve_write_reports(write_reports),
        )
        return result

    def run_realtime(
        self,
        stock_codes: List[str],
        *,
        interval_minutes: int,
        max_cycles: Optional[int] = None,
        account_name: Optional[str] = None,
        notify_enabled: Optional[bool] = None,
        write_reports: Optional[bool] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> List[AgentRunResult]:
        """Run realtime loop and finalize every cycle."""
        runtime_ctx = self._resolve_runtime_context(
            account_name=account_name,
            runtime_config=runtime_config,
        )
        run_results = self.orchestrator.run_realtime(
            stock_codes,
            interval_minutes=interval_minutes,
            max_cycles=max_cycles,
            heartbeat_sleep=5.0,
            account_name=runtime_ctx.account_name,
            initial_cash_override=runtime_ctx.initial_cash_override,
            runtime_config=runtime_ctx.runtime_config,
        )
        should_notify = self._resolve_notify_enabled(notify_enabled)
        should_write_reports = self._resolve_write_reports(write_reports)
        for item in run_results:
            self._finalize_run(
                run_result=item,
                account_name=runtime_ctx.account_name,
                notify_enabled=should_notify,
                write_reports=should_write_reports,
            )
        return run_results

    def run_async(
        self,
        stock_codes: List[str],
        *,
        request_id: Optional[str] = None,
        account_name: Optional[str] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Submit async task via task service."""
        from agent_stock.services.agent_task_service import get_agent_task_service

        runtime_ctx = self._resolve_runtime_context(
            account_name=account_name,
            runtime_config=runtime_config,
        )

        service = get_agent_task_service(
            config=self.config,
            db_manager=self.db,
            execution_repo=self.repo,
            agent_service=self,
        )
        return service.submit_task(
            stock_codes=stock_codes,
            request_id=request_id,
            account_name=runtime_ctx.account_name,
            runtime_config=runtime_config,
        )

    def _resolve_runtime_context(
        self,
        *,
        account_name: Optional[str] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> _ResolvedRuntimeContext:
        """Resolve per-request runtime overrides without mutating global config."""
        parsed_runtime = self._parse_runtime_config(runtime_config)
        runtime_account_name = parsed_runtime.account.account_name if parsed_runtime and parsed_runtime.account else None
        top_level_account_name = str(account_name).strip() if account_name else None
        if top_level_account_name and len(top_level_account_name) > 128:
            raise ValueError("account_name length must be <= 128")

        if runtime_account_name and top_level_account_name and runtime_account_name != top_level_account_name:
            raise ValueError("account_name conflicts with runtime_config.account.account_name")

        resolved_account = (
            runtime_account_name
            or top_level_account_name
            or str(getattr(self.config, "agent_account_name", "paper-default") or "paper-default")
        )

        initial_cash_override = (
            parsed_runtime.account.initial_cash
            if parsed_runtime and parsed_runtime.account and parsed_runtime.account.initial_cash is not None
            else None
        )
        return _ResolvedRuntimeContext(
            account_name=resolved_account,
            initial_cash_override=initial_cash_override,
            runtime_config=parsed_runtime,
        )

    @staticmethod
    def _parse_runtime_config(runtime_config: Optional[Dict[str, Any]]) -> Optional[AgentRuntimeConfig]:
        """Parse request runtime configuration from plain dict."""
        if not runtime_config:
            return None
        if not isinstance(runtime_config, dict):
            raise ValueError("runtime_config must be an object")

        account_cfg: Optional[RuntimeAccountConfig] = None
        llm_cfg: Optional[RuntimeLlmConfig] = None
        strategy_cfg: Optional[RuntimeStrategyConfig] = None
        execution_cfg: Optional[RuntimeExecutionConfig] = None
        context_cfg: Optional[RuntimeContextConfig] = None

        account_raw = runtime_config.get("account")
        if account_raw is not None:
            if not isinstance(account_raw, dict):
                raise ValueError("runtime_config.account must be an object")
            account_name = str(account_raw.get("account_name") or "").strip()
            if not account_name:
                raise ValueError("runtime_config.account.account_name is required")
            if len(account_name) > 128:
                raise ValueError("runtime_config.account.account_name length must be <= 128")
            initial_cash_raw = account_raw.get("initial_cash")
            initial_cash = None
            if initial_cash_raw is None:
                raise ValueError("runtime_config.account.initial_cash is required")
            initial_cash = float(initial_cash_raw)
            if initial_cash <= 0:
                raise ValueError("runtime_config.account.initial_cash must be > 0")
            display_name = account_raw.get("account_display_name")
            account_cfg = RuntimeAccountConfig(
                account_name=account_name,
                initial_cash=initial_cash,
                account_display_name=str(display_name).strip() if display_name else None,
            )

        llm_raw = runtime_config.get("llm")
        if llm_raw is not None:
            if not isinstance(llm_raw, dict):
                raise ValueError("runtime_config.llm must be an object")
            provider = str(llm_raw.get("provider") or "").strip().lower()
            if provider not in {"gemini", "anthropic", "openai", "deepseek", "custom"}:
                raise ValueError("runtime_config.llm.provider must be one of gemini|anthropic|openai|deepseek|custom")
            base_url = str(llm_raw.get("base_url") or "").strip()
            model = str(llm_raw.get("model") or "").strip()
            if not base_url:
                raise ValueError("runtime_config.llm.base_url is required")
            if not model:
                raise ValueError("runtime_config.llm.model is required")
            api_token = llm_raw.get("api_token")
            api_token_text = str(api_token).strip() if api_token else None
            llm_cfg = RuntimeLlmConfig(
                provider=provider,
                base_url=base_url,
                model=model,
                api_token=api_token_text,
                has_token=bool(llm_raw.get("has_token") or api_token_text),
            )

        strategy_raw = runtime_config.get("strategy")
        if strategy_raw is not None:
            if not isinstance(strategy_raw, dict):
                raise ValueError("runtime_config.strategy must be an object")
            try:
                position_max_pct = float(strategy_raw.get("position_max_pct"))
                stop_loss_pct = float(strategy_raw.get("stop_loss_pct"))
                take_profit_pct = float(strategy_raw.get("take_profit_pct"))
            except (TypeError, ValueError) as exc:
                raise ValueError("runtime_config.strategy fields must be numbers") from exc

            if position_max_pct < 0 or position_max_pct > 100:
                raise ValueError("runtime_config.strategy.position_max_pct must be in [0, 100]")
            if stop_loss_pct < 0 or stop_loss_pct > 100:
                raise ValueError("runtime_config.strategy.stop_loss_pct must be in [0, 100]")
            if take_profit_pct < 0 or take_profit_pct > 100:
                raise ValueError("runtime_config.strategy.take_profit_pct must be in [0, 100]")

            strategy_cfg = RuntimeStrategyConfig(
                position_max_pct=position_max_pct,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )

        execution_raw = runtime_config.get("execution")
        if execution_raw is not None:
            if not isinstance(execution_raw, dict):
                raise ValueError("runtime_config.execution must be an object")
            if "credential_ticket" in execution_raw or "ticket_id" in execution_raw:
                raise ValueError("runtime_config.execution.credential_ticket/ticket_id are no longer supported")

            allowed_execution_fields = {"mode", "has_ticket", "broker_account_id"}
            unknown_execution_fields = set(execution_raw.keys()) - allowed_execution_fields
            if unknown_execution_fields:
                field_list = ", ".join(sorted(unknown_execution_fields))
                raise ValueError(f"runtime_config.execution contains unsupported fields: {field_list}")

            mode = str(execution_raw.get("mode") or "").strip().lower()
            if mode not in {"paper", "broker"}:
                raise ValueError("runtime_config.execution.mode must be one of paper|broker")

            broker_account_id_raw = execution_raw.get("broker_account_id")
            broker_account_id: Optional[int] = None
            if broker_account_id_raw is not None:
                try:
                    broker_account_id = int(broker_account_id_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError("runtime_config.execution.broker_account_id must be an integer") from exc
                if broker_account_id <= 0:
                    raise ValueError("runtime_config.execution.broker_account_id must be >= 1")
            if mode == "broker" and broker_account_id is None:
                raise ValueError("runtime_config.execution.broker_account_id is required when mode=broker")

            execution_cfg = RuntimeExecutionConfig(
                mode=mode,
                has_ticket=bool(execution_raw.get("has_ticket")),
                broker_account_id=broker_account_id,
            )

        context_raw = runtime_config.get("context")
        if context_raw is not None:
            if not isinstance(context_raw, dict):
                raise ValueError("runtime_config.context must be an object")
            allowed_context_fields = {"account_snapshot", "summary", "positions"}
            unknown_context_fields = set(context_raw.keys()) - allowed_context_fields
            if unknown_context_fields:
                field_list = ", ".join(sorted(unknown_context_fields))
                raise ValueError(f"runtime_config.context contains unsupported fields: {field_list}")

            account_snapshot = context_raw.get("account_snapshot")
            summary = context_raw.get("summary")
            positions = context_raw.get("positions")

            if account_snapshot is not None and not isinstance(account_snapshot, dict):
                raise ValueError("runtime_config.context.account_snapshot must be an object")
            if summary is not None and not isinstance(summary, dict):
                raise ValueError("runtime_config.context.summary must be an object")
            if positions is not None:
                if not isinstance(positions, list):
                    raise ValueError("runtime_config.context.positions must be a list")
                if any(not isinstance(item, dict) for item in positions):
                    raise ValueError("runtime_config.context.positions items must be objects")

            context_cfg = RuntimeContextConfig(
                account_snapshot=dict(account_snapshot) if isinstance(account_snapshot, dict) else None,
                summary=dict(summary) if isinstance(summary, dict) else None,
                positions=[dict(item) for item in positions] if isinstance(positions, list) else None,
            )

        return AgentRuntimeConfig(
            account=account_cfg,
            llm=llm_cfg,
            strategy=strategy_cfg,
            execution=execution_cfg,
            context=context_cfg,
        )

    def _resolve_notify_enabled(self, notify_enabled: Optional[bool]) -> bool:
        if notify_enabled is not None:
            return bool(notify_enabled)
        return bool(getattr(self.config, "agent_legacy_notify_enabled", False))

    def _resolve_write_reports(self, write_reports: Optional[bool]) -> bool:
        if write_reports is not None:
            return bool(write_reports)
        return bool(getattr(self.config, "agent_write_local_reports", False))

    def _get_notifier(self) -> NotificationService:
        if self.notifier is None:
            self.notifier = NotificationService()
        return self.notifier

    def _finalize_run(
        self,
        *,
        run_result: AgentRunResult,
        account_name: str,
        notify_enabled: bool,
        write_reports: bool,
    ) -> None:
        """Persist snapshots and optionally write reports / send notifications."""
        report_path: Optional[str] = None
        if write_reports:
            markdown_path, csv_path = self._write_reports(run_result)
            run_result.markdown_report_path = str(markdown_path)
            run_result.csv_report_path = str(csv_path)
            report_path = str(markdown_path)

        data_snapshot = {item.code: item.data.to_dict() for item in run_result.results}
        signal_snapshot = {item.code: item.signal.to_dict() for item in run_result.results}
        risk_snapshot = {item.code: item.risk.to_dict() for item in run_result.results}
        execution_snapshot = {item.code: item.execution.to_dict() for item in run_result.results}

        self.repo.save_agent_run(
            run_id=run_result.run_id,
            mode=run_result.mode,
            trade_date=run_result.trade_date,
            stock_codes=[item.code for item in run_result.results],
            account_name=account_name,
            status="completed",
            data_snapshot=data_snapshot,
            signal_snapshot=signal_snapshot,
            risk_snapshot=risk_snapshot,
            execution_snapshot=execution_snapshot,
            account_snapshot=run_result.account_snapshot,
            report_path=report_path,
            error_message=None,
            started_at=run_result.started_at,
            ended_at=run_result.ended_at,
        )

        if not notify_enabled:
            return

        notify_on_no_trade = bool(getattr(self.config, "agent_notify_on_no_trade", False))
        has_trade = any(
            item.execution.action in ("buy", "sell") and item.execution.traded_qty > 0
            for item in run_result.results
        )
        if not has_trade and not notify_on_no_trade:
            return

        notifier = self._get_notifier()
        report_text = self._build_notification_text(run_result, notifier=notifier)
        if notifier.is_available():
            notifier.send(report_text)

    def _build_notification_text(self, run_result: AgentRunResult, *, notifier: NotificationService) -> str:
        """Build concise notification payload for one cycle."""
        if hasattr(notifier, "generate_agent_execution_report"):
            return notifier.generate_agent_execution_report(run_result.to_dict())
        return self._build_markdown_report(run_result)

    def _write_reports(self, run_result: AgentRunResult) -> tuple[Path, Path]:
        """Write markdown + csv report to local filesystem."""
        report_dir = Path(self.config.log_dir) / "agent_reports" / run_result.trade_date.isoformat()
        report_dir.mkdir(parents=True, exist_ok=True)

        markdown_path = report_dir / f"agent_run_{run_result.run_id}.md"
        csv_path = report_dir / f"agent_run_{run_result.run_id}.csv"

        markdown_path.write_text(self._build_markdown_report(run_result), encoding="utf-8")
        self._write_csv_report(csv_path, run_result)
        return markdown_path, csv_path

    def _build_markdown_report(self, run_result: AgentRunResult) -> str:
        """Generate markdown report content."""
        header = [
            f"# Agent Run {run_result.run_id}",
            "",
            f"- Mode: {run_result.mode}",
            f"- Trade Date: {run_result.trade_date.isoformat()}",
            f"- Started: {run_result.started_at.isoformat()}",
            f"- Ended: {run_result.ended_at.isoformat()}",
            "",
            "## Account Snapshot",
            f"- Cash: {run_result.account_snapshot.get('cash', 0):.2f}",
            f"- Market Value: {run_result.account_snapshot.get('total_market_value', 0):.2f}",
            f"- Total Asset: {run_result.account_snapshot.get('total_asset', 0):.2f}",
            "",
            "## Per-Stock Execution",
            "",
            "| Code | Advice | Target Weight | Target Notional | Action | Traded Qty | Fill Price | Position After |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]

        for item in run_result.results:
            header.append(
                "| {code} | {advice} | {weight:.4f} | {notional:.2f} | {action} | {qty} | {price} | {pos} |".format(
                    code=item.code,
                    advice=item.signal.operation_advice,
                    weight=item.risk.target_weight,
                    notional=item.risk.target_notional,
                    action=item.execution.action,
                    qty=item.execution.traded_qty,
                    price=f"{item.execution.fill_price:.4f}" if item.execution.fill_price else "-",
                    pos=item.execution.position_after,
                )
            )

        return "\n".join(header)

    @staticmethod
    def _write_csv_report(path: Path, run_result: AgentRunResult) -> None:
        """Write per-stock csv report."""
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "run_id",
                    "mode",
                    "trade_date",
                    "code",
                    "operation_advice",
                    "sentiment_score",
                    "target_weight",
                    "target_notional",
                    "action",
                    "traded_qty",
                    "fill_price",
                    "fee",
                    "tax",
                    "cash_after",
                    "position_after",
                    "risk_flags",
                ],
            )
            writer.writeheader()
            for item in run_result.results:
                writer.writerow(
                    {
                        "run_id": run_result.run_id,
                        "mode": run_result.mode,
                        "trade_date": run_result.trade_date.isoformat(),
                        "code": item.code,
                        "operation_advice": item.signal.operation_advice,
                        "sentiment_score": item.signal.sentiment_score,
                        "target_weight": item.risk.target_weight,
                        "target_notional": item.risk.target_notional,
                        "action": item.execution.action,
                        "traded_qty": item.execution.traded_qty,
                        "fill_price": item.execution.fill_price,
                        "fee": item.execution.fee,
                        "tax": item.execution.tax,
                        "cash_after": item.execution.cash_after,
                        "position_after": item.execution.position_after,
                        "risk_flags": ",".join(item.risk.risk_flags),
                    }
                )
