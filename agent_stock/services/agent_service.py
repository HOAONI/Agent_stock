# -*- coding: utf-8 -*-
"""应用服务层入口。

CLI、同步 API、异步任务都会汇总到这里。它本身不做具体分析或交易决策，
主要负责解析请求级 `runtime_config`、调用编排器，并统一落库与生成报表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from agent_stock.agents.contracts import AgentRunResult
from agent_stock.agents.orchestrator import AgentOrchestrator
from agent_stock.reporting import write_run_reports
from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.runtime_config import parse_runtime_config
from agent_stock.storage import DatabaseManager
from agent_stock.config import (
    AgentRuntimeConfig,
    Config,
    get_config,
)


@dataclass(frozen=True)
class _ResolvedRuntimeContext:
    """一次运行调用解析后的请求级上下文。"""

    account_name: str
    initial_cash_override: float | None
    runtime_config: AgentRuntimeConfig | None


class AgentService:
    """封装 Agent 编排的高层用例服务。"""

    def __init__(
        self,
        config: Config | None = None,
        db_manager: DatabaseManager | None = None,
        orchestrator: Any | None = None,
        execution_repo: ExecutionRepository | Any | None = None,
    ) -> None:
        """初始化应用服务及其依赖。"""
        self.config = config or get_config()
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = execution_repo or ExecutionRepository(self.db)
        self.orchestrator = orchestrator or AgentOrchestrator(config=self.config, db_manager=self.db, execution_repo=self.repo)

    def run_once(
        self,
        stock_codes: list[str],
        *,
        account_name: str | None = None,
        request_id: str | None = None,
        write_reports: bool | None = None,
        runtime_config: dict[str, Any] | None = None,
        planning_context: dict[str, Any] | None = None,
        paper_order_submitter: Any | None = None,
        stage_observer: Any | None = None,
    ) -> AgentRunResult:
        """执行一次完整周期，并持久化运行快照。"""
        if not [str(code or "").strip() for code in stock_codes if str(code or "").strip()]:
            raise ValueError("stock_codes must not be empty")
        # 先把请求级账户、初始资金和 LLM/策略覆盖项解析清楚，再进入主链路。
        runtime_ctx = self._resolve_runtime_context(
            account_name=account_name,
            runtime_config=runtime_config,
        )
        run_once_kwargs = {
            "account_name": runtime_ctx.account_name,
            "request_id": request_id,
            "initial_cash_override": runtime_ctx.initial_cash_override,
            "runtime_config": runtime_ctx.runtime_config,
            "planning_context": planning_context,
            "stage_observer": stage_observer,
        }
        if paper_order_submitter is not None:
            run_once_kwargs["paper_order_submitter"] = paper_order_submitter
        try:
            result = self.orchestrator.run_once(
                stock_codes,
                **run_once_kwargs,
            )
        except TypeError as exc:
            if "paper_order_submitter" not in str(exc):
                raise
            run_once_kwargs.pop("paper_order_submitter", None)
            result = self.orchestrator.run_once(
                stock_codes,
                **run_once_kwargs,
            )
        self._finalize_run(
            run_result=result,
            account_name=runtime_ctx.account_name,
            write_reports=self._resolve_write_reports(write_reports),
        )
        return result

    def run_realtime(
        self,
        stock_codes: list[str],
        *,
        interval_minutes: int,
        max_cycles: int | None = None,
        account_name: str | None = None,
        write_reports: bool | None = None,
        runtime_config: dict[str, Any] | None = None,
        planning_context: dict[str, Any] | None = None,
        paper_order_submitter: Any | None = None,
        stage_observer: Any | None = None,
    ) -> list[AgentRunResult]:
        """执行实时循环，并在每个周期结束后落库。"""
        if not [str(code or "").strip() for code in stock_codes if str(code or "").strip()]:
            raise ValueError("stock_codes must not be empty")
        runtime_ctx = self._resolve_runtime_context(
            account_name=account_name,
            runtime_config=runtime_config,
        )
        realtime_kwargs = {
            "interval_minutes": interval_minutes,
            "max_cycles": max_cycles,
            "heartbeat_sleep": 5.0,
            "account_name": runtime_ctx.account_name,
            "initial_cash_override": runtime_ctx.initial_cash_override,
            "runtime_config": runtime_ctx.runtime_config,
            "planning_context": planning_context,
            "stage_observer": stage_observer,
        }
        if paper_order_submitter is not None:
            realtime_kwargs["paper_order_submitter"] = paper_order_submitter
        try:
            run_results = self.orchestrator.run_realtime(
                stock_codes,
                **realtime_kwargs,
            )
        except TypeError as exc:
            if "paper_order_submitter" not in str(exc):
                raise
            realtime_kwargs.pop("paper_order_submitter", None)
            run_results = self.orchestrator.run_realtime(
                stock_codes,
                **realtime_kwargs,
            )
        should_write_reports = self._resolve_write_reports(write_reports)
        # 实时模式会产出多个周期结果，但落库和报表写出规则要保持一致。
        for item in run_results:
            self._finalize_run(
                run_result=item,
                account_name=runtime_ctx.account_name,
                write_reports=should_write_reports,
            )
        return run_results

    def run_async(
        self,
        stock_codes: list[str],
        *,
        request_id: str | None = None,
        account_name: str | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> dict:
        """通过任务服务提交异步运行。"""
        if not [str(code or "").strip() for code in stock_codes if str(code or "").strip()]:
            raise ValueError("stock_codes must not be empty")
        from agent_stock.services.agent_task_service import get_agent_task_service

        # 异步模式只负责投递任务，真正执行仍然回到同一个 AgentService。
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
        account_name: str | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> _ResolvedRuntimeContext:
        """解析请求级运行时覆盖项，且不修改全局配置。"""
        parsed_runtime = self._parse_runtime_config(runtime_config)
        runtime_account_name = parsed_runtime.account.account_name if parsed_runtime and parsed_runtime.account else None
        top_level_account_name = str(account_name).strip() if account_name else None
        if top_level_account_name and len(top_level_account_name) > 128:
            raise ValueError("account_name length must be <= 128")

        if runtime_account_name and top_level_account_name and runtime_account_name != top_level_account_name:
            raise ValueError("account_name conflicts with runtime_config.account.account_name")

        # 账户名优先级：runtime_config.account -> 顶层参数 -> 全局默认配置。
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
    def _parse_runtime_config(runtime_config: dict[str, Any] | None) -> AgentRuntimeConfig | None:
        """解析原始运行时配置字典。"""
        return parse_runtime_config(runtime_config)

    def _resolve_write_reports(self, write_reports: bool | None) -> bool:
        """确定本次运行是否需要输出本地报表。"""
        if write_reports is not None:
            return bool(write_reports)
        return bool(getattr(self.config, "agent_write_local_reports", False))

    def _finalize_run(
        self,
        *,
        run_result: AgentRunResult,
        account_name: str,
        write_reports: bool,
    ) -> None:
        """持久化各阶段快照，并按需生成本地报表。"""
        report_path: str | None = None
        if write_reports:
            markdown_path, csv_path = write_run_reports(run_result, self.config.log_dir)
            run_result.markdown_report_path = str(markdown_path)
            run_result.csv_report_path = str(csv_path)
            report_path = str(markdown_path)

        # 统一按阶段拆分快照，方便 API 与问题排查按 data/signal/risk/execution 查看。
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
