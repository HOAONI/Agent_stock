# -*- coding: utf-8 -*-
"""执行智能体。

它接收风控阶段给出的目标金额，决定本次应当生成纸面执行意图，还是调用本地
broker 运行时做一次立即成交的模拟下单。
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any


from data_provider.base import canonical_stock_code

from agent_stock.agents.contracts import AgentState, ExecutionAgentOutput, RiskAgentOutput
from agent_stock.agents.agentic_decision import generate_structured_decision
from agent_stock.analyzer import get_analyzer
from agent_stock.repositories.execution_repo import ExecutionRepository
from agent_stock.services.backtrader_runtime_service import get_backtrader_runtime_service
from agent_stock.storage import DatabaseManager
from agent_stock.config import Config, RuntimeExecutionConfig, get_config


class ExecutionAgent:
    """生成执行意图，或向本地模拟运行时提交订单。"""

    def __init__(
        self,
        config: Config | None = None,
        db_manager: DatabaseManager | None = None,
        execution_repo: ExecutionRepository | None = None,
        runtime_service: Any | None = None,
        analyzer=None,
    ) -> None:
        """初始化执行代理及其运行时依赖。"""
        self.config = config or get_config()
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = execution_repo or ExecutionRepository(self.db)
        self.runtime_service = runtime_service or get_backtrader_runtime_service()
        self.analyzer = analyzer or get_analyzer()

    def run(
        self,
        *,
        run_id: str,
        code: str,
        trade_date: date,
        current_price: float,
        risk_output: RiskAgentOutput,
        account_snapshot: dict[str, Any] | None = None,
        account_name: str | None = None,
        initial_cash_override: float | None = None,
        runtime_execution: RuntimeExecutionConfig | None = None,
        backend_task_id: str | None = None,
        signal_output: Any | None = None,
        data_output: Any | None = None,
    ) -> ExecutionAgentOutput:
        """按纸面模式或 broker 模拟模式执行单只股票决策。"""
        normalized_code = canonical_stock_code(code)
        resolved_account_name = account_name or str(getattr(self.config, "agent_account_name", "paper-default") or "paper-default")
        initial_cash = (
            float(initial_cash_override)
            if initial_cash_override is not None
            else float(getattr(self.config, "agent_initial_cash", 1_000_000.0))
        )
        runtime_mode = str(getattr(runtime_execution, "mode", "paper") or "paper").strip().lower()
        broker_account_id = self._as_int(getattr(runtime_execution, "broker_account_id", None), 0) if runtime_execution else 0

        # broker 模式会真实调用本地运行时服务；paper 模式只做意图推演，不落订单。
        if runtime_mode == "broker":
            if broker_account_id <= 0:
                output = ExecutionAgentOutput(
                    code=normalized_code,
                    trade_date=trade_date,
                    state=AgentState.FAILED,
                    execution_mode="broker",
                    backend_task_id=backend_task_id,
                    broker_requested=False,
                    executed_via="backtrader_internal",
                    broker_ticket_id=None,
                    fallback_reason=None,
                    action="none",
                    reason="invalid_broker_account",
                    error_message="runtime_execution.broker_account_id is required for broker mode",
                )
            else:
                output = self._run_broker_execution(
                    run_id=run_id,
                    code=normalized_code,
                    trade_date=trade_date,
                    current_price=current_price,
                    risk_output=risk_output,
                    account_snapshot=account_snapshot,
                    account_name=resolved_account_name,
                    initial_cash=initial_cash,
                    broker_account_id=broker_account_id,
                    backend_task_id=backend_task_id,
                    signal_output=signal_output,
                    data_output=data_output,
                )
        else:
            output = self._run_paper_intent(
                code=normalized_code,
                trade_date=trade_date,
                current_price=current_price,
                risk_output=risk_output,
                account_name=resolved_account_name,
                initial_cash=initial_cash,
                account_snapshot=account_snapshot,
                signal_output=signal_output,
                data_output=data_output,
            )
            output.execution_mode = "paper"
            output.backend_task_id = backend_task_id
            output.broker_requested = False
            output.executed_via = "paper"
            output.broker_ticket_id = None
            output.fallback_reason = None

        return self._attach_proposal_metadata(output, current_price=current_price)

    def prepare_order(
        self,
        *,
        code: str,
        trade_date: date,
        current_price: float,
        risk_output: RiskAgentOutput,
        account_snapshot: dict[str, Any] | None = None,
        account_name: str | None = None,
        initial_cash_override: float | None = None,
        backend_task_id: str | None = None,
        signal_output: Any | None = None,
        data_output: Any | None = None,
    ) -> ExecutionAgentOutput:
        """Prepare a paper order without touching broker runtime side effects."""
        resolved_account_name = account_name or str(getattr(self.config, "agent_account_name", "paper-default") or "paper-default")
        initial_cash = (
            float(initial_cash_override)
            if initial_cash_override is not None
            else float(getattr(self.config, "agent_initial_cash", 1_000_000.0))
        )
        output = self._run_paper_intent(
            code=canonical_stock_code(code),
            trade_date=trade_date,
            current_price=current_price,
            risk_output=risk_output,
            account_name=resolved_account_name,
            initial_cash=initial_cash,
            account_snapshot=account_snapshot,
            signal_output=signal_output,
            data_output=data_output,
        )
        output.execution_mode = "paper"
        output.backend_task_id = backend_task_id
        output.broker_requested = False
        output.executed_via = "paper"
        output.broker_ticket_id = None
        output.fallback_reason = None
        return self._attach_proposal_metadata(output, current_price=current_price)

    def _attach_proposal_metadata(
        self,
        output: ExecutionAgentOutput,
        *,
        current_price: float,
    ) -> ExecutionAgentOutput:
        paper_submit_result = output.paper_submit_result if isinstance(output.paper_submit_result, dict) else {}
        provider_status = str(paper_submit_result.get("status") or "").strip().lower()

        if isinstance(output.proposed_order, dict) and output.proposal_state:
            if not output.proposal_reason:
                output.proposal_reason = output.adjustment_reason or output.reason
            if provider_status in {"filled", "submitted"}:
                output.proposal_state = "executed" if provider_status == "filled" else "submitted"
            output.status = self._resolve_execution_status(output)
            output.execution_allowed = output.proposal_state in {"proposed", "submitted", "executed"}
            return output

        final_order = output.final_order if isinstance(output.final_order, dict) else None
        proposed_order: dict[str, Any] | None = None
        proposal_state = output.proposal_state
        if isinstance(final_order, dict) and int(final_order.get("quantity") or 0) > 0:
            proposed_order = {
                "code": str(final_order.get("code") or output.code),
                "action": str(final_order.get("action") or output.action or ""),
                "quantity": int(final_order.get("quantity") or 0),
                "target_qty": int(final_order.get("target_qty") or output.target_qty or 0),
                "price": round(float(final_order.get("price") or output.fill_price or current_price or 0.0), 4),
            }
            if output.execution_mode == "paper":
                proposal_state = "proposed"
            elif int(output.traded_qty or 0) > 0:
                proposal_state = "executed"
            else:
                proposal_state = "submitted"
        elif output.action in {"buy", "sell"} and int(output.traded_qty or 0) > 0:
            proposed_order = {
                "code": output.code,
                "action": output.action,
                "quantity": int(output.traded_qty or 0),
                "target_qty": int(output.target_qty or 0),
                "price": round(float(output.fill_price or current_price or 0.0), 4),
            }
            proposal_state = "executed" if output.execution_mode == "broker" else "proposed"
        elif output.reason == "target_matched":
            proposal_state = "not_needed"
        else:
            proposal_state = "blocked"

        output.proposed_order = proposed_order
        output.proposal_state = proposal_state
        output.proposal_reason = output.adjustment_reason or output.reason
        if provider_status in {"filled", "submitted"}:
            output.proposal_state = "executed" if provider_status == "filled" else "submitted"
        output.status = self._resolve_execution_status(output)
        output.execution_allowed = output.proposal_state in {"proposed", "submitted", "executed"}
        return output

    @staticmethod
    def _resolve_execution_status(output: ExecutionAgentOutput) -> str:
        if output.state == AgentState.FAILED:
            return "failed"
        proposal_state = str(output.proposal_state or "").strip()
        if proposal_state == "executed":
            return "executed"
        if proposal_state == "submitted":
            return "submitted"
        if proposal_state == "proposed":
            return "prepared"
        if proposal_state == "not_needed":
            return "not_needed"
        if proposal_state == "blocked":
            return "blocked"
        return "ready" if output.state == AgentState.READY else "blocked"

    def _run_broker_execution(
        self,
        *,
        run_id: str,
        code: str,
        trade_date: date,
        current_price: float,
        risk_output: RiskAgentOutput,
        account_snapshot: dict[str, Any] | None,
        account_name: str,
        initial_cash: float,
        broker_account_id: int,
        backend_task_id: str | None,
        signal_output: Any | None,
        data_output: Any | None,
    ) -> ExecutionAgentOutput:
        """调用本地 broker 运行时执行一笔模拟下单。"""
        # 先用 paper 口径推导出“理论上应该下什么单”，再决定是否真正调用 broker。
        candidate = self._run_paper_intent(
            code=code,
            trade_date=trade_date,
            current_price=current_price,
            risk_output=risk_output,
            account_name=account_name,
            initial_cash=initial_cash,
            account_snapshot=account_snapshot,
            signal_output=signal_output,
            data_output=data_output,
        )

        snapshot_before = self._fetch_broker_snapshot(
            broker_account_id=broker_account_id,
            account_name=account_name,
            initial_cash=initial_cash,
            fallback_snapshot=account_snapshot,
        )
        if snapshot_before:
            candidate.cash_before = round(float(snapshot_before.get("cash") or candidate.cash_before), 4)
            candidate.position_before = int(self._find_position(snapshot_before, code).get("quantity") or candidate.position_before)
            candidate.account_snapshot = snapshot_before

        if candidate.state != AgentState.READY or candidate.action not in {"buy", "sell"} or candidate.traded_qty <= 0:
            cash_before = float(snapshot_before.get("cash") or candidate.cash_before) if snapshot_before else candidate.cash_before
            position_before = int(self._find_position(snapshot_before, code).get("quantity") or candidate.position_before) if snapshot_before else candidate.position_before
            candidate.execution_mode = "broker"
            candidate.backend_task_id = backend_task_id
            candidate.broker_requested = False
            candidate.executed_via = "backtrader_internal"
            candidate.broker_ticket_id = None
            candidate.fallback_reason = None
            candidate.cash_before = round(cash_before, 4)
            candidate.cash_after = round(cash_before if candidate.state != AgentState.READY else candidate.cash_after, 4)
            candidate.position_before = position_before
            candidate.position_after = position_before if candidate.state != AgentState.READY else candidate.position_after
            if snapshot_before:
                candidate.account_snapshot = snapshot_before
            return candidate

        try:
            response = self.runtime_service.place_order(
                {
                    "broker_account_id": broker_account_id,
                    "payload": {
                        "stock_code": code,
                        "direction": candidate.action,
                        "type": "market",
                        "price": current_price,
                        "quantity": candidate.traded_qty,
                    },
                }
            )
        except Exception as exc:
            return self._build_broker_failure_output(
                code=code,
                trade_date=trade_date,
                backend_task_id=backend_task_id,
                candidate=candidate,
                broker_account_id=broker_account_id,
                account_name=account_name,
                initial_cash=initial_cash,
                snapshot_before=snapshot_before,
                reason="broker_runtime_error",
                error_message=str(exc),
            )

        status = str(response.get("status") or response.get("provider_status") or "").strip().lower()
        if status not in {"filled", "submitted"}:
            return self._build_broker_failure_output(
                code=code,
                trade_date=trade_date,
                backend_task_id=backend_task_id,
                candidate=candidate,
                broker_account_id=broker_account_id,
                account_name=account_name,
                initial_cash=initial_cash,
                snapshot_before=snapshot_before,
                reason="broker_rejected",
                error_message=str(response.get("message") or "broker rejected order").strip() or "broker rejected order",
                response=response,
            )

        snapshot_after = self._fetch_broker_snapshot(
            broker_account_id=broker_account_id,
            account_name=account_name,
            initial_cash=initial_cash,
            fallback_snapshot=snapshot_before,
        ) or snapshot_before or candidate.account_snapshot
        position_before = int(self._find_position(snapshot_before, code).get("quantity") or candidate.position_before) if snapshot_before else candidate.position_before
        position_after = int(self._find_position(snapshot_after, code).get("quantity") or candidate.position_after) if snapshot_after else candidate.position_after
        cash_before = self._as_number(response.get("cash_before"), candidate.cash_before)
        cash_after = self._as_number(
            response.get("cash_after"),
            float(snapshot_after.get("cash") or candidate.cash_after) if snapshot_after else candidate.cash_after,
        )
        filled_quantity = max(0, self._as_int(response.get("filled_quantity"), candidate.traded_qty))
        fill_price = self._as_number(response.get("filled_price"), candidate.fill_price or current_price)
        fee = self._as_number(response.get("fee"), candidate.fee)
        tax = self._as_number(response.get("tax"), candidate.tax)
        provider_order_id = self._as_text(response.get("provider_order_id") or response.get("order_id"))

        return ExecutionAgentOutput(
            code=code,
            trade_date=trade_date,
            state=AgentState.READY,
            execution_mode="broker",
            backend_task_id=backend_task_id,
            broker_requested=True,
            executed_via="backtrader_internal",
            broker_ticket_id=provider_order_id,
            fallback_reason=None,
            action=candidate.action,
            reason="broker_executed",
            order_id=self._parse_optional_int(response.get("order_id")),
            trade_id=self._parse_optional_int(response.get("trade_id")),
            target_qty=candidate.target_qty,
            traded_qty=filled_quantity,
            fill_price=round(fill_price, 4) if fill_price is not None else None,
            fee=round(fee, 4),
            tax=round(tax, 4),
            cash_before=round(cash_before, 4),
            cash_after=round(cash_after, 4),
            position_before=position_before,
            position_after=position_after,
            account_snapshot=snapshot_after or candidate.account_snapshot,
        )

    def _build_broker_failure_output(
        self,
        *,
        code: str,
        trade_date: date,
        backend_task_id: str | None,
        candidate: ExecutionAgentOutput,
        broker_account_id: int,
        account_name: str,
        initial_cash: float,
        snapshot_before: dict[str, Any] | None,
        reason: str,
        error_message: str,
        response: dict[str, Any] | None = None,
    ) -> ExecutionAgentOutput:
        """构造 broker 路径失败时的统一执行结果。"""
        snapshot_after = self._fetch_broker_snapshot(
            broker_account_id=broker_account_id,
            account_name=account_name,
            initial_cash=initial_cash,
            fallback_snapshot=snapshot_before or candidate.account_snapshot,
        ) or snapshot_before or candidate.account_snapshot
        cash_before = self._as_number(
            response.get("cash_before") if response else None,
            float(snapshot_before.get("cash") or candidate.cash_before) if snapshot_before else candidate.cash_before,
        )
        cash_after = self._as_number(
            response.get("cash_after") if response else None,
            float(snapshot_after.get("cash") or cash_before) if snapshot_after else cash_before,
        )
        position_before = int(self._find_position(snapshot_before, code).get("quantity") or candidate.position_before) if snapshot_before else candidate.position_before
        position_after = int(self._find_position(snapshot_after, code).get("quantity") or position_before) if snapshot_after else position_before
        provider_order_id = self._as_text(response.get("provider_order_id") or response.get("order_id")) if response else None

        return ExecutionAgentOutput(
            code=code,
            trade_date=trade_date,
            state=AgentState.FAILED,
            execution_mode="broker",
            backend_task_id=backend_task_id,
            broker_requested=True,
            executed_via="backtrader_internal",
            broker_ticket_id=provider_order_id,
            fallback_reason=None,
            action=candidate.action,
            reason=reason,
            order_id=self._parse_optional_int(response.get("order_id")) if response else None,
            trade_id=self._parse_optional_int(response.get("trade_id")) if response else None,
            target_qty=candidate.target_qty,
            traded_qty=0,
            fill_price=None,
            fee=round(self._as_number(response.get("fee") if response else None, 0.0), 4),
            tax=round(self._as_number(response.get("tax") if response else None, 0.0), 4),
            cash_before=round(cash_before, 4),
            cash_after=round(cash_after, 4),
            position_before=position_before,
            position_after=position_after,
            account_snapshot=snapshot_after,
            error_message=error_message.strip()[:500] or reason,
        )

    def _fetch_broker_snapshot(
        self,
        *,
        broker_account_id: int,
        account_name: str,
        initial_cash: float,
        fallback_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """从本地 broker 运行时读取账户快照。"""
        try:
            summary = self.runtime_service.get_account_summary({"broker_account_id": broker_account_id})
            positions_raw = self.runtime_service.get_positions({"broker_account_id": broker_account_id})
        except Exception:
            if isinstance(fallback_snapshot, dict):
                return self._normalize_snapshot(fallback_snapshot, account_name=account_name, initial_cash=initial_cash)
            return None

        positions = []
        for raw in positions_raw:
            if not isinstance(raw, dict):
                continue
            positions.append(
                {
                    "code": raw.get("stock_code") or raw.get("code") or raw.get("symbol"),
                    "quantity": raw.get("quantity"),
                    "available_qty": raw.get("available_qty"),
                    "avg_cost": raw.get("avg_cost"),
                    "last_price": raw.get("last_price"),
                    "market_value": raw.get("market_value"),
                    "unrealized_pnl": raw.get("unrealized_pnl"),
                }
            )

        fallback_name = ""
        if isinstance(fallback_snapshot, dict):
            fallback_name = str(fallback_snapshot.get("name") or "").strip()
        snapshot = {
            "name": fallback_name or account_name,
            "cash": summary.get("cash"),
            "initial_cash": summary.get("initial_capital"),
            "total_market_value": summary.get("market_value"),
            "total_asset": summary.get("total_asset"),
            "realized_pnl": summary.get("realized_pnl"),
            "unrealized_pnl": summary.get("unrealized_pnl"),
            "cumulative_fees": summary.get("cumulative_fees"),
            "positions": positions,
            "snapshot_at": summary.get("snapshot_at"),
            "data_source": "backtrader_internal",
            "broker_account_id": broker_account_id,
            "provider_code": "backtrader_local",
            "provider_name": "Backtrader Local Sim",
            "account_uid": f"bt-{broker_account_id}",
            "account_display_name": fallback_snapshot.get("account_display_name") if isinstance(fallback_snapshot, dict) else account_name,
        }
        return self._normalize_snapshot(snapshot, account_name=account_name, initial_cash=initial_cash)

    def _run_paper_intent(
        self,
        *,
        code: str,
        trade_date: date,
        current_price: float,
        risk_output: RiskAgentOutput,
        account_name: str,
        initial_cash: float,
        account_snapshot: dict[str, Any] | None,
        signal_output: Any | None = None,
        data_output: Any | None = None,
    ) -> ExecutionAgentOutput:
        """在不真正下单的情况下推导纸面交易意图。"""
        if current_price <= 0:
            return ExecutionAgentOutput(
                code=code,
                trade_date=trade_date,
                state=AgentState.FAILED,
                action="none",
                reason="invalid_price",
                error_message="current price is invalid",
                decision={
                    "action": "abort",
                    "summary": "执行阶段拿到无效价格，无法继续生成订单。",
                    "reason": "invalid_price",
                    "next_action": "abort",
                    "confidence": 0.05,
                    "warnings": ["invalid_price"],
                },
                confidence=0.05,
                warnings=["invalid_price"],
                fallback_chain=["paper_execution"],
                next_action="abort",
            )

        snapshot_before = self._normalize_snapshot(account_snapshot, account_name=account_name, initial_cash=initial_cash)
        position_before = self._find_position(snapshot_before, code)
        current_qty = int(position_before.get("quantity") or 0)
        cash_before = float(snapshot_before.get("cash") or 0.0)

        lot = max(1, int(getattr(self.config, "agent_min_trade_lot", 100)))
        slippage_bps = float(getattr(self.config, "agent_slippage_bps", 5.0))
        fee_rate = float(getattr(self.config, "agent_fee_rate", 0.0003))
        sell_tax_rate = float(getattr(self.config, "agent_sell_tax_rate", 0.001))

        buy_price = current_price * (1.0 + slippage_bps / 10000.0)
        sell_price = current_price * (1.0 - slippage_bps / 10000.0)

        target_qty = self._target_qty(
            target_notional=float(risk_output.target_notional),
            effective_price=buy_price,
            lot=lot,
        )

        delta = target_qty - current_qty
        if delta == 0:
            return ExecutionAgentOutput(
                code=code,
                trade_date=trade_date,
                state=AgentState.SKIPPED,
                action="none",
                reason="target_matched",
                target_qty=target_qty,
                traded_qty=0,
                cash_before=cash_before,
                cash_after=cash_before,
                position_before=current_qty,
                position_after=current_qty,
                account_snapshot=snapshot_before,
                decision={
                    "action": "continue",
                    "summary": "当前仓位已与目标仓位一致，无需新增执行动作。",
                    "reason": "target_matched",
                    "next_action": "done",
                    "confidence": 0.95,
                    "warnings": [],
                },
                confidence=0.95,
                fallback_chain=["paper_execution"],
                next_action="done",
            )

        if delta > 0:
            side = "buy"
            affordable_qty = self._target_qty(
                target_notional=cash_before,
                effective_price=buy_price * (1.0 + fee_rate),
                lot=lot,
            )
            trade_qty = min((delta // lot) * lot, affordable_qty)
            if trade_qty < lot:
                return ExecutionAgentOutput(
                    code=code,
                    trade_date=trade_date,
                    state=AgentState.SKIPPED,
                    action="none",
                    reason="insufficient_cash",
                    target_qty=target_qty,
                    traded_qty=0,
                    cash_before=cash_before,
                    cash_after=cash_before,
                    position_before=current_qty,
                    position_after=current_qty,
                    account_snapshot=snapshot_before,
                    decision={
                        "action": "skip",
                        "summary": "执行阶段判断当前现金不足，本轮不继续下单。",
                        "reason": "insufficient_cash",
                        "next_action": "done",
                        "confidence": 0.94,
                        "warnings": ["insufficient_cash"],
                    },
                    confidence=0.94,
                    warnings=["insufficient_cash"],
                    fallback_chain=["paper_execution"],
                    next_action="done",
                )
            fill_price = buy_price
        else:
            side = "sell"
            trade_qty = min((-delta // lot) * lot, (current_qty // lot) * lot)
            if trade_qty < lot:
                return ExecutionAgentOutput(
                    code=code,
                    trade_date=trade_date,
                    state=AgentState.SKIPPED,
                    action="none",
                    reason="insufficient_position",
                    target_qty=target_qty,
                    traded_qty=0,
                    cash_before=cash_before,
                    cash_after=cash_before,
                    position_before=current_qty,
                    position_after=current_qty,
                    account_snapshot=snapshot_before,
                    decision={
                        "action": "skip",
                        "summary": "执行阶段判断可卖持仓不足，本轮不继续下单。",
                        "reason": "insufficient_position",
                        "next_action": "done",
                        "confidence": 0.94,
                        "warnings": ["insufficient_position"],
                    },
                    confidence=0.94,
                    warnings=["insufficient_position"],
                    fallback_chain=["paper_execution"],
                    next_action="done",
                )
            fill_price = sell_price

        original_order = {
            "code": code,
            "action": side,
            "quantity": trade_qty,
            "price": round(fill_price, 4),
            "target_qty": target_qty,
        }
        adjustment = self._resolve_execution_adjustment(
            code=code,
            current_price=current_price,
            lot=lot,
            original_order=original_order,
            risk_output=risk_output,
            signal_output=signal_output,
            data_output=data_output,
        )
        warnings = list(adjustment.get("warnings") or [])
        decision = adjustment.get("decision") if isinstance(adjustment.get("decision"), dict) else {}
        llm_used = bool(adjustment.get("llm_used"))
        adjustment_action = str(adjustment.get("action") or "continue").strip() or "continue"
        adjusted_qty = int(adjustment.get("quantity") or trade_qty)
        adjusted_qty = min(trade_qty, max(0, adjusted_qty))
        adjusted_qty = (adjusted_qty // lot) * lot
        if adjustment_action == "skip" or adjusted_qty < lot:
            return ExecutionAgentOutput(
                code=code,
                trade_date=trade_date,
                state=AgentState.SKIPPED,
                action="none",
                reason=str(adjustment.get("reason") or "execution_skipped"),
                target_qty=target_qty,
                traded_qty=0,
                cash_before=cash_before,
                cash_after=cash_before,
                position_before=current_qty,
                position_after=current_qty,
                account_snapshot=snapshot_before,
                observations=list(adjustment.get("observations") or []),
                decision=decision or {
                    "action": "skip",
                    "summary": "执行阶段因异常波动主动放弃本轮下单。",
                    "reason": "execution_skipped",
                    "next_action": "done",
                    "confidence": 0.88,
                    "warnings": warnings,
                },
                confidence=float((decision or {}).get("confidence") or 0.88),
                warnings=warnings,
                llm_used=llm_used,
                fallback_chain=["paper_execution", *(("llm_execution_planner",) if llm_used else ())],
                next_action="done",
                original_order=original_order,
                final_order=None,
                adjustment_applied=bool(adjustment.get("adjustment_applied")),
                adjustment_reason=str(adjustment.get("adjustment_reason") or adjustment.get("reason") or "execution_skipped"),
                risk_reduction_only=True,
            )

        trade_qty = adjusted_qty
        fill_price = float(adjustment.get("price") or fill_price)
        final_order = {
            "code": code,
            "action": side,
            "quantity": trade_qty,
            "price": round(fill_price, 4),
            "target_qty": target_qty,
        }

        gross_amount = fill_price * float(trade_qty)
        fee = gross_amount * fee_rate
        tax = gross_amount * sell_tax_rate if side == "sell" else 0.0

        if side == "buy":
            cash_after = cash_before - gross_amount - fee
            position_after_qty = current_qty + trade_qty
            avg_cost_before = float(position_before.get("avg_cost") or 0.0)
            new_avg_cost = (
                ((avg_cost_before * current_qty) + (fill_price * trade_qty) + fee) / position_after_qty
                if position_after_qty > 0
                else 0.0
            )
        else:
            cash_after = cash_before + gross_amount - fee - tax
            position_after_qty = max(0, current_qty - trade_qty)
            new_avg_cost = float(position_before.get("avg_cost") or 0.0)

        snapshot_after = self._project_snapshot_after_trade(
            snapshot=snapshot_before,
            code=code,
            current_price=current_price,
            side=side,
            trade_qty=trade_qty,
            cash_after=cash_after,
            avg_cost_after=new_avg_cost,
            position_after_qty=position_after_qty,
        )

        return ExecutionAgentOutput(
            code=code,
            trade_date=trade_date,
            state=AgentState.READY,
            action=side,
            reason=str(adjustment.get("reason") or "intent_generated"),
            target_qty=target_qty,
            traded_qty=trade_qty,
            fill_price=round(fill_price, 4),
            fee=round(fee, 4),
            tax=round(tax, 4),
            cash_before=round(cash_before, 4),
            cash_after=round(cash_after, 4),
            position_before=current_qty,
            position_after=position_after_qty,
            account_snapshot=snapshot_after,
            observations=list(adjustment.get("observations") or []),
            decision=decision or {
                "action": "continue",
                "summary": "执行阶段已生成可提交的保守订单。",
                "reason": "intent_generated",
                "next_action": "done",
                "confidence": 0.8,
                "warnings": warnings,
            },
            confidence=float((decision or {}).get("confidence") or 0.8),
            warnings=warnings,
            llm_used=llm_used,
            fallback_chain=["paper_execution", *(("llm_execution_planner",) if llm_used else ())],
            next_action="done",
            original_order=original_order,
            final_order=final_order,
            adjustment_applied=bool(adjustment.get("adjustment_applied")),
            adjustment_reason=str(adjustment.get("adjustment_reason") or ""),
            risk_reduction_only=bool(adjustment.get("risk_reduction_only", False)),
        )

    def _resolve_execution_adjustment(
        self,
        *,
        code: str,
        current_price: float,
        lot: int,
        original_order: dict[str, Any],
        risk_output: RiskAgentOutput,
        signal_output: Any | None,
        data_output: Any | None,
    ) -> dict[str, Any]:
        """根据波动和风险状态，将原始订单调整为更保守的执行计划。"""
        today = data_output.analysis_context.get("today") if data_output and isinstance(data_output.analysis_context, dict) else {}
        yesterday = data_output.analysis_context.get("yesterday") if data_output and isinstance(data_output.analysis_context, dict) else {}
        change_pct = data_output.realtime_quote.get("change_pct") if data_output and isinstance(data_output.realtime_quote, dict) else None
        if change_pct is None and isinstance(today, dict):
            change_pct = today.get("pct_chg")
        open_price = float(today.get("open") or 0.0) if isinstance(today, dict) else 0.0
        high_price = float(today.get("high") or 0.0) if isinstance(today, dict) else 0.0
        low_price = float(today.get("low") or 0.0) if isinstance(today, dict) else 0.0
        previous_close = float(yesterday.get("close") or 0.0) if isinstance(yesterday, dict) else 0.0
        today_close = float(today.get("close") or 0.0) if isinstance(today, dict) else 0.0
        reference_price = previous_close or today_close or open_price or current_price
        amplitude_pct = ((high_price - low_price) / open_price * 100.0) if open_price > 0 and high_price > 0 and low_price > 0 else 0.0
        deviation_pct = abs((current_price - reference_price) / reference_price * 100.0) if reference_price > 0 else 0.0
        change_pct_value = abs(float(change_pct or 0.0))
        volatility_score = max(change_pct_value, amplitude_pct, deviation_pct)

        observations = [
            {
                "current_price": round(current_price, 4),
                "reference_price": round(reference_price, 4),
                "change_pct": round(float(change_pct or 0.0), 4),
                "amplitude_pct": round(amplitude_pct, 4),
                "deviation_pct": round(deviation_pct, 4),
                "volatility_score": round(volatility_score, 4),
                "risk_flags": list(risk_output.risk_flags or []),
                "signal_advice": getattr(signal_output, "operation_advice", None),
            }
        ]
        warnings: list[str] = []

        default_action = "continue"
        default_factor = 1.0
        adjustment_reason = ""
        if volatility_score >= 9.0 or deviation_pct >= 6.0:
            default_action = "skip"
            default_factor = 0.0
            adjustment_reason = "price_anomaly_skip"
            warnings.append("price_anomaly_skip")
        elif volatility_score >= 5.0 or deviation_pct >= 3.5:
            default_action = "split" if int(original_order.get("quantity") or 0) >= lot * 2 else "reduce"
            default_factor = 0.5
            adjustment_reason = "price_volatility_reduce"
            warnings.append("price_volatility_reduce")

        default_decision = {
            "action": default_action,
            "summary": (
                "价格波动在可接受范围内，保留原执行计划。"
                if default_action == "continue"
                else "当前价格波动偏大，执行阶段将主动降低风险后再执行。"
                if default_action in {"reduce", "split"}
                else "当前价格异常波动较大，执行阶段决定放弃本轮下单。"
            ),
            "reason": adjustment_reason or "execution_ready",
            "next_action": "done",
            "confidence": 0.86 if default_action == "continue" else 0.73 if default_action in {"reduce", "split"} else 0.91,
            "warnings": warnings,
            "requested_notional_factor": default_factor,
            "adjustment_mode": default_action,
            "adjustment_reason": adjustment_reason,
        }
        decision, llm_used = generate_structured_decision(
            analyzer=self.analyzer,
            stage="execution",
            prompt=self._build_execution_stage_prompt(
                code=code,
                original_order=original_order,
                observations=observations,
                risk_output=risk_output,
            ),
            allowed_actions={"continue", "reduce", "split", "skip"},
            default_decision=default_decision,
        )
        for item in decision.get("warnings") or []:
            if isinstance(item, str) and item not in warnings:
                warnings.append(item)

        action = str(decision.get("action") or default_action).strip() or default_action
        try:
            factor = float(decision.get("requested_notional_factor") if decision.get("requested_notional_factor") is not None else default_factor)
        except (TypeError, ValueError):
            factor = default_factor
        factor = max(0.0, min(factor, 1.0))
        original_qty = int(original_order.get("quantity") or 0)
        adjusted_qty = original_qty if action == "continue" else int((original_qty * factor) // lot) * lot
        if action == "reduce" and adjusted_qty >= original_qty:
            adjusted_qty = max(0, original_qty - lot)
        if action == "split":
            if original_qty < lot * 2:
                adjusted_qty = 0
            elif adjusted_qty >= original_qty:
                adjusted_qty = max(lot, (original_qty // 2 // lot) * lot)

        return {
            "action": action,
            "quantity": adjusted_qty,
            "price": float(original_order.get("price") or current_price),
            "reason": str(decision.get("reason") or adjustment_reason or "execution_ready"),
            "decision": decision,
            "warnings": warnings,
            "llm_used": llm_used,
            "observations": observations,
            "adjustment_applied": action != "continue" and (adjusted_qty != original_qty or action == "skip"),
            "adjustment_reason": str(decision.get("adjustment_reason") or adjustment_reason or action),
            "risk_reduction_only": action in {"reduce", "split", "skip"},
        }

    @staticmethod
    def _build_execution_stage_prompt(
        *,
        code: str,
        original_order: dict[str, Any],
        observations: list[dict[str, Any]],
        risk_output: RiskAgentOutput,
    ) -> str:
        return (
            "你是股票执行代理，只输出严格 JSON，不要输出解释、Markdown 或代码块。\n"
            "允许 action 只有：continue, reduce, split, skip。\n"
            "规则：1. 只能维持或降低风险，绝不能放大数量；2. 波动异常时优先 reduce、split 或 skip；"
            "3. requested_notional_factor 必须在 0 到 1 之间。\n\n"
            f"股票代码：{code}\n"
            f"原始订单：{original_order}\n"
            f"执行观测：{observations}\n"
            f"风控输出：{risk_output.to_dict()}\n\n"
            "输出 JSON 字段：action, summary, reason, next_action, confidence, warnings, requested_notional_factor, adjustment_mode, adjustment_reason。"
        )

    @staticmethod
    def _as_number(value: Any, default: float = 0.0) -> float:
        """将输入安全解析为浮点数。"""
        try:
            num = float(value)
            if num != num:
                return default
            return num
        except Exception:
            return default

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        """将输入安全解析为整数。"""
        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _as_text(value: Any) -> str | None:
        """将输入清洗为非空字符串。"""
        text = str(value or "").strip()
        return text or None

    @classmethod
    def _parse_optional_int(cls, value: Any) -> int | None:
        """解析可选整数字段，兼容 `bt-order-*` 格式。"""
        if value is None:
            return None
        if isinstance(value, str) and value.startswith("bt-order-"):
            return cls._as_int(value.replace("bt-order-", "", 1), 0) or None
        parsed = cls._as_int(value, 0)
        return parsed or None

    @classmethod
    def _normalize_positions(cls, value: Any) -> list[dict[str, Any]]:
        """将不同来源的持仓结构规整为统一格式。"""
        if not isinstance(value, list):
            return []
        items: list[dict[str, Any]] = []
        for raw in value:
            if not isinstance(raw, dict):
                continue
            code = str(raw.get("code") or raw.get("stock_code") or raw.get("symbol") or "").strip()
            if not code:
                continue
            qty = max(0, cls._as_int(raw.get("quantity") or raw.get("qty") or raw.get("volume"), 0))
            available_qty = max(0, cls._as_int(raw.get("available_qty") or raw.get("available") or qty, qty))
            avg_cost = cls._as_number(raw.get("avg_cost") or raw.get("cost_price"), 0.0)
            last_price = cls._as_number(raw.get("last_price") or raw.get("price"), 0.0)
            market_value = cls._as_number(raw.get("market_value"), qty * last_price)
            items.append(
                {
                    "code": code,
                    "quantity": qty,
                    "available_qty": min(available_qty, qty),
                    "avg_cost": avg_cost,
                    "last_price": last_price,
                    "market_value": market_value,
                    "unrealized_pnl": cls._as_number(raw.get("unrealized_pnl"), 0.0),
                }
            )
        return items

    @classmethod
    def _normalize_snapshot(
        cls,
        snapshot: dict[str, Any] | None,
        *,
        account_name: str,
        initial_cash: float,
    ) -> dict[str, Any]:
        """归一化账户快照，并补全缺失的汇总字段。"""
        raw = snapshot if isinstance(snapshot, dict) else {}
        positions = cls._normalize_positions(raw.get("positions"))
        inferred_market_value = sum(float(item.get("market_value") or 0.0) for item in positions)
        cash = cls._as_number(raw.get("cash"), initial_cash)
        total_market_value = cls._as_number(raw.get("total_market_value"), inferred_market_value)
        if total_market_value <= 0 and inferred_market_value > 0:
            total_market_value = inferred_market_value
        total_asset = cls._as_number(raw.get("total_asset"), cash + total_market_value)
        if total_asset <= 0:
            total_asset = cash + total_market_value

        return {
            "account_id": raw.get("account_id"),
            "name": str(raw.get("name") or account_name),
            "cash": round(cash, 4),
            "initial_cash": cls._as_number(raw.get("initial_cash"), initial_cash),
            "total_market_value": round(total_market_value, 4),
            "total_asset": round(total_asset, 4),
            "realized_pnl": cls._as_number(raw.get("realized_pnl"), 0.0),
            "unrealized_pnl": cls._as_number(raw.get("unrealized_pnl"), 0.0),
            "cumulative_fees": cls._as_number(raw.get("cumulative_fees"), 0.0),
            "positions": positions,
            "snapshot_at": raw.get("snapshot_at"),
            "data_source": raw.get("data_source"),
            "broker_account_id": raw.get("broker_account_id"),
            "provider_code": raw.get("provider_code"),
            "provider_name": raw.get("provider_name"),
            "account_uid": raw.get("account_uid"),
            "account_display_name": raw.get("account_display_name"),
        }

    @staticmethod
    def _find_position(account_snapshot: dict[str, Any], code: str) -> dict[str, Any]:
        """从账户快照中查找指定股票持仓。"""
        for item in account_snapshot.get("positions", []):
            if str(item.get("code")) == str(code):
                return item
        return {}

    @classmethod
    def _project_snapshot_after_trade(
        cls,
        *,
        snapshot: dict[str, Any],
        code: str,
        current_price: float,
        side: str,
        trade_qty: int,
        cash_after: float,
        avg_cost_after: float,
        position_after_qty: int,
    ) -> dict[str, Any]:
        """基于当前快照推演成交后的账户状态。"""
        positions = [dict(item) for item in snapshot.get("positions", []) if isinstance(item, dict)]
        updated = False
        for idx, pos in enumerate(positions):
            if str(pos.get("code")) != str(code):
                continue
            updated = True
            if position_after_qty <= 0:
                positions.pop(idx)
                break
            pos["quantity"] = position_after_qty
            pos["available_qty"] = position_after_qty
            pos["avg_cost"] = round(avg_cost_after, 4)
            pos["last_price"] = round(current_price, 4)
            pos["market_value"] = round(float(position_after_qty) * current_price, 4)
            pos["unrealized_pnl"] = round((current_price - avg_cost_after) * float(position_after_qty), 4)
            break

        if not updated and side == "buy" and position_after_qty > 0:
            positions.append(
                {
                    "code": code,
                    "quantity": position_after_qty,
                    "available_qty": position_after_qty,
                    "avg_cost": round(avg_cost_after, 4),
                    "last_price": round(current_price, 4),
                    "market_value": round(float(position_after_qty) * current_price, 4),
                    "unrealized_pnl": round((current_price - avg_cost_after) * float(position_after_qty), 4),
                }
            )

        total_market_value = sum(float(item.get("market_value") or 0.0) for item in positions)
        total_asset = float(cash_after) + total_market_value

        next_snapshot = dict(snapshot)
        next_snapshot["cash"] = round(float(cash_after), 4)
        next_snapshot["positions"] = positions
        next_snapshot["total_market_value"] = round(total_market_value, 4)
        next_snapshot["total_asset"] = round(total_asset, 4)
        return next_snapshot

    @staticmethod
    def _target_qty(target_notional: float, effective_price: float, lot: int) -> int:
        """按目标金额、成交价和整手约束计算目标股数。"""
        if target_notional <= 0 or effective_price <= 0 or lot <= 0:
            return 0
        raw_qty = math.floor(target_notional / effective_price)
        return (raw_qty // lot) * lot
