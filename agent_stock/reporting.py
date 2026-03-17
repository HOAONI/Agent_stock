# -*- coding: utf-8 -*-
"""Agent 运行结果的本地报表生成辅助函数。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from agent_stock.agents.contracts import AgentRunResult


def generate_agent_execution_report(run_payload: Mapping[str, Any]) -> str:
    """为单次持久化运行结果渲染精简版 Markdown 报告。"""
    account_snapshot = run_payload.get("account_snapshot") or {}
    lines = [
        "# Multi-Agent Execution Report",
        "",
        f"- Run ID: {run_payload.get('run_id', '-')}",
        f"- Mode: {run_payload.get('mode', '-')}",
        f"- Trade Date: {run_payload.get('trade_date', '-')}",
        "",
        "## Account Snapshot",
        f"- Cash: {float(account_snapshot.get('cash') or 0.0):.2f}",
        f"- Market Value: {float(account_snapshot.get('total_market_value') or 0.0):.2f}",
        f"- Total Asset: {float(account_snapshot.get('total_asset') or 0.0):.2f}",
        "",
        "## Per-Stock Execution",
        "",
        "| Code | Advice | Target Weight | Action | Traded Qty | Fill Price | Position After |",
        "|---|---|---:|---|---:|---:|---:|",
    ]

    for item in run_payload.get("results") or []:
        signal = item.get("signal") or {}
        risk = item.get("risk") or {}
        execution = item.get("execution") or {}
        fill_price = execution.get("fill_price")
        lines.append(
            "| {code} | {advice} | {weight:.4f} | {action} | {qty} | {price} | {position} |".format(
                code=item.get("code", "-"),
                advice=signal.get("operation_advice", "-"),
                weight=float(risk.get("target_weight") or 0.0),
                action=execution.get("action", "-"),
                qty=int(execution.get("traded_qty") or 0),
                price=f"{float(fill_price):.4f}" if fill_price not in (None, "") else "-",
                position=int(execution.get("position_after") or 0),
            )
        )

    return "\n".join(lines)


def render_run_markdown(run_result: AgentRunResult) -> str:
    """为内存中的运行结果渲染 Markdown 摘要。"""
    lines = [
        f"# Agent Run {run_result.run_id}",
        "",
        f"- Mode: {run_result.mode}",
        f"- Trade Date: {run_result.trade_date.isoformat()}",
        f"- Started: {run_result.started_at.isoformat()}",
        f"- Ended: {run_result.ended_at.isoformat()}",
        "",
        "## Account Snapshot",
        f"- Cash: {float(run_result.account_snapshot.get('cash') or 0.0):.2f}",
        f"- Market Value: {float(run_result.account_snapshot.get('total_market_value') or 0.0):.2f}",
        f"- Total Asset: {float(run_result.account_snapshot.get('total_asset') or 0.0):.2f}",
        "",
        "## Per-Stock Execution",
        "",
        "| Code | Advice | Target Weight | Target Notional | Action | Traded Qty | Fill Price | Position After |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]

    for item in run_result.results:
        fill_price = item.execution.fill_price
        lines.append(
            "| {code} | {advice} | {weight:.4f} | {notional:.2f} | {action} | {qty} | {price} | {position} |".format(
                code=item.code,
                advice=item.signal.operation_advice,
                weight=float(item.risk.target_weight or 0.0),
                notional=float(item.risk.target_notional or 0.0),
                action=item.execution.action,
                qty=int(item.execution.traded_qty or 0),
                price=f"{float(fill_price):.4f}" if fill_price is not None else "-",
                position=int(item.execution.position_after or 0),
            )
        )

    return "\n".join(lines)


def write_run_reports(run_result: AgentRunResult, log_dir: str | Path) -> tuple[Path, Path]:
    """为一次运行写出 Markdown 与 CSV 报表。"""
    report_dir = Path(log_dir) / "agent_reports" / run_result.trade_date.isoformat()
    report_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = report_dir / f"agent_run_{run_result.run_id}.md"
    csv_path = report_dir / f"agent_run_{run_result.run_id}.csv"

    markdown_path.write_text(render_run_markdown(run_result), encoding="utf-8")
    _write_csv_report(csv_path, run_result)
    return markdown_path, csv_path


def _write_csv_report(path: Path, run_result: AgentRunResult) -> None:
    """将逐股票执行结果写入 CSV 文件。"""
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
