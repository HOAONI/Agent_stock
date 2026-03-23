# -*- coding: utf-8 -*-
"""风控智能体。

它不负责决定“看多还是看空”，而是把信号阶段的建议映射成可执行的目标仓位，
并叠加账户级约束、止盈止损和请求级策略覆盖项。
"""

from __future__ import annotations

from datetime import date
from typing import Any


from agent_stock.agents.contracts import AgentState, RiskAgentOutput, SignalAgentOutput
from agent_stock.config import Config, RuntimeStrategyConfig, get_config


class RiskAgent:
    """应用固定仓位映射与账户级限制。"""

    def __init__(self, config: Config | None = None) -> None:
        """初始化风控配置。"""
        self.config = config or get_config()

    def run(
        self,
        *,
        code: str,
        trade_date: date,
        current_price: float,
        signal_output: SignalAgentOutput,
        account_snapshot: dict[str, Any],
        current_position_value: float,
        runtime_strategy: RuntimeStrategyConfig | None = None,
    ) -> RiskAgentOutput:
        """计算目标持仓金额，并应用账户级风险限制。"""
        if current_price <= 0:
            return RiskAgentOutput(
                code=code,
                trade_date=trade_date,
                state=AgentState.FAILED,
                target_weight=0.0,
                target_notional=0.0,
                current_price=current_price,
                hard_risk_triggered=True,
                risk_flags=["invalid_price"],
                error_message="current price is invalid",
            )

        flags = []
        base_weight = self._weight_from_advice(signal_output.operation_advice)
        target_weight = base_weight
        strategy_applied = runtime_strategy is not None

        runtime_stop_loss_pct = (
            float(runtime_strategy.stop_loss_pct)
            if runtime_strategy and runtime_strategy.stop_loss_pct is not None
            else None
        )
        runtime_take_profit_pct = (
            float(runtime_strategy.take_profit_pct)
            if runtime_strategy and runtime_strategy.take_profit_pct is not None
            else None
        )

        signal_stop_loss_enabled = not (runtime_stop_loss_pct is not None and runtime_stop_loss_pct == 0.0)
        signal_take_profit_enabled = not (runtime_take_profit_pct is not None and runtime_take_profit_pct == 0.0)

        signal_stop_loss = (
            float(signal_output.resolved_stop_loss)
            if signal_output.resolved_stop_loss is not None
            else (float(signal_output.stop_loss) if signal_output.stop_loss is not None else None)
        )
        signal_take_profit = (
            float(signal_output.resolved_take_profit)
            if signal_output.resolved_take_profit is not None
            else (float(signal_output.take_profit) if signal_output.take_profit is not None else None)
        )

        effective_stop_loss = signal_stop_loss if signal_stop_loss_enabled else None
        effective_take_profit = signal_take_profit if signal_take_profit_enabled else None

        # 显式卖出建议拥有最高优先级，直接把目标仓位压到 0。
        if self._is_sell_advice(signal_output.operation_advice):
            target_weight = 0.0
            flags.append("sell_signal")

        if signal_stop_loss_enabled and signal_stop_loss is not None and current_price <= signal_stop_loss:
            target_weight = 0.0
            flags.append("stop_loss_triggered")
        if signal_take_profit_enabled and signal_take_profit is not None and current_price >= signal_take_profit:
            target_weight = 0.0
            flags.append("take_profit_triggered")

        position = self._find_position(account_snapshot, code)
        position_qty = int(position.get("quantity") or 0)
        avg_cost = float(position.get("avg_cost") or 0.0)
        # 运行时策略覆盖项只对“已有持仓”的止盈止损生效，因为需要依赖持仓成本价。
        if runtime_strategy and position_qty > 0 and avg_cost > 0:
            if runtime_stop_loss_pct is not None and runtime_stop_loss_pct > 0:
                stop_loss_price = avg_cost * (1.0 - runtime_stop_loss_pct / 100.0)
                effective_stop_loss = stop_loss_price
                if current_price <= stop_loss_price:
                    target_weight = 0.0
                    flags.append("runtime_stop_loss_triggered")
            if runtime_take_profit_pct is not None and runtime_take_profit_pct > 0:
                take_profit_price = avg_cost * (1.0 + runtime_take_profit_pct / 100.0)
                effective_take_profit = take_profit_price
                if current_price >= take_profit_price:
                    target_weight = 0.0
                    flags.append("runtime_take_profit_triggered")

        total_asset = float(account_snapshot.get("total_asset") or 0.0)
        total_market_value = float(account_snapshot.get("total_market_value") or 0.0)
        if total_market_value <= 0:
            total_market_value = sum(
                float(item.get("market_value") or 0.0) for item in account_snapshot.get("positions", [])
            )
        if total_asset <= 0:
            total_asset = float(account_snapshot.get("cash") or 0.0)
        if total_asset <= 0:
            total_asset = float(getattr(self.config, "agent_initial_cash", 1_000_000.0))

        max_single = float(getattr(self.config, "agent_max_single_position_pct", 0.5))
        if runtime_strategy and runtime_strategy.position_max_pct is not None:
            max_single = max(0.0, min(float(runtime_strategy.position_max_pct) / 100.0, 1.0))
        position_cap_pct = max_single * 100.0
        max_total = float(getattr(self.config, "agent_max_total_exposure_pct", 0.9))

        target_weight = max(0.0, min(target_weight, max_single))

        target_notional = total_asset * target_weight
        max_single_notional = total_asset * max_single
        target_notional = min(target_notional, max_single_notional)

        portfolio_excluding_current = max(0.0, total_market_value - current_position_value)
        max_total_notional = total_asset * max_total
        allowed_for_code = max(0.0, max_total_notional - portfolio_excluding_current)
        if target_notional > allowed_for_code:
            target_notional = allowed_for_code
            flags.append("total_exposure_clamped")

        target_weight = (target_notional / total_asset) if total_asset > 0 else 0.0

        return RiskAgentOutput(
            code=code,
            trade_date=trade_date,
            state=AgentState.READY,
            target_weight=round(target_weight, 6),
            target_notional=round(target_notional, 2),
            current_price=current_price,
            stop_loss=round(effective_stop_loss, 4) if effective_stop_loss is not None else None,
            take_profit=round(effective_take_profit, 4) if effective_take_profit is not None else None,
            effective_stop_loss=round(effective_stop_loss, 4) if effective_stop_loss is not None else None,
            effective_take_profit=round(effective_take_profit, 4) if effective_take_profit is not None else None,
            position_cap_pct=round(position_cap_pct, 4),
            strategy_applied=strategy_applied,
            hard_risk_triggered=bool(flags),
            risk_flags=flags,
        )

    def _weight_from_advice(self, advice: str) -> float:
        """根据操作建议映射基础目标仓位。"""
        text = (advice or "").strip().lower()

        strong_buy_tokens = ("强烈买入", "strong buy")
        buy_tokens = ("买入", "加仓", "buy", "add")
        hold_tokens = ("持有", "hold")
        wait_tokens = ("观望", "等待", "wait")
        sell_tokens = ("卖出", "减仓", "清仓", "sell", "reduce", "strong sell", "强烈卖出")

        if any(token in text for token in strong_buy_tokens):
            return float(getattr(self.config, "agent_weight_strong_buy", 0.50))
        if any(token in text for token in sell_tokens):
            return float(getattr(self.config, "agent_weight_sell", 0.0))
        if any(token in text for token in buy_tokens):
            return float(getattr(self.config, "agent_weight_buy", 0.30))
        if any(token in text for token in hold_tokens):
            return float(getattr(self.config, "agent_weight_hold", 0.20))
        if any(token in text for token in wait_tokens):
            return float(getattr(self.config, "agent_weight_wait", 0.0))
        return float(getattr(self.config, "agent_weight_wait", 0.0))

    @staticmethod
    def _is_sell_advice(advice: str) -> bool:
        """判断建议是否明确指向卖出/减仓。"""
        text = (advice or "").strip().lower()
        return any(token in text for token in ("卖出", "减仓", "清仓", "sell", "reduce", "strong sell", "强烈卖出"))

    @staticmethod
    def _find_position(account_snapshot: dict[str, Any], code: str) -> dict[str, Any]:
        """从账户快照中提取指定股票的持仓。"""
        for item in account_snapshot.get("positions", []):
            if str(item.get("code")) == str(code):
                return item
        return {}
