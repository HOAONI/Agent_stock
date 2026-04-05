# -*- coding: utf-8 -*-
"""风控智能体。

它不负责决定“看多还是看空”，而是把信号阶段的建议映射成可执行的目标仓位，
并叠加账户级约束、止盈止损和请求级策略覆盖项。
"""

from __future__ import annotations

from datetime import date
from typing import Any


from agent_stock.agents.contracts import AgentState, RiskAgentOutput, SignalAgentOutput
from agent_stock.agents.agentic_decision import generate_structured_decision
from agent_stock.analyzer import get_analyzer
from agent_stock.config import Config, RuntimeStrategyConfig, get_config


class RiskAgent:
    """应用固定仓位映射与账户级限制。"""

    def __init__(self, config: Config | None = None, analyzer=None) -> None:
        """初始化风控配置。"""
        self.config = config or get_config()
        self.analyzer = analyzer or get_analyzer()

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
            decision = {
                "action": "abort",
                "summary": "当前价格无效，风控阶段中止。",
                "reason": "invalid_price",
                "next_action": "abort",
                "confidence": 0.05,
                "warnings": ["invalid_price"],
            }
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
                observations=[{"current_price": current_price, "signal_advice": signal_output.operation_advice}],
                decision=decision,
                confidence=0.05,
                warnings=["invalid_price"],
                llm_used=False,
                fallback_chain=["risk_rule"],
                next_action="abort",
                status="failed",
                risk_level="high",
                execution_allowed=False,
                hard_blocks=["invalid_price"],
                soft_flags=[],
                review_reason="invalid_price",
                suggested_next="abort",
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

        observations = [
            {
                "signal_advice": signal_output.operation_advice,
                "sentiment_score": signal_output.sentiment_score,
                "current_price": current_price,
                "current_position_value": current_position_value,
                "cash": float(account_snapshot.get("cash") or 0.0),
                "total_asset": total_asset,
                "total_market_value": total_market_value,
                "position_qty": position_qty,
                "base_weight": base_weight,
                "resolved_weight": round(target_weight, 6),
                "resolved_notional": round(target_notional, 2),
                "risk_flags": list(flags),
            }
        ]
        flatten_reason_flags = {
            "sell_signal",
            "stop_loss_triggered",
            "take_profit_triggered",
            "runtime_stop_loss_triggered",
            "runtime_take_profit_triggered",
        }
        should_flatten_position = position_qty > 0 and any(flag in flatten_reason_flags for flag in flags)

        if target_notional <= 0 or target_weight <= 0:
            if should_flatten_position:
                default_decision = {
                    "action": "continue_execution",
                    "summary": "风控要求对现有持仓执行减仓或清仓，可继续进入执行阶段。",
                    "reason": "flatten_position",
                    "next_action": "execution",
                    "confidence": 0.93,
                    "warnings": list(flags),
                    "requested_target_weight_pct": 0.0,
                }
            else:
                default_decision = {
                    "action": "skip_execution",
                    "summary": "风控判断当前不应继续下单，本轮只保留分析与警告。",
                    "reason": "risk_blocked",
                    "next_action": "skip_execution",
                    "confidence": 0.9,
                    "warnings": list(flags),
                }
        else:
            default_decision = {
                "action": "continue_execution",
                "summary": "风控已完成，可进入执行阶段。",
                "reason": "risk_ready",
                "next_action": "execution",
                "confidence": 0.8 if not flags else 0.64,
                "warnings": list(flags),
                "requested_target_weight_pct": round(target_weight * 100.0, 4),
            }

        decision, llm_used = generate_structured_decision(
            analyzer=self.analyzer,
            stage="risk",
            prompt=self._build_risk_stage_prompt(
                code=code,
                signal_output=signal_output,
                observations=observations,
                target_weight=target_weight,
                target_notional=target_notional,
                flags=flags,
            ),
            allowed_actions={"continue_execution", "skip_execution", "request_signal_review", "abort"},
            default_decision=default_decision,
        )
        warnings = list(flags)
        for item in decision.get("warnings") or []:
            if isinstance(item, str) and item not in warnings:
                warnings.append(item)

        action = str(decision.get("action") or default_decision["action"]).strip() or default_decision["action"]
        final_target_weight = float(target_weight)
        final_target_notional = float(target_notional)
        next_action = "execution"
        state = AgentState.READY
        if action == "skip_execution":
            final_target_weight = 0.0
            final_target_notional = 0.0
            next_action = "skip_execution"
            state = AgentState.SKIPPED
        elif action == "request_signal_review":
            final_target_weight = 0.0
            final_target_notional = 0.0
            warnings.append("request_signal_review")
            next_action = "signal"
            state = AgentState.SKIPPED
        elif action == "continue_execution":
            requested_weight_pct = decision.get("requested_target_weight_pct")
            try:
                requested_weight = max(0.0, min(float(requested_weight_pct) / 100.0, final_target_weight))
            except (TypeError, ValueError):
                requested_weight = final_target_weight
            if requested_weight < final_target_weight and final_target_weight > 0:
                final_target_weight = requested_weight
                final_target_notional = round(total_asset * final_target_weight, 2)
                warnings.append("llm_weight_reduced")
        else:
            final_target_weight = 0.0
            final_target_notional = 0.0
            next_action = "abort"
            state = AgentState.FAILED
            if "risk_aborted" not in warnings:
                warnings.append("risk_aborted")

        hard_block_tokens = {
            "invalid_price",
            "risk_aborted",
            "stop_loss_triggered",
            "take_profit_triggered",
            "runtime_stop_loss_triggered",
            "runtime_take_profit_triggered",
            "sell_signal",
        }
        hard_blocks = [item for item in warnings if item in hard_block_tokens]
        soft_flags = [item for item in warnings if item not in hard_blocks]
        execution_allowed = state == AgentState.READY and (
            final_target_notional > 0 or (should_flatten_position and any(flag in hard_blocks for flag in flatten_reason_flags))
        )
        if state == AgentState.FAILED or hard_blocks:
            risk_level = "high"
        elif execution_allowed and not soft_flags:
            risk_level = "low"
        elif execution_allowed:
            risk_level = "medium"
        elif state == AgentState.SKIPPED:
            risk_level = "high" if warnings else "medium"
        else:
            risk_level = "medium"
        review_reason = str(decision.get("reason") or "").strip() or None
        status = "failed" if state == AgentState.FAILED else "review_required" if action == "request_signal_review" else "ready" if execution_allowed else "blocked"

        return RiskAgentOutput(
            code=code,
            trade_date=trade_date,
            state=state,
            target_weight=round(final_target_weight, 6),
            target_notional=round(final_target_notional, 2),
            current_price=current_price,
            stop_loss=round(effective_stop_loss, 4) if effective_stop_loss is not None else None,
            take_profit=round(effective_take_profit, 4) if effective_take_profit is not None else None,
            effective_stop_loss=round(effective_stop_loss, 4) if effective_stop_loss is not None else None,
            effective_take_profit=round(effective_take_profit, 4) if effective_take_profit is not None else None,
            position_cap_pct=round(position_cap_pct, 4),
            strategy_applied=strategy_applied,
            hard_risk_triggered=bool(warnings) or state != AgentState.READY,
            risk_flags=warnings,
            observations=observations,
            decision=decision,
            confidence=float(decision.get("confidence") or default_decision["confidence"]),
            warnings=warnings,
            llm_used=llm_used,
            fallback_chain=["risk_rule", *(("llm_risk_planner",) if llm_used else ())],
            next_action=next_action,
            status=status,
            risk_level=risk_level,
            execution_allowed=execution_allowed,
            hard_blocks=hard_blocks,
            soft_flags=soft_flags,
            review_reason=review_reason if action == "request_signal_review" else None,
            suggested_next=next_action,
        )

    @staticmethod
    def _build_risk_stage_prompt(
        *,
        code: str,
        signal_output: SignalAgentOutput,
        observations: list[dict[str, Any]],
        target_weight: float,
        target_notional: float,
        flags: list[str],
    ) -> str:
        return (
            "你是股票风控代理，只输出严格 JSON，不要输出解释、Markdown 或代码块。\n"
            "允许 action 只有：continue_execution, skip_execution, request_signal_review, abort。\n"
            "规则：1. 不能提高风险，只能维持或降低仓位；2. 若仓位、止盈止损或总暴露受限，可直接 skip_execution；"
            "3. 如需回到信号阶段，只能返回 request_signal_review。\n\n"
            f"股票代码：{code}\n"
            f"信号输出：{signal_output.to_dict()}\n"
            f"风险观测：{observations}\n"
            f"当前目标仓位：{round(target_weight * 100.0, 4)}%\n"
            f"当前目标金额：{round(target_notional, 2)}\n"
            f"风险标记：{flags}\n\n"
            "输出 JSON 字段：action, summary, reason, next_action, confidence, warnings, requested_target_weight_pct。"
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
