# -*- coding: utf-8 -*-
"""组合策略 DSL 解析测试。"""

from __future__ import annotations

import pytest

from agent_stock.services.strategy_rule_dsl import build_rule_dsl_from_text, normalize_rule_dsl_params


def test_build_rule_dsl_from_text_parses_combined_entry_and_exit_rules():
    payload = build_rule_dsl_from_text("如果 MACD 金叉且 RSI<30 时买入，跌破 5 日线止损，过去一年收益怎样")

    assert payload is not None
    entry = payload["entry"]
    exit_group = payload["exit"]

    assert entry["operator"] == "and"
    assert len(entry["conditions"]) == 2
    assert entry["conditions"][0]["kind"] == "macd_cross"
    assert entry["conditions"][0]["direction"] == "bullish"
    assert entry["conditions"][1] == {
        "kind": "rsi_threshold",
        "period": 14,
        "operator": "lt",
        "threshold": 30,
    }

    assert exit_group["operator"] == "or"
    assert exit_group["conditions"] == [
        {
            "kind": "price_ma_relation",
            "maWindow": 5,
            "relation": "cross_below",
        }
    ]


def test_build_rule_dsl_from_text_returns_none_for_simple_single_template():
    assert build_rule_dsl_from_text("如果我在每次 MACD 金叉时买入，过去一年收益怎样") is None


def test_normalize_rule_dsl_params_requires_entry_conditions():
    with pytest.raises(ValueError, match="rule_dsl.entry.conditions is required"):
        normalize_rule_dsl_params({"exit": {"operator": "or", "conditions": [{"kind": "stop_loss_pct", "pct": 8}]}})
