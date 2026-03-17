# -*- coding: utf-8 -*-
"""内部回测服务计算测试。

这里主要覆盖确定性回测的统计输出、窗口推断和策略对比逻辑。相比策略模板回测，
这些测试更强调“输入一组候选记录后，服务层如何稳定汇总指标”。
"""

from __future__ import annotations

from datetime import datetime
from datetime import date, timedelta
import unittest

import pandas as pd

from agent_stock.services.backtest_service import BACKTEST_COMPARE_STRATEGY_CODES, BacktestService


class _StubFetcherManager:
    """最小化行情抓取桩，用于给 BacktestService 提供确定性行情。"""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_daily_data(self, stock_code, start_date=None, end_date=None, days=30):  # noqa: ARG002
        return self.frame.copy(), "stub"


class BacktestServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # 这组样本既能覆盖止盈/止损评估，也足以构造数据不足场景。
        frame = pd.DataFrame(
            [
                {"date": "2026-01-02", "high": 10.2, "low": 9.8, "close": 10.0},
                {"date": "2026-01-03", "high": 10.8, "low": 10.1, "close": 10.5},
                {"date": "2026-01-04", "high": 11.2, "low": 10.6, "close": 11.0},
                {"date": "2026-01-05", "high": 11.8, "low": 11.0, "close": 11.5},
                {"date": "2026-01-06", "high": 12.2, "low": 11.6, "close": 12.0},
                {"date": "2026-01-07", "high": 12.6, "low": 12.0, "close": 12.4},
            ]
        )
        self.service = BacktestService(fetcher_manager=_StubFetcherManager(frame))

    def test_run_outputs_completed_and_insufficient(self):
        payload = {
            "eval_window_days": 3,
            "engine_version": "v1",
            "neutral_band_pct": 2.0,
            "candidates": [
                {
                    "analysis_history_id": 1,
                    "owner_user_id": 10,
                    "code": "600519",
                    "created_at": datetime(2026, 1, 3).isoformat(),
                    "context_snapshot": '{"enhanced_context":{"date":"2026-01-03"}}',
                    "operation_advice": "买入",
                    "stop_loss": 9.5,
                    "take_profit": 12.0,
                },
                {
                    "analysis_history_id": 2,
                    "owner_user_id": 10,
                    "code": "600519",
                    "created_at": datetime(2026, 1, 7).isoformat(),
                    "context_snapshot": '{"enhanced_context":{"date":"2026-01-07"}}',
                    "operation_advice": "买入",
                    "stop_loss": 11.0,
                    "take_profit": 13.0,
                },
            ],
        }

        result = self.service.run(payload)

        self.assertEqual(result["processed"], 2)
        self.assertEqual(result["saved"], 2)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["insufficient"], 1)
        self.assertEqual(len(result["items"]), 2)

        completed = next(item for item in result["items"] if item["analysis_history_id"] == 1)
        self.assertEqual(completed["eval_status"], "completed")
        self.assertEqual(completed["position_recommendation"], "long")
        self.assertIn(completed["first_hit"], {"take_profit", "neither", "ambiguous", "stop_loss"})

        insufficient = next(item for item in result["items"] if item["analysis_history_id"] == 2)
        self.assertEqual(insufficient["eval_status"], "insufficient_data")

    def test_compare_generates_all_default_strategies(self):
        compare_payload = {
            "eval_window_days_list": [5],
            "neutral_band_pct": 2.0,
            "rows_by_window": {
                "5": [
                    {
                        "code": "600519",
                        "analysis_date": "2026-01-03",
                        "evaluated_at": "2026-01-10T00:00:00Z",
                        "operation_advice": "买入",
                        "stop_loss": 9.5,
                        "take_profit": 12.5,
                        "simulated_return_pct": 3.0,
                        "stock_return_pct": 2.5,
                        "eval_status": "completed",
                        "position_recommendation": "long",
                        "context_snapshot": '{"enhanced_context":{"raw_data":[{"date":"2026-01-02","close":10.0},{"date":"2026-01-03","close":10.5}]}}',
                    }
                ]
            },
        }

        result = self.service.compare(compare_payload)
        items = result["items"]

        self.assertEqual(len(items), len(BACKTEST_COMPARE_STRATEGY_CODES))
        got_codes = {item["strategy_code"] for item in items}
        self.assertEqual(got_codes, set(BACKTEST_COMPARE_STRATEGY_CODES))
        self.assertTrue(all(item["eval_window_days"] == 5 for item in items))

    def test_summary_and_compare_expose_prediction_and_trade_win_rates(self):
        summary_payload = {
            "scope": "overall",
            "eval_window_days": 5,
            "rows": [
                {
                    "eval_status": "completed",
                    "position_recommendation": "long",
                    "outcome": "win",
                    "direction_correct": True,
                    "stock_return_pct": 3.2,
                    "simulated_return_pct": 2.6,
                },
                {
                    "eval_status": "completed",
                    "position_recommendation": "cash",
                    "outcome": "loss",
                    "direction_correct": False,
                    "stock_return_pct": 4.1,
                    "simulated_return_pct": 0.2,
                },
            ],
        }
        summary = self.service.summary(summary_payload)
        self.assertIn("prediction_win_rate_pct", summary)
        self.assertIn("trade_win_rate_pct", summary)

        compare_payload = {
            "eval_window_days_list": [5],
            "strategy_codes": ["agent_v1"],
            "neutral_band_pct": 2.0,
            "rows_by_window": {
                "5": [
                    {
                        "code": "600519",
                        "analysis_date": "2026-01-03",
                        "evaluated_at": "2026-01-10T00:00:00Z",
                        "operation_advice": "卖出",
                        "simulated_return_pct": 0.0,
                        "stock_return_pct": -3.0,
                        "eval_status": "completed",
                        "position_recommendation": "cash",
                    }
                ]
            },
        }
        compare = self.service.compare(compare_payload)
        self.assertEqual(compare["metric_definition_version"], "v2")
        first = compare["items"][0]
        self.assertEqual(first["direction_accuracy_pct"], 100.0)
        self.assertIn("prediction_win_rate_pct", first)
        self.assertIn("trade_win_rate_pct", first)

    def test_compare_infers_windows_from_rows_by_window_keys(self):
        payload = {
            "rows_by_window": {
                "5": [
                    {
                        "code": "600519",
                        "analysis_date": "2026-01-03",
                        "stock_return_pct": 2.5,
                        "simulated_return_pct": 3.0,
                        "eval_status": "completed",
                    }
                ]
            }
        }
        result = self.service.compare(payload)
        self.assertEqual(len(result["items"]), len(BACKTEST_COMPARE_STRATEGY_CODES))
        self.assertTrue(all(item["eval_window_days"] == 5 for item in result["items"]))

    def test_compare_invalid_window_payload_raises_validation_error(self):
        payload = {"eval_window_days_list": [0, 121, "bad"], "rows_by_window": {"bad": []}}
        with self.assertRaisesRegex(ValueError, "eval_window_days_list must contain integers"):
            self.service.compare(payload)

    def test_compare_empty_window_payload_raises_validation_error(self):
        with self.assertRaisesRegex(ValueError, "eval_window_days_list is required"):
            self.service.compare({"eval_window_days_list": [], "rows_by_window": {}})

    def test_compare_fetch_failure_raises_error(self):
        class _FailingFetcher:
            def get_daily_data(self, stock_code, start_date=None, end_date=None, days=30):  # noqa: ARG002
                raise RuntimeError("network down")

        service = BacktestService(fetcher_manager=_FailingFetcher())
        payload = {
            "eval_window_days_list": [5],
            "rows_by_window": {
                "5": [
                    {
                        "code": "600519",
                        "analysis_date": "2026-01-03",
                        "stock_return_pct": 2.5,
                        "simulated_return_pct": 3.0,
                        "eval_status": "completed",
                    }
                ]
            },
        }
        with self.assertRaisesRegex(ValueError, "compare_fetch_failed"):
            service.compare(payload)

    def test_compare_ma20_uses_cross_up_entry_rule(self):
        rows = []
        start = date(2026, 1, 1)
        price = 100.0
        for idx in range(40):
            day = start + timedelta(days=idx)
            if day.weekday() >= 5:
                continue
            price += 1.0
            rows.append({"date": day.isoformat(), "high": price * 1.01, "low": price * 0.99, "close": price})
        frame = pd.DataFrame(rows)
        service = BacktestService(fetcher_manager=_StubFetcherManager(frame))

        payload = {
            "eval_window_days_list": [5],
            "strategy_codes": ["ma20_trend"],
            "neutral_band_pct": 2.0,
            "rows_by_window": {
                "5": [
                    {
                        "code": "600519",
                        "analysis_date": rows[-10]["date"],
                        "stock_return_pct": 3.0,
                        "simulated_return_pct": 3.0,
                        "eval_status": "completed",
                    }
                ]
            },
        }
        result = service.compare(payload)
        self.assertEqual(len(result["items"]), 1)
        # 在持续上升趋势中，如果分析日没有新的 `cross_up`，
        # 标准 MA20 策略应保持空仓。
        self.assertEqual(result["items"][0]["prediction_win_rate_pct"], 0.0)

    def test_compare_rsi14_uses_lt30_entry_rule(self):
        close_values = [
            100.0,
            99.0749139528992,
            101.12534184434045,
            102.7361133293036,
            101.23839674250524,
            101.21529030968389,
            100.91157128430084,
            101.82667149404978,
            103.58018460768752,
            101.3229768962485,
            98.41825718968939,
            100.38098377573796,
            99.98586911211048,
            101.56404788166175,
            98.65134125032676,
            98.33290740224446,
            99.65688867286597,
            98.03904812690826,
            100.6525162629047,
            103.33942126517477,
            100.45622734160707,
            97.70854817866564,
            99.21439611590913,
            101.17069955921891,
            99.20875653555604,
            97.39213825836694,
            96.48288103962673,
            94.96014977920612,
            96.89486993589695,
            96.67965432631074,
            98.12504833597046,
            97.18442531096302,
            95.86216845205652,
            96.8040498767958,
            95.83810425024688,
            93.91553116551252,
            92.76009155000667,
            92.67534185470648,
            91.1674309144615,
            92.02187733051331,
            92.27808413154286,
        ]
        rows = []
        start = date(2026, 1, 1)
        for idx, close in enumerate(close_values):
            day = start + timedelta(days=idx)
            rows.append({"date": day.isoformat(), "high": close * 1.01, "low": close * 0.99, "close": close})
        frame = pd.DataFrame(rows)
        service = BacktestService(fetcher_manager=_StubFetcherManager(frame))

        payload = {
            "eval_window_days_list": [5],
            "strategy_codes": ["rsi14_mean_reversion"],
            "neutral_band_pct": 2.0,
            "rows_by_window": {
                "5": [
                    {
                        "code": "600519",
                        "analysis_date": rows[35]["date"],
                        "stock_return_pct": 3.0,
                        "simulated_return_pct": 3.0,
                        "eval_status": "completed",
                    }
                ]
            },
        }
        result = service.compare(payload)
        self.assertEqual(len(result["items"]), 1)
        # 分析日 RSI 介于 30 到 40 之间，因此标准 `<30` 规则应保持空仓。
        self.assertEqual(result["items"][0]["prediction_win_rate_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
