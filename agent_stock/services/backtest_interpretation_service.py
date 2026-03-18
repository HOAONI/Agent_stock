# -*- coding: utf-8 -*-
"""回测结果自然语言解读服务。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from agent_stock.analyzer import GeminiAnalyzer
from agent_stock.config import Config, RuntimeLlmConfig, get_config, redact_sensitive_text

logger = logging.getLogger(__name__)

UNAVAILABLE_SUMMARY = "AI 解读暂不可用，请先检查当前运行环境里的模型配置。"
FAILED_SUMMARY = "AI 解读生成失败，请稍后重试。"


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _clean_text(value: Any, max_length: int = 4096) -> str:
    text = str(value or "").strip()
    if len(text) <= max_length:
        return text
    return text[:max_length]


@dataclass(frozen=True)
class InterpretationItem:
    """单条回测解读请求。"""

    item_key: str
    item_type: str
    label: str
    code: str
    requested_range: Dict[str, Any]
    effective_range: Dict[str, Any]
    metrics: Dict[str, Any]
    benchmark: Dict[str, Any]
    context: Dict[str, Any]


class BacktestInterpretationService:
    """把结构化回测结果转换为中文自然语言说明。"""

    def __init__(
        self,
        *,
        config: Optional[Config] = None,
        analyzer_factory: Optional[Callable[[Optional[RuntimeLlmConfig]], Any]] = None,
    ) -> None:
        self.config = config or get_config()
        self._analyzer_factory = analyzer_factory or (
            lambda runtime_llm: GeminiAnalyzer(config=self.config, runtime_llm=runtime_llm)
        )

    def interpret(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """批量生成回测结果中文解读。"""
        items = self._parse_items(payload)
        if not items:
            return {"items": []}

        runtime_llm = self._parse_runtime_llm(_as_dict(payload.get("runtime_llm")))
        analyzer = self._analyzer_factory(runtime_llm)
        if not getattr(analyzer, "is_available", lambda: False)():
            return {
                "items": [
                    self._build_result(
                        item_key=item.item_key,
                        status="unavailable",
                        summary=UNAVAILABLE_SUMMARY,
                    )
                    for item in items
                ]
            }

        prompt = self._build_prompt(items)
        try:
            raw_text = analyzer.generate_text(
                prompt,
                temperature=0.2,
                max_output_tokens=2200,
            )
            parsed = self._parse_response(raw_text)
        except Exception as exc:
            safe_error = redact_sensitive_text(str(exc))
            logger.warning("backtest interpretation failed: %s", safe_error)
            return {
                "items": [
                    self._build_result(
                        item_key=item.item_key,
                        status="failed",
                        summary=FAILED_SUMMARY,
                        error_message=safe_error,
                    )
                    for item in items
                ]
            }

        parsed_by_key = {
            row["item_key"]: row
            for row in parsed
            if _clean_text(row.get("item_key"), 128)
        }
        results: List[Dict[str, Any]] = []
        for item in items:
            matched = parsed_by_key.get(item.item_key)
            if not matched:
                results.append(
                    self._build_result(
                        item_key=item.item_key,
                        status="failed",
                        summary=FAILED_SUMMARY,
                        error_message="missing_item_in_model_response",
                    )
                )
                continue
            results.append(
                self._build_result(
                    item_key=item.item_key,
                    status="ready",
                    verdict=matched.get("verdict"),
                    summary=matched.get("summary"),
                )
            )
        return {"items": results}

    def _parse_items(self, payload: Dict[str, Any]) -> List[InterpretationItem]:
        items: List[InterpretationItem] = []
        for row in _as_list_of_dicts(payload.get("items")):
            item_key = _clean_text(row.get("item_key"), 128)
            if not item_key:
                continue
            label = _clean_text(row.get("label"), 128) or item_key
            items.append(
                InterpretationItem(
                    item_key=item_key,
                    item_type=_clean_text(row.get("item_type"), 32) or "strategy",
                    label=label,
                    code=_clean_text(row.get("code"), 32),
                    requested_range=_as_dict(row.get("requested_range")),
                    effective_range=_as_dict(row.get("effective_range")),
                    metrics=_as_dict(row.get("metrics")),
                    benchmark=_as_dict(row.get("benchmark")),
                    context=_as_dict(row.get("context")),
                )
            )
        return items

    @staticmethod
    def _parse_runtime_llm(payload: Dict[str, Any]) -> RuntimeLlmConfig | None:
        if not payload:
            return None

        provider = _clean_text(payload.get("provider"), 32).lower()
        base_url = _clean_text(payload.get("base_url"), 1024) or None
        model = _clean_text(payload.get("model"), 128) or None
        api_token = _clean_text(payload.get("api_token"), 4096) or None
        if not provider or not base_url or not model:
            raise ValueError("runtime_llm.provider/base_url/model are required")
        return RuntimeLlmConfig(
            provider=provider,
            base_url=base_url,
            model=model,
            api_token=api_token,
            has_token=bool(payload.get("has_token") or api_token),
        )

    def _build_prompt(self, items: List[InterpretationItem]) -> str:
        prompt_payload = [
            {
                "item_key": item.item_key,
                "item_type": item.item_type,
                "label": item.label,
                "code": item.code,
                "requested_range": item.requested_range,
                "effective_range": item.effective_range,
                "metrics": item.metrics,
                "benchmark": item.benchmark,
                "context": item.context,
            }
            for item in items
        ]
        return f"""你是一名量化回测结果解读助手。请基于给定事实，为每个回测结果输出面向普通投资者的中文说明。

硬性规则：
1. 只能依据输入中的区间、指标和上下文，不得编造未来预测、未提供的新闻、基本面或额外事实。
2. 输出中文，不要 Markdown，不要项目符号。
3. 每个 summary 控制在 2-3 句，优先解释总收益、最大回撤、夏普/胜率/交易次数等关键数字对风险收益的含义。
4. 如果 total_trades 为 0 或者样本明显不足，要明确说明“无成交/样本不足”，不要伪造表现判断。
5. verdict 是一句很短的结论标签，例如“表现中等”“收益偏弱”“回撤可控”“样本不足”，控制在 8 个字内。
6. 最大回撤如果是负数，说明时要按其绝对值理解。
7. 不要给出买卖建议，只做历史表现解释。

输出必须是严格 JSON，对象结构如下：
{{"items":[{{"item_key":"原样返回","verdict":"简短标签","summary":"2-3句中文说明"}}]}}

输入数据：
{json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ":"))}
"""

    def _parse_response(self, raw_text: str) -> List[Dict[str, Any]]:
        payload = self._extract_json(raw_text)
        if isinstance(payload, list):
            items = payload
        else:
            items = _as_list_of_dicts(_as_dict(payload).get("items"))
        parsed: List[Dict[str, Any]] = []
        for row in items:
            item_key = _clean_text(row.get("item_key"), 128)
            summary = _clean_text(row.get("summary"), 280)
            if not item_key or not summary:
                continue
            parsed.append(
                {
                    "item_key": item_key,
                    "verdict": _clean_text(row.get("verdict"), 24) or None,
                    "summary": summary,
                }
            )
        if not parsed:
            raise ValueError("invalid_interpretation_response: no valid items")
        return parsed

    @staticmethod
    def _extract_json(raw_text: str) -> Any:
        text = _clean_text(raw_text, 20000)
        fenced = text.replace("```json", "```").replace("```JSON", "```")
        segments = [segment.strip() for segment in fenced.split("```") if segment.strip()]
        candidates = segments + [text]
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                pass

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            try:
                return json.loads(text[first_brace:last_brace + 1])
            except Exception:
                pass

        first_bracket = text.find("[")
        last_bracket = text.rfind("]")
        if first_bracket >= 0 and last_bracket > first_bracket:
            return json.loads(text[first_bracket:last_bracket + 1])
        raise ValueError("invalid_interpretation_response: no json object found")

    @staticmethod
    def _build_result(
        *,
        item_key: str,
        status: str,
        summary: Any,
        verdict: Any = None,
        error_message: Any = None,
    ) -> Dict[str, Any]:
        return {
            "item_key": item_key,
            "status": status,
            "verdict": _clean_text(verdict, 24) or None,
            "summary": _clean_text(summary, 280) or (FAILED_SUMMARY if status == "failed" else UNAVAILABLE_SUMMARY),
            "error_message": _clean_text(error_message, 280) or None,
        }


_backtest_interpretation_service: BacktestInterpretationService | None = None


def get_backtest_interpretation_service() -> BacktestInterpretationService:
    """返回回测解读服务单例。"""
    global _backtest_interpretation_service
    if _backtest_interpretation_service is None:
        _backtest_interpretation_service = BacktestInterpretationService()
    return _backtest_interpretation_service
