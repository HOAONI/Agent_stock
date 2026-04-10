# -*- coding: utf-8 -*-
"""回测结果自然语言解读服务。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Callable


from agent_stock.analyzer import GeminiAnalyzer
from agent_stock.config import Config, RuntimeLlmConfig, get_config, redact_sensitive_text

logger = logging.getLogger(__name__)

UNAVAILABLE_SUMMARY = "AI 解读暂不可用，请先检查当前运行环境里的模型配置。"
FAILED_SUMMARY = "AI 解读生成失败，请稍后重试。"
BACKTEST_INTERPRETATION_SYSTEM_PROMPT = """你是一名量化回测结果解读助手。

你的唯一任务是根据用户给出的结构化回测数据，输出严格 JSON 格式的中文解读结果。

必须遵守：
1. 只能依据输入中的区间、指标、基准和上下文，不得编造未来预测、新闻、基本面或额外事实。
2. 仅输出 JSON，不要 Markdown，不要代码块，不要额外解释。
3. 返回结构必须为 {"items":[{"item_key":"原样返回","verdict":"简短标签","summary":"2-3句中文说明"}]}。
4. summary 只写 2-3 句中文，优先解释总收益、回撤、夏普、胜率、交易次数等风险收益含义。
5. 若 total_trades 为 0 或样本明显不足，必须明确说明“无成交”或“样本不足”，不要伪造表现判断。
6. verdict 控制在 8 个字以内，例如“表现中等”“收益偏弱”“回撤可控”“样本不足”。
7. 若最大回撤为负数，解读时按其绝对值理解。
8. 不要给出买卖建议，只解释历史表现。"""


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
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
    requested_range: dict[str, Any]
    effective_range: dict[str, Any]
    metrics: dict[str, Any]
    benchmark: dict[str, Any]
    context: dict[str, Any]


class BacktestInterpretationService:
    """把结构化回测结果转换为中文自然语言说明。"""

    def __init__(
        self,
        *,
        config: Config | None = None,
        analyzer_factory: Callable[[RuntimeLlmConfig | None], Any] | None = None,
    ) -> None:
        self.config = config or get_config()
        self._analyzer_factory = analyzer_factory or (
            lambda runtime_llm: GeminiAnalyzer(
                config=self.config,
                runtime_llm=runtime_llm,
                system_prompt=BACKTEST_INTERPRETATION_SYSTEM_PROMPT,
            )
        )

    def interpret(self, payload: dict[str, Any]) -> dict[str, Any]:
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
                max_output_tokens=self._resolve_max_output_tokens(len(items)),
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
        results: list[dict[str, Any]] = []
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

    def _parse_items(self, payload: dict[str, Any]) -> list[InterpretationItem]:
        items: list[InterpretationItem] = []
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
    def _parse_runtime_llm(payload: dict[str, Any]) -> RuntimeLlmConfig | None:
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

    def _build_prompt(self, items: list[InterpretationItem]) -> str:
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
        return (
            "请基于下面的输入数据返回严格 JSON。"
            "保持 item_key 原样返回，summary 使用 2-3 句中文，verdict 使用简短中文标签。\n"
            f"输入数据：\n{json.dumps(prompt_payload, ensure_ascii=False, separators=(',', ':'))}"
        )

    @staticmethod
    def _resolve_max_output_tokens(item_count: int) -> int:
        safe_count = max(1, int(item_count or 0))
        return min(1800, 320 + (220 * safe_count))

    def _parse_response(self, raw_text: str) -> list[dict[str, Any]]:
        payload = self._extract_json(raw_text)
        if isinstance(payload, list):
            items = payload
        else:
            items = _as_list_of_dicts(_as_dict(payload).get("items"))
        parsed: list[dict[str, Any]] = []
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
    ) -> dict[str, Any]:
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
