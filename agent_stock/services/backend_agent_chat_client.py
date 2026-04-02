# -*- coding: utf-8 -*-
"""Agent 问股工具对 Backend 的内部桥接客户端。"""

from __future__ import annotations

import os
from typing import Any

import httpx

from agent_stock.config import Config, get_config, redact_sensitive_text


class BackendAgentChatClient:
    """访问 Backend 内部 agent-chat 工具接口。"""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.base_url = (
            os.getenv("BACKEND_INTERNAL_BASE_URL")
            or os.getenv("BACKEND_BASE_URL")
            or "http://127.0.0.1:8002"
        ).rstrip("/")
        self.token = str(getattr(self.config, "agent_service_auth_token", "") or "").strip()
        self.timeout = max(5.0, float(getattr(self.config, "agent_llm_request_timeout_ms", 120000)) / 1000.0)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}{path}",
                    headers=self._headers(),
                    json=payload,
                )
        except Exception as exc:
            raise RuntimeError(f"backend bridge request failed: {redact_sensitive_text(str(exc))}") from exc

        if response.status_code >= 400:
            message = response.text
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    detail = payload.get("message") or payload.get("error") or payload
                    message = str(detail)
            except Exception:
                pass
            raise RuntimeError(
                f"backend bridge request failed with {response.status_code}: {redact_sensitive_text(message)}"
            )

        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("backend bridge returned invalid payload")
        return data

    async def get_runtime_account_context(self, *, owner_user_id: int, refresh: bool = True) -> dict[str, Any]:
        return await self._post(
            "/internal/v1/agent-chat/runtime-context",
            {"owner_user_id": owner_user_id, "refresh": bool(refresh)},
        )

    async def get_analysis_history(
        self,
        *,
        owner_user_id: int,
        stock_codes: list[str] | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        return await self._post(
            "/internal/v1/agent-chat/analysis-history",
            {
                "owner_user_id": owner_user_id,
                "stock_codes": stock_codes or [],
                "limit": max(1, min(int(limit), 20)),
            },
        )

    async def get_backtest_summary(
        self,
        *,
        owner_user_id: int,
        stock_codes: list[str] | None = None,
        limit: int = 6,
    ) -> dict[str, Any]:
        return await self._post(
            "/internal/v1/agent-chat/backtest-summary",
            {
                "owner_user_id": owner_user_id,
                "stock_codes": stock_codes or [],
                "limit": max(1, min(int(limit), 20)),
            },
        )

    async def place_simulated_order(
        self,
        *,
        owner_user_id: int,
        session_id: str,
        candidate_order: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._post(
            "/internal/v1/agent-chat/place-simulated-order",
            {
                "owner_user_id": owner_user_id,
                "session_id": session_id,
                "candidate_order": candidate_order,
            },
        )
