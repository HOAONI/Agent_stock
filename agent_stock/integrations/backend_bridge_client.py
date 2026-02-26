# -*- coding: utf-8 -*-
"""HTTP client for Backend_stock internal agent bridge endpoints."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from src.config import Config, redact_sensitive_payload, redact_sensitive_text


class BackendBridgeError(RuntimeError):
    """Normalized bridge client error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code

    def __str__(self) -> str:
        return self.message


class BackendBridgeClient:
    """Backend internal bridge client with small retry policy."""

    def __init__(
        self,
        *,
        base_url: str,
        service_token: str,
        timeout_seconds: float = 5.0,
        retry_count: int = 1,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.service_token = str(service_token or "").strip()
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.retry_count = max(0, int(retry_count))
        self._session = session or requests.Session()

    @classmethod
    def from_config(cls, config: Config, *, session: Optional[requests.Session] = None) -> "BackendBridgeClient":
        timeout_seconds = float(getattr(config, "agent_backend_request_timeout_ms", 5000)) / 1000.0
        return cls(
            base_url=str(getattr(config, "agent_backend_base_url", "http://127.0.0.1:8002")),
            service_token=str(getattr(config, "agent_service_auth_token", "") or ""),
            timeout_seconds=timeout_seconds,
            retry_count=int(getattr(config, "agent_backend_retry_count", 1)),
            session=session,
        )

    def exchange_credential_ticket(self, ticket: str) -> Dict[str, Any]:
        """Exchange credential ticket for one-time broker credentials."""
        token = str(ticket or "").strip()
        if not token:
            raise BackendBridgeError("credential ticket is empty", status_code=400, error_code="validation_error")
        payload = {"ticket": token}
        return self._post_json("/api/v1/internal/agent/credential-tickets/exchange", payload)

    def post_execution_event(
        self,
        *,
        user_id: int,
        broker_account_id: int,
        task_id: Optional[str],
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Post execution event for backend audit."""
        body: Dict[str, Any] = {
            "user_id": int(user_id),
            "broker_account_id": int(broker_account_id),
            "event_type": str(event_type),
        }
        if task_id:
            body["task_id"] = str(task_id)
        if payload is not None:
            body["payload"] = payload
        if status:
            body["status"] = str(status)
        if error_code:
            body["error_code"] = str(error_code)
        return self._post_json("/api/v1/internal/agent/execution-events", body)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.base_url:
            raise BackendBridgeError("backend bridge base url is not configured", error_code="config_error")
        if not self.service_token:
            raise BackendBridgeError("agent service auth token is not configured", error_code="config_error")

        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.service_token}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.retry_count + 1):
            try:
                response = self._session.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
            except requests.RequestException as exc:
                if attempt < self.retry_count:
                    continue
                message = redact_sensitive_text(str(exc))
                raise BackendBridgeError(
                    f"bridge request failed: {message}",
                    status_code=None,
                    error_code="network_error",
                ) from exc

            if 200 <= response.status_code < 300:
                data = self._safe_json(response)
                if isinstance(data, dict):
                    return data
                return {"ok": True}

            if response.status_code >= 500 and attempt < self.retry_count:
                continue

            data = self._safe_json(response)
            error_code = None
            message: str
            if isinstance(data, dict):
                error_code = str(data.get("error") or "") or None
                message = str(data.get("message") or response.text or f"http_{response.status_code}")
            else:
                message = response.text or f"http_{response.status_code}"
            safe_message = redact_sensitive_text(message)
            raise BackendBridgeError(
                f"bridge http error: {safe_message}",
                status_code=response.status_code,
                error_code=error_code,
            )

        raise BackendBridgeError("unexpected bridge retry state", error_code="internal_error")

    @staticmethod
    def _safe_json(response: requests.Response) -> Any:
        try:
            return redact_sensitive_payload(response.json())
        except ValueError:
            return None
