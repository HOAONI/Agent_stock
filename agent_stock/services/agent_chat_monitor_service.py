# -*- coding: utf-8 -*-
"""Agent 问股监控快照服务。"""

from __future__ import annotations

import asyncio
import copy
import threading
from collections import defaultdict
from typing import Any

from agent_stock.repositories.chat_repo import AgentChatRepository
from agent_stock.time_utils import local_now

STAGE_ORDER = ("data", "signal", "risk", "execution")
STAGE_TITLES = {
    "data": "数据 Agent",
    "signal": "信号 Agent",
    "risk": "风控 Agent",
    "execution": "执行 Agent",
}
FINAL_STAGE_TITLES = {
    "data": "数据获取 Agent",
    "signal": "信号策略 Agent",
    "risk": "风险控制 Agent",
    "execution": "执行 Agent",
}


def _now_iso() -> str:
    return local_now().isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _as_int(value: Any) -> int | None:
    try:
        parsed = int(float(value))
    except Exception:
        return None
    return parsed if parsed >= 0 else None


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed:
        return None
    return parsed


def _copy_snapshot(value: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(value)


def _empty_session() -> dict[str, Any]:
    return {
        "session_id": "",
        "title": "",
        "user_message": "",
        "source": "agent_chat",
        "live_status": "idle",
        "started_at": None,
        "ended_at": None,
        "updated_at": None,
        "interrupted_reason": None,
        "superseded_run": None,
    }


def _empty_agent_card(code: str, total_calls: int = 0) -> dict[str, Any]:
    return {
        "code": code,
        "title": STAGE_TITLES[code],
        "status": "idle",
        "latest_task": "",
        "latest_duration_ms": None,
        "total_calls": int(total_calls),
        "stock_code": "",
        "confidence": None,
        "warnings": [],
    }


def _empty_stock_detail(stock_code: str) -> dict[str, Any]:
    return {
        "stock_code": stock_code,
        "stock_name": "",
        "planner_trace": [],
        "condition_evaluations": [],
        "stage_visits": [],
        "final_stages": [_empty_final_stage_item(code) for code in STAGE_ORDER],
        "candidate_order": None,
        "execution_result": None,
    }


def _empty_final_stage_item(code: str) -> dict[str, Any]:
    return {
        "code": code,
        "title": FINAL_STAGE_TITLES[code],
        "status": "pending",
        "summary": "",
        "duration_ms": None,
        "input": None,
        "output": None,
        "error_message": None,
        "raw": None,
    }


def _empty_snapshot(total_calls: dict[str, int] | None = None) -> dict[str, Any]:
    counts = total_calls or {}
    return {
        "session": _empty_session(),
        "agent_cards": [_empty_agent_card(code, counts.get(code, 0)) for code in STAGE_ORDER],
        "controller_plan": {
            "goal": "",
            "stage_priority": list(STAGE_ORDER),
            "policy_snapshot": {},
        },
        "execution_chain": [],
        "stock_details": [],
    }


def _analysis_payload_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    structured = _as_dict(meta.get("structured_result"))
    analysis = structured.get("analysis")
    if isinstance(analysis, dict):
        return analysis
    if any(key in structured for key in ("stocks", "stage_traces", "controller_plan")):
        return structured
    return {}


def _message_has_monitor_payload(meta: dict[str, Any]) -> bool:
    analysis = _analysis_payload_from_meta(meta)
    if analysis:
        return True
    if _as_text(_as_dict(meta.get("structured_result")).get("intent")) == "portfolio_health":
        return True
    return False


def _map_trace_status(trace: dict[str, Any]) -> str:
    state = _as_text(trace.get("state")).lower()
    if state == "failed" or _as_text(trace.get("error_message")):
        return "error"
    return "completed"


def _map_session_status(snapshot: dict[str, Any]) -> str:
    session = _as_dict(snapshot.get("session"))
    current_status = _as_text(session.get("live_status")).lower()
    cards = _as_list(snapshot.get("agent_cards"))
    if any(_as_text(card.get("status")) == "error" for card in cards if isinstance(card, dict)):
        return "error"
    if any(_as_text(card.get("status")) == "running" for card in cards if isinstance(card, dict)):
        return "running"
    if current_status in {"error", "running", "interrupted"}:
        return current_status
    if _as_list(snapshot.get("execution_chain")):
        return "completed"
    if current_status == "completed":
        return "completed"
    return "idle"


def _count_calls_from_messages(messages: list[dict[str, Any]]) -> dict[str, int]:
    counts = {code: 0 for code in STAGE_ORDER}
    for message in messages:
        meta = _as_dict(message.get("meta"))
        if not _message_has_monitor_payload(meta):
            continue
        analysis = _analysis_payload_from_meta(meta)
        for trace in _as_list(analysis.get("stage_traces")):
            if not isinstance(trace, dict):
                continue
            code = _as_text(trace.get("stage"))
            if code in counts:
                counts[code] += 1
    return counts


class AgentChatMonitorService:
    """维护最近一轮 Agent 问股的快照与 SSE 订阅。"""

    def __init__(self, repo: AgentChatRepository) -> None:
        self.repo = repo
        self._lock = threading.Lock()
        self._live_runs: dict[str, dict[str, Any]] = {}
        self._cached_snapshots: dict[str, dict[str, Any]] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)

    @staticmethod
    def _owner_key(owner_user_id: int | str) -> str:
        return str(owner_user_id).strip()

    def subscribe(self, owner_user_id: int | str) -> asyncio.Queue[dict[str, Any]]:
        owner_key = self._owner_key(owner_user_id)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        with self._lock:
            self._subscribers[owner_key].add(queue)
        return queue

    def unsubscribe(self, owner_user_id: int | str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        owner_key = self._owner_key(owner_user_id)
        with self._lock:
            queues = self._subscribers.get(owner_key)
            if not queues:
                return
            queues.discard(queue)
            if not queues:
                self._subscribers.pop(owner_key, None)

    def _publish_locked(self, owner_key: str, snapshot: dict[str, Any]) -> None:
        payload = _copy_snapshot(snapshot)
        for queue in list(self._subscribers.get(owner_key, set())):
            try:
                queue.put_nowait(payload)
            except Exception:
                continue

    def _get_historical_counts(self, owner_key: str) -> dict[str, int]:
        cached = self._cached_snapshots.get(owner_key)
        if isinstance(cached, dict):
            counts: dict[str, int] = {}
            for item in _as_list(cached.get("agent_cards")):
                if not isinstance(item, dict):
                    continue
                code = _as_text(item.get("code"))
                if code in STAGE_ORDER:
                    counts[code] = int(item.get("total_calls") or 0)
            if counts:
                return {code: int(counts.get(code, 0)) for code in STAGE_ORDER}
        messages = self.repo.list_messages_by_owner(owner_key, role="assistant", descending=True)
        counts = _count_calls_from_messages(messages)
        return {code: int(counts.get(code, 0)) for code in STAGE_ORDER}

    def start_run(
        self,
        *,
        owner_user_id: int | str,
        session_id: str,
        title: str,
        user_message: str,
    ) -> dict[str, Any] | None:
        owner_key = self._owner_key(owner_user_id)
        now = _now_iso()
        with self._lock:
            previous_run = self._live_runs.get(owner_key)
            superseded_run: dict[str, Any] | None = None
            if isinstance(previous_run, dict):
                previous_snapshot = previous_run.get("snapshot") if isinstance(previous_run.get("snapshot"), dict) else {}
                previous_session = _as_dict(previous_snapshot.get("session"))
                previous_snapshot.setdefault("session", {})
                previous_snapshot["session"]["live_status"] = "interrupted"
                previous_snapshot["session"]["updated_at"] = now
                previous_snapshot["session"]["ended_at"] = now
                previous_snapshot["session"]["interrupted_reason"] = "superseded_by_new_run"
                previous_snapshot["session"]["superseded_run"] = {
                    "session_id": _as_text(session_id),
                    "user_message": _as_text(user_message),
                    "at": now,
                }
                self._cached_snapshots[owner_key] = _copy_snapshot(previous_snapshot)
                self._publish_locked(owner_key, previous_snapshot)
                superseded_run = {
                    "session_id": _as_text(previous_session.get("session_id")),
                    "title": _as_text(previous_session.get("title")),
                    "user_message": _as_text(previous_session.get("user_message")),
                    "started_at": previous_session.get("started_at"),
                    "updated_at": previous_session.get("updated_at"),
                    "ended_at": now,
                    "interrupted_reason": "superseded_by_new_run",
                }
            total_calls = self._get_historical_counts(owner_key)
            snapshot = _empty_snapshot(total_calls)
            snapshot["session"] = {
                "session_id": session_id,
                "title": _as_text(title),
                "user_message": _as_text(user_message),
                "source": "agent_chat",
                "live_status": "running",
                "started_at": now,
                "ended_at": None,
                "updated_at": now,
                "interrupted_reason": None,
                "superseded_run": superseded_run,
            }
            live_run = {
                "owner_key": owner_key,
                "session_id": session_id,
                "snapshot": snapshot,
                "node_index": {},
                "seen_stage_visits": set(),
                "stock_index": {},
            }
            self._live_runs[owner_key] = live_run
            self._publish_locked(owner_key, snapshot)
            return superseded_run

    def get_snapshot(self, owner_user_id: int | str) -> dict[str, Any]:
        owner_key = self._owner_key(owner_user_id)
        with self._lock:
            live = self._live_runs.get(owner_key)
            if isinstance(live, dict):
                return _copy_snapshot(live["snapshot"])
            cached = self._cached_snapshots.get(owner_key)
            if isinstance(cached, dict):
                return _copy_snapshot(cached)
        snapshot = self._build_persisted_snapshot(owner_key)
        with self._lock:
            self._cached_snapshots[owner_key] = _copy_snapshot(snapshot)
        return snapshot

    def record_event(
        self,
        *,
        owner_user_id: int | str,
        session_id: str,
        event_name: str,
        payload: dict[str, Any],
    ) -> None:
        owner_key = self._owner_key(owner_user_id)
        with self._lock:
            live = self._live_runs.get(owner_key)
            if not isinstance(live, dict) or _as_text(live.get("session_id")) != _as_text(session_id):
                return
            snapshot = live["snapshot"]
            snapshot["session"]["updated_at"] = _now_iso()
            if event_name in {"thinking", "tool_start", "tool_done", "message_start", "message_delta"}:
                snapshot["session"]["live_status"] = "running"
            elif event_name == "supervisor_plan":
                snapshot["controller_plan"] = {
                    "goal": _as_text(payload.get("goal")),
                    "stage_priority": [
                        item for item in (_as_list(payload.get("stage_priority")) or list(STAGE_ORDER))
                        if _as_text(item) in STAGE_ORDER
                    ] or list(STAGE_ORDER),
                    "policy_snapshot": _as_dict(payload.get("policy_snapshot")),
                }
            elif event_name in {"planner_step", "planner_replan", "policy_block"}:
                stock_code = _as_text(payload.get("stock_code"))
                if stock_code:
                    detail = self._ensure_stock_detail_locked(live, stock_code)
                    detail["planner_trace"].append({"event": event_name, **copy.deepcopy(payload)})
            elif event_name == "condition_eval":
                stock_code = _as_text(payload.get("stock_code"))
                if stock_code:
                    detail = self._ensure_stock_detail_locked(live, stock_code)
                    detail["condition_evaluations"].append(copy.deepcopy(payload))
            elif event_name == "stage_start":
                self._apply_stage_start_locked(live, payload)
            elif event_name == "stage_update":
                self._apply_stage_update_locked(live, payload)
            elif event_name == "warning":
                self._apply_warning_locked(live, payload)
            elif event_name == "error":
                snapshot["session"]["live_status"] = "error"
            elif event_name == "done":
                snapshot["session"]["live_status"] = "completed"
            snapshot["session"]["live_status"] = _map_session_status(snapshot)
            self._publish_locked(owner_key, snapshot)

    def finalize_run(self, owner_user_id: int | str, session_id: str) -> dict[str, Any]:
        owner_key = self._owner_key(owner_user_id)
        snapshot = self._build_persisted_snapshot(owner_key, preferred_session_id=_as_text(session_id))
        with self._lock:
            live = self._live_runs.get(owner_key)
            if isinstance(live, dict) and _as_text(live.get("session_id")) == _as_text(session_id):
                self._live_runs.pop(owner_key, None)
            snapshot.setdefault("session", {})
            session = _as_dict(snapshot.get("session"))
            snapshot["session"] = {
                **session,
                "ended_at": session.get("ended_at") or _now_iso(),
                "interrupted_reason": session.get("interrupted_reason"),
                "superseded_run": session.get("superseded_run"),
            }
            self._cached_snapshots[owner_key] = _copy_snapshot(snapshot)
            self._publish_locked(owner_key, snapshot)
        return snapshot

    def _ensure_stock_detail_locked(self, live: dict[str, Any], stock_code: str) -> dict[str, Any]:
        stock_code = _as_text(stock_code)
        stock_index = live["stock_index"]
        existing = stock_index.get(stock_code)
        if isinstance(existing, dict):
            return existing
        detail = _empty_stock_detail(stock_code)
        stock_index[stock_code] = detail
        live["snapshot"]["stock_details"].append(detail)
        return detail

    def _ensure_node_locked(self, live: dict[str, Any], payload: dict[str, Any], *, increment_total_calls: bool) -> dict[str, Any] | None:
        stage = _as_text(payload.get("stage"))
        stock_code = _as_text(payload.get("stock_code"))
        visit = _as_int(payload.get("visit"))
        if stage not in STAGE_ORDER or not stock_code or visit is None:
            return None
        key = f"{stock_code}:{stage}:{visit}"
        node_index = live["node_index"]
        existing = node_index.get(key)
        if isinstance(existing, dict):
            return existing

        node = {
            "node_id": f"{_as_text(live.get('session_id'))}:{stock_code}:{stage}:{visit}",
            "stock_code": stock_code,
            "stage": stage,
            "visit": visit,
            "status": "running",
            "summary": f"{STAGE_TITLES[stage]} 运行中",
            "duration_ms": None,
            "confidence": None,
            "warnings": [],
            "started_at": _as_text(payload.get("started_at")) or _now_iso(),
            "finished_at": None,
            "detail": {
                "stock_code": stock_code,
                "stage": stage,
                "visit": visit,
                "status": "running",
                "summary": f"{STAGE_TITLES[stage]} 运行中",
                "duration_ms": None,
                "confidence": None,
                "warnings": [],
                "input": None,
                "output": None,
                "error_message": None,
                "decision": {},
                "observations": [],
                "fallback_chain": [],
                "next_action": "",
                "llm_used": False,
                "started_at": _as_text(payload.get("started_at")) or _now_iso(),
                "finished_at": None,
                "raw": {},
            },
        }
        node_index[key] = node
        live["snapshot"]["execution_chain"].append(node)
        stock_detail = self._ensure_stock_detail_locked(live, stock_code)
        stock_detail["stage_visits"].append(node)
        if increment_total_calls and key not in live["seen_stage_visits"]:
            live["seen_stage_visits"].add(key)
            for card in live["snapshot"]["agent_cards"]:
                if _as_text(card.get("code")) == stage:
                    card["total_calls"] = int(card.get("total_calls") or 0) + 1
                    break
        return node

    def _sync_agent_card_locked(self, live: dict[str, Any], stage: str, *, status: str, stock_code: str, summary: str, duration_ms: int | None, confidence: float | None, warnings: list[str]) -> None:
        for card in live["snapshot"]["agent_cards"]:
            if _as_text(card.get("code")) != stage:
                continue
            card["status"] = status
            card["stock_code"] = stock_code
            card["latest_task"] = summary
            card["latest_duration_ms"] = duration_ms
            card["confidence"] = confidence
            card["warnings"] = list(warnings or [])
            return

    def _update_final_stage_locked(self, live: dict[str, Any], stock_code: str, stage: str, detail: dict[str, Any]) -> None:
        stock_detail = self._ensure_stock_detail_locked(live, stock_code)
        items = stock_detail["final_stages"]
        existing = next((item for item in items if _as_text(item.get("code")) == stage), None)
        if not isinstance(existing, dict):
            existing = _empty_final_stage_item(stage)
            items.append(existing)
        existing.update(
            {
                "status": "failed" if _as_text(detail.get("status")) == "error" else "done",
                "summary": _as_text(detail.get("summary")),
                "duration_ms": _as_int(detail.get("duration_ms")),
                "input": copy.deepcopy(detail.get("input")),
                "output": copy.deepcopy(detail.get("output")),
                "error_message": _as_text(detail.get("error_message")) or None,
                "raw": copy.deepcopy(detail.get("raw")),
            }
        )

    def _apply_stage_start_locked(self, live: dict[str, Any], payload: dict[str, Any]) -> None:
        node = self._ensure_node_locked(live, payload, increment_total_calls=True)
        if not isinstance(node, dict):
            return
        stage = _as_text(node.get("stage"))
        stock_code = _as_text(node.get("stock_code"))
        summary = _as_text(payload.get("summary")) or f"{STAGE_TITLES.get(stage, stage)} 运行中"
        node["summary"] = summary
        detail = _as_dict(node.get("detail"))
        detail.update(
            {
                "summary": summary,
                "started_at": _as_text(payload.get("started_at")) or _as_text(detail.get("started_at")) or _now_iso(),
                "raw": {**copy.deepcopy(detail.get("raw") or {}), **copy.deepcopy(payload)},
            }
        )
        self._sync_agent_card_locked(
            live,
            stage,
            status="running",
            stock_code=stock_code,
            summary=summary,
            duration_ms=None,
            confidence=None,
            warnings=[],
        )

    def _apply_stage_update_locked(self, live: dict[str, Any], payload: dict[str, Any]) -> None:
        node = self._ensure_node_locked(live, payload, increment_total_calls=True)
        if not isinstance(node, dict):
            return
        stage = _as_text(node.get("stage"))
        stock_code = _as_text(node.get("stock_code"))
        warnings = [_as_text(item) for item in _as_list(payload.get("warnings")) if _as_text(item)]
        detail = {
            "stock_code": stock_code,
            "stage": stage,
            "visit": _as_int(payload.get("visit")) or 1,
            "status": _map_trace_status(payload),
            "summary": _as_text(payload.get("summary")) or f"{STAGE_TITLES.get(stage, stage)} 已完成",
            "duration_ms": _as_int(payload.get("duration_ms")),
            "confidence": _as_float(payload.get("confidence")),
            "warnings": warnings,
            "input": copy.deepcopy(payload.get("input")),
            "output": copy.deepcopy(payload.get("output")),
            "error_message": _as_text(payload.get("error_message")) or None,
            "decision": copy.deepcopy(_as_dict(payload.get("decision"))),
            "observations": copy.deepcopy(_as_list(payload.get("observations"))),
            "fallback_chain": copy.deepcopy(_as_list(payload.get("fallback_chain"))),
            "next_action": _as_text(payload.get("next_action")),
            "llm_used": bool(payload.get("llm_used")),
            "started_at": _as_text(payload.get("started_at")) or _as_text(node.get("started_at")) or _now_iso(),
            "finished_at": _as_text(payload.get("finished_at")) or _now_iso(),
            "raw": copy.deepcopy(payload),
        }
        node.update(
            {
                "status": detail["status"],
                "summary": detail["summary"],
                "duration_ms": detail["duration_ms"],
                "confidence": detail["confidence"],
                "warnings": warnings,
                "started_at": detail["started_at"],
                "finished_at": detail["finished_at"],
                "detail": detail,
            }
        )
        self._sync_agent_card_locked(
            live,
            stage,
            status=detail["status"],
            stock_code=stock_code,
            summary=detail["summary"],
            duration_ms=detail["duration_ms"],
            confidence=detail["confidence"],
            warnings=warnings,
        )
        self._update_final_stage_locked(live, stock_code, stage, detail)

    def _apply_warning_locked(self, live: dict[str, Any], payload: dict[str, Any]) -> None:
        stage = _as_text(payload.get("stage"))
        stock_code = _as_text(payload.get("stock_code"))
        message = _as_text(payload.get("message"))
        if stage not in STAGE_ORDER or not stock_code or not message:
            return
        for node in reversed(live["snapshot"]["execution_chain"]):
            if _as_text(node.get("stage")) != stage or _as_text(node.get("stock_code")) != stock_code:
                continue
            if message not in node["warnings"]:
                node["warnings"].append(message)
            detail = _as_dict(node.get("detail"))
            warnings = [_as_text(item) for item in _as_list(detail.get("warnings")) if _as_text(item)]
            if message not in warnings:
                warnings.append(message)
            detail["warnings"] = warnings
            self._sync_agent_card_locked(
                live,
                stage,
                status=_as_text(node.get("status")) or "running",
                stock_code=stock_code,
                summary=_as_text(node.get("summary")),
                duration_ms=_as_int(node.get("duration_ms")),
                confidence=_as_float(node.get("confidence")),
                warnings=node["warnings"],
            )
            return

    def _build_persisted_snapshot(self, owner_key: str, *, preferred_session_id: str = "") -> dict[str, Any]:
        messages = self.repo.list_messages_by_owner(owner_key, role="assistant", descending=True)
        counts = _count_calls_from_messages(messages)
        preferred = _as_text(preferred_session_id)
        selected_message: dict[str, Any] | None = None
        for message in messages:
            meta = _as_dict(message.get("meta"))
            if not _message_has_monitor_payload(meta):
                continue
            if preferred and _as_text(message.get("session_id")) == preferred:
                selected_message = message
                break
            if selected_message is None:
                selected_message = message
        if not isinstance(selected_message, dict):
            return _empty_snapshot(counts)

        session_id = _as_text(selected_message.get("session_id"))
        session_header = self.repo.get_session(owner_key, session_id) or {}
        session_messages = self.repo.list_messages(owner_key, session_id)
        user_message = self._resolve_user_message(session_messages, int(selected_message.get("id") or 0))
        return self._build_snapshot_from_message(
            owner_key=owner_key,
            message=selected_message,
            session_header=session_header,
            session_messages=session_messages,
            user_message=user_message,
            total_calls=counts,
        )

    @staticmethod
    def _resolve_user_message(session_messages: list[dict[str, Any]], assistant_message_id: int) -> str:
        last_user_message = ""
        for item in session_messages:
            if _as_text(item.get("role")) == "user":
                last_user_message = _as_text(item.get("content"))
            if int(item.get("id") or 0) == assistant_message_id:
                break
        return last_user_message

    def _build_snapshot_from_message(
        self,
        *,
        owner_key: str,
        message: dict[str, Any],
        session_header: dict[str, Any],
        session_messages: list[dict[str, Any]],
        user_message: str,
        total_calls: dict[str, int],
    ) -> dict[str, Any]:
        meta = _as_dict(message.get("meta"))
        analysis = _analysis_payload_from_meta(meta)
        snapshot = _empty_snapshot(total_calls)
        stage_traces = [copy.deepcopy(item) for item in _as_list(analysis.get("stage_traces")) if isinstance(item, dict)]
        controller_plan = _as_dict(analysis.get("controller_plan"))
        snapshot["controller_plan"] = {
            "goal": _as_text(controller_plan.get("goal")),
            "stage_priority": [
                item for item in (_as_list(controller_plan.get("stage_priority")) or list(STAGE_ORDER))
                if _as_text(item) in STAGE_ORDER
            ] or list(STAGE_ORDER),
            "policy_snapshot": copy.deepcopy(_as_dict(controller_plan.get("policy_snapshot"))),
        }

        stock_detail_index: dict[str, dict[str, Any]] = {}
        execution_chain: list[dict[str, Any]] = []
        stage_latest: dict[str, dict[str, Any]] = {}
        for trace in stage_traces:
            stage = _as_text(trace.get("stage"))
            stock_code = _as_text(trace.get("stock_code"))
            visit = _as_int(trace.get("visit")) or 1
            if stage not in STAGE_ORDER or not stock_code:
                continue
            warnings = [_as_text(item) for item in _as_list(trace.get("warnings")) if _as_text(item)]
            node = {
                "node_id": f"{_as_text(message.get('session_id'))}:{stock_code}:{stage}:{visit}",
                "stock_code": stock_code,
                "stage": stage,
                "visit": visit,
                "status": _map_trace_status(trace),
                "summary": _as_text(trace.get("summary")) or f"{STAGE_TITLES.get(stage, stage)} 已完成",
                "duration_ms": _as_int(trace.get("duration_ms")),
                "confidence": _as_float(trace.get("confidence")),
                "warnings": warnings,
                "started_at": _as_text(trace.get("started_at")) or None,
                "finished_at": _as_text(trace.get("finished_at")) or None,
                "detail": {
                    "stock_code": stock_code,
                    "stage": stage,
                    "visit": visit,
                    "status": _map_trace_status(trace),
                    "summary": _as_text(trace.get("summary")) or f"{STAGE_TITLES.get(stage, stage)} 已完成",
                    "duration_ms": _as_int(trace.get("duration_ms")),
                    "confidence": _as_float(trace.get("confidence")),
                    "warnings": warnings,
                    "input": copy.deepcopy(trace.get("input")),
                    "output": copy.deepcopy(trace.get("output")),
                    "error_message": _as_text(trace.get("error_message")) or None,
                    "decision": copy.deepcopy(_as_dict(trace.get("decision"))),
                    "observations": copy.deepcopy(_as_list(trace.get("observations"))),
                    "fallback_chain": copy.deepcopy(_as_list(trace.get("fallback_chain"))),
                    "next_action": _as_text(trace.get("next_action")),
                    "llm_used": bool(trace.get("llm_used")),
                    "started_at": _as_text(trace.get("started_at")) or None,
                    "finished_at": _as_text(trace.get("finished_at")) or None,
                    "raw": copy.deepcopy(trace),
                },
            }
            execution_chain.append(node)
            stage_latest[stage] = node
            detail = stock_detail_index.get(stock_code)
            if not isinstance(detail, dict):
                detail = _empty_stock_detail(stock_code)
                stock_detail_index[stock_code] = detail
            detail["stage_visits"].append(node)
            self._apply_final_stage_from_node(detail, node)

        top_planner_trace = [copy.deepcopy(item) for item in _as_list(analysis.get("planner_trace")) if isinstance(item, dict)]
        top_condition_evaluations = [copy.deepcopy(item) for item in _as_list(analysis.get("condition_evaluations")) if isinstance(item, dict)]
        for stock in [copy.deepcopy(item) for item in _as_list(analysis.get("stocks")) if isinstance(item, dict)]:
            stock_code = _as_text(stock.get("code"))
            if not stock_code:
                continue
            detail = stock_detail_index.get(stock_code)
            if not isinstance(detail, dict):
                detail = _empty_stock_detail(stock_code)
                stock_detail_index[stock_code] = detail
            detail["stock_name"] = _as_text(stock.get("name"))
            detail["planner_trace"] = [copy.deepcopy(item) for item in _as_list(stock.get("planner_trace")) if isinstance(item, dict)] or [
                item for item in top_planner_trace if _as_text(item.get("stock_code")) == stock_code
            ]
            detail["condition_evaluations"] = [
                copy.deepcopy(item) for item in _as_list(stock.get("condition_evaluations")) if isinstance(item, dict)
            ] or [
                item for item in top_condition_evaluations if _as_text(item.get("stock_code")) == stock_code
            ]
            detail["candidate_order"] = copy.deepcopy(stock.get("candidate_order")) if isinstance(stock.get("candidate_order"), dict) else None
            detail["execution_result"] = copy.deepcopy(stock.get("execution_result")) if isinstance(stock.get("execution_result"), dict) else None
            raw = _as_dict(stock.get("raw"))
            if raw:
                detail["final_stages"] = self._build_final_stages_from_raw(raw)

        snapshot["session"] = {
            "session_id": _as_text(message.get("session_id")),
            "title": _as_text(session_header.get("title")),
            "user_message": _as_text(user_message),
            "source": "agent_chat",
            "live_status": "completed",
            "started_at": execution_chain[0]["started_at"] if execution_chain else (_as_text(message.get("created_at")) or None),
            "updated_at": _as_text(session_header.get("updated_at")) or _as_text(message.get("created_at")) or None,
        }
        snapshot["execution_chain"] = execution_chain
        snapshot["stock_details"] = list(stock_detail_index.values())
        for card in snapshot["agent_cards"]:
            code = _as_text(card.get("code"))
            latest = stage_latest.get(code)
            if not isinstance(latest, dict):
                continue
            card["status"] = _as_text(latest.get("status")) or "completed"
            card["latest_task"] = _as_text(latest.get("summary"))
            card["latest_duration_ms"] = _as_int(latest.get("duration_ms"))
            card["stock_code"] = _as_text(latest.get("stock_code"))
            card["confidence"] = _as_float(latest.get("confidence"))
            card["warnings"] = list(_as_list(latest.get("warnings")))
        snapshot["session"]["live_status"] = _map_session_status(snapshot)
        return snapshot

    @staticmethod
    def _apply_final_stage_from_node(stock_detail: dict[str, Any], node: dict[str, Any]) -> None:
        stage = _as_text(node.get("stage"))
        detail = _as_dict(node.get("detail"))
        for item in stock_detail["final_stages"]:
            if _as_text(item.get("code")) != stage:
                continue
            item.update(
                {
                    "status": "failed" if _as_text(node.get("status")) == "error" else "done",
                    "summary": _as_text(node.get("summary")),
                    "duration_ms": _as_int(node.get("duration_ms")),
                    "input": copy.deepcopy(detail.get("input")),
                    "output": copy.deepcopy(detail.get("output")),
                    "error_message": _as_text(detail.get("error_message")) or None,
                    "raw": copy.deepcopy(detail.get("raw")),
                }
            )
            return

    @staticmethod
    def _build_final_stages_from_raw(raw: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for code in STAGE_ORDER:
            payload = _as_dict(raw.get(code))
            error_message = _as_text(payload.get("error_message"))
            status = "pending"
            if payload:
                status = "failed" if error_message else "done"
            summary = _as_text(_as_dict(payload.get("decision")).get("summary")) or error_message or ""
            items.append(
                {
                    "code": code,
                    "title": FINAL_STAGE_TITLES[code],
                    "status": status,
                    "summary": summary,
                    "duration_ms": _as_int(payload.get("duration_ms")),
                    "input": copy.deepcopy(payload.get("input")),
                    "output": copy.deepcopy(payload.get("output")),
                    "error_message": error_message or None,
                    "raw": copy.deepcopy(payload) if payload else None,
                }
            )
        return items
