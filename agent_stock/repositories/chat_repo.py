# -*- coding: utf-8 -*-
"""聊天会话仓储。"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import delete, desc, select

from agent_stock.storage import AgentChatMessage, AgentChatSession, DatabaseManager
from agent_stock.time_utils import local_now


class AgentChatRepository:
    """管理聊天会话和消息持久化。"""

    def __init__(self, db_manager: DatabaseManager | None = None) -> None:
        self.db = db_manager or DatabaseManager.get_instance()

    @staticmethod
    def _normalize_owner_user_id(owner_user_id: int | str) -> str:
        return str(owner_user_id).strip()

    @staticmethod
    def _safe_json_dumps(payload: Any) -> str | None:
        if payload is None:
            return None
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _safe_json_loads(payload: str | None) -> dict[str, Any] | None:
        if not payload:
            return None
        try:
            value = json.loads(payload)
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _build_preview(content: str, limit: int = 160) -> str:
        text = " ".join(str(content or "").strip().split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 1]}..."

    @classmethod
    def _build_session_title(cls, title: str | None = None, context: dict[str, Any] | None = None) -> str:
        explicit_title = cls._build_preview(str(title or "").strip(), 48)
        if explicit_title:
            return explicit_title

        if isinstance(context, dict):
            stock_code = str(context.get("stock_code") or context.get("stockCode") or "").strip()
            if stock_code:
                return cls._build_preview(f"{stock_code} 问股", 48) or "Agent问股"

        return "Agent问股"

    def ensure_session(
        self,
        *,
        owner_user_id: int | str,
        session_id: str,
        title: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        owner_id = self._normalize_owner_user_id(owner_user_id)
        now = local_now()
        resolved_title = self._build_session_title(title, context)
        with self.db.get_session() as session:
            row = session.execute(
                select(AgentChatSession).where(
                    AgentChatSession.owner_user_id == owner_id,
                    AgentChatSession.session_id == session_id,
                )
            ).scalar_one_or_none()
            if row is None:
                row = AgentChatSession(
                    owner_user_id=owner_id,
                    session_id=session_id,
                    title=resolved_title,
                    latest_message_preview="",
                    context_json=self._safe_json_dumps(context),
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                if title:
                    row.title = resolved_title
                elif not row.title:
                    row.title = resolved_title
                if context is not None:
                    row.context_json = self._safe_json_dumps(context)
                row.updated_at = now
            session.commit()
            return self.get_session(owner_id, session_id) or {}

    def get_session(self, owner_user_id: int | str, session_id: str) -> dict[str, Any] | None:
        owner_id = self._normalize_owner_user_id(owner_user_id)
        with self.db.get_session() as session:
            row = session.execute(
                select(AgentChatSession).where(
                    AgentChatSession.owner_user_id == owner_id,
                    AgentChatSession.session_id == session_id,
                )
            ).scalar_one_or_none()
            if row is None:
                return None

            message_count = session.execute(
                select(AgentChatMessage)
                .where(
                    AgentChatMessage.owner_user_id == owner_id,
                    AgentChatMessage.session_id == session_id,
                )
                .order_by(desc(AgentChatMessage.created_at))
            ).scalars().all()

            return {
                "session_id": row.session_id,
                "title": row.title,
                "latest_message_preview": row.latest_message_preview,
                "message_count": len(message_count),
                "context": self._safe_json_loads(row.context_json),
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }

    def list_sessions(self, owner_user_id: int | str, limit: int = 50) -> list[dict[str, Any]]:
        owner_id = self._normalize_owner_user_id(owner_user_id)
        with self.db.get_session() as session:
            rows = session.execute(
                select(AgentChatSession)
                .where(AgentChatSession.owner_user_id == owner_id)
                .order_by(desc(AgentChatSession.updated_at), desc(AgentChatSession.created_at))
                .limit(limit)
            ).scalars().all()
            items: list[dict[str, Any]] = []
            for row in rows:
                message_count = session.execute(
                    select(AgentChatMessage.id).where(
                        AgentChatMessage.owner_user_id == owner_id,
                        AgentChatMessage.session_id == row.session_id,
                    )
                ).all()
                items.append(
                    {
                        "session_id": row.session_id,
                        "title": row.title,
                        "latest_message_preview": row.latest_message_preview,
                        "message_count": len(message_count),
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    }
                )
            return items

    def list_messages(self, owner_user_id: int | str, session_id: str) -> list[dict[str, Any]]:
        owner_id = self._normalize_owner_user_id(owner_user_id)
        with self.db.get_session() as session:
            rows = session.execute(
                select(AgentChatMessage)
                .where(
                    AgentChatMessage.owner_user_id == owner_id,
                    AgentChatMessage.session_id == session_id,
                )
                .order_by(AgentChatMessage.created_at.asc(), AgentChatMessage.id.asc())
            ).scalars().all()
            return [
                {
                    "id": row.id,
                    "session_id": row.session_id,
                    "role": row.role,
                    "content": row.content,
                    "meta": self._safe_json_loads(row.meta_json),
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    def add_message(
        self,
        *,
        owner_user_id: int | str,
        session_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        owner_id = self._normalize_owner_user_id(owner_user_id)
        now = local_now()
        clean_role = str(role or "").strip().lower() or "assistant"
        clean_content = str(content or "").strip()

        with self.db.get_session() as session:
            header = session.execute(
                select(AgentChatSession).where(
                    AgentChatSession.owner_user_id == owner_id,
                    AgentChatSession.session_id == session_id,
                )
            ).scalar_one_or_none()
            if header is None:
                header = AgentChatSession(
                    owner_user_id=owner_id,
                    session_id=session_id,
                    title=self._build_preview(clean_content, 48) if clean_role == "user" else "Agent问股",
                    latest_message_preview=self._build_preview(clean_content),
                    created_at=now,
                    updated_at=now,
                )
                session.add(header)
                session.flush()

            message = AgentChatMessage(
                owner_user_id=owner_id,
                session_id=session_id,
                role=clean_role,
                content=clean_content,
                meta_json=self._safe_json_dumps(meta),
                created_at=now,
            )
            session.add(message)

            if not header.title:
                header.title = self._build_preview(clean_content, 48) if clean_role == "user" and clean_content else "Agent问股"
            header.latest_message_preview = self._build_preview(clean_content)
            header.updated_at = now
            session.commit()

            return {
                "id": message.id,
                "session_id": session_id,
                "role": clean_role,
                "content": clean_content,
                "meta": meta,
                "created_at": now.isoformat(),
            }

    def update_session_context(
        self,
        *,
        owner_user_id: int | str,
        session_id: str,
        context: dict[str, Any] | None,
    ) -> None:
        owner_id = self._normalize_owner_user_id(owner_user_id)
        with self.db.get_session() as session:
            row = session.execute(
                select(AgentChatSession).where(
                    AgentChatSession.owner_user_id == owner_id,
                    AgentChatSession.session_id == session_id,
                )
            ).scalar_one_or_none()
            if row is None:
                return
            row.context_json = self._safe_json_dumps(context)
            row.updated_at = local_now()
            session.commit()

    def delete_session(self, owner_user_id: int | str, session_id: str) -> bool:
        owner_id = self._normalize_owner_user_id(owner_user_id)
        with self.db.get_session() as session:
            deleted_messages = session.execute(
                delete(AgentChatMessage).where(
                    AgentChatMessage.owner_user_id == owner_id,
                    AgentChatMessage.session_id == session_id,
                )
            )
            deleted_sessions = session.execute(
                delete(AgentChatSession).where(
                    AgentChatSession.owner_user_id == owner_id,
                    AgentChatSession.session_id == session_id,
                )
            )
            session.commit()
            return bool(deleted_messages.rowcount or deleted_sessions.rowcount)

    def get_latest_assistant_message(self, owner_user_id: int | str, session_id: str) -> dict[str, Any] | None:
        owner_id = self._normalize_owner_user_id(owner_user_id)
        with self.db.get_session() as session:
            row = session.execute(
                select(AgentChatMessage)
                .where(
                    AgentChatMessage.owner_user_id == owner_id,
                    AgentChatMessage.session_id == session_id,
                    AgentChatMessage.role == "assistant",
                )
                .order_by(desc(AgentChatMessage.created_at), desc(AgentChatMessage.id))
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            return {
                "id": row.id,
                "session_id": row.session_id,
                "role": row.role,
                "content": row.content,
                "meta": self._safe_json_loads(row.meta_json),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }

    def list_recent_assistant_messages(
        self,
        owner_user_id: int | str,
        session_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        owner_id = self._normalize_owner_user_id(owner_user_id)
        with self.db.get_session() as session:
            rows = session.execute(
                select(AgentChatMessage)
                .where(
                    AgentChatMessage.owner_user_id == owner_id,
                    AgentChatMessage.session_id == session_id,
                    AgentChatMessage.role == "assistant",
                )
                .order_by(desc(AgentChatMessage.created_at), desc(AgentChatMessage.id))
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "id": row.id,
                    "session_id": row.session_id,
                    "role": row.role,
                    "content": row.content,
                    "meta": self._safe_json_loads(row.meta_json),
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
