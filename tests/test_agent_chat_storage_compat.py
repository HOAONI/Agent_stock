# -*- coding: utf-8 -*-
"""聊天存储兼容迁移测试。"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from agent_stock.config import Config
from agent_stock.repositories.chat_repo import AgentChatRepository
from agent_stock.storage import DatabaseManager


class AgentChatStorageCompatTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "legacy_chat.db")
        os.environ["DATABASE_PATH"] = self.db_path
        os.environ["DATABASE_URL"] = ""

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE agent_chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_id VARCHAR(64) NOT NULL UNIQUE,
                    title VARCHAR(255) NOT NULL,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE agent_chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_id VARCHAR(64) NOT NULL,
                    role VARCHAR(16) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO agent_chat_sessions (
                    user_id, session_id, title, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (7, "legacy-session", "老会话", "2026-04-02T09:30:00", "2026-04-02T09:31:00"),
            )
            conn.execute(
                """
                INSERT INTO agent_chat_messages (
                    user_id, session_id, role, content, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (7, "legacy-session", "assistant", "历史消息", "2026-04-02T09:31:00"),
            )
            conn.commit()
        finally:
            conn.close()

        Config.reset_instance()
        DatabaseManager.reset_instance()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        os.environ.pop("DATABASE_URL", None)
        self.temp_dir.cleanup()

    def test_legacy_chat_tables_are_backfilled_and_missing_columns_are_added(self):
        db = DatabaseManager.get_instance()
        repo = AgentChatRepository(db)

        conn = sqlite3.connect(self.db_path)
        try:
            session_columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_chat_sessions)").fetchall()}
            message_columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_chat_messages)").fetchall()}
        finally:
            conn.close()

        self.assertIn("owner_user_id", session_columns)
        self.assertIn("latest_message_preview", session_columns)
        self.assertIn("context_json", session_columns)
        self.assertIn("owner_user_id", message_columns)
        self.assertIn("meta_json", message_columns)

        sessions = repo.list_sessions(7, limit=10)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], "legacy-session")
        self.assertIsNone(sessions[0]["latest_message_preview"])

        detail = repo.get_session(7, "legacy-session")
        assert detail is not None
        self.assertEqual(detail["title"], "老会话")

        messages = repo.list_messages(7, "legacy-session")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "历史消息")
        self.assertIsNone(messages[0]["meta"])

        repo.add_message(
            owner_user_id=7,
            session_id="legacy-session",
            role="user",
            content="帮我看下 600519",
        )
        updated = repo.get_session(7, "legacy-session")
        assert updated is not None
        self.assertEqual(updated["latest_message_preview"], "帮我看下 600519")

        created = repo.ensure_session(
            owner_user_id=7,
            session_id="fresh-session",
            context={"stock_code": "600519"},
        )
        self.assertEqual(created["title"], "600519 问股")
        self.assertEqual(created["latest_message_preview"], "")


if __name__ == "__main__":
    unittest.main()
