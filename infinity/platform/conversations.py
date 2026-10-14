"""Normal-chat conversation history. Every query is scoped to the owning user."""

from __future__ import annotations

import time

from .db import PlatformDB

MAX_TITLE = 80


class ConversationError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def title_from(text: str) -> str:
    t = " ".join(text.split())
    return (t[: MAX_TITLE - 1] + "…") if len(t) > MAX_TITLE else (t or "New chat")


class Conversations:
    def __init__(self, db: PlatformDB) -> None:
        self.db = db

    def list(self, user_id: int, limit: int = 200) -> list[dict]:
        return self.db.all(
            "SELECT c.id, c.title, c.created_at, c.updated_at, "
            "(SELECT COUNT(*) FROM conversation_messages m WHERE m.conversation_id = c.id) AS messages "
            "FROM conversations c WHERE c.user_id = ? ORDER BY c.updated_at DESC, c.id DESC LIMIT ?", (user_id, limit))

    def _own(self, user_id: int, cid: int) -> dict:
        c = self.db.one("SELECT * FROM conversations WHERE id = ? AND user_id = ?", (cid, user_id))
        if c is None:
            raise ConversationError("conversation not found", 404)
        return c

    def create(self, user_id: int, title: str) -> dict:
        now = time.time()
        cur = self.db.run("INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?,?,?,?)",
                          (user_id, title_from(title), now, now))
        return self._own(user_id, int(cur.lastrowid))

    def get(self, user_id: int, cid: int) -> dict:
        c = self._own(user_id, cid)
        c["messages"] = self.db.all(
            "SELECT role, content, model, created_at FROM conversation_messages WHERE conversation_id = ? ORDER BY position",
            (cid,))
        return c

    def replace_messages(self, user_id: int, cid: int, messages: list[dict]) -> None:
        """Store the conversation as shown to the user (supports edit-and-resend and regenerate)."""
        self._own(user_id, cid)
        now = time.time()
        with self.db.tx():
            self.db.conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (cid,))
            for i, m in enumerate(messages):
                self.db.conn.execute(
                    "INSERT INTO conversation_messages (conversation_id, position, role, content, model, created_at) "
                    "VALUES (?,?,?,?,?,?)", (cid, i, m["role"], m["content"], m.get("model"), now))
            self.db.conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid))

    def rename(self, user_id: int, cid: int, title: str) -> dict:
        self._own(user_id, cid)
        if not title.strip():
            raise ConversationError("title must not be empty")
        self.db.run("UPDATE conversations SET title = ? WHERE id = ?", (title_from(title), cid))
        return self._own(user_id, cid)

    def delete(self, user_id: int, cid: int) -> None:
        self._own(user_id, cid)
        self.db.run("DELETE FROM conversations WHERE id = ?", (cid,))
