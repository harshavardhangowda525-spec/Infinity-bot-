"""INFINITY MEMORY — local SQLite store.

Layers:
- session: sessions, messages, tasks, tool_calls
- project: project_notes (architecture, commands, conventions, bugs, decisions)
- user:    preferences, trusted_workspaces
- experience: experiences (task -> plan -> outcome), retrieved before new tasks

This is ordinary database storage. It does not change model weights.
Anything that looks like a secret is refused for notes/preferences and
redacted from tool outputs before it is stored.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .redact import looks_secret, redact

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, workspace TEXT NOT NULL, mode TEXT, model TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  task TEXT NOT NULL, mode TEXT, status TEXT, summary TEXT, reason TEXT, plan TEXT,
  steps INTEGER DEFAULT 0, tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0,
  tokens_exact INTEGER DEFAULT 0, started_at REAL, finished_at REAL);
CREATE TABLE IF NOT EXISTS tool_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
  step INTEGER, tool TEXT, args TEXT, ok INTEGER, output TEXT, duration_ms INTEGER, created_at REAL);
CREATE TABLE IF NOT EXISTS project_notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, workspace TEXT NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL,
  value TEXT NOT NULL, created_at REAL, updated_at REAL, UNIQUE(workspace, kind, key));
CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL);
CREATE TABLE IF NOT EXISTS trusted_workspaces (path TEXT PRIMARY KEY, mode TEXT NOT NULL, approved_at REAL);
CREATE TABLE IF NOT EXISTS experiences (
  id INTEGER PRIMARY KEY AUTOINCREMENT, workspace TEXT, task TEXT, plan TEXT, outcome TEXT,
  errors TEXT, verification TEXT, feedback TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_task ON tool_calls(task_id);
"""
SCHEMA_VERSION = "1"
NOTE_KINDS = {"architecture", "command", "convention", "bug", "decision", "dependency", "note"}


class SecretRefused(ValueError):
    pass


def default_db_path() -> Path:
    env = os.environ.get("INFINITY_DB")
    return Path(env).expanduser() if env else Path.home() / ".infinity" / "infinity.db"


class Memory:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
        self.db.commit()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.db.execute(sql, params)
            self.db.commit()
            return cur

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.db.execute(sql, params).fetchall()]

    # ------------------------------------------------------------- sessions
    def new_session(self, workspace: str, mode: str, model: str, session_id: str | None = None) -> str:
        sid = session_id or uuid.uuid4().hex[:12]
        self._exec("INSERT INTO sessions VALUES (?,?,?,?,?)", (sid, workspace, mode, model, time.time()))
        return sid

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._exec("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                   (session_id, role, redact(content), time.time()))

    def messages(self, session_id: str) -> list[dict]:
        return self._rows("SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id", (session_id,))

    def start_task(self, session_id: str, task: str, mode: str) -> str:
        tid = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO tasks (id, session_id, task, mode, status, started_at) VALUES (?,?,?,?,?,?)",
                   (tid, session_id, redact(task), mode, "running", time.time()))
        return tid

    def finish_task(self, task_id: str, status: str, summary: str, reason: str, plan: list[str], steps: int,
                    tokens_in: int, tokens_out: int, tokens_exact: bool) -> None:
        self._exec(
            "UPDATE tasks SET status=?, summary=?, reason=?, plan=?, steps=?, tokens_in=?, tokens_out=?, tokens_exact=?, finished_at=? WHERE id=?",
            (status, redact(summary), redact(reason), json.dumps(plan), steps, tokens_in, tokens_out,
             int(tokens_exact), time.time(), task_id))

    def record_tool_call(self, task_id: str, step: int, tool: str, args: dict, ok: bool, output: str, duration_ms: int) -> None:
        self._exec(
            "INSERT INTO tool_calls (task_id, step, tool, args, ok, output, duration_ms, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (task_id, step, tool, redact(json.dumps(args)[:4000]), int(ok), redact(output[:4000]), duration_ms, time.time()))

    def tasks(self, limit: int = 50) -> list[dict]:
        return self._rows("SELECT * FROM tasks ORDER BY started_at DESC LIMIT ?", (limit,))

    def tool_calls(self, task_id: str) -> list[dict]:
        return self._rows("SELECT * FROM tool_calls WHERE task_id=? ORDER BY id", (task_id,))

    # -------------------------------------------------------- project memory
    def set_note(self, workspace: str, kind: str, key: str, value: str) -> None:
        if kind not in NOTE_KINDS:
            raise ValueError(f"kind must be one of {sorted(NOTE_KINDS)}")
        if looks_secret(key) or looks_secret(value):
            raise SecretRefused("refusing to store something that looks like a secret")
        now = time.time()
        self._exec(
            "INSERT INTO project_notes (workspace, kind, key, value, created_at, updated_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(workspace, kind, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (workspace, kind, key, value, now, now))

    def notes(self, workspace: str | None = None) -> list[dict]:
        if workspace is None:
            return self._rows("SELECT * FROM project_notes ORDER BY workspace, kind, key")
        return self._rows("SELECT * FROM project_notes WHERE workspace=? ORDER BY kind, key", (workspace,))

    def delete_note(self, note_id: int) -> bool:
        return self._exec("DELETE FROM project_notes WHERE id=?", (note_id,)).rowcount > 0

    # ------------------------------------------------------ user preferences
    def set_preference(self, key: str, value: str) -> None:
        if looks_secret(key) or looks_secret(value):
            raise SecretRefused("refusing to store something that looks like a secret")
        self._exec("INSERT OR REPLACE INTO preferences VALUES (?,?,?)", (key, value, time.time()))

    def preferences(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self._rows("SELECT key, value FROM preferences ORDER BY key")}

    def delete_preference(self, key: str) -> bool:
        return self._exec("DELETE FROM preferences WHERE key=?", (key,)).rowcount > 0

    def trust_workspace(self, path: str, mode: str) -> None:
        self._exec("INSERT OR REPLACE INTO trusted_workspaces VALUES (?,?,?)", (path, mode, time.time()))

    def workspace_trust(self, path: str) -> str | None:
        rows = self._rows("SELECT mode FROM trusted_workspaces WHERE path=?", (path,))
        return rows[0]["mode"] if rows else None

    def untrust_workspace(self, path: str) -> bool:
        return self._exec("DELETE FROM trusted_workspaces WHERE path=?", (path,)).rowcount > 0

    # ------------------------------------------------------------ experience
    def add_experience(self, workspace: str, task: str, plan: list[str], outcome: str, errors: list[str],
                       verification: list[dict], feedback: str = "") -> int:
        cur = self._exec(
            "INSERT INTO experiences (workspace, task, plan, outcome, errors, verification, feedback, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (workspace, redact(task), json.dumps(plan), outcome, redact(json.dumps(errors[-10:])),
             redact(json.dumps(verification[-10:])), redact(feedback), time.time()))
        return int(cur.lastrowid)

    def relevant_experiences(self, task: str, workspace: str | None = None, limit: int = 3) -> list[dict]:
        """Keyword-overlap retrieval (no embeddings yet — see docs)."""
        words = {w for w in re.findall(r"[a-zA-Z_]{3,}", task.lower())}
        if not words:
            return []
        rows = self._rows("SELECT * FROM experiences ORDER BY created_at DESC LIMIT 500")
        scored = []
        for r in rows:
            text = (r["task"] or "").lower()
            score = sum(1 for w in words if w in text) + (0.5 if workspace and r["workspace"] == workspace else 0)
            if score >= 1:
                scored.append((score, r))
        scored.sort(key=lambda s: (-s[0], -s[1]["created_at"]))
        return [r for _, r in scored[:limit]]

    def set_feedback(self, experience_id: int, feedback: str) -> bool:
        return self._exec("UPDATE experiences SET feedback=? WHERE id=?", (redact(feedback), experience_id)).rowcount > 0

    # ------------------------------------------------------- inspect/export
    TABLES = ("sessions", "messages", "tasks", "tool_calls", "project_notes", "preferences", "trusted_workspaces", "experiences")

    def export(self) -> dict:
        return {t: self._rows(f"SELECT * FROM {t}") for t in self.TABLES}

    def delete_session(self, session_id: str) -> bool:
        return self._exec("DELETE FROM sessions WHERE id=?", (session_id,)).rowcount > 0

    def wipe(self) -> None:
        with self._lock:
            for t in reversed(self.TABLES):
                self.db.execute(f"DELETE FROM {t}")
            self.db.commit()

    def close(self) -> None:
        self.db.close()
