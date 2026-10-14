"""Platform database (accounts, subscriptions, payments, ledger, keys, audit).

Schema changes are applied as numbered migrations recorded in
``schema_migrations``; never edit an applied migration — add a new one.
"""

from __future__ import annotations

import os
import time
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          email TEXT NOT NULL UNIQUE COLLATE NOCASE,
          password_hash TEXT NOT NULL,
          role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'owner')),
          status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended')),
          workspace_roots TEXT NOT NULL DEFAULT '[]',
          created_at REAL NOT NULL,
          last_login_at REAL,
          last_active_at REAL
        );
        CREATE TABLE auth_sessions (
          id TEXT PRIMARY KEY,                 -- sha256 of the cookie token
          user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          csrf TEXT NOT NULL,
          created_at REAL NOT NULL,
          expires_at REAL NOT NULL,
          last_seen_at REAL NOT NULL,
          ip TEXT, user_agent TEXT,
          revoked INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE login_attempts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          email TEXT, ip TEXT, ok INTEGER NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE plans (
          id TEXT PRIMARY KEY,
          position INTEGER NOT NULL,
          data TEXT NOT NULL,                  -- JSON plan definition
          accepting_new INTEGER NOT NULL DEFAULT 1,
          updated_at REAL NOT NULL
        );
        CREATE TABLE subscriptions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL REFERENCES users(id),
          plan TEXT NOT NULL REFERENCES plans(id),
          status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'canceled', 'expired', 'payment_failed')),
          billing_period TEXT NOT NULL DEFAULT 'monthly',
          price_inr INTEGER NOT NULL,
          source TEXT NOT NULL,                -- 'payment' | 'owner_grant' | 'checkout_dev'
          current_period_start REAL,
          current_period_end REAL,
          cancel_at_period_end INTEGER NOT NULL DEFAULT 0,
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL
        );
        CREATE TABLE payments (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL REFERENCES users(id),
          subscription_id INTEGER REFERENCES subscriptions(id),
          provider TEXT NOT NULL,
          provider_event_id TEXT NOT NULL UNIQUE,
          kind TEXT NOT NULL CHECK (kind IN ('payment', 'refund')),
          status TEXT NOT NULL CHECK (status IN ('verified', 'failed')),
          amount_inr INTEGER NOT NULL,
          currency TEXT NOT NULL,
          plan TEXT,
          created_at REAL NOT NULL
        );
        CREATE TABLE ledger (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL REFERENCES users(id),
          delta INTEGER NOT NULL,
          kind TEXT NOT NULL CHECK (kind IN ('allocation', 'usage', 'refund', 'adjustment', 'expiry')),
          reason TEXT NOT NULL,
          ref TEXT,
          idempotency_key TEXT NOT NULL UNIQUE,
          actor_id INTEGER,
          created_at REAL NOT NULL
        );
        CREATE INDEX idx_ledger_user ON ledger(user_id, created_at);
        CREATE TABLE api_keys (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL REFERENCES users(id),
          name TEXT NOT NULL,
          prefix TEXT NOT NULL,
          key_hash TEXT NOT NULL UNIQUE,
          scopes TEXT NOT NULL DEFAULT '["generate"]',
          status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked')),
          rate_limit_per_min INTEGER NOT NULL DEFAULT 30,
          created_at REAL NOT NULL,
          last_used_at REAL,
          revoked_at REAL,
          request_count INTEGER NOT NULL DEFAULT 0,
          error_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE usage_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL REFERENCES users(id),
          api_key_id INTEGER REFERENCES api_keys(id),
          source TEXT NOT NULL CHECK (source IN ('cook', 'api')),
          ref TEXT NOT NULL,
          tokens_in INTEGER NOT NULL,
          tokens_out INTEGER NOT NULL,
          exact INTEGER NOT NULL,
          ok INTEGER NOT NULL DEFAULT 1,
          created_at REAL NOT NULL
        );
        CREATE INDEX idx_usage_user ON usage_events(user_id, created_at);
        CREATE TABLE audit_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          actor_id INTEGER,
          action TEXT NOT NULL,
          target TEXT,
          success INTEGER NOT NULL,
          reason TEXT,
          request_id TEXT,
          created_at REAL NOT NULL,
          prev_hash TEXT NOT NULL,
          hash TEXT NOT NULL
        );
        CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
        CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
        CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
        CREATE TRIGGER ledger_no_delete BEFORE DELETE ON ledger BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
        """,
    ),
    (
        2,
        # Normal chat usage ('chat'). SQLite cannot alter a CHECK constraint, so the table is
        # rebuilt with every row copied unchanged.
        """
        CREATE TABLE usage_events_v2 (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL REFERENCES users(id),
          api_key_id INTEGER REFERENCES api_keys(id),
          source TEXT NOT NULL CHECK (source IN ('cook', 'api', 'chat')),
          ref TEXT NOT NULL,
          tokens_in INTEGER NOT NULL,
          tokens_out INTEGER NOT NULL,
          exact INTEGER NOT NULL,
          ok INTEGER NOT NULL DEFAULT 1,
          created_at REAL NOT NULL
        );
        INSERT INTO usage_events_v2 (id, user_id, api_key_id, source, ref, tokens_in, tokens_out, exact, ok, created_at)
          SELECT id, user_id, api_key_id, source, ref, tokens_in, tokens_out, exact, ok, created_at FROM usage_events;
        DROP TABLE usage_events;
        ALTER TABLE usage_events_v2 RENAME TO usage_events;
        CREATE INDEX idx_usage_user ON usage_events(user_id, created_at);
        """,
    ),
    (
        3,
        # Normal-chat conversation history (per user; users can rename and delete their own).
        """
        CREATE TABLE conversations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL REFERENCES users(id),
          title TEXT NOT NULL,
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL
        );
        CREATE INDEX idx_conversations_user ON conversations(user_id, updated_at);
        CREATE TABLE conversation_messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
          position INTEGER NOT NULL,
          role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
          content TEXT NOT NULL,
          model TEXT,
          created_at REAL NOT NULL
        );
        CREATE INDEX idx_conversation_messages ON conversation_messages(conversation_id, position);
        """,
    ),
]


def default_platform_db() -> Path:
    env = os.environ.get("INFINITY_PLATFORM_DB")
    return Path(env).expanduser() if env else Path.home() / ".infinity" / "platform.db"


class PlatformDB:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_platform_db()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if str(self.path) != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.migrate()

    def migrate(self) -> list[int]:
        with self._lock:
            self.conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at REAL)")
            done = {r[0] for r in self.conn.execute("SELECT version FROM schema_migrations")}
            applied = []
            for version, sql in MIGRATIONS:
                if version in done:
                    continue
                with self.tx():
                    for stmt in _split_sql(sql):
                        self.conn.execute(stmt)
                    self.conn.execute("INSERT INTO schema_migrations VALUES (?, ?)", (version, time.time()))
                applied.append(version)
            return applied

    @contextmanager
    def tx(self):
        """IMMEDIATE transaction: takes the write lock up front (atomic check-then-write)."""
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def one(self, sql: str, params: tuple = ()) -> dict | None:
        with self._lock:
            r = self.conn.execute(sql, params).fetchone()
            return dict(r) if r else None

    def all(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def run(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def migrations(self) -> list[dict]:
        return self.all("SELECT version, applied_at FROM schema_migrations ORDER BY version")


def _split_sql(sql: str) -> list[str]:
    """Split a migration into statements, keeping trigger bodies intact."""
    out, buf, depth = [], [], 0
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buf.append(line)
        upper = stripped.upper()
        if upper.startswith("CREATE TRIGGER"):
            depth += upper.count(" BEGIN ") + (1 if upper.endswith(" BEGIN") else 0)
            if " END;" in upper or upper.endswith("END;"):
                depth = 0
                out.append("\n".join(buf))
                buf = []
            continue
        if depth == 0 and stripped.endswith(";"):
            out.append("\n".join(buf))
            buf = []
    if buf:
        out.append("\n".join(buf))
    return [s for s in out if s.strip().strip(";").strip()]
