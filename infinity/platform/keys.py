"""Agent API keys and usage metering.

A key is shown exactly once at creation; only its sha256 is stored. Keys can
be rotated (new secret, old one revoked atomically) and revoked. Each key has
a per-minute request limit enforced in memory, and usage is recorded per key.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
from collections import defaultdict, deque

from .audit import Audit
from .db import PlatformDB

KEY_PREFIX = "inf_"
NAME_RE = re.compile(r"^[A-Za-z0-9 _.\-]{1,48}$")
ALLOWED_SCOPES = {"generate"}


class KeyError_(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class RateLimiter:
    def __init__(self) -> None:
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, bucket: str, limit: int, window_s: float = 60.0) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[bucket]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True


class Keys:
    def __init__(self, db: PlatformDB, audit: Audit) -> None:
        self.db = db
        self.audit = audit
        self.limiter = RateLimiter()

    def create(self, user_id: int, name: str, scopes: list[str] | None = None, rate_limit_per_min: int = 30) -> dict:
        if not NAME_RE.match(name or ""):
            raise KeyError_("name: 1-48 letters, digits, spaces, . _ -")
        scopes = scopes or ["generate"]
        if set(scopes) - ALLOWED_SCOPES:
            raise KeyError_(f"unknown scopes: {sorted(set(scopes) - ALLOWED_SCOPES)}")
        if not 1 <= rate_limit_per_min <= 600:
            raise KeyError_("rate limit must be between 1 and 600 requests/minute")
        if self.db.one("SELECT COUNT(*) AS n FROM api_keys WHERE user_id = ? AND status = 'active'", (user_id,))["n"] >= 25:
            raise KeyError_("maximum of 25 active keys", 409)
        secret = KEY_PREFIX + secrets.token_urlsafe(32)
        prefix = secret[:12]
        cur = self.db.run(
            "INSERT INTO api_keys (user_id, name, prefix, key_hash, scopes, rate_limit_per_min, created_at) VALUES (?,?,?,?,?,?,?)",
            (user_id, name, prefix, _hash(secret), json.dumps(scopes), rate_limit_per_min, time.time()))
        kid = int(cur.lastrowid)
        self.audit.record(user_id, "apikey.create", f"key:{kid}", reason=name)
        return {**self.get(user_id, kid), "secret": secret}

    def get(self, user_id: int, key_id: int) -> dict:
        r = self.db.one("SELECT id, user_id, name, prefix, scopes, status, rate_limit_per_min, created_at, last_used_at, revoked_at, "
                        "request_count, error_count FROM api_keys WHERE id = ? AND user_id = ?", (key_id, user_id))
        if not r:
            raise KeyError_("no such key", 404)
        r["scopes"] = json.loads(r["scopes"])
        r.update(self._usage(key_id))
        return r

    def _usage(self, key_id: int) -> dict:
        u = self.db.one("SELECT COALESCE(SUM(tokens_in),0) AS tin, COALESCE(SUM(tokens_out),0) AS tout FROM usage_events WHERE api_key_id = ?", (key_id,))
        return {"tokens_in": u["tin"], "tokens_out": u["tout"]}

    def list(self, user_id: int | None = None) -> list[dict]:
        sql = ("SELECT id, user_id, name, prefix, scopes, status, rate_limit_per_min, created_at, last_used_at, revoked_at, "
               "request_count, error_count FROM api_keys")
        rows = self.db.all(sql + (" WHERE user_id = ?" if user_id is not None else "") + " ORDER BY id DESC",
                           (user_id,) if user_id is not None else ())
        for r in rows:
            r["scopes"] = json.loads(r["scopes"])
            r.update(self._usage(r["id"]))
        return rows

    def revoke(self, user_id: int, key_id: int, actor_id: int, reason: str = "") -> dict:
        k = self.db.one("SELECT * FROM api_keys WHERE id = ? AND user_id = ?", (key_id, user_id))
        if not k:
            raise KeyError_("no such key", 404)
        self.db.run("UPDATE api_keys SET status = 'revoked', revoked_at = ? WHERE id = ?", (time.time(), key_id))
        self.audit.record(actor_id, "apikey.revoke", f"key:{key_id}", reason=reason or k["name"])
        return self.get(user_id, key_id)

    def rotate(self, user_id: int, key_id: int) -> dict:
        k = self.db.one("SELECT * FROM api_keys WHERE id = ? AND user_id = ? AND status = 'active'", (key_id, user_id))
        if not k:
            raise KeyError_("no such active key", 404)
        new = self.create(user_id, k["name"], json.loads(k["scopes"]), k["rate_limit_per_min"])
        self.db.run("UPDATE api_keys SET status = 'revoked', revoked_at = ? WHERE id = ?", (time.time(), key_id))
        self.audit.record(user_id, "apikey.rotate", f"key:{key_id}", reason=f"replaced by key:{new['id']}")
        return new

    def authenticate(self, secret: str | None) -> dict | None:
        if not secret or not secret.startswith(KEY_PREFIX):
            return None
        k = self.db.one("SELECT * FROM api_keys WHERE key_hash = ?", (_hash(secret),))
        if not k or k["status"] != "active":
            return None
        u = self.db.one("SELECT status FROM users WHERE id = ?", (k["user_id"],))
        if not u or u["status"] != "active":
            return None
        k["scopes"] = json.loads(k["scopes"])
        return k

    def touch(self, key_id: int, ok: bool) -> None:
        self.db.run("UPDATE api_keys SET last_used_at = ?, request_count = request_count + 1, error_count = error_count + ? WHERE id = ?",
                    (time.time(), 0 if ok else 1, key_id))

    # ---------------------------------------------------------------- usage
    def record_usage(self, user_id: int, source: str, ref: str, tokens_in: int, tokens_out: int, exact: bool,
                     api_key_id: int | None = None, ok: bool = True) -> None:
        self.db.run(
            "INSERT INTO usage_events (user_id, api_key_id, source, ref, tokens_in, tokens_out, exact, ok, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, api_key_id, source, ref, tokens_in, tokens_out, int(exact), int(ok), time.time()))

    def usage_summary(self, user_id: int | None = None, days: int = 30) -> dict:
        since = time.time() - days * 86400
        cond, params = ("WHERE created_at >= ?" + (" AND user_id = ?" if user_id is not None else ""),
                        (since,) + ((user_id,) if user_id is not None else ()))
        totals = self.db.one(f"SELECT COALESCE(SUM(tokens_in),0) AS tin, COALESCE(SUM(tokens_out),0) AS tout, COUNT(*) AS n, "
                             f"COALESCE(SUM(1-exact),0) AS estimated, COALESCE(SUM(1-ok),0) AS failed FROM usage_events {cond}", params)
        daily = self.db.all(f"SELECT CAST(created_at/86400 AS INTEGER)*86400 AS day, SUM(tokens_in) AS tin, SUM(tokens_out) AS tout, "
                            f"COUNT(*) AS n FROM usage_events {cond} GROUP BY day ORDER BY day", params)
        by_source = self.db.all(f"SELECT source, SUM(tokens_in) AS tin, SUM(tokens_out) AS tout, COUNT(*) AS n FROM usage_events {cond} GROUP BY source", params)
        return {"days": days, "tokens_in": totals["tin"], "tokens_out": totals["tout"], "requests": totals["n"],
                "estimated_events": totals["estimated"], "failed_events": totals["failed"], "daily": daily, "by_source": by_source}
