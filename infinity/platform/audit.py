"""Append-only, hash-chained audit log.

Each record's hash covers its content and the previous record's hash, so any
edit or removal of history is detectable with ``verify()``. The database also
refuses UPDATE/DELETE on the table via triggers.
"""

from __future__ import annotations

import hashlib
import json
import time

from .db import PlatformDB

GENESIS = "0" * 64


def _digest(prev: str, rec: dict) -> str:
    blob = json.dumps({k: rec[k] for k in ("actor_id", "action", "target", "success", "reason", "request_id", "created_at")},
                      sort_keys=True, default=str)
    return hashlib.sha256((prev + blob).encode()).hexdigest()


class Audit:
    def __init__(self, db: PlatformDB) -> None:
        self.db = db

    def record(self, actor_id: int | None, action: str, target: str | None = None, success: bool = True,
               reason: str | None = None, request_id: str | None = None) -> None:
        with self.db.tx() as c:
            last = c.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
            prev = last[0] if last else GENESIS
            rec = {"actor_id": actor_id, "action": action, "target": target, "success": int(success),
                   "reason": reason, "request_id": request_id, "created_at": time.time()}
            h = _digest(prev, rec)
            c.execute(
                "INSERT INTO audit_log (actor_id, action, target, success, reason, request_id, created_at, prev_hash, hash) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (actor_id, action, target, int(success), reason, request_id, rec["created_at"], prev, h))

    def list(self, limit: int = 100, offset: int = 0, action: str | None = None) -> list[dict]:
        if action:
            return self.db.all("SELECT * FROM audit_log WHERE action LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?",
                               (f"{action}%", limit, offset))
        return self.db.all("SELECT * FROM audit_log ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))

    def verify(self) -> dict:
        prev = GENESIS
        n = 0
        for r in self.db.all("SELECT * FROM audit_log ORDER BY id"):
            if r["prev_hash"] != prev or _digest(prev, r) != r["hash"]:
                return {"ok": False, "broken_at": r["id"], "checked": n}
            prev = r["hash"]
            n += 1
        return {"ok": True, "checked": n}
