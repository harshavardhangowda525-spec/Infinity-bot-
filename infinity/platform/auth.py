"""Accounts, password hashing, sessions, login rate limiting.

- Passwords: scrypt (n=2**14, r=8, p=1) with a per-user random salt.
- Sessions: random 256-bit token in an HttpOnly, SameSite=Strict cookie; only
  its sha256 is stored. Absolute lifetime + idle timeout. Each session has its
  own CSRF token, required on every state-changing request.
- Brute force: failed logins are limited per e-mail and per IP.
- There is no default account. The owner is created explicitly
  (``infinity admin create-owner``) with a password typed by the owner.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass

from .audit import Audit
from .db import PlatformDB

SESSION_LIFETIME_S = 12 * 3600
SESSION_IDLE_S = 2 * 3600
MAX_FAILS_PER_EMAIL = 5
MAX_FAILS_PER_IP = 20
FAIL_WINDOW_S = 15 * 60
MIN_PASSWORD_LEN = 10
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt$16384$8$1${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, digest = stored.split("$")
        if algo != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=len(digest) // 2)
        return hmac.compare_digest(dk.hex(), digest)
    except (ValueError, TypeError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass
class User:
    id: int
    email: str
    role: str
    status: str
    workspace_roots: list[str]
    created_at: float

    @property
    def is_owner(self) -> bool:
        return self.role == "owner"

    def public(self) -> dict:
        return {"id": self.id, "email": self.email, "role": self.role, "status": self.status,
                "workspace_roots": self.workspace_roots, "created_at": self.created_at}


def _user(row: dict) -> User:
    return User(row["id"], row["email"], row["role"], row["status"], json.loads(row["workspace_roots"] or "[]"), row["created_at"])


class Auth:
    def __init__(self, db: PlatformDB, audit: Audit, allow_registration: bool = True) -> None:
        self.db = db
        self.audit = audit
        self.allow_registration = allow_registration

    # ----------------------------------------------------------- accounts
    def _validate(self, email: str, password: str) -> str:
        email = email.strip().lower()
        if not EMAIL_RE.match(email) or len(email) > 254:
            raise AuthError("enter a valid e-mail address")
        if len(password) < MIN_PASSWORD_LEN:
            raise AuthError(f"password must be at least {MIN_PASSWORD_LEN} characters")
        if len(password) > 1024:
            raise AuthError("password too long")
        return email

    def create_user(self, email: str, password: str, role: str = "user", actor_id: int | None = None) -> User:
        email = self._validate(email, password)
        if role not in ("user", "owner"):
            raise AuthError("invalid role")
        if self.db.one("SELECT id FROM users WHERE email = ?", (email,)):
            raise AuthError("an account with this e-mail already exists", 409)
        cur = self.db.run("INSERT INTO users (email, password_hash, role, created_at) VALUES (?,?,?,?)",
                          (email, hash_password(password), role, time.time()))
        uid = int(cur.lastrowid)
        self.audit.record(actor_id if actor_id is not None else uid, f"account.create.{role}", f"user:{uid}")
        return self.get(uid)

    def register(self, email: str, password: str) -> User:
        if not self.allow_registration:
            raise AuthError("registration is disabled", 403)
        return self.create_user(email, password, "user")

    def owner_exists(self) -> bool:
        return self.db.one("SELECT 1 AS x FROM users WHERE role = 'owner'") is not None

    def get(self, user_id: int) -> User | None:
        row = self.db.one("SELECT * FROM users WHERE id = ?", (user_id,))
        return _user(row) if row else None

    def by_email(self, email: str) -> User | None:
        row = self.db.one("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
        return _user(row) if row else None

    def set_status(self, user_id: int, status: str, actor_id: int, reason: str) -> None:
        if status not in ("active", "suspended"):
            raise AuthError("invalid status")
        target = self.get(user_id)
        if target is None:
            raise AuthError("no such user", 404)
        if target.is_owner:
            raise AuthError("the owner account cannot be suspended", 403)
        self.db.run("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
        if status == "suspended":
            self.revoke_sessions(user_id, actor_id, "account suspended")
        self.audit.record(actor_id, f"account.{status}", f"user:{user_id}", reason=reason)

    def set_workspace_roots(self, user_id: int, roots: list[str], actor_id: int, reason: str) -> None:
        from pathlib import Path

        clean = []
        for r in roots:
            p = Path(r).expanduser().resolve()
            if not p.is_dir():
                raise AuthError(f"not a directory: {r}")
            clean.append(str(p))
        self.db.run("UPDATE users SET workspace_roots = ? WHERE id = ?", (json.dumps(clean), user_id))
        self.audit.record(actor_id, "account.workspace_roots", f"user:{user_id}", reason=f"{reason} :: {clean}")

    # ------------------------------------------------------------ sessions
    def _too_many_failures(self, email: str, ip: str | None) -> bool:
        since = time.time() - FAIL_WINDOW_S
        by_email = self.db.one("SELECT COUNT(*) AS n FROM login_attempts WHERE email = ? AND ok = 0 AND created_at > ?",
                               (email, since))["n"]
        by_ip = self.db.one("SELECT COUNT(*) AS n FROM login_attempts WHERE ip = ? AND ok = 0 AND created_at > ?",
                            (ip or "", since))["n"]
        return by_email >= MAX_FAILS_PER_EMAIL or by_ip >= MAX_FAILS_PER_IP

    def login(self, email: str, password: str, ip: str | None = None, user_agent: str | None = None) -> tuple[str, dict]:
        email = email.strip().lower()
        if self._too_many_failures(email, ip):
            self.audit.record(None, "auth.login.throttled", f"email:{email}", success=False, reason=f"ip={ip}")
            raise AuthError("too many failed attempts; try again in 15 minutes", 429)
        row = self.db.one("SELECT * FROM users WHERE email = ?", (email,))
        ok = bool(row) and verify_password(password, row["password_hash"])
        if not row:
            verify_password(password, hash_password("timing-equaliser"))  # similar timing for unknown users
        self.db.run("INSERT INTO login_attempts (email, ip, ok, created_at) VALUES (?,?,?,?)", (email, ip or "", int(ok), time.time()))
        if not ok:
            self.audit.record(row["id"] if row else None, "auth.login", f"email:{email}", success=False, reason="bad credentials")
            raise AuthError("incorrect e-mail or password", 401)
        if row["status"] != "active":
            self.audit.record(row["id"], "auth.login", f"user:{row['id']}", success=False, reason="account suspended")
            raise AuthError("this account is suspended", 403)
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        now = time.time()
        self.db.run(
            "INSERT INTO auth_sessions (id, user_id, csrf, created_at, expires_at, last_seen_at, ip, user_agent) VALUES (?,?,?,?,?,?,?,?)",
            (_token_hash(token), row["id"], csrf, now, now + SESSION_LIFETIME_S, now, ip, (user_agent or "")[:300]))
        self.db.run("UPDATE users SET last_login_at = ?, last_active_at = ? WHERE id = ?", (now, now, row["id"]))
        self.audit.record(row["id"], "auth.login", f"user:{row['id']}", reason=f"ip={ip}")
        return token, {"csrf": csrf, "expires_at": now + SESSION_LIFETIME_S}

    def session(self, token: str | None) -> tuple[User, dict] | None:
        if not token:
            return None
        s = self.db.one("SELECT * FROM auth_sessions WHERE id = ?", (_token_hash(token),))
        now = time.time()
        if not s or s["revoked"] or s["expires_at"] < now or now - s["last_seen_at"] > SESSION_IDLE_S:
            return None
        user = self.get(s["user_id"])
        if user is None or user.status != "active":
            return None
        if now - s["last_seen_at"] > 60:  # throttle writes
            self.db.run("UPDATE auth_sessions SET last_seen_at = ? WHERE id = ?", (now, s["id"]))
            self.db.run("UPDATE users SET last_active_at = ? WHERE id = ?", (now, user.id))
        return user, s

    def logout(self, token: str) -> None:
        s = self.db.one("SELECT user_id FROM auth_sessions WHERE id = ?", (_token_hash(token),))
        self.db.run("UPDATE auth_sessions SET revoked = 1 WHERE id = ?", (_token_hash(token),))
        if s:
            self.audit.record(s["user_id"], "auth.logout", f"user:{s['user_id']}")

    def revoke_sessions(self, user_id: int, actor_id: int, reason: str) -> int:
        n = self.db.run("UPDATE auth_sessions SET revoked = 1 WHERE user_id = ? AND revoked = 0", (user_id,)).rowcount
        self.audit.record(actor_id, "auth.sessions.revoke", f"user:{user_id}", reason=f"{reason} ({n} sessions)")
        return n

    def login_history(self, user_id: int | None = None, limit: int = 50) -> list[dict]:
        if user_id is None:
            return self.db.all("SELECT * FROM login_attempts ORDER BY id DESC LIMIT ?", (limit,))
        u = self.get(user_id)
        return self.db.all("SELECT * FROM login_attempts WHERE email = ? ORDER BY id DESC LIMIT ?", (u.email if u else "", limit))
