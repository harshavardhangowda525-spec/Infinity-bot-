"""INFINITY STUDIO backend.

Security model:
- binds to loopback by default; Host/Origin headers must be local (anti DNS rebinding / CSRF);
- accounts with scrypt passwords; session cookie (HttpOnly, SameSite=Strict);
- every state-changing request carries the session's CSRF token (X-Infinity-Token);
- every protected endpoint re-checks the user, role and subscription on the server;
- COOK (agent tasks on this machine) needs an active paid plan with credits, and
  a workspace inside a root the owner authorised for that account (owner: any);
- agent API keys authenticate /api/v1/* only, are rate limited and metered;
- admin endpoints are owner-only and audited.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agent.checkpoints import CheckpointStore, list_sessions
from ..agent.permissions import BINARY_EXTS, IGNORED_DIRS, PathError, is_sensitive_path
from ..agent.project import detect_project
from ..platform import AuthError, BillingError, KeyError_, Platform, User
from .runtime import RuntimeHub, find_checkpoints, hyper_adapters, open_models

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "testserver"}
COOKIE = "infinity_session"
DEFAULT_MAX_STEPS = 25


# ------------------------------------------------------------------ bodies
class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=1024)


class WorkspaceIn(BaseModel):
    path: str
    permission: str | None = Field(default=None, pattern="^(read-only|ask|auto)$")


class ModelIn(BaseModel):
    checkpoint: str | None = None
    open_model: str | None = Field(default=None, max_length=100)
    hyper_adapter: str | None = Field(default=None, max_length=100)


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=12000)


class ChatIn(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=60)
    max_new_tokens: int = Field(default=512, ge=16, le=2048)


class TaskIn(BaseModel):
    task: str = Field(min_length=1, max_length=20000)
    mode: str = Field(default="code", pattern="^(ask|plan|review|code|debug|build|autonomous)$")
    max_steps: int = Field(default=DEFAULT_MAX_STEPS, ge=1, le=200)


class ApprovalIn(BaseModel):
    approve: bool


class RollbackIn(BaseModel):
    session: str = Field(pattern="^[a-f0-9]{6,32}$")


class CheckoutIn(BaseModel):
    plan: str = Field(pattern="^[a-z]{1,20}$")
    period: str = Field(default="monthly", pattern="^(monthly|yearly)$")


class KeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=48)
    rate_limit_per_min: int = Field(default=30, ge=1, le=600)


class GenerateIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    max_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)


class ReasonIn(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class StatusIn(ReasonIn):
    status: str = Field(pattern="^(active|suspended)$")


class CreditsIn(ReasonIn):
    delta: int = Field(ge=-10_000_000, le=10_000_000)


class GrantIn(ReasonIn):
    plan: str = Field(pattern="^[a-z]{1,20}$")


class RootsIn(ReasonIn):
    roots: list[str] = Field(max_length=20)


class PlanPatch(ReasonIn):
    changes: dict
    confirm_price_change: bool = False


def _local(value: str | None) -> bool:
    if not value:
        return False
    host = value.split("://", 1)[-1].split("/", 1)[0]
    if host.startswith("["):
        host = host.split("]")[0] + "]"
    else:
        host = host.rsplit(":", 1)[0]
    return host in ALLOWED_HOSTS


def create_app(hub: RuntimeHub, platform: Platform, static_dir: Path | None = None, secure_cookies: bool = False) -> FastAPI:
    app = FastAPI(title="INFINITY STUDIO", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.hub = hub
    app.state.platform = platform
    P = platform

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if not _local(request.headers.get("host")):
            return JSONResponse({"detail": "forbidden host"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and not _local(origin):
            return JSONResponse({"detail": "forbidden origin"}, status_code=403)
        request.state.request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.exception_handler(AuthError)
    @app.exception_handler(BillingError)
    @app.exception_handler(KeyError_)
    async def domain_error(_: Request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    # -------------------------------------------------------------- auth deps
    def current(request: Request) -> tuple[User, dict]:
        res = P.auth.session(request.cookies.get(COOKIE))
        if res is None:
            raise HTTPException(401, "sign in required")
        return res

    def user_dep(request: Request) -> User:
        return current(request)[0]

    def write_dep(request: Request) -> User:
        user, sess = current(request)
        if not secrets.compare_digest(request.headers.get("x-infinity-token", ""), sess["csrf"]):
            raise HTTPException(403, "missing or invalid CSRF token")
        return user

    def owner_read(user: User = Depends(user_dep)) -> User:
        if not user.is_owner:
            raise HTTPException(403, "owner only")
        return user

    def owner_write(user: User = Depends(write_dep)) -> User:
        if not user.is_owner:
            raise HTTPException(403, "owner only")
        return user

    def entitlement_view(user: User) -> dict:
        ent = P.billing.entitlement(user.id)
        return {
            "plan": {k: ent.plan.get(k) for k in ("id", "name", "monthly_price_inr", "monthly_token_credits", "max_task_steps")},
            "subscription": ent.subscription,
            "balance": ent.balance,
            "cook": ent.cook or user.is_owner,
            "cook_reason": None if (ent.cook or user.is_owner) else ent.reason,
            "api": ent.api,
            "owner_exempt": user.is_owner,
        }

    # ------------------------------------------------------------------ auth
    @app.get("/api/setup")
    def setup():
        return {"owner_exists": P.auth.owner_exists(), "registration": P.auth.allow_registration}

    @app.post("/api/auth/register")
    def register(body: Credentials, request: Request):
        if not _local(request.headers.get("origin") or request.headers.get("host")):
            raise HTTPException(403, "forbidden origin")
        u = P.auth.register(body.email, body.password)
        return {"user": u.public()}

    @app.post("/api/auth/login")
    def login(body: Credentials, request: Request, response: Response):
        ip = request.client.host if request.client else None
        token, info = P.auth.login(body.email, body.password, ip=ip, user_agent=request.headers.get("user-agent"))
        response.set_cookie(COOKIE, token, httponly=True, samesite="strict", secure=secure_cookies, path="/",
                            max_age=int(info["expires_at"] - time.time()))
        return {"ok": True}

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response, user: User = Depends(write_dep)):
        P.auth.logout(request.cookies.get(COOKIE, ""))
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(request: Request):
        user, sess = current(request)
        return {"user": user.public(), "csrf": sess["csrf"], "entitlement": entitlement_view(user)}

    # --------------------------------------------------------------- billing
    @app.get("/api/plans")
    def plans():
        return P.billing.catalog()

    @app.get("/api/billing/me")
    def billing_me(user: User = Depends(user_dep), limit: int = 50, offset: int = 0):
        return {
            "entitlement": entitlement_view(user),
            "ledger": P.billing.ledger(user.id, min(limit, 200), max(offset, 0)),
            "usage": P.keys.usage_summary(user.id, 30),
            "subscriptions": P.billing.subscriptions(user.id, limit=20),
        }

    @app.post("/api/billing/checkout")
    def checkout(body: CheckoutIn, user: User = Depends(write_dep)):
        return P.billing.start_checkout(user.id, body.plan, body.period)

    @app.post("/api/billing/cancel")
    def cancel_sub(user: User = Depends(write_dep)):
        return {"subscription": P.billing.cancel(user.id)}

    @app.post("/api/billing/webhook")
    async def webhook(request: Request):
        body = await request.body()
        return P.billing.handle_webhook(body, request.headers.get("x-infinity-signature"))

    # ------------------------------------------------------------- API keys
    @app.get("/api/keys")
    def list_keys(user: User = Depends(user_dep)):
        return {"keys": P.keys.list(user.id)}

    @app.post("/api/keys")
    def create_key(body: KeyIn, user: User = Depends(write_dep)):
        return {"key": P.keys.create(user.id, body.name, rate_limit_per_min=body.rate_limit_per_min),
                "notice": "Copy this key now. It will not be shown again."}

    @app.post("/api/keys/{key_id}/rotate")
    def rotate_key(key_id: int, user: User = Depends(write_dep)):
        return {"key": P.keys.rotate(user.id, key_id), "notice": "Copy this key now. It will not be shown again."}

    @app.post("/api/keys/{key_id}/revoke")
    def revoke_key(key_id: int, user: User = Depends(write_dep)):
        return {"key": P.keys.revoke(user.id, key_id, user.id)}

    @app.post("/api/v1/generate")
    def v1_generate(body: GenerateIn, request: Request):
        auth = request.headers.get("authorization", "")
        key = P.keys.authenticate(auth[7:] if auth.lower().startswith("bearer ") else None)
        if key is None:
            raise HTTPException(401, "invalid or revoked API key")
        if "generate" not in key["scopes"]:
            raise HTTPException(403, "key lacks the 'generate' scope")
        owner = P.auth.get(key["user_id"])
        ent = P.billing.entitlement(key["user_id"])
        if not (ent.api or owner.is_owner):
            P.keys.touch(key["id"], False)
            raise HTTPException(402, ent.reason or "API access requires an active plan with credits")
        rpm = ent.plan.get("requests_per_minute")
        limit = min(key["rate_limit_per_min"], rpm) if rpm else key["rate_limit_per_min"]
        if not P.keys.limiter.allow(f"key:{key['id']}", limit):
            P.keys.touch(key["id"], False)
            raise HTTPException(429, f"rate limit exceeded ({limit}/min)")
        model = hub.model
        if model is None or not hasattr(model, "engine"):
            P.keys.touch(key["id"], False)
            raise HTTPException(503, f"INFINITY HYPER 5.5 unavailable: {hub.model_error}")
        from ..core.inference import SamplingConfig

        eng = model.engine
        text = eng.generate(body.prompt, SamplingConfig(max_new_tokens=body.max_tokens, temperature=body.temperature), allow_special=False)
        tin = min(len(eng.tokenizer.encode(body.prompt)) + 1, eng.run_cfg.model.context_length)
        tout = len(eng.tokenizer.encode(text))
        ref = f"api:{request.state.request_id}"
        P.keys.record_usage(key["user_id"], "api", ref, tin, tout, True, api_key_id=key["id"])
        charged = 0
        if not owner.is_owner:
            P.billing.charge(key["user_id"], tin + tout, f"usage:{ref}", f"API generate via key {key['prefix']}", ref)
            charged = tin + tout
        P.keys.touch(key["id"], True)
        return {"model": model.info(), "text": text, "usage": {"tokens_in": tin, "tokens_out": tout, "exact": True,
                                                                "credits_charged": charged}}

    # ----------------------------------------------------------- runtime
    def rt_for(user: User):
        return hub.get(user.id)

    def need_workspace(user: User):
        rt = rt_for(user)
        if rt.workspace is None:
            raise HTTPException(409, "no workspace selected")
        return rt, rt.workspace

    @app.get("/api/status")
    def status(user: User = Depends(user_dep)):
        st = rt_for(user).status()
        st["model"] = hub.model_status()
        st["account"] = {"email": user.email, "role": user.role}
        st["entitlement"] = entitlement_view(user)
        return st

    @app.get("/api/models")
    def models(user: User = Depends(user_dep)):
        return {"active": hub.model_status(), "checkpoints": find_checkpoints(hub.base_dir),
                "hyper_adapters": hyper_adapters(hub.base_dir), "open_models": open_models(hub.base_dir),
                "can_switch": user.is_owner}

    @app.post("/api/model")
    def set_model(body: ModelIn, request: Request, user: User = Depends(owner_write)):
        if sum(bool(x) for x in (body.checkpoint, body.open_model, body.hyper_adapter)) != 1:
            raise HTTPException(422, "give exactly one of checkpoint, hyper_adapter or open_model")
        target = body.checkpoint or (f"open:{body.open_model}" if body.open_model else f"hyper-adapter:{body.hyper_adapter}")
        try:
            if body.hyper_adapter:
                if body.hyper_adapter not in {e["name"] for e in hyper_adapters(hub.base_dir)}:
                    raise HTTPException(404, "unknown adapter")
                hub.load_hyper_adapter(body.hyper_adapter, remember=True)
            elif body.open_model:
                if body.open_model not in {e["id"] for e in open_models(hub.base_dir)}:
                    raise HTTPException(404, "unknown open model")
                hub.load_open(body.open_model, remember=True)
            else:
                if str(Path(body.checkpoint)) not in {c["path"] for c in find_checkpoints(hub.base_dir)}:
                    raise HTTPException(404, "unknown checkpoint")
                hub.load_model(body.checkpoint, remember=True)
        except RuntimeError as e:  # includes ModelUnavailable (the previous model stays loaded)
            P.audit.record(user.id, "model.switch", target, success=False, reason=str(e)[:300],
                           request_id=request.state.request_id)
            raise HTTPException(409, str(e))
        P.audit.record(user.id, "model.switch", target, request_id=request.state.request_id)
        return hub.model_status()

    @app.post("/api/chat")
    def chat(body: ChatIn, request: Request, user: User = Depends(write_dep)):
        """Normal chat: a plain conversation with the loaded model (no files, no tools).

        Available to every signed-in account. Usage is always recorded. Credits are charged
        only on paid plans (free accounts are not charged), and the owner is exempt.
        """
        from ..agent.models import Message
        from ..hf.chat import system_prompt_for

        model = hub.model
        if model is None:
            raise HTTPException(503, f"no model loaded: {hub.model_error}")
        info = model.info()
        system = Message("system", system_prompt_for(info))
        turns = [Message(m.role, m.content) for m in body.messages]
        if turns[-1].role != "user":
            raise HTTPException(422, "the last message must be from the user")
        max_new = max(16, min(body.max_new_tokens, model.max_context_tokens // 2))
        budget = model.max_context_tokens - max_new - 32
        count = lambda ms: sum(model.count_tokens(m.content)[0] + 8 for m in ms)
        while len(turns) > 1 and count([system] + turns) > budget:
            turns = turns[2:] if len(turns) > 2 else turns[1:]  # drop the oldest exchange
        try:
            comp = model.complete([system] + turns, max_new_tokens=max_new)
        except Exception as e:
            raise HTTPException(500, f"generation failed: {type(e).__name__}: {e}")
        ref = f"chat:{request.state.request_id}"
        P.keys.record_usage(user.id, "chat", ref, comp.tokens_in, comp.tokens_out, comp.tokens_exact)
        ent = P.billing.entitlement(user.id)
        charged = 0
        if not user.is_owner and (ent.plan.get("monthly_price_inr") or 0) > 0:
            charged = comp.tokens_in + comp.tokens_out
            P.billing.charge(user.id, charged, f"usage:{ref}", "Normal chat", ref)
        return {"reply": comp.text, "model": {"product": info.get("product"), "family": info.get("family", "infinity")},
                "usage": {"tokens_in": comp.tokens_in, "tokens_out": comp.tokens_out, "exact": comp.tokens_exact,
                          "credits_charged": charged, "context_turns_used": len(turns)}}

    @app.get("/api/intelligence")
    def intelligence(user: User = Depends(user_dep)):
        import json as _json

        from ..bench.runner import BenchStore

        rt = rt_for(user)
        model = hub.model_status()
        active_file = hub.base_dir / "runs" / "ACTIVE.json"
        store_path = hub.base_dir / "runs" / "benchmarks.db"
        benchmarks = BenchStore(store_path).list(limit=100) if store_path.exists() else []
        index = None
        if rt.workspace is not None:
            try:
                from ..agent.index import CodeIndex

                idx = CodeIndex(rt.workspace.root)
                idx.update()
                index = idx.stats()
                idx.close()
            except Exception as e:  # report, never hide
                index = {"error": str(e)}
        mem = hub.memory
        ws = str(rt.workspace.root) if rt.workspace else None
        memory = {
            "experiences": mem._rows("SELECT COUNT(*) AS n FROM experiences")[0]["n"],
            "project_notes": len(mem.notes(ws)) if ws else 0,
            "relevant_to_last_task": len(mem.relevant_experiences(rt.task["task"], ws)) if rt.task else 0,
        }
        return {
            "product": "INFINITY HYPER 5.5",
            "model": model,
            "promotion": _json.loads(active_file.read_text()) if active_file.exists() else None,
            "benchmarks": benchmarks,
            "index": index,
            "memory": memory,
            "agent": rt.status()["agent"],
            "plan": (rt.task or {}).get("report", {}).get("plan") if rt.task and rt.task.get("report") else None,
        }

    @app.get("/api/projects")
    def projects(user: User = Depends(user_dep)):
        rt = rt_for(user)
        recent = rt.recent_projects()
        if not user.is_owner:
            recent = [r for r in recent if _within(r["path"], user.workspace_roots)]
        return {"recent": recent, "current": str(rt.workspace.root) if rt.workspace else None,
                "authorized_roots": None if user.is_owner else user.workspace_roots}

    @app.get("/api/project")
    def project(user: User = Depends(user_dep)):
        rt, ws = need_workspace(user)
        return {"path": str(ws.root), "permission": rt.permission.value, **detect_project(ws.root)}

    @app.get("/api/tree")
    def tree(path: str = ".", depth: int = 1, user: User = Depends(user_dep)):
        _, ws = need_workspace(user)
        try:
            root = ws.resolve(path)
        except PathError as e:
            raise HTTPException(403, str(e))
        if not root.is_dir():
            raise HTTPException(404, "not a directory")
        depth = max(1, min(depth, 4))

        def walk(d: Path, level: int) -> list[dict]:
            items = []
            try:
                entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                return items
            for e in entries[:500]:
                if e.name in IGNORED_DIRS:
                    continue
                rel = str(ws.rel(e))
                if e.is_dir():
                    node = {"name": e.name, "path": rel, "type": "dir"}
                    if level + 1 < depth:
                        node["children"] = walk(e, level + 1)
                    items.append(node)
                else:
                    items.append({"name": e.name, "path": rel, "type": "file", "sensitive": is_sensitive_path(ws.rel(e)), "size": e.stat().st_size})
            return items

        return {"path": str(ws.rel(root)), "children": walk(root, 0)}

    @app.get("/api/file")
    def file(path: str, user: User = Depends(user_dep)):
        _, ws = need_workspace(user)
        try:
            p = ws.resolve(path)
        except PathError as e:
            raise HTTPException(403, str(e))
        if not p.is_file():
            raise HTTPException(404, "not a file")
        if is_sensitive_path(ws.rel(p)):
            raise HTTPException(403, "secrets files are not shown")
        if p.suffix.lower() in BINARY_EXTS or p.stat().st_size > 1_000_000:
            raise HTTPException(415, "binary or too large to preview")
        raw = p.read_bytes()
        if b"\x00" in raw[:8192]:
            raise HTTPException(415, "binary file")
        return {"path": str(ws.rel(p)), "content": raw.decode("utf-8", errors="replace"), "size": len(raw)}

    @app.get("/api/tasks/current")
    def current_task(user: User = Depends(user_dep)):
        return {"task": rt_for(user).task}

    @app.get("/api/checkpoints")
    def checkpoints(user: User = Depends(user_dep)):
        _, ws = need_workspace(user)
        return {"sessions": list_sessions(ws.root)}

    def require_cook(user: User) -> dict:
        ent = entitlement_view(user)
        if not ent["cook"]:
            raise HTTPException(402, ent["cook_reason"] or "COOK requires an active paid plan")
        return ent

    @app.post("/api/workspace")
    def set_workspace(body: WorkspaceIn, request: Request, user: User = Depends(write_dep)):
        require_cook(user)
        path = str(Path(os.path.expanduser(body.path)).resolve())
        if not user.is_owner and not _within(path, user.workspace_roots):
            P.audit.record(user.id, "workspace.open", path, success=False, reason="outside authorised roots",
                           request_id=request.state.request_id)
            raise HTTPException(403, "this folder is not inside a workspace root the owner authorised for your account")
        try:
            return rt_for(user).set_workspace(path, body.permission)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/tasks")
    def start_task(body: TaskIn, request: Request, user: User = Depends(write_dep)):
        ent = require_cook(user)
        plan_steps = ent["plan"].get("max_task_steps")
        steps = min(body.max_steps, plan_steps) if plan_steps else body.max_steps

        def on_done(report: dict) -> dict:
            ref = f"task:{report['task_id']}"
            tin, tout = int(report["tokens_in"]), int(report["tokens_out"])
            ok = report["status"] not in ("blocked", "error")
            P.keys.record_usage(user.id, "cook", ref, tin, tout, bool(report["tokens_exact"]), ok=ok)
            if user.is_owner:
                return {"charged": 0, "owner_exempt": True, "tokens": tin + tout}
            P.billing.charge(user.id, tin + tout, f"usage:{ref}", f"COOK task ({report['status']})", ref)
            return {"charged": tin + tout, "balance": P.billing.balance(user.id)}

        try:
            return rt_for(user).start_task(body.task, body.mode, steps, on_done=on_done)
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/tasks/cancel")
    def cancel_task(user: User = Depends(write_dep)):
        return {"cancelled": rt_for(user).cancel_task()}

    @app.post("/api/approvals/{aid}")
    def approve(aid: str, body: ApprovalIn, user: User = Depends(write_dep)):
        if not rt_for(user).resolve_approval(aid, body.approve):
            raise HTTPException(404, "no such pending approval")
        return {"ok": True}

    @app.post("/api/rollback")
    def rollback(body: RollbackIn, user: User = Depends(write_dep)):
        rt, ws = need_workspace(user)
        if rt.busy:
            raise HTTPException(409, "cannot roll back while a task is running")
        store = CheckpointStore(ws.root, body.session)
        if not store.index:
            raise HTTPException(404, "no checkpoint for that session")
        restored = store.restore()
        rt.emit({"type": "rollback", "session": body.session, "restored": restored})
        return {"restored": restored}

    # ------------------------------------------------------------ admin
    @app.get("/api/admin/overview")
    def admin_overview(user: User = Depends(owner_read)):
        db = P.db
        now = time.time()
        month = now - 30 * 86400
        return {
            "users": {"total": db.one("SELECT COUNT(*) AS n FROM users")["n"],
                      "active_24h": db.one("SELECT COUNT(*) AS n FROM users WHERE last_active_at > ?", (now - 86400,))["n"],
                      "suspended": db.one("SELECT COUNT(*) AS n FROM users WHERE status='suspended'")["n"]},
            "subscriptions": {r["status"]: r["n"] for r in db.all("SELECT status, COUNT(*) AS n FROM subscriptions GROUP BY status")},
            "active_by_plan": {r["plan"]: r["n"] for r in db.all("SELECT plan, COUNT(*) AS n FROM subscriptions WHERE status IN ('active','canceled') GROUP BY plan")},
            "revenue_30d": P.billing.revenue(month),
            "usage_30d": P.keys.usage_summary(None, 30),
            "model": hub.model_status(),
            "tasks": hub.active_tasks(),
            "security": P.db.all("SELECT * FROM audit_log WHERE success = 0 ORDER BY id DESC LIMIT 10"),
            "audit_chain": P.audit.verify(),
            "payment_provider": P.billing.provider_status(),
            "ledger_duplicates": P.billing.duplicate_refs(),
            "database": {"migrations": P.db.migrations(), "path": str(P.db.path)},
        }

    @app.get("/api/admin/users")
    def admin_users(q: str = "", limit: int = 50, offset: int = 0, user: User = Depends(owner_read)):
        rows = P.db.all("SELECT id, email, role, status, workspace_roots, created_at, last_login_at, last_active_at FROM users "
                        "WHERE email LIKE ? ORDER BY id LIMIT ? OFFSET ?", (f"%{q}%", min(limit, 200), max(offset, 0)))
        for r in rows:
            ent = P.billing.entitlement(r["id"])
            r["plan"] = ent.subscription["plan"] if ent.subscription else "free"
            r["subscription_status"] = ent.subscription["status"] if ent.subscription else None
            r["balance"] = ent.balance
            u = P.keys.usage_summary(r["id"], 30)
            r["usage_30d"] = u["tokens_in"] + u["tokens_out"]
        return {"users": rows}

    @app.get("/api/admin/users/{uid}")
    def admin_user(uid: int, user: User = Depends(owner_read)):
        u = P.auth.get(uid)
        if not u:
            raise HTTPException(404, "no such user")
        return {"user": u.public(), "entitlement": entitlement_view(u), "ledger": P.billing.ledger(uid, 100),
                "subscriptions": P.billing.subscriptions(uid), "keys": P.keys.list(uid), "logins": P.auth.login_history(uid, 20),
                "usage": P.keys.usage_summary(uid, 30)}

    @app.post("/api/admin/users/{uid}/status")
    def admin_status(uid: int, body: StatusIn, user: User = Depends(owner_write)):
        P.auth.set_status(uid, body.status, user.id, body.reason)
        return {"user": P.auth.get(uid).public()}

    @app.post("/api/admin/users/{uid}/revoke-sessions")
    def admin_revoke(uid: int, body: ReasonIn, user: User = Depends(owner_write)):
        return {"revoked": P.auth.revoke_sessions(uid, user.id, body.reason)}

    @app.post("/api/admin/users/{uid}/credits")
    def admin_credits(uid: int, body: CreditsIn, request: Request, user: User = Depends(owner_write)):
        if not P.auth.get(uid):
            raise HTTPException(404, "no such user")
        P.billing.adjust(uid, body.delta, user.id, body.reason, request.state.request_id)
        return {"balance": P.billing.balance(uid)}

    @app.post("/api/admin/users/{uid}/grant")
    def admin_grant(uid: int, body: GrantIn, user: User = Depends(owner_write)):
        if not P.auth.get(uid):
            raise HTTPException(404, "no such user")
        return {"subscription": P.billing.owner_grant(uid, body.plan, user.id, body.reason)}

    @app.post("/api/admin/users/{uid}/workspace-roots")
    def admin_roots(uid: int, body: RootsIn, user: User = Depends(owner_write)):
        P.auth.set_workspace_roots(uid, body.roots, user.id, body.reason)
        return {"user": P.auth.get(uid).public()}

    @app.patch("/api/admin/plans/{plan_id}")
    def admin_plan(plan_id: str, body: PlanPatch, user: User = Depends(owner_write)):
        return {"plan": P.billing.update_plan(plan_id, body.changes, user.id, body.reason, body.confirm_price_change)}

    @app.get("/api/admin/payments")
    def admin_payments(limit: int = 100, offset: int = 0, user: User = Depends(owner_read)):
        return {"payments": P.billing.payments(min(limit, 500), max(offset, 0)), "revenue": P.billing.revenue()}

    @app.get("/api/admin/subscriptions")
    def admin_subs(status: str | None = None, limit: int = 100, offset: int = 0, user: User = Depends(owner_read)):
        return {"subscriptions": P.billing.subscriptions(None, status, min(limit, 500), max(offset, 0))}

    @app.get("/api/admin/keys")
    def admin_keys(user: User = Depends(owner_read)):
        return {"keys": P.keys.list(None)}

    @app.post("/api/admin/keys/{key_id}/revoke")
    def admin_revoke_key(key_id: int, body: ReasonIn, user: User = Depends(owner_write)):
        k = P.db.one("SELECT user_id FROM api_keys WHERE id = ?", (key_id,))
        if not k:
            raise HTTPException(404, "no such key")
        return {"key": P.keys.revoke(k["user_id"], key_id, user.id, body.reason)}

    @app.get("/api/admin/audit")
    def admin_audit(limit: int = 100, offset: int = 0, action: str | None = None, user: User = Depends(owner_read)):
        return {"events": P.audit.list(min(limit, 500), max(offset, 0), action), "chain": P.audit.verify()}

    # ------------------------------------------------------------ events
    @app.websocket("/api/events")
    async def events(ws: WebSocket):
        if not _local(ws.headers.get("host")) or (ws.headers.get("origin") and not _local(ws.headers.get("origin"))):
            await ws.close(code=4403)
            return
        res = P.auth.session(ws.cookies.get(COOKIE))
        if res is None:
            await ws.close(code=4401)
            return
        rt = hub.get(res[0].id)
        await ws.accept()
        try:
            since = int(ws.query_params.get("since", "0"))
        except ValueError:
            since = 0
        try:
            checks = 0
            while True:
                batch = rt.events_since(since)
                if batch:
                    since = batch[-1]["seq"]
                    await ws.send_json({"events": batch})
                await asyncio.sleep(0.1)
                checks += 1
                if checks % 50 == 0 and P.auth.session(ws.cookies.get(COOKIE)) is None:  # logged out / suspended
                    await ws.close(code=4401)
                    return
        except (WebSocketDisconnect, RuntimeError):
            return

    # ------------------------------------------------------------ static
    if static_dir and (static_dir / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(404, "not found")
            candidate = (static_dir / full_path).resolve()
            if full_path and candidate.is_file() and candidate.is_relative_to(static_dir.resolve()):
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")

    return app


def _within(path: str, roots: list[str]) -> bool:
    p = Path(path).resolve()
    return any(p == Path(r) or p.is_relative_to(Path(r)) for r in roots)


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    from ..agent import Memory

    ap = argparse.ArgumentParser(prog="infinity studio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--checkpoint", default=None, help="checkpoint/run dir (default: newest under runs/)")
    ap.add_argument("--base", default=".", help="repository root containing runs/")
    ap.add_argument("--db", default=None, help="agent memory database")
    ap.add_argument("--platform-db", default=None, help="accounts/billing database (default ~/.infinity/platform.db)")
    ap.add_argument("--no-registration", action="store_true", help="disable self-service sign-up")
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("refusing to bind a non-loopback address: INFINITY STUDIO controls files and a terminal on this machine")
        return 2
    base = Path(args.base).resolve()
    platform = Platform(args.platform_db, allow_registration=not args.no_registration)
    if not platform.auth.owner_exists():
        print("No owner account yet. Create one first:  infinity admin create-owner --email you@example.com")
    hub = RuntimeHub(base, Memory(args.db))
    if args.checkpoint:
        try:
            hub.load_model(args.checkpoint)
        except Exception as e:
            print(f"model not loaded: {e}")
    else:
        hub.autoload_latest()
    static = base / "studio" / "dist"
    app = create_app(hub, platform, static if static.exists() else None)
    print(f"INFINITY STUDIO on http://{args.host}:{args.port}  model: {hub.model_status().get('state')}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
