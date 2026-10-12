"""INFINITY STUDIO backend (local only).

Security model for a local developer tool:
- binds to 127.0.0.1 by default;
- every request must carry a Host header of localhost/127.0.0.1 (blocks DNS
  rebinding from a malicious website);
- every state-changing request and the event WebSocket require the session
  token, which only same-origin pages can obtain (no CORS headers are sent);
- file reads go through the same workspace / secrets rules as the agent.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agent.checkpoints import CheckpointStore, list_sessions
from ..agent.permissions import BINARY_EXTS, IGNORED_DIRS, PathError, is_sensitive_path
from ..agent.project import detect_project
from .runtime import Runtime, find_checkpoints

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


class WorkspaceIn(BaseModel):
    path: str
    permission: str | None = Field(default=None, pattern="^(read-only|ask|auto)$")


class ModelIn(BaseModel):
    checkpoint: str


class TaskIn(BaseModel):
    task: str = Field(min_length=1, max_length=20000)
    mode: str = Field(default="code", pattern="^(ask|plan|review|code|debug|build|autonomous)$")
    max_steps: int = Field(default=25, ge=1, le=200)


class ApprovalIn(BaseModel):
    approve: bool


class RollbackIn(BaseModel):
    session: str = Field(pattern="^[a-f0-9]{6,32}$")


def create_app(runtime: Runtime, static_dir: Path | None = None, token: str | None = None) -> FastAPI:
    app = FastAPI(title="INFINITY STUDIO", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.runtime = runtime
    app.state.token = token or secrets.token_urlsafe(32)

    @app.middleware("http")
    async def host_guard(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        if host not in ALLOWED_HOSTS and host != "testserver":
            return JSONResponse({"detail": "forbidden host"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin.split("://", 1)[-1].rsplit(":", 1)[0] not in ALLOWED_HOSTS:
            return JSONResponse({"detail": "forbidden origin"}, status_code=403)
        return await call_next(request)

    def require_token(request: Request) -> None:
        if not secrets.compare_digest(request.headers.get("x-infinity-token", ""), app.state.token):
            raise HTTPException(401, "missing or invalid session token")

    rt = runtime

    def need_workspace():
        if rt.workspace is None:
            raise HTTPException(409, "no workspace selected")
        return rt.workspace

    # ------------------------------------------------------------ read
    @app.get("/api/session")
    def session():
        return {"token": app.state.token}

    @app.get("/api/status")
    def status():
        return rt.status()

    @app.get("/api/models")
    def models():
        return {"active": rt.model_status(), "checkpoints": find_checkpoints(rt.base_dir)}

    @app.get("/api/projects")
    def projects():
        return {"recent": rt.recent_projects(), "current": str(rt.workspace.root) if rt.workspace else None}

    @app.get("/api/project")
    def project():
        ws = need_workspace()
        return {"path": str(ws.root), "permission": rt.permission.value, **detect_project(ws.root)}

    @app.get("/api/tree")
    def tree(path: str = ".", depth: int = 1):
        ws = need_workspace()
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
                    items.append({"name": e.name, "path": rel, "type": "file",
                                  "sensitive": is_sensitive_path(ws.rel(e)), "size": e.stat().st_size})
            return items

        return {"path": str(ws.rel(root)), "children": walk(root, 0)}

    @app.get("/api/file")
    def file(path: str):
        ws = need_workspace()
        try:
            p = ws.resolve(path)
        except PathError as e:
            raise HTTPException(403, str(e))
        if not p.is_file():
            raise HTTPException(404, "not a file")
        rel = ws.rel(p)
        if is_sensitive_path(rel):
            raise HTTPException(403, "secrets files are not shown")
        if p.suffix.lower() in BINARY_EXTS or p.stat().st_size > 1_000_000:
            raise HTTPException(415, "binary or too large to preview")
        raw = p.read_bytes()
        if b"\x00" in raw[:8192]:
            raise HTTPException(415, "binary file")
        return {"path": str(rel), "content": raw.decode("utf-8", errors="replace"), "size": len(raw)}

    @app.get("/api/tasks/current")
    def current_task():
        return {"task": rt.task}

    @app.get("/api/tasks/history")
    def history(limit: int = 20):
        return {"tasks": rt.memory.tasks(min(limit, 100))}

    @app.get("/api/checkpoints")
    def checkpoints():
        ws = need_workspace()
        return {"sessions": list_sessions(ws.root)}

    # ----------------------------------------------------------- write
    @app.post("/api/workspace", dependencies=[Depends(require_token)])
    def set_workspace(body: WorkspaceIn):
        try:
            return rt.set_workspace(os.path.expanduser(body.path), body.permission)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/model", dependencies=[Depends(require_token)])
    def set_model(body: ModelIn):
        allowed = {c["path"] for c in find_checkpoints(rt.base_dir)}
        if str(Path(body.checkpoint)) not in allowed:
            raise HTTPException(404, "unknown checkpoint")
        try:
            rt.load_model(body.checkpoint)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return rt.model_status()

    @app.post("/api/tasks", dependencies=[Depends(require_token)])
    def start_task(body: TaskIn):
        try:
            return rt.start_task(body.task, body.mode, body.max_steps)
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/tasks/cancel", dependencies=[Depends(require_token)])
    def cancel_task():
        return {"cancelled": rt.cancel_task()}

    @app.post("/api/approvals/{aid}", dependencies=[Depends(require_token)])
    def approve(aid: str, body: ApprovalIn):
        if not rt.resolve_approval(aid, body.approve):
            raise HTTPException(404, "no such pending approval")
        return {"ok": True}

    @app.post("/api/rollback", dependencies=[Depends(require_token)])
    def rollback(body: RollbackIn):
        ws = need_workspace()
        if rt.busy:
            raise HTTPException(409, "cannot roll back while a task is running")
        store = CheckpointStore(ws.root, body.session)
        if not store.index:
            raise HTTPException(404, "no checkpoint for that session")
        restored = store.restore()
        rt.emit({"type": "rollback", "session": body.session, "restored": restored})
        return {"restored": restored}

    # ------------------------------------------------------------ events
    @app.websocket("/api/events")
    async def events(ws: WebSocket):
        host = (ws.headers.get("host") or "").rsplit(":", 1)[0]
        origin = ws.headers.get("origin")
        ok_origin = origin is None or origin.split("://", 1)[-1].rsplit(":", 1)[0] in ALLOWED_HOSTS
        token_ok = secrets.compare_digest(ws.query_params.get("token", ""), app.state.token)
        if (host not in ALLOWED_HOSTS and host != "testserver") or not ok_origin or not token_ok:
            await ws.close(code=4401)
            return
        await ws.accept()
        try:
            since = int(ws.query_params.get("since", "0"))
        except ValueError:
            since = 0
        try:
            while True:
                batch = rt.events_since(since)
                if batch:
                    since = batch[-1]["seq"]
                    await ws.send_json({"events": batch})
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, RuntimeError):
            return

    # ------------------------------------------------------------ static
    if static_dir and (static_dir / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            candidate = (static_dir / full_path).resolve()
            if full_path and candidate.is_file() and candidate.is_relative_to(static_dir.resolve()):
                return FileResponse(candidate)
            return FileResponse(static_dir / "index.html")

    return app


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    from ..agent import Memory

    ap = argparse.ArgumentParser(prog="infinity-studio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--permission", default=None, choices=["read-only", "ask", "auto"])
    ap.add_argument("--checkpoint", default=None, help="checkpoint/run dir (default: newest under runs/)")
    ap.add_argument("--base", default=".", help="repository root containing runs/")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("refusing to bind a non-loopback address: INFINITY STUDIO controls your files and terminal")
        return 2
    base = Path(args.base).resolve()
    rt = Runtime(base, Memory(args.db))
    if args.checkpoint:
        try:
            rt.load_model(args.checkpoint)
        except Exception as e:
            print(f"model not loaded: {e}")
    else:
        rt.autoload_latest()
    if args.workspace:
        rt.set_workspace(args.workspace, args.permission)
    static = base / "studio" / "dist"
    app = create_app(rt, static if static.exists() else None)
    print(f"INFINITY STUDIO on http://{args.host}:{args.port}  model: {rt.model_status().get('state')}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
