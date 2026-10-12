"""Server-side runtime state shared by the API: model, workspace, task, events, approvals."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from pathlib import Path

from ..agent import Agent, Limits, Memory, Mode, PermissionMode, PermissionPolicy, Workspace
from ..agent.models import ModelAdapter, ModelUnavailable, NeuralCoreModel
from ..agent.permissions import ApprovalRequest
from ..agent.project import detect_project

APPROVAL_TIMEOUT_S = 300.0


def find_checkpoints(base: Path) -> list[dict]:
    """All verified-looking checkpoints under <base>/runs/*/checkpoints."""
    out = []
    for manifest in sorted(base.glob("runs/*/checkpoints/step_*/manifest.json")):
        import json

        try:
            m = json.loads(manifest.read_text())
        except ValueError:
            continue
        out.append({"path": str(manifest.parent), "run": manifest.parent.parent.parent.name, "step": m.get("step"),
                    "parameters": m.get("parameters"), "metrics": m.get("metrics", {})})
    return out


class Runtime:
    def __init__(self, base_dir: Path, memory: Memory | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.memory = memory or Memory()
        self.lock = threading.RLock()
        self.model: ModelAdapter | None = None
        self.model_error: str | None = "no checkpoint loaded"
        self.model_loading = False
        self.workspace: Workspace | None = None
        self.permission = PermissionMode.READ_ONLY
        self.task: dict | None = None  # current/last task summary
        self.thread: threading.Thread | None = None
        self.cancel = threading.Event()
        self.events: deque = deque(maxlen=5000)
        self.seq = 0
        self.approvals: dict[str, dict] = {}
        self.errors: deque = deque(maxlen=50)
        self.started_at = time.time()

    # ------------------------------------------------------------ events
    def emit(self, event: dict) -> None:
        with self.lock:
            self.seq += 1
            event = {"seq": self.seq, "time": time.time(), **event}
            self.events.append(event)
            if event.get("type") == "task_end" and event.get("status") in ("blocked", "partial"):
                self.errors.append({"time": event["time"], "message": f"task {event.get('status')}: {event.get('reason')}"})

    def events_since(self, seq: int) -> list[dict]:
        with self.lock:
            return [e for e in self.events if e["seq"] > seq]

    # ------------------------------------------------------------- model
    def load_model(self, checkpoint: str) -> None:
        with self.lock:
            if self.busy:
                raise RuntimeError("cannot switch model while a task is running")
            self.model_loading = True
        try:
            model = NeuralCoreModel(checkpoint)
            with self.lock:
                self.model, self.model_error = model, None
            self.emit({"type": "model", "info": model.info()})
        except ModelUnavailable as e:
            with self.lock:
                self.model, self.model_error = None, str(e)
                self.errors.append({"time": time.time(), "message": str(e)})
            self.emit({"type": "model", "info": None, "error": str(e)})
            raise
        finally:
            self.model_loading = False

    def autoload_latest(self) -> None:
        ckpts = find_checkpoints(self.base_dir)
        if not ckpts:
            self.model_error = "no trained checkpoint found under runs/ — run `infinity train` first"
            return
        best = max(ckpts, key=lambda c: (c["run"], c["step"] or 0))
        try:
            self.load_model(best["path"])
        except ModelUnavailable:
            pass

    def model_status(self) -> dict:
        with self.lock:
            if self.model_loading:
                return {"state": "loading"}
            if self.model is None:
                return {"state": "unavailable", "reason": self.model_error}
            return {"state": "ready", "locality": "local", **self.model.info()}

    # --------------------------------------------------------- workspace
    def set_workspace(self, path: str, permission: str | None) -> dict:
        with self.lock:
            if self.busy:
                raise RuntimeError("cannot change workspace while a task is running")
            ws = Workspace(Path(path))
            trusted = self.memory.workspace_trust(str(ws.root))
            mode = PermissionMode(permission) if permission else PermissionMode(trusted or "read-only")
            if permission and permission != "read-only":
                self.memory.trust_workspace(str(ws.root), permission)
            self.workspace, self.permission = ws, mode
        info = detect_project(ws.root)
        self.emit({"type": "workspace", "path": str(ws.root), "permission": mode.value, "project": info})
        return {"path": str(ws.root), "permission": mode.value, "project": info}

    def recent_projects(self, limit: int = 12) -> list[dict]:
        rows = self.memory._rows(
            "SELECT workspace, MAX(created_at) AS last FROM sessions GROUP BY workspace ORDER BY last DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            p = Path(r["workspace"])
            last_task = self.memory._rows(
                "SELECT t.task, t.status FROM tasks t JOIN sessions s ON s.id=t.session_id WHERE s.workspace=? "
                "ORDER BY t.started_at DESC LIMIT 1", (r["workspace"],))
            out.append({"path": str(p), "name": p.name, "exists": p.is_dir(), "last_used": r["last"],
                        "trust": self.memory.workspace_trust(str(p)), "last_task": last_task[0] if last_task else None})
        return out

    # -------------------------------------------------------------- tasks
    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _approver(self, req: ApprovalRequest) -> bool:
        aid = uuid.uuid4().hex[:10]
        entry = {"id": aid, "event": threading.Event(), "approved": False}
        with self.lock:
            self.approvals[aid] = entry
        self.emit({"type": "approval_request", "id": aid, "tool": req.tool, "category": req.category.value,
                   "risk": req.risk.value if req.risk else None, "summary": req.summary, "reasons": req.reasons,
                   "diff": req.diff, "timeout_s": APPROVAL_TIMEOUT_S})
        deadline = time.monotonic() + APPROVAL_TIMEOUT_S
        while not entry["event"].wait(0.2):
            if self.cancel.is_set() or time.monotonic() > deadline:
                break
        with self.lock:
            self.approvals.pop(aid, None)
        decided = entry["event"].is_set()
        self.emit({"type": "approval_resolved", "id": aid, "approved": decided and entry["approved"],
                   "reason": "decided" if decided else ("cancelled" if self.cancel.is_set() else "timed out")})
        return decided and entry["approved"]

    def resolve_approval(self, aid: str, approved: bool) -> bool:
        with self.lock:
            entry = self.approvals.get(aid)
            if entry is None:
                return False
            entry["approved"] = bool(approved)
            entry["event"].set()
            return True

    def start_task(self, text: str, mode: str, max_steps: int = 25, on_done=None) -> dict:
        with self.lock:
            if self.busy:
                raise RuntimeError("a task is already running")
            if self.workspace is None:
                raise RuntimeError("select a project workspace first")
            if self.model is None:
                raise RuntimeError(f"INFINITY HYPER 5.5 is not available: {self.model_error}")
            self.cancel = threading.Event()
            policy = PermissionPolicy(self.permission, approver=self._approver)
            agent = Agent(self.model, self.workspace, policy, Mode(mode), memory=self.memory,
                          limits=Limits(max_steps=max_steps), on_event=self.emit, cancel=self.cancel)
            self.task = {"task": text, "mode": mode, "state": "running", "started_at": time.time(), "report": None}

            def run():
                try:
                    report = agent.run(text)
                    rd = report.to_dict()
                    if on_done is not None:
                        try:
                            rd["billing"] = on_done(rd)
                        except Exception as e:  # metering failure is reported, never hidden
                            rd["billing"] = {"error": f"{type(e).__name__}: {e}"}
                            self.errors.append({"time": time.time(), "message": f"usage metering failed: {e}"})
                    with self.lock:
                        self.task.update(state="finished", report=rd, finished_at=time.time())
                    self.emit({"type": "report", "report": rd})
                except Exception as e:  # report, never crash the server
                    with self.lock:
                        self.task.update(state="error", error=f"{type(e).__name__}: {e}")
                        self.errors.append({"time": time.time(), "message": f"agent crashed: {e}"})
                    self.emit({"type": "task_end", "status": "error", "reason": f"{type(e).__name__}: {e}", "summary": ""})

            self.thread = threading.Thread(target=run, daemon=True, name="infinity-agent")
            self.thread.start()
            return dict(self.task)

    def cancel_task(self) -> bool:
        if not self.busy:
            return False
        self.cancel.set()
        return True

    def status(self) -> dict:
        with self.lock:
            pending = [{"id": a} for a in self.approvals]
            task = None
            if self.task:
                task = {k: v for k, v in self.task.items() if k != "report"}
                if self.task.get("report"):
                    r = self.task["report"]
                    task["status"] = r["status"]
                    task["reason"] = r["reason"]
            return {
                "product": "INFINITY BOT",
                "model": self.model_status(),
                "runtime": {"connected": True, "uptime_s": round(time.time() - self.started_at)},
                "workspace": str(self.workspace.root) if self.workspace else None,
                "permission": self.permission.value,
                "agent": {"busy": self.busy, "awaiting_approval": bool(pending), "task": task},
                "approvals": pending,
                "recent_errors": list(self.errors)[-10:],
                "last_seq": self.seq,
            }


class RuntimeHub:
    """One agent runtime per user (own workspace, task, events, approvals); one shared model."""

    def __init__(self, base_dir: Path, memory: Memory | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.memory = memory or Memory()
        self.lock = threading.RLock()
        self.runtimes: dict[int, Runtime] = {}
        self.model: ModelAdapter | None = None
        self.model_error: str | None = "no checkpoint loaded"

    def get(self, user_id: int) -> Runtime:
        with self.lock:
            rt = self.runtimes.get(user_id)
            if rt is None:
                rt = Runtime(self.base_dir, self.memory)
                rt.model, rt.model_error = self.model, self.model_error
                self.runtimes[user_id] = rt
            return rt

    def set_model(self, model: ModelAdapter | None, error: str | None) -> None:
        with self.lock:
            self.model, self.model_error = model, error
            for rt in self.runtimes.values():
                rt.model, rt.model_error = model, error
                rt.emit({"type": "model", "info": model.info() if model else None, "error": error})

    def any_busy(self) -> bool:
        return any(rt.busy for rt in self.runtimes.values())

    def load_model(self, checkpoint: str) -> None:
        if self.any_busy():
            raise RuntimeError("cannot switch model while a task is running")
        try:
            model = NeuralCoreModel(checkpoint)
        except ModelUnavailable as e:
            self.set_model(None, str(e))
            raise
        self.set_model(model, None)

    def autoload_latest(self) -> None:
        """Load the explicitly promoted checkpoint (runs/ACTIVE.json) if any, else the newest one."""
        active = self.base_dir / "runs" / "ACTIVE.json"
        if active.exists():
            import json as _json

            try:
                self.load_model(_json.loads(active.read_text())["checkpoint"])
            except (ModelUnavailable, KeyError, ValueError) as e:
                self.set_model(None, f"promoted checkpoint failed to load: {e}")
            return
        ckpts = find_checkpoints(self.base_dir)
        if not ckpts:
            self.set_model(None, "no trained checkpoint found under runs/ — run `infinity train` first")
            return
        best = max(ckpts, key=lambda c: (c["run"], c["step"] or 0))
        try:
            self.load_model(best["path"])
        except ModelUnavailable:
            pass

    def model_status(self) -> dict:
        if self.model is None:
            return {"state": "unavailable", "reason": self.model_error}
        return {"state": "ready", "locality": "local", **self.model.info()}

    def active_tasks(self) -> list[dict]:
        out = []
        for uid, rt in self.runtimes.items():
            if rt.task:
                t = {k: v for k, v in rt.task.items() if k != "report"}
                t["user_id"] = uid
                t["busy"] = rt.busy
                if rt.task.get("report"):
                    t["status"] = rt.task["report"]["status"]
                out.append(t)
        return out
