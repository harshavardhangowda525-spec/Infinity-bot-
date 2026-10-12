import json
import sys
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from infinity.agent import Memory
from infinity.agent.models import ScriptedModel
from infinity.server.app import create_app
from infinity.server.runtime import Runtime

PY = sys.executable


def act(tool=None, args=None, final=None):
    return json.dumps({"thought": "t", "final": final} if final else {"thought": "t", "tool": tool, "args": args or {}})


@pytest.fixture()
def setup(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "app.py").write_text("print('hi')\n")
    (proj / ".env").write_text("SECRET=1\n")
    (proj / "package.json").write_text('{"dependencies": {"react": "18"}, "scripts": {"test": "x"}}')
    rt = Runtime(tmp_path, Memory(tmp_path / "m.db"))
    app = create_app(rt, token="tok")
    client = TestClient(app)
    return rt, client, proj


H = {"x-infinity-token": "tok"}


def wait_idle(rt, timeout=10):
    t0 = time.time()
    while rt.busy and time.time() - t0 < timeout:
        time.sleep(0.05)


def test_status_reports_unavailable_model_honestly(setup):
    rt, c, _ = setup
    s = c.get("/api/status").json()
    assert s["model"]["state"] == "unavailable" and s["model"]["reason"]
    assert s["agent"]["busy"] is False


def test_host_and_token_guards(setup):
    rt, c, proj = setup
    assert c.get("/api/status", headers={"host": "evil.example.com"}).status_code == 403
    assert c.get("/api/status", headers={"origin": "https://evil.example.com"}).status_code == 403
    assert c.post("/api/workspace", json={"path": str(proj)}).status_code == 401
    assert c.post("/api/workspace", json={"path": str(proj)}, headers={"x-infinity-token": "nope"}).status_code == 401
    assert c.post("/api/workspace", json={"path": str(proj)}, headers=H).status_code == 200


def test_workspace_tree_and_file_rules(setup):
    rt, c, proj = setup
    r = c.post("/api/workspace", json={"path": str(proj)}, headers=H).json()
    assert r["permission"] == "read-only"  # untrusted default
    assert "React" in r["project"]["frameworks"]
    tree = c.get("/api/tree").json()["children"]
    assert {n["name"] for n in tree} >= {"app.py", ".env"}
    assert next(n for n in tree if n["name"] == ".env")["sensitive"] is True
    assert c.get("/api/file", params={"path": "app.py"}).json()["content"] == "print('hi')\n"
    assert c.get("/api/file", params={"path": ".env"}).status_code == 403
    assert c.get("/api/file", params={"path": "../m.db"}).status_code == 403


def test_task_requires_model_and_workspace(setup):
    rt, c, proj = setup
    assert c.post("/api/tasks", json={"task": "x"}, headers=H).status_code == 409


def test_task_runs_with_approval_and_events(setup):
    rt, c, proj = setup
    c.post("/api/workspace", json={"path": str(proj), "permission": "ask"}, headers=H)
    rt.model = ScriptedModel([
        act("read_file", {"path": "app.py"}),
        act("edit_file", {"path": "app.py", "old": "hi", "new": "hello"}),
        act("run_command", {"command": f"{PY} app.py"}),
        act(final={"status": "completed", "summary": "changed greeting"}),
    ])
    rt.model_error = None
    r = c.post("/api/tasks", json={"task": "say hello", "mode": "code"}, headers=H)
    assert r.status_code == 200
    # approve every request the agent raises (edit needs approval in ask mode)
    approved = 0
    t0 = time.time()
    while rt.busy and time.time() - t0 < 10:
        for a in list(rt.approvals):
            assert c.post(f"/api/approvals/{a}", json={"approve": True}, headers=H).status_code == 200
            approved += 1
        time.sleep(0.05)
    wait_idle(rt)
    assert approved == 2  # the edit and the (MEDIUM) python command
    assert (proj / "app.py").read_text() == "print('hello')\n"
    task = c.get("/api/tasks/current").json()["task"]
    assert task["report"]["status"] == "completed_unverified"  # `python app.py` is not a check
    types = [e["type"] for e in rt.events_since(0)]
    for t in ("task_start", "approval_request", "approval_resolved", "tool_result", "task_end", "report"):
        assert t in types
    diff_events = [e for e in rt.events_since(0) if e["type"] == "tool_result" and e.get("diff")]
    assert "+print('hello')" in diff_events[0]["diff"]
    # rollback restores the original file
    session = task["report"]["session_id"]
    assert c.post("/api/rollback", json={"session": session}, headers=H).json()["restored"] == ["app.py"]
    assert (proj / "app.py").read_text() == "print('hi')\n"


def test_cancel_running_task(setup):
    rt, c, proj = setup
    c.post("/api/workspace", json={"path": str(proj), "permission": "auto"}, headers=H)
    rt.model = ScriptedModel(lambda m: act("list_dir", {}))
    rt.model_error = None
    c.post("/api/tasks", json={"task": "loop", "max_steps": 200}, headers=H)
    assert c.post("/api/tasks/cancel", headers=H).json()["cancelled"] is True
    wait_idle(rt)
    assert rt.task["report"]["status"] == "cancelled"


def test_events_websocket_requires_token(setup):
    rt, c, _ = setup
    rt.emit({"type": "hello"})
    with c.websocket_connect("/api/events?token=tok&since=0") as ws:
        msg = ws.receive_json()
        assert msg["events"][0]["type"] == "hello"
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/api/events?token=bad") as ws:
            ws.receive_json()


def test_model_switch_only_to_known_checkpoints(setup):
    rt, c, _ = setup
    assert c.post("/api/model", json={"checkpoint": "/etc"}, headers=H).status_code == 404
