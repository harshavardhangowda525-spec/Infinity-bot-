import hashlib
import hmac
import json
import sys
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from infinity.agent import Memory
from infinity.agent.models import ScriptedModel
from infinity.platform import Platform
from infinity.server.app import create_app
from infinity.server.runtime import RuntimeHub

PY = sys.executable
PW = "correct horse battery"


def act(tool=None, args=None, final=None):
    return json.dumps({"thought": "t", "final": final} if final else {"thought": "t", "tool": tool, "args": args or {}})


class Env:
    def __init__(self, tmp_path):
        self.proj = tmp_path / "proj"
        self.proj.mkdir()
        (self.proj / "app.py").write_text("print('hi')\n")
        (self.proj / ".env").write_text("SECRET=1\n")
        (self.proj / "package.json").write_text('{"dependencies": {"react": "18"}}')
        self.other = tmp_path / "other"
        self.other.mkdir()
        self.platform = Platform(tmp_path / "platform.db")
        self.hub = RuntimeHub(tmp_path, Memory(tmp_path / "m.db"))
        self.app = create_app(self.hub, self.platform)
        self.owner = self.platform.auth.create_user("owner@example.com", PW, "owner")

    def client(self, email, password=PW):
        c = TestClient(self.app)
        r = c.post("/api/auth/login", json={"email": email, "password": password})
        assert r.status_code == 200, r.text
        c.headers["x-infinity-token"] = c.get("/api/auth/me").json()["csrf"]
        return c

    def user(self, email="user@example.com"):
        u = self.platform.auth.create_user(email, PW)
        return u, self.client(email)

    def scripted(self, script):
        self.hub.set_model(ScriptedModel(script), None)

    def paid(self, u, plan="plus", credits=10_000):
        self.platform.billing.update_plan(plan, {"monthly_token_credits": credits}, self.owner.id, "test config")
        self.platform.billing.owner_grant(u.id, plan, self.owner.id, "test")
        self.platform.auth.set_workspace_roots(u.id, [str(self.proj)], self.owner.id, "test")


@pytest.fixture()
def env(tmp_path):
    return Env(tmp_path)


def wait_idle(rt, timeout=10):
    t0 = time.time()
    while rt.busy and time.time() - t0 < timeout:
        time.sleep(0.05)


def test_unauthenticated_access_is_refused(env):
    c = TestClient(env.app)
    assert c.get("/api/setup").json()["owner_exists"] is True
    assert c.get("/api/plans").status_code == 200  # public catalogue
    for path in ("/api/status", "/api/models", "/api/billing/me", "/api/keys", "/api/admin/overview"):
        assert c.get(path).status_code == 401, path
    assert c.post("/api/tasks", json={"task": "x"}).status_code == 401


def test_host_origin_and_csrf_guards(env):
    c = env.client("owner@example.com")
    assert c.get("/api/status", headers={"host": "evil.example.com"}).status_code == 403
    assert c.get("/api/status", headers={"origin": "https://evil.example.com"}).status_code == 403
    no_csrf = c.post("/api/workspace", json={"path": str(env.proj)}, headers={"x-infinity-token": "wrong"})
    assert no_csrf.status_code == 403


def test_login_errors_and_logout(env):
    c = TestClient(env.app)
    assert c.post("/api/auth/login", json={"email": "owner@example.com", "password": "nope nope nope"}).status_code == 401
    oc = env.client("owner@example.com")
    assert oc.post("/api/auth/logout").status_code == 200
    assert oc.get("/api/status").status_code == 401


def test_free_user_cannot_use_cook_even_via_direct_api(env):
    u, c = env.user()
    me = c.get("/api/auth/me").json()
    assert me["entitlement"]["cook"] is False and "subscription" in me["entitlement"]["cook_reason"]
    assert c.post("/api/workspace", json={"path": str(env.proj)}).status_code == 402
    assert c.post("/api/tasks", json={"task": "x"}).status_code == 402
    # dev checkout does not unlock anything
    r = c.post("/api/billing/checkout", json={"plan": "pro"}).json()
    assert r["mode"] == "development" and r["activated"] is False
    assert c.post("/api/tasks", json={"task": "x"}).status_code == 402
    # cannot reach admin APIs
    assert c.get("/api/admin/overview").status_code == 403
    assert c.post(f"/api/admin/users/{u.id}/credits", json={"delta": 999, "reason": "free money"}).status_code == 403
    assert c.post(f"/api/admin/users/{u.id}/grant", json={"plan": "max", "reason": "self grant"}).status_code == 403


def test_paid_plan_with_unconfigured_credits_is_refused_honestly(env):
    u, c = env.user()
    env.platform.billing.owner_grant(u.id, "pro", env.owner.id, "t")
    r = c.post("/api/tasks", json={"task": "x"})
    assert r.status_code == 402 and "not been configured" in r.json()["detail"]


@pytest.mark.parametrize("plan", ["plus", "pro", "max"])
def test_each_paid_plan_unlocks_cook(env, plan):
    u, c = env.user(f"{plan}@example.com")
    env.paid(u, plan)
    assert c.get("/api/auth/me").json()["entitlement"]["cook"] is True
    assert c.post("/api/workspace", json={"path": str(env.proj), "permission": "auto"}).status_code == 200


def test_workspace_must_be_inside_authorised_roots(env):
    u, c = env.user()
    env.paid(u)
    r = c.post("/api/workspace", json={"path": str(env.other)})
    assert r.status_code == 403
    assert env.platform.audit.list(action="workspace.open")[0]["success"] == 0


def test_task_runs_meters_usage_and_charges_credits(env):
    u, c = env.user()
    env.paid(u, "plus", 10_000)
    c.post("/api/workspace", json={"path": str(env.proj), "permission": "ask"})
    env.scripted([
        act("read_file", {"path": "app.py"}),
        act("edit_file", {"path": "app.py", "old": "hi", "new": "hello"}),
        act(final={"status": "completed", "summary": "changed greeting"}),
    ])
    assert c.post("/api/tasks", json={"task": "say hello", "mode": "code"}).status_code == 200
    rt = env.hub.get(u.id)
    t0 = time.time()
    while rt.busy and time.time() - t0 < 10:
        for a in list(rt.approvals):
            assert c.post(f"/api/approvals/{a}", json={"approve": True}).status_code == 200
        time.sleep(0.05)
    wait_idle(rt)
    report = c.get("/api/tasks/current").json()["task"]["report"]
    assert report["status"] == "completed_unverified"
    used = report["tokens_in"] + report["tokens_out"]
    assert report["billing"]["charged"] == used
    me = c.get("/api/billing/me").json()
    assert me["entitlement"]["balance"] == 10_000 - used
    assert me["usage"]["requests"] == 1 and me["usage"]["estimated_events"] == 1  # scripted model: estimated tokens
    assert env.platform.billing.duplicate_refs() == []
    session = report["session_id"]
    assert c.post("/api/rollback", json={"session": session}).json()["restored"] == ["app.py"]


def test_credit_exhaustion_blocks_new_tasks(env):
    u, c = env.user()
    env.paid(u, "plus", 5)
    c.post("/api/workspace", json={"path": str(env.proj), "permission": "auto"})
    env.scripted([act(final={"status": "completed", "summary": "ok"})])
    c.post("/api/tasks", json={"task": "a long task description that costs tokens"})
    wait_idle(env.hub.get(u.id))
    assert c.get("/api/billing/me").json()["entitlement"]["balance"] <= 0
    r = c.post("/api/tasks", json={"task": "again"})
    assert r.status_code == 402 and "no credits" in r.json()["detail"]


def test_users_are_isolated(env):
    a, ca = env.user("a@example.com")
    b, cb = env.user("b@example.com")
    env.paid(a)
    ca.post("/api/workspace", json={"path": str(env.proj), "permission": "auto"})
    env.hub.get(a.id).emit({"type": "private", "secret": "a-only"})
    assert cb.get("/api/status").json()["workspace"] is None
    assert cb.get("/api/tree").status_code == 409
    assert all(e["type"] != "private" for e in env.hub.get(b.id).events_since(0))
    ka = ca.post("/api/keys", json={"name": "JARVIS"}).json()["key"]
    assert cb.post(f"/api/keys/{ka['id']}/revoke").status_code == 404


def test_owner_admin_flow_and_audit(env):
    u, _ = env.user()
    oc = env.client("owner@example.com")
    ov = oc.get("/api/admin/overview").json()
    assert ov["users"]["total"] == 2 and ov["revenue_30d"]["net_inr"] == 0 and ov["audit_chain"]["ok"]
    assert oc.patch("/api/admin/plans/pro", json={"changes": {"monthly_price_inr": 1500}, "reason": "raise"}).status_code == 409
    assert oc.patch("/api/admin/plans/pro", json={"changes": {"monthly_token_credits": 20000}, "reason": "configure"}).status_code == 200
    assert oc.post(f"/api/admin/users/{u.id}/grant", json={"plan": "pro", "reason": "beta"}).status_code == 200
    assert oc.post(f"/api/admin/users/{u.id}/credits", json={"delta": 500, "reason": "goodwill"}).json()["balance"] == 20500
    users = oc.get("/api/admin/users").json()["users"]
    assert next(x for x in users if x["id"] == u.id)["plan"] == "pro"
    assert oc.post(f"/api/admin/users/{u.id}/status", json={"status": "suspended", "reason": "abuse test"}).status_code == 200
    actions = [e["action"] for e in oc.get("/api/admin/audit").json()["events"]]
    for a in ("account.suspended", "credits.adjust", "billing.grant", "plan.update"):
        assert a in actions


def test_api_key_generate_requires_plan_and_meters(env, tmp_path):
    u, c = env.user()
    key = c.post("/api/keys", json={"name": "JARVIS", "rate_limit_per_min": 2}).json()["key"]
    h = {"authorization": f"Bearer {key['secret']}"}
    assert TestClient(env.app).post("/api/v1/generate", json={"prompt": "def"}, headers=h).status_code == 402  # free plan
    env.paid(u, "plus", 100)
    assert TestClient(env.app).post("/api/v1/generate", json={"prompt": "def"}, headers=h).status_code == 503  # no model
    assert TestClient(env.app).post("/api/v1/generate", json={"prompt": "def"}, headers={"authorization": "Bearer inf_bogus"}).status_code == 401
    assert TestClient(env.app).post("/api/v1/generate", json={"prompt": "def"}, headers=h).status_code == 503  # 2nd counted call
    assert TestClient(env.app).post("/api/v1/generate", json={"prompt": "def"}, headers=h).status_code == 429  # limit 2/min


def test_api_generate_with_real_neural_core_meters_exact_tokens(env, tmp_path):
    torch = pytest.importorskip("torch")
    from infinity.agent.models import NeuralCoreModel
    from infinity.core.checkpoint import save_checkpoint
    from infinity.core.config import ModelConfig, RunConfig
    from infinity.core.model import InfinityNeuralCore
    from infinity.core.tokenizer import BPETokenizer

    tok = BPETokenizer.train(["def add(a, b):\n    return a + b\n"] * 20, vocab_size=300)
    cfg = RunConfig(name="t", model=ModelConfig(vocab_size=tok.vocab_size, context_length=64, d_model=32, n_layers=1, n_heads=4, d_ff=64))
    torch.manual_seed(0)
    path = save_checkpoint(tmp_path / "ck", 1, InfinityNeuralCore(cfg.model), None, cfg, tok, {})
    env.hub.set_model(NeuralCoreModel(str(path)), None)
    u, c = env.user()
    env.paid(u, "plus", 1000)
    key = c.post("/api/keys", json={"name": "ASTON"}).json()["key"]
    r = TestClient(env.app).post("/api/v1/generate", json={"prompt": "def add", "max_tokens": 8},
                                 headers={"authorization": f"Bearer {key['secret']}"})
    assert r.status_code == 200, r.text
    usage = r.json()["usage"]
    assert usage["exact"] is True and usage["credits_charged"] == usage["tokens_in"] + usage["tokens_out"]
    assert c.get("/api/billing/me").json()["entitlement"]["balance"] == 1000 - usage["credits_charged"]
    k = c.get("/api/keys").json()["keys"][0]
    assert k["request_count"] == 1 and k["tokens_in"] == usage["tokens_in"]


def test_webhook_endpoint_activates_only_with_valid_signature(env, monkeypatch):
    u, c = env.user()
    body = json.dumps({"id": "evt_9", "type": "payment.succeeded", "user_id": u.id, "plan": "plus", "amount_inr": 500, "currency": "INR"}).encode()
    anon = TestClient(env.app)
    assert anon.post("/api/billing/webhook", content=body).status_code == 503
    monkeypatch.setenv("INFINITY_BILLING_WEBHOOK_SECRET", "whsec_x")
    assert anon.post("/api/billing/webhook", content=body, headers={"x-infinity-signature": "00"}).status_code == 401
    sig = hmac.new(b"whsec_x", body, hashlib.sha256).hexdigest()
    assert anon.post("/api/billing/webhook", content=body, headers={"x-infinity-signature": sig}).json()["activated"]
    assert c.get("/api/auth/me").json()["entitlement"]["subscription"]["plan"] == "plus"


def test_events_websocket_requires_session(env):
    u, c = env.user()
    env.hub.get(u.id).emit({"type": "hello"})
    with c.websocket_connect("/api/events?since=0") as ws:
        assert ws.receive_json()["events"][0]["type"] == "hello"
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with TestClient(env.app).websocket_connect("/api/events") as ws:
            ws.receive_json()


def test_only_owner_switches_model(env):
    u, c = env.user()
    assert c.post("/api/model", json={"checkpoint": "/x"}).status_code == 403
    oc = env.client("owner@example.com")
    assert oc.post("/api/model", json={"checkpoint": "/etc"}).status_code == 404


def test_cancel_running_task(env):
    oc = env.client("owner@example.com")  # owner is exempt from plan gating
    oc.post("/api/workspace", json={"path": str(env.proj), "permission": "auto"})
    env.scripted(lambda m: act("list_dir", {}))
    oc.post("/api/tasks", json={"task": "loop", "max_steps": 200})
    assert oc.post("/api/tasks/cancel").json()["cancelled"] is True
    rt = env.hub.get(env.owner.id)
    wait_idle(rt)
    assert rt.task["report"]["status"] == "cancelled"
    assert rt.task["report"]["billing"]["owner_exempt"] is True
