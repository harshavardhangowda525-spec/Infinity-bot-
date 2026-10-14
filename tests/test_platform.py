import hashlib
import hmac
import json
import threading
import time

import pytest

from infinity.platform import AuthError, BillingError, KeyError_, Platform
from infinity.platform.auth import hash_password, verify_password

PW = "correct horse battery"


@pytest.fixture()
def p(tmp_path):
    return Platform(tmp_path / "platform.db")


def mkuser(p, email="user@example.com", role="user"):
    return p.auth.create_user(email, PW, role)


def configure_credits(p, plan="plus", credits=1000):
    owner = p.auth.by_email("owner@example.com") or mkuser(p, "owner@example.com", "owner")
    p.billing.update_plan(plan, {"monthly_token_credits": credits}, owner.id, "configure for test")
    return owner


# ------------------------------------------------------------------ auth
def test_password_hashing():
    h = hash_password(PW)
    assert h.startswith("scrypt$") and PW not in h
    assert verify_password(PW, h) and not verify_password("wrong password!", h)
    assert hash_password(PW) != h  # salted


def test_no_default_accounts_and_validation(p):
    assert not p.auth.owner_exists()
    assert p.db.one("SELECT COUNT(*) AS n FROM users")["n"] == 0
    with pytest.raises(AuthError):
        p.auth.register("not-an-email", PW)
    with pytest.raises(AuthError):
        p.auth.register("a@b.co", "short")
    p.auth.register("a@b.co", PW)
    with pytest.raises(AuthError) as e:
        p.auth.register("A@B.CO", PW)
    assert e.value.status == 409


def test_login_session_logout(p):
    u = mkuser(p)
    token, info = p.auth.login("USER@example.com", PW, ip="1.2.3.4")
    user, sess = p.auth.session(token)
    assert user.id == u.id and sess["csrf"] == info["csrf"]
    assert p.db.one("SELECT id FROM auth_sessions")["id"] != token  # only the hash is stored
    p.auth.logout(token)
    assert p.auth.session(token) is None


def test_login_failures_are_throttled(p):
    mkuser(p)
    for _ in range(5):
        with pytest.raises(AuthError) as e:
            p.auth.login("user@example.com", "wrong password!", ip="9.9.9.9")
        assert e.value.status == 401
    with pytest.raises(AuthError) as e:
        p.auth.login("user@example.com", PW, ip="9.9.9.9")  # even the right password
    assert e.value.status == 429


def test_session_expiry_and_suspension(p):
    owner = mkuser(p, "owner@example.com", "owner")
    u = mkuser(p)
    token, _ = p.auth.login(u.email, PW)
    p.db.run("UPDATE auth_sessions SET last_seen_at = last_seen_at - 3*3600")
    assert p.auth.session(token) is None  # idle timeout
    token, _ = p.auth.login(u.email, PW)
    p.auth.set_status(u.id, "suspended", owner.id, "abuse")
    assert p.auth.session(token) is None
    with pytest.raises(AuthError) as e:
        p.auth.login(u.email, PW)
    assert e.value.status == 403
    with pytest.raises(AuthError):
        p.auth.set_status(owner.id, "suspended", owner.id, "x")


# --------------------------------------------------------------- plans
def test_plan_catalog_from_single_source(p):
    cat = p.billing.catalog()
    by = {x["id"]: x for x in cat["plans"]}
    assert cat["currency"] == "INR" and cat["yearly_billing"]["enabled"] is False
    assert (by["plus"]["monthly_price_inr"], by["pro"]["monthly_price_inr"], by["max"]["monthly_price_inr"]) == (500, 1000, 2000)
    assert by["pro"]["popular"] and all(by[k]["cook"] for k in ("plus", "pro", "max")) and not by["free"]["cook"]
    assert all(by[k]["monthly_token_credits"] is None for k in by)  # not invented
    assert cat["payment_provider"]["configured"] is False


def test_plan_update_rules_and_audit(p):
    owner = mkuser(p, "owner@example.com", "owner")
    with pytest.raises(BillingError) as e:
        p.billing.update_plan("pro", {"monthly_price_inr": 1200}, owner.id, "raise")
    assert e.value.status == 409  # needs explicit confirmation
    with pytest.raises(BillingError):
        p.billing.update_plan("pro", {"monthly_price_inr": -1}, owner.id, "x", confirm_price_change=True)
    with pytest.raises(BillingError):
        p.billing.update_plan("pro", {"id": "hacked"}, owner.id, "x")
    with pytest.raises(BillingError):
        p.billing.update_plan("pro", {"monthly_token_credits": 5}, owner.id, "   ")
    p.billing.update_plan("pro", {"monthly_token_credits": 50000, "accepting_new": False}, owner.id, "launch")
    assert p.billing.plan("pro")["monthly_token_credits"] == 50000 and not p.billing.plan("pro")["accepting_new"]
    assert p.audit.list(action="plan.update")[0]["reason"].startswith("launch")


# -------------------------------------------------------- subscriptions
def test_dev_checkout_never_activates(p):
    u = mkuser(p)
    r = p.billing.start_checkout(u.id, "pro")
    assert r["mode"] == "development" and r["activated"] is False
    assert r["summary"]["price_inr"] == 1000 and r["summary"]["billing_period"] == "monthly"
    assert p.billing.current_subscription(u.id) is None
    assert not p.billing.entitlement(u.id).cook
    with pytest.raises(BillingError):
        p.billing.start_checkout(u.id, "pro", "yearly")
    with pytest.raises(BillingError):
        p.billing.start_checkout(u.id, "free")


def test_owner_grant_and_entitlement_with_unconfigured_credits(p):
    owner = mkuser(p, "owner@example.com", "owner")
    u = mkuser(p)
    p.billing.owner_grant(u.id, "plus", owner.id, "beta tester")
    ent = p.billing.entitlement(u.id)
    assert ent.subscription["plan"] == "plus" and ent.balance == 0
    assert not ent.cook and "not been configured" in ent.reason  # honest: no invented credits


def test_allocation_charge_and_idempotency(p):
    owner = configure_credits(p, "plus", 1000)
    u = mkuser(p)
    sub = p.billing.owner_grant(u.id, "plus", owner.id, "test")
    assert p.billing.balance(u.id) == 1000
    p.billing.allocate_period(sub)  # repeat: no double allocation
    assert p.billing.balance(u.id) == 1000
    p.billing.charge(u.id, 300, "usage:task1", "COOK task", "task:1")
    p.billing.charge(u.id, 300, "usage:task1", "COOK task", "task:1")  # retried -> idempotent
    assert p.billing.balance(u.id) == 700
    assert p.billing.duplicate_refs() == []
    assert p.billing.entitlement(u.id).cook


def test_concurrent_charges_are_atomic(p):
    owner = configure_credits(p, "plus", 10_000)
    u = mkuser(p)
    p.billing.owner_grant(u.id, "plus", owner.id, "test")
    keys = [f"usage:{i % 20}" for i in range(100)]  # 100 attempts, 20 distinct operations

    def worker(k):
        p.billing.charge(u.id, 10, k, "x", k)

    ts = [threading.Thread(target=worker, args=(k,)) for k in keys]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert p.billing.balance(u.id) == 10_000 - 20 * 10


def test_ledger_is_append_only(p):
    u = mkuser(p)
    owner = mkuser(p, "owner@example.com", "owner")
    p.billing.adjust(u.id, 50, owner.id, "goodwill", "req-1")
    import sqlite3

    with pytest.raises(sqlite3.DatabaseError):
        p.db.run("UPDATE ledger SET delta = 999999")
    with pytest.raises(BillingError):
        p.billing.adjust(u.id, 5, owner.id, "", "req-2")


def sign(body: bytes, secret="whsec_test"):
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_webhook_verification_and_activation(p, monkeypatch):
    configure_credits(p, "pro", 5000)
    u = mkuser(p)
    body = json.dumps({"id": "evt_1", "type": "payment.succeeded", "user_id": u.id, "plan": "pro", "amount_inr": 1000, "currency": "INR"}).encode()
    with pytest.raises(BillingError) as e:
        p.billing.handle_webhook(body, sign(body))
    assert e.value.status == 503  # provider not configured
    monkeypatch.setenv("INFINITY_BILLING_WEBHOOK_SECRET", "whsec_test")
    with pytest.raises(BillingError) as e:
        p.billing.handle_webhook(body, "deadbeef")
    assert e.value.status == 401
    assert p.billing.handle_webhook(body, sign(body))["activated"]
    assert p.billing.handle_webhook(body, sign(body)) == {"duplicate": True}
    assert p.billing.entitlement(u.id).cook and p.billing.balance(u.id) == 5000
    assert p.billing.revenue()["net_inr"] == 1000


def test_webhook_rejects_wrong_amount_and_failed_payment_grants_nothing(p, monkeypatch):
    monkeypatch.setenv("INFINITY_BILLING_WEBHOOK_SECRET", "whsec_test")
    u = mkuser(p)
    cheap = json.dumps({"id": "evt_2", "type": "payment.succeeded", "user_id": u.id, "plan": "max", "amount_inr": 1, "currency": "INR"}).encode()
    with pytest.raises(BillingError) as e:
        p.billing.handle_webhook(cheap, sign(cheap))
    assert e.value.status == 422
    failed = json.dumps({"id": "evt_3", "type": "payment.failed", "user_id": u.id, "plan": "plus", "amount_inr": 500, "currency": "INR"}).encode()
    assert p.billing.handle_webhook(failed, sign(failed)) == {"activated": False}
    assert p.billing.current_subscription(u.id) is None
    rev = p.billing.revenue()
    assert rev["net_inr"] == 0 and rev["failed_payments"] == 2


def test_refund_revokes_and_reduces_revenue(p, monkeypatch):
    monkeypatch.setenv("INFINITY_BILLING_WEBHOOK_SECRET", "whsec_test")
    u = mkuser(p)
    pay = json.dumps({"id": "evt_p", "type": "payment.succeeded", "user_id": u.id, "plan": "plus", "amount_inr": 500, "currency": "INR"}).encode()
    p.billing.handle_webhook(pay, sign(pay))
    ref = json.dumps({"id": "evt_r", "type": "refund.succeeded", "user_id": u.id, "amount_inr": 500}).encode()
    p.billing.handle_webhook(ref, sign(ref))
    assert p.billing.current_subscription(u.id) is None
    assert p.billing.revenue()["net_inr"] == 0


def test_subscription_expiry_and_cancel(p):
    owner = configure_credits(p, "plus", 100)
    u = mkuser(p)
    p.billing.owner_grant(u.id, "plus", owner.id, "t")
    p.billing.cancel(u.id)
    assert p.billing.entitlement(u.id).cook  # canceled: access until period end
    p.db.run("UPDATE subscriptions SET current_period_end = ?", (time.time() - 1,))
    ent = p.billing.entitlement(u.id)
    assert not ent.cook and ent.subscription is None


# ---------------------------------------------------------------- keys
def test_api_keys_lifecycle(p):
    u = mkuser(p)
    k = p.keys.create(u.id, "JARVIS")
    assert k["secret"].startswith("inf_") and k["prefix"] == k["secret"][:12]
    assert "secret" not in p.keys.list(u.id)[0]
    assert p.db.one("SELECT key_hash FROM api_keys")["key_hash"] != k["secret"]
    assert p.keys.authenticate(k["secret"])["id"] == k["id"]
    new = p.keys.rotate(u.id, k["id"])
    assert p.keys.authenticate(k["secret"]) is None and p.keys.authenticate(new["secret"])
    p.keys.revoke(u.id, new["id"], u.id)
    assert p.keys.authenticate(new["secret"]) is None
    other = mkuser(p, "other@example.com")
    with pytest.raises(KeyError_):
        p.keys.revoke(other.id, k["id"], other.id)  # cannot touch someone else's key
    with pytest.raises(KeyError_):
        p.keys.create(u.id, "<script>")


def test_rate_limiter():
    from infinity.platform.keys import RateLimiter

    rl = RateLimiter()
    assert all(rl.allow("k", 3) for _ in range(3))
    assert not rl.allow("k", 3)


def test_audit_chain_detects_tampering(p):
    mkuser(p)
    mkuser(p, "b@example.com")
    assert p.audit.verify()["ok"]
    p.db.run("DROP TRIGGER audit_no_update")  # simulate an attacker with raw DB access
    p.db.run("UPDATE audit_log SET reason = 'edited' WHERE id = 1")
    v = p.audit.verify()
    assert not v["ok"] and v["broken_at"] == 1


def test_migration_2_keeps_usage_rows_and_allows_chat(tmp_path, monkeypatch):
    import sqlite3

    from infinity.platform import db as dbmod

    path = tmp_path / "old.db"
    monkeypatch.setattr(dbmod, "MIGRATIONS", dbmod.MIGRATIONS[:1])
    old = dbmod.PlatformDB(path)
    old.run("INSERT INTO users (email, password_hash, role, status, created_at) VALUES ('a@b.c', 'x', 'user', 'active', 1)")
    old.run("INSERT INTO usage_events (user_id, source, ref, tokens_in, tokens_out, exact, ok, created_at) "
            "VALUES (1, 'cook', 'task:1', 5, 7, 1, 1, 2)")
    with pytest.raises(sqlite3.IntegrityError):
        old.run("INSERT INTO usage_events (user_id, source, ref, tokens_in, tokens_out, exact, ok, created_at) "
                "VALUES (1, 'chat', 'c', 1, 1, 1, 1, 3)")
    old.conn.close()
    monkeypatch.undo()
    new = dbmod.PlatformDB(path)
    assert [m["version"] for m in new.migrations()] == [1, 2]
    rows = new.all("SELECT source, ref, tokens_in, tokens_out FROM usage_events")
    assert [dict(r) for r in rows] == [{"source": "cook", "ref": "task:1", "tokens_in": 5, "tokens_out": 7}]
    new.run("INSERT INTO usage_events (user_id, source, ref, tokens_in, tokens_out, exact, ok, created_at) "
            "VALUES (1, 'chat', 'c', 1, 1, 1, 1, 3)")
    with pytest.raises(sqlite3.IntegrityError):
        new.run("INSERT INTO usage_events (user_id, source, ref, tokens_in, tokens_out, exact, ok, created_at) "
                "VALUES (1, 'other', 'c', 1, 1, 1, 1, 3)")
