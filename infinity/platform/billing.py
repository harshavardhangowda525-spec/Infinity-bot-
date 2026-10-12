"""Plans, subscriptions, verified payments, credit ledger, entitlements.

Trust rules:
- Plan prices/allowances come from the database (seeded once from
  configs/plans.json). The client never supplies a price or plan definition.
- A subscription becomes ACTIVE only through (a) a payment event whose HMAC
  signature verifies against INFINITY_BILLING_WEBHOOK_SECRET and whose amount
  and currency match the server-side plan, or (b) an audited owner grant.
  Checkout in development mode never activates anything.
- Credits live in an append-only ledger. Every entry has a unique idempotency
  key; balance checks and deductions happen inside one IMMEDIATE transaction.
- Allowances left as null are "not configured": no credits are invented.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .audit import Audit
from .db import PlatformDB

PERIOD_S = 30 * 24 * 3600
EDITABLE_PLAN_FIELDS = {
    "name": str, "tagline": str, "features": list, "monthly_price_inr": int, "monthly_token_credits": (int, type(None)),
    "max_task_steps": (int, type(None)), "requests_per_minute": (int, type(None)), "cook": bool, "api_access": bool,
    "priority": str, "popular": bool, "cta": (str, type(None)),
}
PRICE_FIELDS = {"monthly_price_inr"}


class BillingError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def default_plans_file() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "plans.json"


@dataclass
class Entitlement:
    plan: dict
    subscription: dict | None
    balance: int
    cook: bool
    api: bool
    reason: str | None  # why access is not granted (None when granted)


class Billing:
    def __init__(self, db: PlatformDB, audit: Audit, plans_file: Path | None = None) -> None:
        self.db = db
        self.audit = audit
        self._seed(plans_file or default_plans_file())

    # ---------------------------------------------------------------- plans
    def _seed(self, path: Path) -> None:
        if self.db.one("SELECT 1 AS x FROM plans LIMIT 1"):
            return
        cfg = json.loads(Path(path).read_text())
        if cfg.get("currency") != "INR":
            raise BillingError("plans.json must use INR")
        now = time.time()
        with self.db.tx() as c:
            for i, p in enumerate(cfg["plans"]):
                c.execute("INSERT INTO plans (id, position, data, accepting_new, updated_at) VALUES (?,?,?,?,?)",
                          (p["id"], i, json.dumps(p), 1, now))
            c.execute("CREATE TABLE IF NOT EXISTS billing_settings (key TEXT PRIMARY KEY, value TEXT)")
            c.execute("INSERT OR REPLACE INTO billing_settings VALUES ('yearly_billing', ?)", (json.dumps(cfg.get("yearly_billing", {"enabled": False})),))

    def plans(self) -> list[dict]:
        out = []
        for r in self.db.all("SELECT * FROM plans ORDER BY position"):
            p = json.loads(r["data"])
            p["accepting_new"] = bool(r["accepting_new"])
            out.append(p)
        return out

    def plan(self, plan_id: str) -> dict:
        r = self.db.one("SELECT * FROM plans WHERE id = ?", (plan_id,))
        if not r:
            raise BillingError(f"unknown plan {plan_id!r}", 404)
        p = json.loads(r["data"])
        p["accepting_new"] = bool(r["accepting_new"])
        return p

    def catalog(self) -> dict:
        yearly = self.db.one("SELECT value FROM billing_settings WHERE key = 'yearly_billing'")
        return {
            "currency": "INR",
            "yearly_billing": json.loads(yearly["value"]) if yearly else {"enabled": False},
            "payment_provider": self.provider_status(),
            "plans": self.plans(),
        }

    def update_plan(self, plan_id: str, changes: dict, actor_id: int, reason: str, confirm_price_change: bool = False) -> dict:
        p = self.plan(plan_id)
        unknown = set(changes) - set(EDITABLE_PLAN_FIELDS) - {"accepting_new"}
        if unknown:
            raise BillingError(f"fields not editable: {sorted(unknown)}")
        for k, v in changes.items():
            if k == "accepting_new":
                if not isinstance(v, bool):
                    raise BillingError("accepting_new must be boolean")
                continue
            typ = EDITABLE_PLAN_FIELDS[k]
            if isinstance(v, bool) and bool not in (typ if isinstance(typ, tuple) else (typ,)):
                raise BillingError(f"{k} has the wrong type")
            if not isinstance(v, typ):
                raise BillingError(f"{k} has the wrong type")
            if isinstance(v, int) and not isinstance(v, bool) and v < 0:
                raise BillingError(f"{k} must not be negative")
        if PRICE_FIELDS & set(changes) and changes.get("monthly_price_inr") != p.get("monthly_price_inr") and not confirm_price_change:
            raise BillingError("changing a price requires explicit confirmation (confirm_price_change=true)", 409)
        if not reason.strip():
            raise BillingError("a reason is required")
        accepting = changes.pop("accepting_new", None)
        p.pop("accepting_new", None)
        before = dict(p)
        p.update(changes)
        with self.db.tx() as c:
            c.execute("UPDATE plans SET data = ?, updated_at = ? WHERE id = ?", (json.dumps(p), time.time(), plan_id))
            if accepting is not None:
                c.execute("UPDATE plans SET accepting_new = ? WHERE id = ?", (int(accepting), plan_id))
        diff = {k: [before.get(k), p.get(k)] for k in changes if before.get(k) != p.get(k)}
        if accepting is not None:
            diff["accepting_new"] = accepting
        # Existing subscribers keep the price they signed up at (stored on the subscription row).
        self.audit.record(actor_id, "plan.update", f"plan:{plan_id}", reason=f"{reason} :: {json.dumps(diff)}")
        return self.plan(plan_id)

    # --------------------------------------------------------- subscription
    def current_subscription(self, user_id: int) -> dict | None:
        now = time.time()
        # Lazily expire subscriptions whose period ended.
        self.db.run("UPDATE subscriptions SET status = 'expired', updated_at = ? WHERE user_id = ? AND status IN ('active','canceled') "
                    "AND current_period_end IS NOT NULL AND current_period_end < ?", (now, user_id, now))
        return self.db.one(
            "SELECT * FROM subscriptions WHERE user_id = ? AND status IN ('active','canceled') ORDER BY id DESC LIMIT 1",
            (user_id,))

    def subscriptions(self, user_id: int | None = None, status: str | None = None, limit: int = 100, offset: int = 0) -> list[dict]:
        where, params = [], []
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)
        if status:
            where.append("status = ?")
            params.append(status)
        sql = "SELECT * FROM subscriptions" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT ? OFFSET ?"
        return self.db.all(sql, tuple(params) + (limit, offset))

    def _activate(self, user_id: int, plan_id: str, source: str, price_inr: int) -> dict:
        now = time.time()
        with self.db.tx() as c:
            c.execute("UPDATE subscriptions SET status = 'expired', updated_at = ? WHERE user_id = ? AND status IN ('active','canceled','pending')",
                      (now, user_id))
            cur = c.execute(
                "INSERT INTO subscriptions (user_id, plan, status, billing_period, price_inr, source, current_period_start, "
                "current_period_end, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (user_id, plan_id, "active", "monthly", price_inr, source, now, now + PERIOD_S, now, now))
            sub_id = int(cur.lastrowid)
        sub = self.db.one("SELECT * FROM subscriptions WHERE id = ?", (sub_id,))
        self.allocate_period(sub)
        return sub

    def allocate_period(self, sub: dict) -> int | None:
        """Grant the plan's monthly credits for this period (idempotent). None if not configured."""
        plan = self.plan(sub["plan"])
        credits = plan.get("monthly_token_credits")
        if credits is None:
            return None
        key = f"alloc:sub{sub['id']}:{int(sub['current_period_start'])}"
        self.ledger_entry(sub["user_id"], int(credits), "allocation", f"{plan['name']} monthly allowance", key, ref=f"sub:{sub['id']}")
        return int(credits)

    def start_checkout(self, user_id: int, plan_id: str, period: str = "monthly") -> dict:
        plan = self.plan(plan_id)
        if not plan.get("paid"):
            raise BillingError("this plan cannot be purchased")
        if not plan.get("accepting_new"):
            raise BillingError("this plan is not accepting new subscriptions", 409)
        if period != "monthly":
            raise BillingError("yearly billing is unavailable until yearly prices are configured", 409)
        provider = self.provider_status()
        summary = {"plan": plan_id, "plan_name": plan["name"], "price_inr": plan["monthly_price_inr"], "currency": "INR",
                   "billing_period": "monthly", "monthly_token_credits": plan.get("monthly_token_credits")}
        if not provider["configured"]:
            self.audit.record(user_id, "billing.checkout.dev", f"plan:{plan_id}", reason="no payment provider configured")
            return {"mode": "development", "activated": False, "summary": summary,
                    "message": "Development mode: no payment provider is configured, so no payment was taken and no "
                               "subscription was activated."}
        raise BillingError("payment provider checkout is not implemented yet", 501)

    def cancel(self, user_id: int) -> dict:
        sub = self.current_subscription(user_id)
        if not sub or sub["status"] != "active":
            raise BillingError("no active subscription", 404)
        self.db.run("UPDATE subscriptions SET status = 'canceled', cancel_at_period_end = 1, updated_at = ? WHERE id = ?",
                    (time.time(), sub["id"]))
        self.audit.record(user_id, "billing.cancel", f"sub:{sub['id']}", reason="access continues until period end")
        return self.db.one("SELECT * FROM subscriptions WHERE id = ?", (sub["id"],))

    def owner_grant(self, user_id: int, plan_id: str, actor_id: int, reason: str) -> dict:
        if not reason.strip():
            raise BillingError("a reason is required")
        plan = self.plan(plan_id)
        if not plan.get("paid"):
            raise BillingError("grant a paid plan")
        sub = self._activate(user_id, plan_id, "owner_grant", 0)
        self.audit.record(actor_id, "billing.grant", f"user:{user_id}", reason=f"{plan_id} :: {reason}")
        return sub

    # ------------------------------------------------------------- payments
    @staticmethod
    def provider_status() -> dict:
        configured = bool(os.environ.get("INFINITY_BILLING_WEBHOOK_SECRET"))
        return {"configured": configured, "mode": "live-webhook" if configured else "development"}

    def handle_webhook(self, body: bytes, signature: str | None) -> dict:
        secret = os.environ.get("INFINITY_BILLING_WEBHOOK_SECRET")
        if not secret:
            raise BillingError("payment provider not configured", 503)
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        if not signature or not hmac.compare_digest(expected, signature):
            self.audit.record(None, "billing.webhook", None, success=False, reason="bad signature")
            raise BillingError("invalid signature", 401)
        try:
            ev = json.loads(body)
            eid, etype, uid = str(ev["id"]), ev["type"], int(ev["user_id"])
        except (ValueError, KeyError, TypeError):
            raise BillingError("malformed event")
        if self.db.one("SELECT id FROM payments WHERE provider_event_id = ?", (eid,)):
            return {"duplicate": True}
        if not self.db.one("SELECT id FROM users WHERE id = ?", (uid,)):
            raise BillingError("unknown user", 404)
        now = time.time()
        if etype == "payment.succeeded":
            plan = self.plan(ev.get("plan", ""))
            amount, currency = int(ev.get("amount_inr", -1)), ev.get("currency")
            if currency != "INR" or amount != plan["monthly_price_inr"] or not plan.get("paid"):
                self._payment(uid, None, eid, "payment", "failed", amount, str(currency), plan["id"], now)
                self.audit.record(None, "billing.webhook.mismatch", f"user:{uid}", success=False,
                                  reason=f"amount/currency mismatch for {plan['id']}: {amount} {currency}")
                raise BillingError("amount or currency does not match the plan", 422)
            sub = self._activate(uid, plan["id"], "payment", amount)
            self._payment(uid, sub["id"], eid, "payment", "verified", amount, "INR", plan["id"], now)
            self.audit.record(None, "billing.payment.verified", f"user:{uid}", reason=f"{plan['id']} {amount} INR event={eid}")
            return {"activated": True, "subscription_id": sub["id"]}
        if etype == "payment.failed":
            self._payment(uid, None, eid, "payment", "failed", int(ev.get("amount_inr", 0)), str(ev.get("currency", "INR")), ev.get("plan"), now)
            self.audit.record(None, "billing.payment.failed", f"user:{uid}", success=False, reason=f"event={eid}")
            return {"activated": False}
        if etype == "refund.succeeded":
            amount = int(ev.get("amount_inr", 0))
            sub = self.current_subscription(uid)
            self._payment(uid, sub["id"] if sub else None, eid, "refund", "verified", amount, "INR", ev.get("plan"), now)
            if sub and ev.get("revoke_access", True):
                self.db.run("UPDATE subscriptions SET status = 'expired', updated_at = ? WHERE id = ?", (now, sub["id"]))
            self.audit.record(None, "billing.refund", f"user:{uid}", reason=f"{amount} INR event={eid}")
            return {"refunded": True}
        raise BillingError(f"unsupported event type {etype!r}")

    def _payment(self, uid, sub_id, eid, kind, status, amount, currency, plan, now) -> None:
        self.db.run(
            "INSERT INTO payments (user_id, subscription_id, provider, provider_event_id, kind, status, amount_inr, currency, plan, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)", (uid, sub_id, "webhook", eid, kind, status, amount, currency, plan, now))

    def revenue(self, since: float | None = None) -> dict:
        """Verified money only: verified payments minus verified refunds. Failed payments are excluded."""
        since = since or 0
        pay = self.db.one("SELECT COALESCE(SUM(amount_inr),0) AS s, COUNT(*) AS n FROM payments WHERE kind='payment' AND status='verified' AND created_at >= ?", (since,))
        ref = self.db.one("SELECT COALESCE(SUM(amount_inr),0) AS s, COUNT(*) AS n FROM payments WHERE kind='refund' AND status='verified' AND created_at >= ?", (since,))
        failed = self.db.one("SELECT COUNT(*) AS n FROM payments WHERE status='failed' AND created_at >= ?", (since,))
        return {"currency": "INR", "gross_inr": pay["s"], "refunds_inr": ref["s"], "net_inr": pay["s"] - ref["s"],
                "verified_payments": pay["n"], "refunds": ref["n"], "failed_payments": failed["n"], "since": since}

    def payments(self, limit: int = 100, offset: int = 0) -> list[dict]:
        return self.db.all("SELECT * FROM payments ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))

    # --------------------------------------------------------------- ledger
    def balance(self, user_id: int) -> int:
        return int(self.db.one("SELECT COALESCE(SUM(delta), 0) AS b FROM ledger WHERE user_id = ?", (user_id,))["b"])

    def ledger_entry(self, user_id: int, delta: int, kind: str, reason: str, idempotency_key: str, ref: str | None = None,
                     actor_id: int | None = None) -> dict:
        """Insert one entry; repeating the same idempotency key returns the original entry instead of duplicating."""
        try:
            with self.db.tx() as c:
                c.execute("INSERT INTO ledger (user_id, delta, kind, reason, ref, idempotency_key, actor_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
                          (user_id, delta, kind, reason, ref, idempotency_key, actor_id, time.time()))
        except sqlite3.IntegrityError:
            pass
        return self.db.one("SELECT * FROM ledger WHERE idempotency_key = ?", (idempotency_key,))

    def charge(self, user_id: int, tokens: int, idempotency_key: str, reason: str, ref: str) -> dict:
        """Deduct actual usage. Atomic and idempotent. Usage already incurred is recorded even if it overdraws."""
        tokens = max(0, int(tokens))
        return self.ledger_entry(user_id, -tokens, "usage", reason, idempotency_key, ref=ref)

    def adjust(self, user_id: int, delta: int, actor_id: int, reason: str, request_id: str) -> dict:
        if not reason.strip():
            raise BillingError("a reason is required")
        if delta == 0:
            raise BillingError("delta must not be zero")
        e = self.ledger_entry(user_id, delta, "adjustment", reason, f"adjust:{request_id}", actor_id=actor_id)
        self.audit.record(actor_id, "credits.adjust", f"user:{user_id}", reason=f"{delta:+d} :: {reason}", request_id=request_id)
        return e

    def ledger(self, user_id: int, limit: int = 50, offset: int = 0) -> list[dict]:
        return self.db.all("SELECT * FROM ledger WHERE user_id = ? ORDER BY id DESC LIMIT ? OFFSET ?", (user_id, limit, offset))

    def duplicate_refs(self) -> list[dict]:
        """Usage refs charged more than once (should be empty thanks to idempotency keys)."""
        return self.db.all("SELECT user_id, ref, COUNT(*) AS n FROM ledger WHERE kind='usage' GROUP BY user_id, ref HAVING n > 1")

    # --------------------------------------------------------- entitlement
    def entitlement(self, user_id: int) -> Entitlement:
        sub = self.current_subscription(user_id)
        plan = self.plan(sub["plan"]) if sub else self.plan("free")
        bal = self.balance(user_id)
        reason = None
        if not sub:
            reason = "COOK requires an active Plus, Pro or Max subscription"
        elif not plan.get("cook"):
            reason = f"the {plan['name']} plan does not include COOK"
        elif bal <= 0:
            reason = ("no credits remaining" if plan.get("monthly_token_credits") is not None
                      else "your plan's monthly credit allowance has not been configured by the platform owner yet")
        api_reason_ok = bool(sub) and plan.get("api_access") and bal > 0
        return Entitlement(plan, sub, bal, cook=reason is None, api=bool(api_reason_ok), reason=reason)
