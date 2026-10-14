import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { allowance, date, inr, num, modelName } from "../lib/format";
import type { AdminOverview, AdminUser, AuditEvent, Plan } from "../lib/types";

type Tab = "overview" | "users" | "plans" | "audit";

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-line bg-hover p-3">
      <div className="text-[10px] tracking-[0.18em] text-muted">{label}</div>
      <div className="mt-1 text-xl font-semibold text-fg">{value}</div>
      {sub && <div className="mt-0.5 text-[11px] text-muted">{sub}</div>}
    </div>
  );
}

/** Prompt for a reason (required by the server for every privileged action). */
function useReasoned() {
  const [pending, setPending] = useState<{ title: string; extra?: "delta" | "plan" | "roots"; run: (reason: string, extra: string) => Promise<unknown> } | null>(null);
  const [reason, setReason] = useState("");
  const [extra, setExtra] = useState("");
  const [error, setError] = useState<string | null>(null);
  const dialog = pending && (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4">
      <form
        role="dialog"
        aria-modal="true"
        aria-label={pending.title}
        className="glass-strong popover w-full max-w-md rounded-2xl p-5 text-sm"
        onSubmit={async (e) => {
          e.preventDefault();
          try {
            await pending.run(reason, extra);
            setPending(null);
            setReason("");
            setExtra("");
            setError(null);
          } catch (err) {
            setError((err as Error).message);
          }
        }}
      >
        <h3 className="text-xs font-semibold tracking-[0.16em] text-fg">{pending.title.toUpperCase()}</h3>
        {pending.extra === "delta" && (
          <label className="mt-3 block text-xs text-muted">Credit change (negative to deduct)
            <input type="number" required value={extra} onChange={(e) => setExtra(e.target.value)} className="mt-1 w-full rounded-lg border border-line bg-surface-2 px-3 py-2 text-fg" />
          </label>
        )}
        {pending.extra === "plan" && (
          <label className="mt-3 block text-xs text-muted">Plan
            <select required value={extra} onChange={(e) => setExtra(e.target.value)} className="mt-1 w-full rounded-lg border border-line bg-surface-2 px-3 py-2 text-fg">
              <option value="">Select…</option><option value="plus">PLUS</option><option value="pro">PRO</option><option value="max">MAX</option>
            </select>
          </label>
        )}
        {pending.extra === "roots" && (
          <label className="mt-3 block text-xs text-muted">Authorised workspace roots (one absolute path per line; empty = none)
            <textarea rows={3} value={extra} onChange={(e) => setExtra(e.target.value)} className="mt-1 w-full rounded-lg border border-line bg-surface-2 px-3 py-2 font-mono text-xs text-fg" />
          </label>
        )}
        <label className="mt-3 block text-xs text-muted">Reason (recorded in the audit log)
          <input required minLength={3} value={reason} onChange={(e) => setReason(e.target.value)} className="mt-1 w-full rounded-lg border border-line bg-surface-2 px-3 py-2 text-fg" />
        </label>
        {error && <p role="alert" className="mt-2 text-xs text-bad">{error}</p>}
        <div className="mt-4 flex gap-2">
          <button className="rounded-lg bg-cyan-glow/90 px-3 py-1.5 text-xs font-semibold text-white">Confirm</button>
          <button type="button" onClick={() => setPending(null)} className="rounded-lg border border-line px-3 py-1.5 text-xs">Cancel</button>
        </div>
      </form>
    </div>
  );
  return { ask: setPending, dialog };
}

function PlanEditor({ plan, onSaved }: { plan: Plan; onSaved: () => void }) {
  const [credits, setCredits] = useState(plan.monthly_token_credits?.toString() ?? "");
  const [steps, setSteps] = useState(plan.max_task_steps?.toString() ?? "");
  const [rpm, setRpm] = useState(plan.requests_per_minute?.toString() ?? "");
  const [price, setPrice] = useState(plan.monthly_price_inr.toString());
  const [accepting, setAccepting] = useState(plan.accepting_new);
  const [confirmPrice, setConfirmPrice] = useState(false);
  const [reason, setReason] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const toNum = (s: string) => (s.trim() === "" ? null : Number(s));
  const save = async () => {
    setMsg(null);
    const changes: Partial<Plan> = { monthly_token_credits: toNum(credits), max_task_steps: toNum(steps), requests_per_minute: toNum(rpm), accepting_new: accepting };
    if (Number(price) !== plan.monthly_price_inr) changes.monthly_price_inr = Number(price);
    try {
      await api.adminPlan(plan.id, changes, reason, confirmPrice);
      setMsg("Saved and audited.");
      onSaved();
    } catch (e) {
      setMsg((e as Error).message);
    }
  };
  const field = (label: string, v: string, set: (s: string) => void, hint: string) => (
    <label className="text-xs text-muted">{label}
      <input type="number" min={0} value={v} onChange={(e) => set(e.target.value)} placeholder="not configured"
        className="mt-1 block w-full rounded-lg border border-line bg-surface-2 px-2.5 py-1.5 text-sm text-fg" />
      <span className="text-[10px] text-muted/70">{hint}</span>
    </label>
  );
  return (
    <div className="rounded-xl border border-line p-4">
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-semibold tracking-[0.2em] text-fg">{plan.name}</h3>
        <span className="text-xs text-muted">{inr(plan.monthly_price_inr)}/month · {allowance(plan.monthly_token_credits)}</span>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {field("Monthly price (₹)", price, setPrice, "existing subscribers keep their price")}
        {field("Monthly credits", credits, setCredits, "empty = not configured")}
        {field("Max task steps", steps, setSteps, "empty = runtime default")}
        {field("API requests/min", rpm, setRpm, "empty = per-key limit only")}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-fg/80">
        <label className="flex items-center gap-1.5"><input type="checkbox" checked={accepting} onChange={(e) => setAccepting(e.target.checked)} />Accepting new subscriptions</label>
        {Number(price) !== plan.monthly_price_inr && (
          <label className="flex items-center gap-1.5 text-amber-diag"><input type="checkbox" checked={confirmPrice} onChange={(e) => setConfirmPrice(e.target.checked)} />I confirm the price change</label>
        )}
        <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="Reason (required)" className="min-w-[180px] flex-1 rounded-lg border border-line bg-surface-2 px-2.5 py-1.5" />
        <button onClick={save} disabled={reason.trim().length < 3} className="rounded-lg bg-cyan-glow/90 px-3 py-1.5 font-semibold text-white disabled:opacity-30">Save</button>
      </div>
      {msg && <p className="mt-2 text-xs text-muted">{msg}</p>}
    </div>
  );
}

export default function AdminView() {
  const [tab, setTab] = useState<Tab>("overview");
  const [ov, setOv] = useState<AdminOverview | null>(null);
  const [users, setUsers] = useState<AdminUser[] | null>(null);
  const [plans, setPlans] = useState<Plan[] | null>(null);
  const [audit, setAudit] = useState<{ events: AuditEvent[]; chain: { ok: boolean; checked: number } } | null>(null);
  const [q, setQ] = useState("");
  const [error, setError] = useState<string | null>(null);
  const { ask, dialog } = useReasoned();

  const load = useCallback(async () => {
    setError(null);
    try {
      if (tab === "overview") setOv(await api.adminOverview());
      if (tab === "users") setUsers((await api.adminUsers(q)).users);
      if (tab === "plans") setPlans((await api.plans()).plans.filter((p) => p.paid));
      if (tab === "audit") setAudit(await api.adminAudit());
    } catch (e) {
      setError((e as Error).message);
    }
  }, [tab, q]);
  useEffect(() => {
    load();
  }, [load]);

  const wrap = (fn: (r: string, x: string) => Promise<unknown>) => async (r: string, x: string) => {
    await fn(r, x);
    await load();
  };

  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl px-4 pb-16 pt-6 sm:px-6">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="mr-3 text-xs font-semibold tracking-[0.25em] text-fg">ADMIN CONSOLE</h1>
          {(["overview", "users", "plans", "audit"] as Tab[]).map((t) => (
            <button key={t} onClick={() => setTab(t)} aria-current={tab === t}
              className={`rounded-lg px-3 py-1.5 text-xs capitalize ${tab === t ? "bg-cyan-glow/15 text-cyan-glow" : "text-muted hover:bg-hover"}`}>{t}</button>
          ))}
          <button onClick={load} className="ml-auto rounded-lg border border-line px-2.5 py-1.5 text-xs text-muted hover:text-fg">Refresh</button>
        </div>
        {error && <p role="alert" className="mt-3 text-sm text-bad">{error}</p>}

        {tab === "overview" && ov && (
          <div className="mt-4 space-y-4">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Metric label="REGISTERED ACCOUNTS" value={num(ov.users.total)} sub={`${ov.users.active_24h} active in 24h · ${ov.users.suspended} suspended`} />
              <Metric label="ACTIVE SUBSCRIPTIONS" value={num((ov.subscriptions.active ?? 0) + (ov.subscriptions.canceled ?? 0))}
                sub={Object.entries(ov.active_by_plan).map(([k, v]) => `${k}: ${v}`).join(" · ") || "none"} />
              <Metric label="VERIFIED REVENUE · 30D" value={inr(ov.revenue_30d.net_inr)}
                sub={`${ov.revenue_30d.verified_payments} payments · ${inr(ov.revenue_30d.refunds_inr)} refunded · ${ov.revenue_30d.failed_payments} failed`} />
              <Metric label="TOKENS · 30D" value={num(ov.usage_30d.tokens_in + ov.usage_30d.tokens_out)} sub={`${num(ov.usage_30d.requests)} requests`} />
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="glass rounded-2xl p-4 text-xs text-fg/80">
                <h2 className="mb-2 text-[10px] tracking-[0.18em] text-muted">SYSTEM</h2>
                <p>Model: {ov.model.state === "ready" ? modelName(ov.model) : `unavailable — ${ov.model.reason}`}</p>
                <p>Payment provider: {ov.payment_provider.configured ? "webhook configured" : "not connected (development mode)"}</p>
                <p>Audit chain: {ov.audit_chain.ok ? `intact (${ov.audit_chain.checked} records)` : `BROKEN at record ${ov.audit_chain.broken_at}`}</p>
                <p>Ledger duplicate charges: {ov.ledger_duplicates.length}</p>
                <p>Database migrations: {ov.database.migrations.map((m) => m.version).join(", ")}</p>
                <p>Active COOK tasks: {ov.tasks.filter((t) => t.busy).length} running · {ov.tasks.length} sessions</p>
              </div>
              <div className="glass rounded-2xl p-4 text-xs">
                <h2 className="mb-2 text-[10px] tracking-[0.18em] text-muted">RECENT SECURITY EVENTS</h2>
                {ov.security.length === 0 ? <p className="text-muted">No failed privileged or login events.</p> : (
                  <ul className="space-y-1">{ov.security.map((e) => <li key={e.id} className="text-bad">{date(e.created_at)} — {e.action} {e.target ?? ""} {e.reason ? `(${e.reason})` : ""}</li>)}</ul>
                )}
              </div>
            </div>
          </div>
        )}

        {tab === "users" && (
          <div className="mt-4">
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search by e-mail" aria-label="Search users"
              className="mb-3 w-full max-w-sm rounded-lg border border-line bg-surface-2 px-3 py-2 text-sm text-fg" />
            <div className="glass scroll-thin overflow-x-auto rounded-2xl">
              <table className="w-full min-w-[900px] text-left text-sm">
                <thead className="text-xs text-muted"><tr className="border-b border-line">
                  {["ID", "E-mail", "Role", "Status", "Plan", "Credits", "Usage 30d", "Last active", "Actions"].map((h) => <th key={h} className="p-3 font-normal">{h}</th>)}
                </tr></thead>
                <tbody>
                  {(users ?? []).map((u) => (
                    <tr key={u.id} className="border-b border-line last:border-0 text-fg/80">
                      <td className="p-3 text-xs">{u.id}</td>
                      <td className="p-3">{u.email}</td>
                      <td className="p-3 text-xs">{u.role}</td>
                      <td className={`p-3 text-xs ${u.status === "active" ? "text-ok" : "text-bad"}`}>{u.status}</td>
                      <td className="p-3 text-xs uppercase">{u.plan}{u.subscription_status && u.subscription_status !== "active" ? ` (${u.subscription_status})` : ""}</td>
                      <td className="p-3 font-mono text-xs">{num(u.balance)}</td>
                      <td className="p-3 font-mono text-xs">{num(u.usage_30d)}</td>
                      <td className="p-3 text-xs text-muted">{date(u.last_active_at)}</td>
                      <td className="p-3">
                        <div className="flex flex-wrap gap-1 text-[11px]">
                          <button className="rounded border border-line px-2 py-0.5 hover:text-fg" onClick={() => ask({ title: `Grant plan to ${u.email}`, extra: "plan", run: wrap((r, x) => api.adminGrant(u.id, x, r)) })}>Grant plan</button>
                          <button className="rounded border border-line px-2 py-0.5 hover:text-fg" onClick={() => ask({ title: `Adjust credits for ${u.email}`, extra: "delta", run: wrap((r, x) => api.adminCredits(u.id, Number(x), r)) })}>Credits</button>
                          <button className="rounded border border-line px-2 py-0.5 hover:text-fg" onClick={() => ask({ title: `Workspace roots for ${u.email}`, extra: "roots", run: wrap((r, x) => api.adminRoots(u.id, x.split("\n").map((s) => s.trim()).filter(Boolean), r)) })}>Workspaces</button>
                          {u.role !== "owner" && (
                            <>
                              <button className="rounded border border-line px-2 py-0.5 hover:text-fg" onClick={() => ask({ title: `Revoke sessions of ${u.email}`, run: wrap((r) => api.adminRevokeSessions(u.id, r)) })}>Sign out</button>
                              <button className={`rounded border px-2 py-0.5 ${u.status === "active" ? "border-bad/30 text-bad" : "border-ok/30 text-ok"}`}
                                onClick={() => ask({ title: `${u.status === "active" ? "Suspend" : "Restore"} ${u.email}`, run: wrap((r) => api.adminStatus(u.id, u.status === "active" ? "suspended" : "active", r)) })}>
                                {u.status === "active" ? "Suspend" : "Restore"}
                              </button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {users && users.length === 0 && <p className="p-4 text-sm text-muted">No matching accounts.</p>}
            </div>
          </div>
        )}

        {tab === "plans" && (
          <div className="mt-4 space-y-3">
            <p className="text-xs text-muted">Every change needs a reason and is audited. Price changes need explicit confirmation and never alter existing subscribers' price.</p>
            {(plans ?? []).map((p) => <PlanEditor key={p.id} plan={p} onSaved={load} />)}
          </div>
        )}

        {tab === "audit" && audit && (
          <div className="mt-4">
            <p className={`mb-3 text-xs ${audit.chain.ok ? "text-ok" : "text-bad"}`}>
              Hash chain {audit.chain.ok ? `verified (${audit.chain.checked} records)` : "verification FAILED"}
            </p>
            <div className="glass scroll-thin overflow-x-auto rounded-2xl">
              <table className="w-full min-w-[720px] text-left text-xs">
                <thead className="text-muted"><tr className="border-b border-line">{["Time", "Actor", "Action", "Target", "Result", "Reason"].map((h) => <th key={h} className="p-2.5 font-normal">{h}</th>)}</tr></thead>
                <tbody>
                  {audit.events.map((e) => (
                    <tr key={e.id} className="border-b border-line last:border-0 text-fg/80">
                      <td className="p-2.5 text-muted">{date(e.created_at)}</td>
                      <td className="p-2.5">{e.actor_id ?? "system"}</td>
                      <td className="p-2.5 font-mono">{e.action}</td>
                      <td className="p-2.5">{e.target}</td>
                      <td className={`p-2.5 ${e.success ? "text-ok" : "text-bad"}`}>{e.success ? "ok" : "failed"}</td>
                      <td className="max-w-[320px] truncate p-2.5 text-muted" title={e.reason ?? ""}>{e.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
      {dialog}
    </div>
  );
}
