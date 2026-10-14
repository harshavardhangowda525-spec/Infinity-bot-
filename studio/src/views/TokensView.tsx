import { useEffect, useState } from "react";
import UsageChart from "../components/UsageChart";
import { api } from "../lib/api";
import { allowance, date, inr, num } from "../lib/format";
import type { BillingMe } from "../lib/types";

const LOW_CREDIT_RATIO = 0.1;

export default function TokensView({ onPlans }: { onPlans: () => void }) {
  const [data, setData] = useState<BillingMe | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState(false);
  const load = () => api.billing().then(setData, (e) => setError((e as Error).message));
  useEffect(() => {
    load();
  }, []);

  if (error) return <p className="p-8 text-center text-sm text-bad">{error}</p>;
  if (!data) return <p className="p-8 text-center text-sm text-muted">Loading…</p>;
  const ent = data.entitlement;
  const alloc = ent.plan.monthly_token_credits;
  const low = alloc !== null && alloc > 0 && ent.balance < alloc * LOW_CREDIT_RATIO;
  const sub = ent.subscription;

  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto grid max-w-6xl gap-4 px-4 pb-16 pt-6 sm:px-6 lg:grid-cols-3">
        <section className="glass rounded-2xl p-5 lg:col-span-1" aria-labelledby="plan-h">
          <h2 id="plan-h" className="text-[11px] font-semibold tracking-[0.2em] text-muted">CURRENT PLAN</h2>
          <p className="mt-2 text-2xl font-semibold tracking-[0.12em] text-cyan-glow">{sub ? ent.plan.name : "FREE"}</p>
          {sub ? (
            <p className="mt-1 text-xs text-muted">
              {sub.source === "owner_grant" ? "Granted by owner" : `${inr(sub.price_inr)} / month`} · {sub.status}
              {sub.cancel_at_period_end ? " (ends at period end)" : ""} · renews/ends {date(sub.current_period_end)}
            </p>
          ) : (
            <p className="mt-1 text-xs text-muted">No active subscription.</p>
          )}
          <h2 className="mt-6 text-[11px] font-semibold tracking-[0.2em] text-muted">REMAINING CREDITS</h2>
          <p className="mt-2 text-3xl font-semibold text-fg">
            {num(ent.balance)}
            {alloc !== null && <span className="text-base font-normal text-muted"> / {num(alloc)}</span>}
          </p>
          <p className="mt-1 text-xs text-muted">Monthly allowance: {allowance(alloc)}</p>
          {ent.owner_exempt && <p className="mt-2 text-xs text-muted">Owner account: usage is recorded but not charged.</p>}
          {alloc !== null && alloc > 0 && (
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-hover" role="meter" aria-valuemin={0} aria-valuemax={alloc} aria-valuenow={Math.max(0, ent.balance)} aria-label="Credits remaining">
              <div className="h-full rounded-full bg-cyan-glow/80" style={{ width: `${Math.max(0, Math.min(100, (ent.balance / alloc) * 100))}%` }} />
            </div>
          )}
          {!ent.cook && ent.cook_reason && (
            <div className="mt-4 rounded-xl border border-amber-diag/30 bg-amber-diag/[0.06] p-3 text-xs text-amber-diag">
              COOK unavailable: {ent.cook_reason}.
              <button onClick={onPlans} className="ml-1 underline">View plans</button>
            </div>
          )}
        </section>

        <section className="glass rounded-2xl p-5 lg:col-span-2" aria-labelledby="usage-h">
          <div className="mb-3 flex items-baseline gap-4">
            <h2 id="usage-h" className="text-[11px] font-semibold tracking-[0.2em] text-muted">TOKEN USAGE · 30 DAYS</h2>
            <span className="text-xs text-muted">in {num(data.usage.tokens_in)} · out {num(data.usage.tokens_out)} · {num(data.usage.requests)} requests</span>
          </div>
          <UsageChart usage={data.usage} />
          {data.usage.by_source.length > 0 && (
            <p className="mt-3 text-xs text-muted">
              {data.usage.by_source.map((s) => `${s.source.toUpperCase()}: ${num(s.tin + s.tout)} tokens / ${s.n} requests`).join(" · ")}
              {data.usage.failed_events > 0 && ` · ${data.usage.failed_events} failed/blocked`}
            </p>
          )}
        </section>

        <section className="glass rounded-2xl p-5 lg:col-span-3" aria-labelledby="ledger-h">
          <h2 id="ledger-h" className="mb-3 text-[11px] font-semibold tracking-[0.2em] text-muted">RECENT LEDGER TRANSACTIONS</h2>
          {data.ledger.length === 0 ? (
            <p className="text-sm text-muted">No ledger entries yet.</p>
          ) : (
            <ul className="divide-y divide-line">
              {data.ledger.map((e) => (
                <li key={e.id} className="flex items-center gap-3 py-2 text-sm">
                  <span className="w-24 shrink-0 text-[11px] uppercase tracking-wider text-muted">{e.kind}</span>
                  <span className="min-w-0 flex-1 truncate text-fg/80">{e.reason}</span>
                  <span className="text-[11px] text-muted">{date(e.created_at)}</span>
                  <span className={`w-24 text-right font-mono ${e.delta >= 0 ? "text-ok" : "text-fg"}`}>
                    {e.delta >= 0 ? "+" : ""}
                    {num(e.delta)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
      {low && !dismissed && (
        <div role="alert" className="glass-strong popover fixed bottom-24 left-4 z-40 max-w-xs rounded-xl p-3 text-xs text-fg">
          <div className="flex items-center">
            <span className="font-semibold tracking-wider text-amber-diag">LOW CREDIT WARNING</span>
            <button onClick={() => setDismissed(true)} className="ml-auto text-muted" aria-label="Dismiss">✕</button>
          </div>
          <p className="mt-1 text-muted">Less than {LOW_CREDIT_RATIO * 100}% of your monthly allowance remains.</p>
        </div>
      )}
    </div>
  );
}
