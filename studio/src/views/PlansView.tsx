import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { allowance, inr } from "../lib/format";
import type { Catalog, CheckoutResult, Entitlement, Plan } from "../lib/types";

const POLICIES: Record<string, string> = {
  "Subscription terms": "subscription terms",
  "Privacy policy": "privacy policy",
  "Refund & cancellation": "refund and cancellation policy",
  "Help & support": "support contact",
};

function Check() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" className="mt-0.5 shrink-0 text-cyan-glow" aria-hidden>
      <path d="M5 12l5 5 9-11" />
    </svg>
  );
}

export function ConfirmDialog({ plan, period, onClose }: { plan: Plan; period: "monthly"; onClose: () => void }) {
  const [state, setState] = useState<"confirm" | "working" | "done" | "error">("confirm");
  const [result, setResult] = useState<CheckoutResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const first = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    first.current?.focus();
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", esc);
    return () => document.removeEventListener("keydown", esc);
  }, [onClose]);
  const confirm = async () => {
    setState("working");
    try {
      setResult(await api.checkout(plan.id, period));
      setState("done");
    } catch (e) {
      setError((e as Error).message);
      setState("error");
    }
  };
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <motion.div role="dialog" aria-modal="true" aria-labelledby="confirm-title" initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}
        className="glass-strong popover w-full max-w-md rounded-2xl p-6">
        <h2 id="confirm-title" className="text-sm font-semibold tracking-[0.16em] text-fg">CONFIRM {plan.name}</h2>
        <dl className="mt-4 space-y-2 text-sm">
          <div className="flex justify-between"><dt className="text-muted">Plan</dt><dd className="text-fg">{plan.name}</dd></div>
          <div className="flex justify-between"><dt className="text-muted">Price</dt><dd className="text-fg">{inr(plan.monthly_price_inr)} / month</dd></div>
          <div className="flex justify-between"><dt className="text-muted">Billing period</dt><dd className="text-fg">Monthly</dd></div>
          <div className="flex justify-between"><dt className="text-muted">Included credits</dt><dd className="text-fg">{allowance(plan.monthly_token_credits)}</dd></div>
        </dl>
        {state === "done" && result && (
          <div role="status" className={`mt-4 rounded-xl border p-3 text-xs ${result.activated ? "border-ok/30 text-ok" : "border-amber-diag/40 bg-amber-diag/[0.06] text-amber-diag"}`}>
            {result.mode === "development" && <div className="mb-1 font-semibold tracking-wider">DEVELOPMENT MODE</div>}
            {result.message}
          </div>
        )}
        {state === "error" && <p role="alert" className="mt-4 text-xs text-bad">{error}</p>}
        <div className="mt-5 flex gap-2">
          {state !== "done" ? (
            <>
              <button ref={first} onClick={onClose} className="flex-1 rounded-xl border border-line py-2.5 text-sm text-fg hover:bg-hover">Back</button>
              <button onClick={confirm} disabled={state === "working"} className="flex-1 rounded-xl bg-accent py-2.5 text-sm font-semibold text-white disabled:opacity-40">
                {state === "working" ? "…" : "Continue to checkout"}
              </button>
            </>
          ) : (
            <button ref={first} onClick={onClose} className="flex-1 rounded-xl border border-line py-2.5 text-sm text-fg hover:bg-hover">Close</button>
          )}
        </div>
      </motion.div>
    </div>
  );
}

export default function PlansView({ entitlement, onBack }: { entitlement: Entitlement | null; onBack: () => void }) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [period, setPeriod] = useState<"monthly" | "yearly">("monthly");
  const [selected, setSelected] = useState<Plan | null>(null);
  const [policy, setPolicy] = useState<string | null>(null);

  useEffect(() => {
    api.plans().then(setCatalog, (e) => setError((e as Error).message));
  }, []);

  const paid = catalog?.plans.filter((p) => p.paid) ?? [];
  const yearly = catalog?.yearly_billing.enabled ?? false;
  const current = entitlement?.subscription?.plan ?? null;

  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl px-4 pb-16 pt-6 sm:px-6">
        <header className="flex items-center gap-3">
          <button onClick={onBack} className="btn-secondary px-3 py-1.5 text-xs" aria-label="Back to chat">← Back</button>
        </header>

        <section className="relative mt-6 flex flex-col items-center text-center">
          <h1 className="text-3xl font-semibold tracking-tight text-fg sm:text-4xl">Plans</h1>
          <p className="mt-2 text-sm text-muted sm:text-base">Normal chat is included for every account. Cook, API access and higher limits come with a plan.</p>
          <div role="radiogroup" aria-label="Billing period" className="glass relative mt-6 flex items-center gap-1 rounded-xl p-1 text-xs">
            <button role="radio" aria-checked={period === "monthly"} onClick={() => setPeriod("monthly")}
              className={`rounded-lg px-4 py-1.5 ${period === "monthly" ? "bg-cyan-glow/15 text-cyan-glow" : "text-muted"}`}>Monthly</button>
            <button role="radio" aria-checked={period === "yearly"} disabled={!yearly} title={catalog?.yearly_billing.note}
              className="rounded-lg px-4 py-1.5 text-muted disabled:cursor-not-allowed">Yearly <span className="ml-1 text-[10px] text-muted/70">unavailable</span></button>
          </div>
          {!yearly && catalog && <p className="relative mt-2 text-[11px] text-muted">{catalog.yearly_billing.note}</p>}
        </section>

        {error && <p role="alert" className="mt-8 text-center text-sm text-bad">{error}</p>}
        {!catalog && !error && <p className="mt-8 text-center text-sm text-muted">Loading plans…</p>}

        <section className="mt-10 grid gap-5 md:grid-cols-3" aria-label="Plans">
          {paid.map((p, i) => (
            <motion.article
              key={p.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06 }}
              className={`glass relative flex flex-col rounded-2xl p-6 ${p.popular ? "border-cyan-glow/40 shadow-sm md:-translate-y-2" : ""}`}
            >
              {p.popular && (
                <span className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-full border border-cyan-glow/40 bg-surface px-3 py-0.5 text-[10px] tracking-[0.2em] text-cyan-glow">MOST POPULAR</span>
              )}
              <h2 className="text-sm font-semibold tracking-[0.25em] text-fg">{p.name}</h2>
              <p className="mt-1 text-xs text-muted">{p.tagline}</p>
              <AnimatePresence mode="wait">
                <motion.p key={period} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} className="mt-5">
                  <span className="text-4xl font-semibold text-fg">{inr(p.monthly_price_inr)}</span>
                  <span className="ml-1 text-sm text-muted">/ month</span>
                </motion.p>
              </AnimatePresence>
              <p className="mt-1 text-[11px] text-muted">Billed monthly</p>
              <p className="mt-4 rounded-lg bg-hover px-3 py-2 text-xs text-fg/80">
                Token allowance: <span className={p.monthly_token_credits === null ? "text-muted" : "text-fg"}>{allowance(p.monthly_token_credits)}</span>
              </p>
              <ul className="mt-4 flex-1 space-y-2 text-sm text-fg/80">
                {p.features.map((f) => (
                  <li key={f} className="flex gap-2"><Check />{f}</li>
                ))}
              </ul>
              <button
                onClick={() => setSelected(p)}
                disabled={!p.accepting_new || current === p.id}
                className={`mt-6 rounded-xl py-2.5 text-sm font-semibold tracking-wider transition disabled:cursor-not-allowed disabled:opacity-40 ${
                  p.popular ? "bg-accent text-white" : "border border-line text-fg hover:border-cyan-glow/40"
                }`}
              >
                {current === p.id ? "CURRENT PLAN" : !p.accepting_new ? "NOT AVAILABLE" : p.cta}
              </button>
            </motion.article>
          ))}
        </section>

        {paid.length > 0 && (
          <section className="mt-14" aria-labelledby="compare">
            <h2 id="compare" className="mb-3 text-[11px] font-semibold tracking-[0.2em] text-muted">COMPARE PLANS</h2>
            <div className="glass scroll-thin overflow-x-auto rounded-2xl">
              <table className="w-full min-w-[560px] text-left text-sm">
                <thead>
                  <tr className="border-b border-line text-xs text-muted">
                    <th className="p-3 font-normal"> </th>
                    {paid.map((p) => <th key={p.id} scope="col" className={`p-3 font-semibold tracking-[0.15em] ${p.popular ? "text-cyan-glow" : "text-fg"}`}>{p.name}</th>)}
                  </tr>
                </thead>
                <tbody className="text-fg/80">
                  {[
                    ["Monthly price", (p: Plan) => inr(p.monthly_price_inr)],
                    ["Included token credits", (p: Plan) => allowance(p.monthly_token_credits)],
                    ["COOK coding agent", (p: Plan) => (p.cook ? "Included" : "—")],
                    ["Task step limit", (p: Plan) => (p.max_task_steps === null ? "Not configured yet" : String(p.max_task_steps))],
                    ["API rate limit", (p: Plan) => (p.requests_per_minute === null ? "Not configured yet" : `${p.requests_per_minute}/min`)],
                    ["Agent API keys", (p: Plan) => (p.api_access ? "Included" : "—")],
                    ["Response priority", (p: Plan) => p.priority],
                    ["Billing period", () => "Monthly"],
                  ].map(([label, fn]) => (
                    <tr key={label as string} className="border-b border-line last:border-0">
                      <th scope="row" className="p-3 text-xs font-normal text-muted">{label as string}</th>
                      {paid.map((p) => <td key={p.id} className="p-3">{(fn as (p: Plan) => string)(p)}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {catalog && !catalog.payment_provider.configured && (
              <p className="mt-3 text-center text-[11px] text-amber-diag/80">Development mode: no payment provider is configured. Checkout will not charge you or activate a plan.</p>
            )}
          </section>
        )}

        <footer className="mt-14 flex flex-wrap justify-center gap-x-6 gap-y-2 text-[11px] text-muted">
          {Object.keys(POLICIES).map((k) => (
            <button key={k} onClick={() => setPolicy(k)} className="hover:text-fg">{k}</button>
          ))}
        </footer>
      </div>

      {selected && <ConfirmDialog plan={selected} period="monthly" onClose={() => setSelected(null)} />}
      {policy && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4" onMouseDown={(e) => e.target === e.currentTarget && setPolicy(null)}>
          <div role="dialog" aria-modal="true" aria-label={policy} className="glass-strong popover max-w-sm rounded-2xl p-5 text-sm text-fg/80">
            <h2 className="mb-2 text-xs font-semibold tracking-[0.16em] text-fg">{policy.toUpperCase()}</h2>
            <p>The platform owner has not published a {POLICIES[policy]} yet.</p>
            <button onClick={() => setPolicy(null)} className="mt-4 rounded-lg border border-line px-3 py-1.5 text-xs">Close</button>
          </div>
        </div>
      )}
    </div>
  );
}
