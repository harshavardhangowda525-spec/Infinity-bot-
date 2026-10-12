import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { allowance, inr } from "../lib/format";
import type { Plan } from "../lib/types";
import { ConfirmDialog } from "../views/PlansView";

/** Shown when a user without COOK access tries to use it. The server enforces the same rule. */
export default function UpgradeOverlay({ reason, onClose, onComparePlans }: { reason: string; onClose: () => void; onComparePlans: () => void }) {
  const [plans, setPlans] = useState<Plan[] | null>(null);
  const [selected, setSelected] = useState<Plan | null>(null);
  useEffect(() => {
    api.plans().then((c) => setPlans(c.plans.filter((p) => p.paid)), () => setPlans([]));
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", esc);
    return () => document.removeEventListener("keydown", esc);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <motion.div role="dialog" aria-modal="true" aria-labelledby="upgrade-h" initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}
        className="glass-strong popover w-full max-w-2xl rounded-2xl p-6">
        <div className="flex items-start">
          <div>
            <h2 id="upgrade-h" className="text-sm font-semibold tracking-[0.2em] text-cyan-glow">UNLOCK COOK</h2>
            <p className="mt-2 max-w-lg text-sm text-slate-300">
              Unlock COOK to build applications, edit project files, run tests, and debug code with your AI coding agent.
            </p>
            <p className="mt-1 text-xs text-amber-diag/90">{reason}.</p>
          </div>
          <button onClick={onClose} aria-label="Close" className="ml-auto rounded p-1 text-slate-400 hover:text-white">✕</button>
        </div>
        <div className="mt-5 grid gap-3 sm:grid-cols-3">
          {plans === null && <p className="text-sm text-slate-500">Loading plans…</p>}
          {plans?.map((p) => (
            <div key={p.id} className={`rounded-xl border p-4 ${p.popular ? "border-cyan-glow/40 shadow-glow" : "border-white/10"}`}>
              <div className="text-xs font-semibold tracking-[0.2em] text-slate-100">{p.name}</div>
              <div className="mt-2 text-2xl font-semibold text-white">{inr(p.monthly_price_inr)}<span className="text-xs font-normal text-slate-400"> /month</span></div>
              <div className="mt-1 text-[11px] text-slate-500">{allowance(p.monthly_token_credits)}</div>
              <button onClick={() => setSelected(p)} disabled={!p.accepting_new}
                className={`mt-3 w-full rounded-lg py-2 text-xs font-semibold disabled:opacity-30 ${p.popular ? "bg-gradient-to-r from-cyan-glow to-electric text-obsidian" : "border border-white/15 text-slate-100"}`}>
                Choose plan
              </button>
            </div>
          ))}
        </div>
        <div className="mt-4 flex">
          <button onClick={onComparePlans} className="text-xs text-slate-400 underline hover:text-white">Compare plans in detail</button>
          <button onClick={onClose} className="ml-auto rounded-lg border border-white/10 px-3 py-1.5 text-xs text-slate-200">Back</button>
        </div>
      </motion.div>
      {selected && <ConfirmDialog plan={selected} period="monthly" onClose={() => setSelected(null)} />}
    </div>
  );
}
