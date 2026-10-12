import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { date, num } from "../lib/format";
import type { ApiKey } from "../lib/types";

function SecretModal({ secret, onClose }: { secret: string; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div role="dialog" aria-modal="true" aria-labelledby="secret-h" className="glass-strong popover w-full max-w-lg rounded-2xl p-5">
        <h2 id="secret-h" className="text-xs font-semibold tracking-[0.16em] text-slate-100">SECURE KEY — SHOWN ONCE</h2>
        <p className="mt-2 text-xs text-slate-400">Copy it now. Only a hash is stored; it cannot be shown again. Treat it like a password.</p>
        <code className="mt-3 block break-all rounded-lg bg-black/40 p-3 font-mono text-xs text-cyan-glow">{secret}</code>
        <div className="mt-4 flex gap-2">
          <button
            onClick={async () => {
              await navigator.clipboard.writeText(secret);
              setCopied(true);
            }}
            className="rounded-lg bg-cyan-glow/90 px-3 py-1.5 text-xs font-semibold text-obsidian"
          >
            {copied ? "Copied" : "Copy key"}
          </button>
          <button onClick={onClose} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-slate-200">I saved it</button>
        </div>
      </div>
    </div>
  );
}

export default function KeysView({ apiAllowed, onPlans }: { apiAllowed: boolean; onPlans: () => void }) {
  const [keys, setKeys] = useState<ApiKey[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [rate, setRate] = useState(30);
  const [secret, setSecret] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<{ id: number; action: "revoke" | "rotate" } | null>(null);
  const load = () => api.keys().then((r) => setKeys(r.keys), (e) => setError((e as Error).message));
  useEffect(() => {
    load();
  }, []);

  const run = async (fn: () => Promise<unknown>) => {
    setError(null);
    try {
      await fn();
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl px-4 pb-32 pt-16 sm:px-6">
        <section className="glass rounded-2xl p-5">
          <h2 className="text-[11px] font-semibold tracking-[0.2em] text-slate-400">AGENT API KEY MANAGEMENT</h2>
          <p className="mt-1 text-xs text-slate-500">
            Keys let your own agents (for example JARVIS or ASTON) call <code className="font-mono">POST /api/v1/generate</code> on this machine. Usage is metered against your credits.
          </p>
          {!apiAllowed && (
            <p className="mt-3 rounded-lg border border-amber-diag/30 bg-amber-diag/[0.06] p-2.5 text-xs text-amber-diag">
              Requests with your keys are refused until you have an active plan with API access and credits.{" "}
              <button onClick={onPlans} className="underline">View plans</button>
            </p>
          )}
          <form
            className="mt-4 flex flex-wrap items-end gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              run(async () => {
                const r = await api.createKey(name.trim(), rate);
                setSecret(r.key.secret);
                setName("");
              });
            }}
          >
            <label className="text-xs text-slate-400">
              Agent / key name
              <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={48} placeholder="JARVIS"
                className="mt-1 block w-56 rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-slate-100" />
            </label>
            <label className="text-xs text-slate-400">
              Requests / minute
              <input type="number" min={1} max={600} value={rate} onChange={(e) => setRate(Number(e.target.value))}
                className="mt-1 block w-28 rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-slate-100" />
            </label>
            <button className="rounded-lg bg-gradient-to-r from-cyan-glow to-electric px-4 py-2 text-xs font-semibold text-obsidian">GENERATE NEW KEY</button>
          </form>
          {error && <p role="alert" className="mt-3 text-xs text-red-300">{error}</p>}
        </section>

        <section className="glass mt-4 overflow-hidden rounded-2xl">
          {keys === null ? (
            <p className="p-5 text-sm text-slate-500">Loading…</p>
          ) : keys.length === 0 ? (
            <p className="p-5 text-sm text-slate-500">No keys yet. Agents appear here only after you create a key for them.</p>
          ) : (
            <div className="scroll-thin overflow-x-auto">
              <table className="w-full min-w-[720px] text-left text-sm">
                <thead className="text-xs text-slate-500">
                  <tr className="border-b border-white/5">
                    {["Key name", "Prefix", "Status", "Last used", "Requests", "Tokens (in/out)", "Limit", ""].map((h) => (
                      <th key={h} className="p-3 font-normal">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {keys.map((k) => (
                    <tr key={k.id} className="border-b border-white/5 last:border-0">
                      <td className="p-3 text-slate-100">{k.name}</td>
                      <td className="p-3 font-mono text-xs text-slate-400">{k.prefix}…</td>
                      <td className="p-3">
                        <span className={`rounded px-1.5 py-0.5 text-[10px] tracking-wider ${k.status === "active" ? "bg-emerald-400/10 text-emerald-300" : "bg-red-400/10 text-red-300"}`}>
                          {k.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="p-3 text-xs text-slate-400">{k.last_used_at ? date(k.last_used_at) : "never"}</td>
                      <td className="p-3 text-xs text-slate-300">{num(k.request_count)}{k.error_count ? ` (${k.error_count} refused)` : ""}</td>
                      <td className="p-3 text-xs text-slate-300">{num(k.tokens_in)} / {num(k.tokens_out)}</td>
                      <td className="p-3 text-xs text-slate-400">{k.rate_limit_per_min}/min</td>
                      <td className="p-3 text-right">
                        {k.status === "active" && (
                          <span className="flex justify-end gap-1">
                            <button onClick={() => setConfirm({ id: k.id, action: "rotate" })} className="rounded border border-white/10 px-2 py-1 text-[11px] text-slate-300 hover:text-white">Rotate</button>
                            <button onClick={() => setConfirm({ id: k.id, action: "revoke" })} className="rounded border border-red-300/30 px-2 py-1 text-[11px] text-red-300 hover:bg-red-400/10">Revoke</button>
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
      {secret && <SecretModal secret={secret} onClose={() => setSecret(null)} />}
      {confirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div role="alertdialog" aria-modal="true" className="glass-strong popover max-w-sm rounded-2xl p-5 text-sm text-slate-300">
            <p>
              {confirm.action === "revoke"
                ? "Revoke this key? Any agent using it will be rejected immediately. This cannot be undone."
                : "Rotate this key? A new secret is issued and the old one stops working immediately."}
            </p>
            <div className="mt-4 flex gap-2">
              <button
                onClick={() => {
                  const c = confirm;
                  setConfirm(null);
                  run(async () => {
                    if (c.action === "revoke") await api.revokeKey(c.id);
                    else setSecret((await api.rotateKey(c.id)).key.secret);
                  });
                }}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${confirm.action === "revoke" ? "bg-red-400/80 text-obsidian" : "bg-cyan-glow/90 text-obsidian"}`}
              >
                {confirm.action === "revoke" ? "Revoke key" : "Rotate key"}
              </button>
              <button onClick={() => setConfirm(null)} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs">Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
