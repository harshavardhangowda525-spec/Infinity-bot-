import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { date, num } from "../lib/format";
import type { ApiKey } from "../lib/types";

function SecretModal({ secret, onClose }: { secret: string; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4">
      <div role="dialog" aria-modal="true" aria-labelledby="secret-h" className="glass-strong popover w-full max-w-lg rounded-2xl p-5">
        <h2 id="secret-h" className="text-xs font-semibold tracking-[0.16em] text-fg">SECURE KEY — SHOWN ONCE</h2>
        <p className="mt-2 text-xs text-muted">Copy it now. Only a hash is stored; it cannot be shown again. Treat it like a password.</p>
        <code className="mt-3 block break-all rounded-lg bg-surface-2 p-3 font-mono text-xs text-cyan-glow">{secret}</code>
        <div className="mt-4 flex gap-2">
          <button
            onClick={async () => {
              await navigator.clipboard.writeText(secret);
              setCopied(true);
            }}
            className="rounded-lg bg-cyan-glow/90 px-3 py-1.5 text-xs font-semibold text-white"
          >
            {copied ? "Copied" : "Copy key"}
          </button>
          <button onClick={onClose} className="rounded-lg border border-line px-3 py-1.5 text-xs text-fg">I saved it</button>
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
      <div className="mx-auto max-w-5xl px-4 pb-16 pt-6 sm:px-6">
        <section className="glass rounded-2xl p-5">
          <h2 className="text-[11px] font-semibold tracking-[0.2em] text-muted">AGENT API KEY MANAGEMENT</h2>
          <p className="mt-1 text-xs text-muted">
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
            <label className="text-xs text-muted">
              Agent / key name
              <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={48} placeholder="JARVIS"
                className="mt-1 block w-56 rounded-lg border border-line bg-surface-2 px-3 py-2 text-sm text-fg" />
            </label>
            <label className="text-xs text-muted">
              Requests / minute
              <input type="number" min={1} max={600} value={rate} onChange={(e) => setRate(Number(e.target.value))}
                className="mt-1 block w-28 rounded-lg border border-line bg-surface-2 px-3 py-2 text-sm text-fg" />
            </label>
            <button className="rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-white">GENERATE NEW KEY</button>
          </form>
          {error && <p role="alert" className="mt-3 text-xs text-bad">{error}</p>}
        </section>

        <section className="glass mt-4 overflow-hidden rounded-2xl">
          {keys === null ? (
            <p className="p-5 text-sm text-muted">Loading…</p>
          ) : keys.length === 0 ? (
            <p className="p-5 text-sm text-muted">No keys yet. Agents appear here only after you create a key for them.</p>
          ) : (
            <div className="scroll-thin overflow-x-auto">
              <table className="w-full min-w-[720px] text-left text-sm">
                <thead className="text-xs text-muted">
                  <tr className="border-b border-line">
                    {["Key name", "Prefix", "Status", "Last used", "Requests", "Tokens (in/out)", "Limit", ""].map((h) => (
                      <th key={h} className="p-3 font-normal">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {keys.map((k) => (
                    <tr key={k.id} className="border-b border-line last:border-0">
                      <td className="p-3 text-fg">{k.name}</td>
                      <td className="p-3 font-mono text-xs text-muted">{k.prefix}…</td>
                      <td className="p-3">
                        <span className={`rounded px-1.5 py-0.5 text-[10px] tracking-wider ${k.status === "active" ? "bg-ok/10 text-ok" : "bg-bad/10 text-bad"}`}>
                          {k.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="p-3 text-xs text-muted">{k.last_used_at ? date(k.last_used_at) : "never"}</td>
                      <td className="p-3 text-xs text-fg/80">{num(k.request_count)}{k.error_count ? ` (${k.error_count} refused)` : ""}</td>
                      <td className="p-3 text-xs text-fg/80">{num(k.tokens_in)} / {num(k.tokens_out)}</td>
                      <td className="p-3 text-xs text-muted">{k.rate_limit_per_min}/min</td>
                      <td className="p-3 text-right">
                        {k.status === "active" && (
                          <span className="flex justify-end gap-1">
                            <button onClick={() => setConfirm({ id: k.id, action: "rotate" })} className="rounded border border-line px-2 py-1 text-[11px] text-fg/80 hover:text-fg">Rotate</button>
                            <button onClick={() => setConfirm({ id: k.id, action: "revoke" })} className="rounded border border-bad/30 px-2 py-1 text-[11px] text-bad hover:bg-bad/10">Revoke</button>
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
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4">
          <div role="alertdialog" aria-modal="true" className="glass-strong popover max-w-sm rounded-2xl p-5 text-sm text-fg/80">
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
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${confirm.action === "revoke" ? "bg-bad/80 text-white" : "bg-cyan-glow/90 text-white"}`}
              >
                {confirm.action === "revoke" ? "Revoke key" : "Rotate key"}
              </button>
              <button onClick={() => setConfirm(null)} className="rounded-lg border border-line px-3 py-1.5 text-xs">Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
