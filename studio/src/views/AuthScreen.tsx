import { motion } from "framer-motion";
import { useEffect, useState, type FormEvent } from "react";
import { api } from "../lib/api";

export default function AuthScreen({ onAuthed }: { onAuthed: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [setup, setSetup] = useState<{ owner_exists: boolean; registration: boolean } | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    api.setup().then(setSetup, () => setOffline(true));
  }, []);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "register") await api.register(email, password);
      await api.login(email, password);
      onAuthed();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="atmosphere flex h-full items-center justify-center p-4">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass-strong w-full max-w-sm rounded-2xl p-6">
        <div className="mb-5 flex flex-col items-center gap-2">
          <svg width="56" height="28" viewBox="0 0 64 32" aria-hidden>
            <path d="M32 16c-6-8-12-12-18-12a12 12 0 0 0 0 24c6 0 12-4 18-12s12-12 18-12a12 12 0 0 1 0 24c-6 0-12-4-18-12z" fill="none" stroke="#5ee7ff" strokeWidth="3" />
          </svg>
          <h1 className="text-sm font-medium tracking-[0.25em] text-slate-100">INFINITY BOT</h1>
          <p className="text-[11px] tracking-[0.18em] text-slate-500">INFINITY HYPER 5.5</p>
        </div>
        {offline ? (
          <p className="text-center text-sm text-red-300">
            Cannot reach the local INFINITY runtime. Start it with <code className="font-mono">infinity studio</code>.
          </p>
        ) : setup && !setup.owner_exists ? (
          <div className="space-y-2 text-sm text-slate-300">
            <p>No owner account exists yet. Create it on the machine running INFINITY BOT:</p>
            <pre className="rounded-lg bg-black/40 p-3 font-mono text-xs text-cyan-glow">infinity admin create-owner --email you@example.com</pre>
            <p className="text-xs text-slate-500">You will be asked for a password there. No default account exists.</p>
            <button onClick={() => api.setup().then(setSetup)} className="mt-2 w-full rounded-xl border border-white/10 py-2 text-xs text-slate-200 hover:bg-white/5">
              I created it, continue
            </button>
          </div>
        ) : (
          <form onSubmit={submit} className="space-y-3">
            <div role="tablist" className="grid grid-cols-2 gap-1 rounded-xl bg-black/30 p-1 text-xs">
              {(["login", "register"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  role="tab"
                  aria-selected={mode === m}
                  disabled={m === "register" && setup?.registration === false}
                  onClick={() => setMode(m)}
                  className={`rounded-lg py-1.5 ${mode === m ? "bg-cyan-glow/15 text-cyan-glow" : "text-slate-400"} disabled:opacity-30`}
                >
                  {m === "login" ? "Sign in" : "Create account"}
                </button>
              ))}
            </div>
            <label className="block text-xs text-slate-400">
              E-mail
              <input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)}
                className="mt-1 w-full rounded-xl border border-white/10 bg-black/30 px-3 py-2.5 text-sm text-slate-100 focus:border-cyan-glow/40 focus:outline-none" />
            </label>
            <label className="block text-xs text-slate-400">
              Password {mode === "register" && <span className="text-slate-600">(at least 10 characters)</span>}
              <input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} required minLength={mode === "register" ? 10 : 1}
                value={password} onChange={(e) => setPassword(e.target.value)}
                className="mt-1 w-full rounded-xl border border-white/10 bg-black/30 px-3 py-2.5 text-sm text-slate-100 focus:border-cyan-glow/40 focus:outline-none" />
            </label>
            {error && <p role="alert" className="text-xs text-red-300">{error}</p>}
            <button disabled={busy} className="w-full rounded-xl bg-gradient-to-r from-cyan-glow to-electric py-2.5 text-sm font-semibold text-obsidian disabled:opacity-40">
              {busy ? "…" : mode === "login" ? "Sign in" : "Create account"}
            </button>
          </form>
        )}
      </motion.div>
    </div>
  );
}
