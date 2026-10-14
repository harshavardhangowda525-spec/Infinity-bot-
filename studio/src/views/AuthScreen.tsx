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
    <div className="flex h-full items-center justify-center bg-bg p-4">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="card w-full max-w-sm rounded-2xl p-7">
        <div className="mb-5 flex flex-col items-center gap-2">
          <svg width="56" height="28" viewBox="0 0 64 32" aria-hidden>
            <path d="M32 16c-6-8-12-12-18-12a12 12 0 0 0 0 24c6 0 12-4 18-12s12-12 18-12a12 12 0 0 1 0 24c-6 0-12-4-18-12z" fill="none" stroke="currentColor" strokeWidth="4" className="text-accent" />
          </svg>
          <h1 className="text-lg font-semibold text-fg">Welcome to Infinity Bot</h1>
          <p className="text-sm text-muted">Sign in to continue</p>
        </div>
        {offline ? (
          <p className="text-center text-sm text-bad">
            Cannot reach the local INFINITY runtime. Start it with <code className="font-mono">infinity studio</code>.
          </p>
        ) : setup && !setup.owner_exists ? (
          <div className="space-y-2 text-sm text-fg/80">
            <p>No owner account exists yet. Create it on the machine running INFINITY BOT:</p>
            <pre className="rounded-lg bg-surface-2 p-3 font-mono text-xs text-fg">infinity admin create-owner --email you@example.com</pre>
            <p className="text-xs text-muted">You will be asked for a password there. No default account exists.</p>
            <button onClick={() => api.setup().then(setSetup)} className="mt-2 w-full rounded-xl border border-line py-2 text-xs text-fg hover:bg-hover">
              I created it, continue
            </button>
          </div>
        ) : (
          <form onSubmit={submit} className="space-y-3">
            <div role="tablist" className="grid grid-cols-2 gap-1 rounded-xl bg-surface-2 p-1 text-xs">
              {(["login", "register"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  role="tab"
                  aria-selected={mode === m}
                  disabled={m === "register" && setup?.registration === false}
                  onClick={() => setMode(m)}
                  className={`rounded-lg py-1.5 ${mode === m ? "bg-surface font-medium text-fg shadow-sm" : "text-muted"} disabled:opacity-30`}
                >
                  {m === "login" ? "Sign in" : "Create account"}
                </button>
              ))}
            </div>
            <label className="block text-xs text-muted">
              E-mail
              <input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)}
                className="input mt-1 py-2.5" />
            </label>
            <label className="block text-xs text-muted">
              Password {mode === "register" && <span className="text-muted/70">(at least 10 characters)</span>}
              <input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} required minLength={mode === "register" ? 10 : 1}
                value={password} onChange={(e) => setPassword(e.target.value)}
                className="input mt-1 py-2.5" />
            </label>
            {error && <p role="alert" className="text-xs text-bad">{error}</p>}
            <button disabled={busy} className="btn-primary w-full py-2.5">
              {busy ? "…" : mode === "login" ? "Sign in" : "Create account"}
            </button>
          </form>
        )}
      </motion.div>
    </div>
  );
}
