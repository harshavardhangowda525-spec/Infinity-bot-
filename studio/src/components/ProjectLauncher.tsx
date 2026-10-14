import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { Permission, Project, RecentProject } from "../lib/types";

const PERMS: { id: Permission; label: string; hint: string }[] = [
  { id: "read-only", label: "Read-only", hint: "Inspect only. Default for unfamiliar projects." },
  { id: "ask", label: "Ask", hint: "Every edit and non-trivial command asks you first." },
  { id: "auto", label: "Auto", hint: "Edits and checks run freely; risky commands still ask." },
];

export default function ProjectLauncher({ onClose, onOpened, current }: {
  onClose: () => void;
  onOpened: (p: Project) => void;
  current: string | null;
}) {
  const [recent, setRecent] = useState<RecentProject[] | null>(null);
  const [path, setPath] = useState(current ?? "");
  const [perm, setPerm] = useState<Permission>("read-only");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [project, setProject] = useState<Project | null>(null);

  useEffect(() => {
    api.projects().then((r) => setRecent(r.recent), (e) => setError((e as Error).message));
    if (current) api.project().then(setProject, () => undefined);
  }, [current]);

  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", esc);
    return () => document.removeEventListener("keydown", esc);
  }, [onClose]);

  const open = async (p: string, permission?: Permission) => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.setWorkspace(p, permission);
      setProject({ ...r.project, path: r.path, permission: r.permission });
      onOpened({ ...r.project, path: r.path, permission: r.permission });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-slate-900/40 p-4" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <motion.div
        role="dialog"
        aria-modal="true"
        aria-label="Project launcher"
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        className="glass-strong w-full max-w-xl rounded-2xl p-5"
      >
        <div className="mb-4 flex items-center">
          <h2 className="text-sm font-semibold tracking-[0.14em] text-fg">PROJECT</h2>
          <button onClick={onClose} aria-label="Close" className="ml-auto rounded p-1 text-muted hover:text-fg">
            ✕
          </button>
        </div>
        <label htmlFor="ws-path" className="text-xs text-muted">
          Local folder (absolute path on the machine running INFINITY BOT)
        </label>
        <input
          id="ws-path"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && path && open(path, perm)}
          placeholder="/home/you/code/my-app"
          className="mt-1 w-full rounded-xl border border-line bg-surface-2 px-3 py-2.5 font-mono text-sm text-fg focus:border-cyan-glow/40 focus:outline-none"
        />
        <div role="radiogroup" aria-label="Permission" className="mt-3 grid grid-cols-3 gap-2">
          {PERMS.map((p) => (
            <button
              key={p.id}
              role="radio"
              aria-checked={perm === p.id}
              onClick={() => setPerm(p.id)}
              title={p.hint}
              className={`rounded-xl border px-3 py-2 text-left text-xs ${perm === p.id ? "border-cyan-glow/50 bg-cyan-glow/10 text-fg" : "border-line text-muted hover:text-fg"}`}
            >
              <div className="font-semibold">{p.label}</div>
              <div className="mt-0.5 text-[11px] text-muted">{p.hint}</div>
            </button>
          ))}
        </div>
        <p className="mt-2 text-[11px] text-muted">
          To create a new project, enter a folder that already exists (an empty one is fine). INFINITY BOT never creates or deletes project roots on its own.
        </p>
        <button
          disabled={!path || busy}
          onClick={() => open(path, perm)}
          className="mt-3 w-full rounded-xl bg-accent py-2.5 text-sm font-semibold text-white disabled:opacity-30"
        >
          {busy ? "Opening…" : "Open project"}
        </button>
        {error && <p className="mt-2 text-xs text-bad">{error}</p>}

        {project && (
          <div className="mt-4 rounded-xl border border-line p-3 text-xs">
            <div className="font-mono text-fg">{project.path}</div>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {[...project.languages, ...project.frameworks, ...project.package_managers].map((t) => (
                <span key={t} className="rounded border border-line px-1.5 py-0.5 text-fg/80">
                  {t}
                </span>
              ))}
              {project.git && <span className="rounded border border-line px-1.5 py-0.5 text-fg/80">git</span>}
            </div>
            <div className="mt-1.5 text-muted">
              {project.file_count}
              {project.file_count_capped ? "+" : ""} files · permission: {project.permission}
              {project.test_commands.length ? ` · checks: ${project.test_commands.join(", ")}` : ""}
            </div>
          </div>
        )}

        <h3 className="mb-1.5 mt-5 text-[11px] font-semibold tracking-[0.14em] text-muted">RECENT</h3>
        {recent === null ? (
          <p className="text-xs text-muted">Loading…</p>
        ) : recent.length === 0 ? (
          <p className="text-xs text-muted">No recent projects yet.</p>
        ) : (
          <ul className="scroll-thin max-h-52 space-y-1 overflow-auto">
            {recent.map((r) => (
              <li key={r.path}>
                <button
                  disabled={!r.exists || busy}
                  onClick={() => open(r.path)}
                  className="w-full rounded-lg px-2.5 py-2 text-left hover:bg-hover disabled:opacity-40"
                >
                  <div className="flex items-center gap-2 text-sm text-fg">
                    {r.name}
                    <span className="text-[11px] text-muted">{r.trust ?? "read-only"}</span>
                    {!r.exists && <span className="text-[11px] text-bad">missing</span>}
                  </div>
                  <div className="truncate font-mono text-[11px] text-muted">{r.path}</div>
                  {r.last_task && (
                    <div className="truncate text-[11px] text-muted">
                      last task: {r.last_task.task} — {r.last_task.status}
                    </div>
                  )}
                </button>
              </li>
            ))}
          </ul>
        )}
      </motion.div>
    </div>
  );
}
