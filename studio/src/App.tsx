import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import CommandDock from "./components/CommandDock";
import InfinityCore from "./components/InfinityCore";
import ModelStatus from "./components/ModelStatus";
import ProjectLauncher from "./components/ProjectLauncher";
import SettingsPopover, { loadSettings, saveSettings, type Settings } from "./components/SettingsPopover";
import TaskWorkspace, { PANELS, type PanelId } from "./components/TaskWorkspace";
import { api } from "./lib/api";
import { useStore } from "./lib/store";
import type { Checkpoint, Mode } from "./lib/types";

const ALL_PANELS: PanelId[] = ["task", "review", "files", "code", "stream"];

export default function App() {
  const store = useStore();
  const { status, connected } = store;
  const [mode, setMode] = useState<Mode>("code");
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [launcher, setLauncher] = useState(false);
  const [settings, setSettingsState] = useState<Settings>(loadSettings);
  const [open, setOpen] = useState<Set<PanelId>>(() => new Set(loadSettings().view === "workspace" ? ALL_PANELS : []));
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [sweep, setSweep] = useState(0);
  const spokenFor = useRef<string | null>(null);

  const setSettings = (s: Settings) => {
    setSettingsState(s);
    saveSettings(s);
    setOpen(new Set(s.view === "workspace" ? ALL_PANELS : []));
  };

  const loadModels = useCallback(() => {
    api.models().then((m) => setCheckpoints(m.checkpoints), () => setCheckpoints([]));
  }, []);
  useEffect(loadModels, [loadModels, connected]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "o") {
        e.preventDefault();
        setLauncher(true);
      } else if ((e.ctrlKey || e.metaKey) && e.key === ".") {
        e.preventDefault();
        setOpen((o) => (o.size === ALL_PANELS.length ? new Set() : new Set(ALL_PANELS)));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Optional: read the final summary aloud (speaking state is driven by the real utterance).
  useEffect(() => {
    const r = store.report;
    if (!settings.readAloud || !r || spokenFor.current === r.task_id || !("speechSynthesis" in window)) return;
    spokenFor.current = r.task_id;
    const u = new SpeechSynthesisUtterance(`${r.status.replace(/_/g, " ")}. ${r.summary}`);
    let decay: ReturnType<typeof setInterval> | undefined;
    u.onstart = () => {
      store.setSpeaking(true);
      decay = setInterval(() => (store.amplitude.current *= 0.82), 50);
    };
    u.onboundary = () => (store.amplitude.current = 0.9);
    u.onend = u.onerror = () => {
      clearInterval(decay);
      store.amplitude.current = 0;
      store.setSpeaking(false);
    };
    window.speechSynthesis.speak(u);
  }, [store.report, settings.readAloud, store]);

  const busy = status?.agent.busy ?? false;
  const disabledReason = !connected
    ? "INFINITY runtime is not reachable — start it with `infinity studio`"
    : !status?.workspace
      ? "Attach a project first (paperclip below, or Ctrl+O)"
      : status.model.state !== "ready"
        ? `INFINITY HYPER 5.5 unavailable: ${status.model.reason ?? "loading"}`
        : null;

  const submit = async (text: string) => {
    setNotice(null);
    try {
      store.clearTaskView();
      await api.startTask(text, mode);
      setOpen((o) => new Set([...o, "task", "stream"]));
    } catch (e) {
      setNotice((e as Error).message);
      throw e;
    }
  };

  const selectModel = async (path: string) => {
    try {
      await api.setModel(path);
      await store.refresh();
      setSweep((s) => s + 1);
    } catch (e) {
      setNotice((e as Error).message);
    }
  };

  const workspaceName = status?.workspace ? status.workspace.split("/").filter(Boolean).pop() ?? status.workspace : null;
  const workspaceOpen = open.size > 0 || store.approvals.length > 0;

  return (
    <div className="atmosphere relative flex h-full flex-col">
      {/* TOP EDGE */}
      <header className="relative z-40 flex items-center gap-3 px-4 pt-3 sm:px-5">
        <div className="flex items-center gap-2">
          <svg width="26" height="14" viewBox="0 0 64 32" aria-hidden>
            <path d="M32 16c-6-8-12-12-18-12a12 12 0 0 0 0 24c6 0 12-4 18-12s12-12 18-12a12 12 0 0 1 0 24c-6 0-12-4-18-12z" fill="none" stroke="#5ee7ff" strokeWidth="4" />
          </svg>
          <span className="hidden text-sm font-medium tracking-[0.2em] text-slate-200 sm:inline">INFINITY BOT</span>
        </div>
        <div className="flex flex-1 justify-center">
          <ModelStatus status={status} connected={connected} />
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`glass hidden items-center gap-1.5 rounded-xl px-2.5 py-1.5 text-[11px] tracking-wider sm:flex ${connected ? "text-emerald-300" : "text-red-300"}`}
            role="status"
          >
            <span className={`h-1.5 w-1.5 rounded-full ${connected ? "bg-emerald-300" : "bg-red-400"}`} />
            {connected ? "CONNECTED" : "OFFLINE"}
          </span>
          <button
            onClick={() => setOpen(open.size === ALL_PANELS.length ? new Set() : new Set(ALL_PANELS))}
            aria-pressed={open.size === ALL_PANELS.length}
            title="Toggle development workspace (Ctrl+.)"
            className={`glass rounded-xl p-2 ${workspaceOpen ? "text-cyan-glow" : "text-slate-300 hover:text-white"}`}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
              <rect x="3" y="4" width="7" height="16" rx="1.5" />
              <rect x="14" y="4" width="7" height="7" rx="1.5" />
              <rect x="14" y="13" width="7" height="7" rx="1.5" />
            </svg>
            <span className="sr-only">Workspace</span>
          </button>
          <SettingsPopover settings={settings} onChange={setSettings} />
        </div>
      </header>

      {/* CENTER */}
      <main className="relative z-10 flex flex-1 flex-col items-center justify-center px-4">
        <div className="relative">
          <InfinityCore
            state={store.core}
            amplitude={store.amplitude}
            compact={workspaceOpen}
            onActivate={() => setOpen((o) => new Set([...o, "task", "stream"]))}
          />
          <AnimatePresence>
            {sweep > 0 && (
              <motion.div
                key={sweep}
                aria-hidden
                initial={{ x: "-60%", opacity: 0 }}
                animate={{ x: "60%", opacity: [0, 0.7, 0] }}
                transition={{ duration: 1.1, ease: "easeInOut" }}
                className="pointer-events-none absolute inset-y-0 left-1/4 w-1/2 bg-gradient-to-r from-transparent via-cyan-glow/25 to-transparent blur-xl"
              />
            )}
          </AnimatePresence>
        </div>
        <p className="mt-1 flex items-center gap-2 text-sm tracking-[0.18em] text-slate-200" aria-live="polite">
          {busy && !store.approvals.length && (
            <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-cyan-glow/30 border-t-cyan-glow motion-reduce:animate-none" aria-hidden />
          )}
          {store.label}
        </p>
        {store.taskText && (busy || store.core === "success" || store.core === "error") && (
          <p className="mt-2 max-w-xl truncate text-center text-xs text-slate-400">TASK: {store.taskText}</p>
        )}
        {!connected && <p className="mt-3 text-xs text-red-300">Cannot reach the local INFINITY runtime. Run <code className="font-mono">infinity studio</code>.</p>}
      </main>

      {/* BOTTOM DOCK */}
      <footer className="relative z-40 flex flex-col items-center gap-2 px-3 pb-[max(12px,env(safe-area-inset-bottom))] sm:px-4">
        {notice && (
          <div role="alert" className="glass max-w-[760px] rounded-xl px-3 py-2 text-xs text-amber-diag">
            {notice}
            <button onClick={() => setNotice(null)} className="ml-3 text-slate-400 hover:text-white" aria-label="Dismiss">
              ✕
            </button>
          </div>
        )}
        {!workspaceOpen && store.taskText && (
          <div className="flex gap-1.5" aria-label="Open workspace panels">
            {PANELS.map((p) => (
              <button
                key={p.id}
                onClick={() => setOpen(new Set([p.id]))}
                className="glass flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-slate-400 hover:text-white"
              >
                {p.icon}
                {p.label}
              </button>
            ))}
          </div>
        )}
        <CommandDock
          mode={mode}
          onMode={setMode}
          onSubmit={submit}
          onCancel={() => api.cancel().catch((e) => setNotice((e as Error).message))}
          busy={busy}
          disabledReason={disabledReason}
          model={status?.model}
          checkpoints={checkpoints}
          onSelectModel={selectModel}
          onAttach={() => setLauncher(true)}
          workspaceName={workspaceName}
          taskText={store.taskText}
          onListening={store.setListening}
          amplitude={store.amplitude}
        />
      </footer>

      <TaskWorkspace open={open} setOpen={setOpen} selectedFile={selectedFile} setSelectedFile={setSelectedFile} />

      {launcher && (
        <ProjectLauncher
          current={status?.workspace ?? null}
          onClose={() => setLauncher(false)}
          onOpened={() => {
            setSweep((s) => s + 1);
            setSelectedFile(null);
            store.refresh();
          }}
        />
      )}
    </div>
  );
}
