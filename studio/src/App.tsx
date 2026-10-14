import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import ChatPanel from "./components/ChatPanel";
import CommandDock from "./components/CommandDock";
import InfinityCore from "./components/InfinityCore";
import ModelStatus from "./components/ModelStatus";
import ProjectLauncher from "./components/ProjectLauncher";
import SettingsPopover, { loadSettings, saveSettings, type Settings } from "./components/SettingsPopover";
import TaskWorkspace, { PANELS, type PanelId } from "./components/TaskWorkspace";
import BottomNav, { type View } from "./components/BottomNav";
import UpgradeOverlay from "./components/UpgradeOverlay";
import { api } from "./lib/api";
import { modelName, num } from "./lib/format";
import { useStore } from "./lib/store";
import type { ChatTurn, Checkpoint, DockMode, Me, OpenModel } from "./lib/types";
import AdminView from "./views/AdminView";
import IntelligenceView from "./views/IntelligenceView";
import KeysView from "./views/KeysView";
import PlansView from "./views/PlansView";
import TokensView from "./views/TokensView";

const ALL_PANELS: PanelId[] = ["task", "review", "files", "code", "stream"];

export default function App({ me, refreshMe, onLoggedOut }: { me: Me; refreshMe: () => void; onLoggedOut: () => void }) {
  const store = useStore();
  const [view, setView] = useState<View>("home");
  const [upgrade, setUpgrade] = useState(false);
  const { status, connected } = store;
  // Accounts without COOK start in Normal chat, which every signed-in account can use.
  const [mode, setMode] = useState<DockMode>(me.entitlement.cook ? "code" : "chat");
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [openModels, setOpenModels] = useState<OpenModel[]>([]);
  const [chat, setChat] = useState<ChatTurn[]>([]);
  const [chatWaiting, setChatWaiting] = useState(false);
  const [switching, setSwitching] = useState(false);
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
    api.models().then(
      (m) => {
        setCheckpoints(m.checkpoints);
        setOpenModels(m.open_models ?? []);
      },
      () => setCheckpoints([]),
    );
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

  const ent = (status as (typeof status & { entitlement?: Me["entitlement"] }) | null)?.entitlement ?? me.entitlement;
  const isOwner = me.user.role === "owner";
  const busy = status?.agent.busy ?? false;
  const isChat = mode === "chat";
  const disabledReason = switching
    ? "Switching model…"
    : isChat
    ? !connected
      ? "INFINITY runtime is not reachable — start it with `infinity studio`"
      : status?.model.state !== "ready"
        ? `No model loaded: ${status?.model.reason ?? "loading"}`
        : null
    : !ent.cook
    ? null // gate shown as a hint; submitting opens the upgrade screen (the server enforces it too)
    : !connected
    ? "INFINITY runtime is not reachable — start it with `infinity studio`"
    : !status?.workspace
      ? "Attach a project first (paperclip below, or Ctrl+O)"
      : status.model.state !== "ready"
        ? `Model unavailable: ${status.model.reason ?? "loading"}`
        : null;

  const sendChat = async (text: string) => {
    const history: ChatTurn[] = [...chat, { role: "user", content: text }];
    setChat(history);
    setChatWaiting(true);
    try {
      const r = await api.chat(history);
      setChat([...history, { role: "assistant", content: r.reply, model: r.model.product }]);
    } catch (e) {
      setNotice((e as Error).message);
    } finally {
      setChatWaiting(false);
    }
  };

  const submit = async (text: string) => {
    setNotice(null);
    if (mode === "chat") return sendChat(text);
    if (!ent.cook) {
      setUpgrade(true);
      return;
    }
    try {
      store.clearTaskView();
      await api.startTask(text, mode);
      setOpen((o) => new Set([...o, "task", "stream"]));
    } catch (e) {
      setNotice((e as Error).message);
      throw e;
    }
  };

  const selectModel = async (value: string) => {
    if (value.startsWith("open:")) setNotice("Loading the open model… the first time, it is downloaded (about 1 GB). This can take a few minutes.");
    setSwitching(true);
    try {
      await api.setModel(value);
      setNotice(null);
      await store.refresh();
      setSweep((s) => s + 1);
    } catch (e) {
      setNotice((e as Error).message);
    } finally {
      setSwitching(false);
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
        <div className="hidden lg:block">
          <BottomNav view={view} onView={setView} isOwner={isOwner} email={me.user.email}
            onLogout={() => api.logout().finally(onLoggedOut)} />
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`glass hidden items-center gap-1.5 rounded-xl px-2.5 py-1.5 text-[11px] tracking-wider sm:flex ${connected ? "text-emerald-300" : "text-red-300"}`}
            role="status"
          >
            <span className={`h-1.5 w-1.5 rounded-full ${connected ? "bg-emerald-300" : "bg-red-400"}`} />
            {connected ? "CONNECTED" : "OFFLINE"}
          </span>
          <button onClick={() => setView("tokens")} title="Plan and remaining credits"
            className="glass hidden items-center gap-1.5 whitespace-nowrap rounded-xl px-2.5 py-1.5 text-[11px] tracking-wider text-slate-300 hover:text-white md:flex">
            {isOwner ? "OWNER · usage not charged" : `${ent.subscription ? ent.plan.name : "FREE"} · ${num(ent.balance)} credits`}
          </button>
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

      {view !== "home" && (
        <div className="absolute inset-0 z-20">
          {view === "plans" && <PlansView entitlement={ent} onBack={() => setView("home")} />}
          {view === "intelligence" && <IntelligenceView />}
          {view === "tokens" && <TokensView onPlans={() => setView("plans")} />}
          {view === "keys" && <KeysView apiAllowed={ent.api || isOwner} onPlans={() => setView("plans")} />}
          {view === "admin" && isOwner && <AdminView />}
        </div>
      )}
      <div className="fixed bottom-3 left-1/2 z-50 -translate-x-1/2 lg:hidden">
        {view !== "home" && (
          <BottomNav view={view} onView={setView} isOwner={isOwner} email={me.user.email} onLogout={() => api.logout().finally(onLoggedOut)} />
        )}
      </div>

      {view === "home" && (<>
      {/* CENTER */}
      <main className="relative z-10 flex flex-1 flex-col items-center justify-center px-4">
        <div className="relative">
          <InfinityCore
            state={store.core}
            amplitude={store.amplitude}
            compact={workspaceOpen || (isChat && chat.length > 0)}
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
        {isChat && (chat.length > 0 || chatWaiting) && (
          <ChatPanel turns={chat} waiting={chatWaiting} onClear={() => setChat([])} />
        )}
        {!isChat && !workspaceOpen && store.taskText && (
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
          busy={isChat ? false : busy}
          disabledReason={disabledReason}
          model={status?.model}
          checkpoints={checkpoints}
          openModels={openModels}
          onSelectModel={selectModel}
          onAttach={() => (ent.cook ? setLauncher(true) : setUpgrade(true))}
          workspaceName={workspaceName}
          taskText={store.taskText}
          onListening={store.setListening}
          amplitude={store.amplitude}
          placeholder={isChat ? `Message ${modelName(status?.model)}…` : ent.cook ? undefined : `COOK locked — ${ent.cook_reason ?? "paid plan required"}. Press Enter to see plans.`}
        />
      </footer>

      <TaskWorkspace open={open} setOpen={setOpen} selectedFile={selectedFile} setSelectedFile={setSelectedFile} />
      </>)}
      {view === "home" && (
        <div className="fixed left-3 top-14 z-40 lg:hidden">
          <BottomNav view={view} onView={setView} isOwner={isOwner} email={me.user.email} onLogout={() => api.logout().finally(onLoggedOut)} />
        </div>
      )}
      {upgrade && (
        <UpgradeOverlay reason={ent.cook_reason ?? "COOK requires an active Plus, Pro or Max subscription"} onClose={() => { setUpgrade(false); refreshMe(); }}
          onComparePlans={() => { setUpgrade(false); setView("plans"); }} />
      )}

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
