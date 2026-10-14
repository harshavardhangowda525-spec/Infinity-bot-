import { useCallback, useEffect, useRef, useState } from "react";
import ModelStatus from "./components/ModelStatus";
import ProjectLauncher from "./components/ProjectLauncher";
import UpgradeOverlay from "./components/UpgradeOverlay";
import { api } from "./lib/api";
import { modelName, num } from "./lib/format";
import { loadPrefs, savePrefs, type Prefs } from "./lib/prefs";
import { useStore } from "./lib/store";
import { applyTheme, loadTheme, saveTheme, type Theme } from "./lib/theme";
import type { ChatTurn, Checkpoint, ConversationSummary, HyperAdapter, Me, Mode, OpenModel } from "./lib/types";
import AdminView from "./views/AdminView";
import HistoryView from "./views/HistoryView";
import IntelligenceView from "./views/IntelligenceView";
import KeysView from "./views/KeysView";
import PlansView from "./views/PlansView";
import SettingsView from "./views/SettingsView";
import TokensView from "./views/TokensView";
import ChatThread from "./ui/ChatThread";
import Composer, { type WorkMode } from "./ui/Composer";
import CookWorkspace from "./ui/CookWorkspace";
import { Icon } from "./ui/Icons";
import ModelSelect from "./ui/ModelSelect";
import Sidebar, { parseRoute, routeHash, type Route } from "./ui/Sidebar";

const SUGGESTIONS = [
  { label: "Build a website", text: "Help me build a simple website for " },
  { label: "Explain a concept", text: "Explain this concept in simple terms: " },
  { label: "Write some code", text: "Write a Python function that " },
  { label: "Brainstorm an idea", text: "Brainstorm ideas for " },
];

const TITLES: Record<Route["name"], string> = {
  chat: "", history: "Chat history", cook: "Cook", settings: "Settings", plans: "Plans & billing",
  usage: "Usage", keys: "API keys", intelligence: "Model & benchmarks", admin: "Admin",
};

interface ChatState {
  id: number | null;
  title: string | null;
  turns: ChatTurn[];
}

export default function App({ me, refreshMe, onLoggedOut }: { me: Me; refreshMe: () => void; onLoggedOut: () => void }) {
  const store = useStore();
  const { status, connected } = store;
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));
  const [theme, setThemeState] = useState<Theme>(loadTheme);
  const [prefs, setPrefsState] = useState<Prefs>(loadPrefs);
  const [mobileNav, setMobileNav] = useState(false);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [openModels, setOpenModels] = useState<OpenModel[]>([]);
  const [hyperAdapters, setHyperAdapters] = useState<HyperAdapter[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [chat, setChat] = useState<ChatState>({ id: null, title: null, turns: [] });
  const [waiting, setWaiting] = useState(false);
  const [prefill, setPrefill] = useState<{ text: string; n: number } | undefined>();
  const [agentMode, setAgentMode] = useState<Mode>("code");
  const [launcher, setLauncher] = useState(false);
  const [upgrade, setUpgrade] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const request = useRef(0);
  const spokenFor = useRef<string | null>(null);

  const ent = (status as (typeof status & { entitlement?: Me["entitlement"] }) | null)?.entitlement ?? me.entitlement;
  const isOwner = me.user.role === "owner";
  const agentBusy = status?.agent.busy ?? false;
  const model = status?.model;

  // ---------------------------------------------------------------- routing, theme, prefs
  useEffect(() => {
    const on = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const go = useCallback((r: Route) => {
    const h = routeHash(r);
    if (window.location.hash !== h) window.location.hash = h;
    setRoute(r);
  }, []);
  useEffect(() => applyTheme(theme), [theme]);
  useEffect(() => setMobileNav(false), [route]);
  useEffect(() => {
    if (!mobileNav) return;
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setMobileNav(false);
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [mobileNav]);
  const setTheme = (t: Theme) => {
    setThemeState(t);
    saveTheme(t);
  };
  const setPrefs = (p: Partial<Prefs>) =>
    setPrefsState((old) => {
      const n = { ...old, ...p };
      savePrefs(n);
      return n;
    });

  // ---------------------------------------------------------------- data
  const loadModels = useCallback(() => {
    api.models().then(
      (m) => {
        setCheckpoints(m.checkpoints);
        setOpenModels(m.open_models ?? []);
        setHyperAdapters(m.hyper_adapters ?? []);
      },
      () => setCheckpoints([]),
    );
  }, []);
  useEffect(loadModels, [loadModels, connected]);
  const loadConversations = useCallback(() => {
    api.conversations().then((r) => setConversations(r.conversations), () => undefined);
  }, []);
  useEffect(loadConversations, [loadConversations]);

  // Open a saved conversation when the route points at one.
  useEffect(() => {
    if (route.name !== "chat") return;
    if (!route.id) return;
    if (route.id === chat.id) return;
    const n = ++request.current;
    setWaiting(false);
    api.conversation(route.id).then(
      (c) => n === request.current && setChat({ id: c.id, title: c.title, turns: c.messages }),
      (e) => {
        setNotice((e as Error).message);
        go({ name: "chat" });
      },
    );
  }, [route, chat.id, go]);

  // ---------------------------------------------------------------- keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "o" && !e.shiftKey) {
        e.preventDefault();
        ent.cook ? setLauncher(true) : setUpgrade(true);
      } else if (mod && e.shiftKey && e.key.toLowerCase() === "o") {
        e.preventDefault();
        newChat();
      } else if (mod && e.key.toLowerCase() === "b") {
        e.preventDefault();
        setPrefs({ sidebarCollapsed: !prefs.sidebarCollapsed });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // ---------------------------------------------------------------- read aloud (opt-in)
  const speak = (text: string) => {
    if (!prefs.readAloud || !("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text.replace(/```[\s\S]*?```/g, " code block "));
    u.onstart = () => store.setSpeaking(true);
    u.onend = u.onerror = () => store.setSpeaking(false);
    window.speechSynthesis.speak(u);
  };
  useEffect(() => {
    const r = store.report;
    if (!r || spokenFor.current === r.task_id) return;
    spokenFor.current = r.task_id;
    speak(`${r.status.replace(/_/g, " ")}. ${r.summary}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [store.report]);

  // ---------------------------------------------------------------- chat
  const newChat = () => {
    if (waiting) api.chatCancel().catch(() => undefined);
    request.current++;
    setWaiting(false);
    setChat({ id: null, title: null, turns: [] });
    setNotice(null);
    go({ name: "chat" });
  };

  const sendHistory = async (history: ChatTurn[], restoreText?: string) => {
    const before = chat.turns;
    const n = ++request.current;
    setChat((c) => ({ ...c, turns: history }));
    setWaiting(true);
    setNotice(null);
    try {
      const r = await api.chat(history, chat.id);
      if (n !== request.current) return;
      setChat({ id: r.conversation.id, title: r.conversation.title,
        turns: [...history, { role: "assistant", content: r.reply, model: r.model.product, stopped: r.stopped }] });
      if (route.name === "chat" && route.id !== r.conversation.id) {
        window.history.replaceState(null, "", routeHash({ name: "chat", id: r.conversation.id }));
        setRoute({ name: "chat", id: r.conversation.id });
      }
      loadConversations();
      speak(r.reply);
    } catch (e) {
      if (n !== request.current) return;
      setChat((c) => ({ ...c, turns: before }));
      if (restoreText) setPrefill({ text: restoreText, n: Date.now() });
      setNotice((e as Error).message);
    } finally {
      if (n === request.current) setWaiting(false);
    }
  };

  const stopChat = () => api.chatCancel().catch((e) => setNotice((e as Error).message));
  const regenerate = () => {
    const last = chat.turns.map((t) => t.role).lastIndexOf("assistant");
    if (last > 0) sendHistory(chat.turns.slice(0, last));
  };
  const editAndResend = (i: number, text: string) => sendHistory([...chat.turns.slice(0, i), { role: "user", content: text }]);

  // ---------------------------------------------------------------- cook
  const startCook = async (text: string) => {
    setNotice(null);
    if (!ent.cook) {
      setUpgrade(true);
      return;
    }
    try {
      store.clearTaskView();
      await api.startTask(text, agentMode);
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
    } catch (e) {
      setNotice((e as Error).message);
    } finally {
      setSwitching(false);
    }
  };

  // ---------------------------------------------------------------- composer
  const isCook = route.name === "cook";
  const offline = "Infinity Bot runtime is not reachable — start it with `infinity studio`";
  const disabledReason = switching
    ? "Switching model…"
    : !connected
      ? offline
      : model?.state !== "ready"
        ? `No model loaded: ${model?.reason ?? "loading"}`
        : isCook && ent.cook && !status?.workspace
          ? "Open a project first (folder button below, or Ctrl+O)"
          : null;
  const onMode = (m: WorkMode) => (m === "cook" ? go({ name: "cook" }) : go(chat.id ? { name: "chat", id: chat.id } : { name: "chat" }));
  const modelSlot = (
    <ModelSelect model={model} checkpoints={checkpoints} hyperAdapters={hyperAdapters} openModels={openModels}
      onSelect={selectModel} disabled={agentBusy || waiting || switching} />
  );
  const workspaceName = status?.workspace ? status.workspace.split(/[\\/]/).filter(Boolean).pop() ?? status.workspace : null;
  const composer = (size: "hero" | "dock") => (
    <Composer
      mode={isCook ? "cook" : "normal"}
      onMode={onMode}
      agentMode={agentMode}
      onAgentMode={setAgentMode}
      onSubmit={isCook ? startCook : (t) => sendHistory([...chat.turns, { role: "user", content: t }], t)}
      onStop={isCook ? () => api.cancel().catch((e) => setNotice((e as Error).message)) : stopChat}
      busy={isCook ? agentBusy : waiting}
      disabledReason={isCook && !ent.cook ? null : disabledReason}
      placeholder={isCook && !ent.cook ? "Cook needs a Plus, Pro or Max plan — press Enter to see plans" : undefined}
      modelSlot={modelSlot}
      onAttachProject={() => (ent.cook ? setLauncher(true) : setUpgrade(true))}
      workspaceName={workspaceName}
      size={size}
      autoFocus={size === "hero"}
      prefill={prefill}
      onListening={store.setListening}
    />
  );

  const title = route.name === "chat" ? chat.title ?? "" : TITLES[route.name];
  const planLabel = isOwner ? "Owner · usage not charged" : `${ent.subscription ? ent.plan.name : "Free"} · ${num(ent.balance)} credits`;
  const logout = () => api.logout().finally(onLoggedOut);

  return (
    <div className="flex h-full bg-bg text-fg">
      <Sidebar
        route={route}
        go={go}
        collapsed={prefs.sidebarCollapsed}
        onToggle={() => setPrefs({ sidebarCollapsed: !prefs.sidebarCollapsed })}
        conversations={conversations}
        onNewChat={newChat}
        onProjects={() => (ent.cook ? setLauncher(true) : setUpgrade(true))}
        user={me.user}
        planLabel={planLabel}
        isOwner={isOwner}
        theme={theme}
        onTheme={setTheme}
        onLogout={logout}
        mobileOpen={mobileNav}
        onMobileClose={() => setMobileNav(false)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center gap-2 px-3 sm:px-5">
          <button onClick={() => setMobileNav(true)} aria-label="Open menu" className="rounded-lg p-2 text-muted hover:bg-hover hover:text-fg md:hidden">
            {Icon.menu}
          </button>
          <h1 className="min-w-0 truncate text-sm font-medium text-fg">{title}</h1>
          <div className="ml-auto flex items-center gap-2">
            {!connected && (
              <span role="status" className="rounded-md bg-bad/10 px-2 py-1 text-xs text-bad">Offline</span>
            )}
            <ModelStatus status={status} connected={connected} />
          </div>
        </header>

        {notice && (
          <div role="alert" className="mx-auto mb-2 flex w-full max-w-3xl items-start gap-2 px-4 sm:px-6">
            <p className="flex-1 rounded-lg border border-warn/30 bg-warn/5 px-3 py-2 text-sm text-fg">{notice}</p>
            <button onClick={() => setNotice(null)} aria-label="Dismiss" className="rounded-md p-2 text-muted hover:bg-hover hover:text-fg">
              {Icon.close}
            </button>
          </div>
        )}

        <main className="flex min-h-0 flex-1 flex-col">
          {route.name === "chat" &&
            (chat.turns.length === 0 && !waiting ? (
              <div className="scroll-thin flex flex-1 flex-col items-center justify-center overflow-y-auto px-4 pb-16">
                <div className="w-full max-w-2xl">
                  <h2 className="text-center text-3xl font-semibold tracking-tight text-fg sm:text-[34px]">What can I help you build?</h2>
                  <p className="mt-2 text-center text-[15px] text-muted">Ask anything, explore ideas, or start building something new.</p>
                  <div className="mt-8">{composer("hero")}</div>
                  <div className="mt-4 flex flex-wrap justify-center gap-2">
                    {SUGGESTIONS.map((s) => (
                      <button key={s.label} onClick={() => setPrefill({ text: s.text, n: Date.now() })}
                        className="rounded-full border border-line bg-surface px-3 py-1.5 text-[13px] text-fg/75 transition hover:bg-hover hover:text-fg">
                        {s.label}
                      </button>
                    ))}
                  </div>
                  {model?.state === "ready" && (
                    <p className="mt-8 text-center text-xs text-muted">{modelName(model)} runs locally on this computer.</p>
                  )}
                </div>
              </div>
            ) : (
              <>
                <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
                  <ChatThread turns={chat.turns} waiting={waiting} onRegenerate={regenerate} onEdit={editAndResend} />
                </div>
                <div className="px-3 pb-3 sm:px-6 sm:pb-5">
                  <div className="mx-auto max-w-3xl">{composer("dock")}</div>
                  <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-muted">
                    Replies come from a local model and can be wrong. Check important answers.
                  </p>
                </div>
              </>
            ))}
          {route.name === "cook" && (
            <CookWorkspace
              composer={composer("dock")}
              locked={!ent.cook}
              lockedReason={ent.cook_reason}
              onUnlock={() => setUpgrade(true)}
              onOpenProject={() => (ent.cook ? setLauncher(true) : setUpgrade(true))}
              permission={status?.workspace ? status.permission : null}
            />
          )}
          {route.name === "history" && (
            <HistoryView conversations={conversations} onOpen={(id) => go({ name: "chat", id })}
              onChanged={() => { loadConversations(); }} />
          )}
          {route.name === "settings" && (
            <SettingsView me={me} theme={theme} onTheme={setTheme} readAloud={prefs.readAloud}
              onReadAloud={(v) => setPrefs({ readAloud: v })} model={model} onLogout={logout} onPlans={() => go({ name: "plans" })} />
          )}
          {route.name === "plans" && <div className="min-h-0 flex-1"><PlansView entitlement={ent} onBack={() => go({ name: "chat" })} /></div>}
          {route.name === "usage" && <div className="min-h-0 flex-1"><TokensView onPlans={() => go({ name: "plans" })} /></div>}
          {route.name === "keys" && <div className="min-h-0 flex-1"><KeysView apiAllowed={ent.api || isOwner} onPlans={() => go({ name: "plans" })} /></div>}
          {route.name === "intelligence" && <div className="min-h-0 flex-1"><IntelligenceView /></div>}
          {route.name === "admin" && (isOwner ? <div className="min-h-0 flex-1"><AdminView /></div>
            : <p className="p-8 text-center text-sm text-muted">The admin console is for the owner account.</p>)}
        </main>
      </div>

      {upgrade && (
        <UpgradeOverlay reason={ent.cook_reason ?? "Cook requires an active Plus, Pro or Max subscription"}
          onClose={() => { setUpgrade(false); refreshMe(); }}
          onComparePlans={() => { setUpgrade(false); go({ name: "plans" }); }} />
      )}
      {launcher && (
        <ProjectLauncher current={status?.workspace ?? null} onClose={() => setLauncher(false)}
          onOpened={() => { store.refresh(); }} />
      )}
    </div>
  );
}
