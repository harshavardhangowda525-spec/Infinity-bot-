import { AnimatePresence } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { useStore } from "../lib/store";
import type { AgentEvent } from "../lib/types";
import ChangeReview from "./ChangeReview";
import { FileView } from "./CodeViewer";
import ExecutionStream from "./ExecutionStream";
import FileExplorer from "./FileExplorer";
import Panel, { Icon } from "./Panel";
import TaskPanel from "./TaskPanel";

export type PanelId = "task" | "review" | "files" | "code" | "stream";

const PANELS: { id: PanelId; label: string; icon: JSX.Element }[] = [
  { id: "task", label: "Task", icon: Icon.list },
  { id: "review", label: "Changes", icon: Icon.diff },
  { id: "files", label: "Files", icon: Icon.folder },
  { id: "code", label: "Code", icon: Icon.code },
  { id: "stream", label: "Stream", icon: Icon.terminal },
];

function useSideLayout() {
  const calc = () => {
    const free = (window.innerWidth - 800) / 2 - 24; // dock is max 760px wide, centred
    return free >= 300 ? { width: Math.min(420, free), bottom: 16 } : { width: 360, bottom: 200 };
  };
  const [l, setL] = useState(calc);
  useEffect(() => {
    const on = () => setL(calc());
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return l;
}

function useIsDesktop() {
  const q = "(min-width: 1024px)";
  const [m, setM] = useState(() => window.matchMedia?.(q).matches ?? true);
  useEffect(() => {
    const mq = window.matchMedia?.(q);
    const on = () => setM(mq.matches);
    mq?.addEventListener?.("change", on);
    return () => mq?.removeEventListener?.("change", on);
  }, []);
  return m;
}

export default function TaskWorkspace({ open, setOpen, selectedFile, setSelectedFile }: {
  open: Set<PanelId>;
  setOpen: (s: Set<PanelId>) => void;
  selectedFile: string | null;
  setSelectedFile: (p: string | null) => void;
}) {
  const store = useStore();
  const desktop = useIsDesktop();
  const side = useSideLayout();
  const [mobileTab, setMobileTab] = useState<PanelId>("task");
  const busy = store.status?.agent.busy ?? false;
  const changed = useMemo(() => new Set(store.changes.map((c) => c.path)), [store.changes]);
  const taskEvents: AgentEvent[] = useMemo(() => {
    const start = [...store.events].reverse().find((e) => e.type === "task_start");
    return start ? store.events.filter((e) => e.seq >= start.seq) : [];
  }, [store.events]);
  // File views refresh whenever the agent changes something or a rollback happens.
  const refreshKey = useMemo(
    () => store.events.filter((e) => (e.type === "tool_result" && e.diff) || e.type === "rollback" || e.type === "workspace").length,
    [store.events],
  );

  const close = (id: PanelId) => {
    const n = new Set(open);
    n.delete(id);
    setOpen(n);
  };
  const openFile = (p: string) => {
    setSelectedFile(p);
    setOpen(new Set([...open, "code"]));
    setMobileTab("code");
  };

  const render = (id: PanelId, onClose?: () => void, className = "") => {
    switch (id) {
      case "task":
        return (
          <Panel key="task" title="TASK" icon={Icon.list} onClose={onClose} className={className}>
            <TaskPanel taskText={store.taskText} plan={store.plan} events={taskEvents} report={store.report} busy={busy} />
          </Panel>
        );
      case "review":
        return (
          <Panel
            key="review"
            title={`CHANGE REVIEW${store.approvals.length ? ` · ${store.approvals.length} PENDING` : ""}`}
            icon={Icon.diff}
            onClose={onClose}
            className={className}
          >
            <ChangeReview approvals={store.approvals} changes={store.changes} report={store.report} busy={busy} onInspect={openFile} onRolledBack={store.refresh} />
          </Panel>
        );
      case "files":
        return (
          <Panel key="files" title="FILE EXPLORER" icon={Icon.folder} onClose={onClose} className={className}>
            <FileExplorer workspace={store.status?.workspace ?? null} onOpen={openFile} selected={selectedFile} changed={changed} refreshKey={refreshKey} />
          </Panel>
        );
      case "code":
        return (
          <Panel key="code" title={selectedFile ? `CODE · ${selectedFile}` : "CODE VIEWER"} icon={Icon.code} onClose={onClose} className={className}>
            {selectedFile ? <FileView path={selectedFile} refreshKey={refreshKey} /> : <p className="p-4 text-sm text-slate-500">Select a file in the explorer.</p>}
          </Panel>
        );
      case "stream":
        return (
          <Panel key="stream" title="EXECUTION STREAM" icon={Icon.terminal} onClose={onClose} className={className} bodyClass="overflow-hidden">
            <ExecutionStream events={taskEvents.length ? taskEvents : store.events} />
          </Panel>
        );
    }
  };

  // Pending approvals always surface, even in the minimal view.
  const visible = new Set(open);
  if (store.approvals.length) visible.add("review");

  if (!desktop) {
    if (visible.size === 0) return null;
    const tab = visible.has(mobileTab) ? mobileTab : [...visible][0];
    return (
      <div className="fixed inset-x-2 bottom-[178px] top-14 z-30 flex flex-col gap-2">
        <nav className="glass flex gap-1 overflow-x-auto rounded-xl p-1" aria-label="Workspace panels">
          {PANELS.map((p) => (
            <button
              key={p.id}
              onClick={() => {
                setMobileTab(p.id);
                setOpen(new Set([...open, p.id]));
              }}
              aria-current={tab === p.id}
              className={`flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs ${tab === p.id ? "bg-cyan-glow/15 text-cyan-glow" : "text-slate-400"}`}
            >
              {p.icon}
              {p.label}
            </button>
          ))}
          <button onClick={() => setOpen(new Set())} className="ml-auto rounded-lg px-2.5 text-xs text-slate-400" aria-label="Close workspace">
            ✕
          </button>
        </nav>
        {render(tab, undefined, "flex-1")}
      </div>
    );
  }

  const left = (["task", "review"] as PanelId[]).filter((p) => visible.has(p));
  const right = (["files", "code", "stream"] as PanelId[]).filter((p) => visible.has(p));
  return (
    <>
      <div className="pointer-events-none fixed left-4 top-16 z-30 flex flex-col gap-3" style={{ width: side.width, bottom: side.bottom }}>
        <AnimatePresence>{left.map((p) => <div key={p} className="pointer-events-auto flex min-h-0 flex-1 flex-col">{render(p, () => close(p), "flex-1")}</div>)}</AnimatePresence>
      </div>
      <div className="pointer-events-none fixed right-4 top-16 z-30 flex flex-col gap-3" style={{ width: side.width, bottom: side.bottom }}>
        <AnimatePresence>{right.map((p) => <div key={p} className="pointer-events-auto flex min-h-0 flex-1 flex-col">{render(p, () => close(p), "flex-1")}</div>)}</AnimatePresence>
      </div>
    </>
  );
}

export { PANELS };
