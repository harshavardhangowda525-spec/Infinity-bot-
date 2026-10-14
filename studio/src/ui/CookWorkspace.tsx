import { useMemo, useState, type ReactNode } from "react";
import ChangeReview from "../components/ChangeReview";
import { FileView } from "../components/CodeViewer";
import ExecutionStream from "../components/ExecutionStream";
import FileExplorer from "../components/FileExplorer";
import TaskPanel from "../components/TaskPanel";
import { useStore } from "../lib/store";
import type { AgentEvent } from "../lib/types";
import { Icon } from "./Icons";

type Tab = "files" | "code" | "changes" | "log";

/**
 * Cook: the coding workspace. Activity (instruction, plan, steps, approvals) on the left, project tools on
 * the right. Everything shown comes from the real agent runtime's events; nothing is simulated.
 */
export default function CookWorkspace({
  composer, locked, lockedReason, onUnlock, onOpenProject, permission,
}: {
  composer: ReactNode;
  locked: boolean;
  lockedReason: string | null;
  onUnlock: () => void;
  onOpenProject: () => void;
  permission: string | null;
}) {
  const store = useStore();
  const [tab, setTab] = useState<Tab>("files");
  const [mobileTab, setMobileTab] = useState<"activity" | Tab>("activity");
  const [selected, setSelected] = useState<string | null>(null);
  const busy = store.status?.agent.busy ?? false;
  const workspace = store.status?.workspace ?? null;
  const changed = useMemo(() => new Set(store.changes.map((c) => c.path)), [store.changes]);
  const taskEvents: AgentEvent[] = useMemo(() => {
    const start = [...store.events].reverse().find((e) => e.type === "task_start");
    return start ? store.events.filter((e) => e.seq >= start.seq) : [];
  }, [store.events]);
  const refreshKey = useMemo(
    () => store.events.filter((e) => (e.type === "tool_result" && e.diff) || e.type === "rollback" || e.type === "workspace").length,
    [store.events],
  );
  const openFile = (p: string) => {
    setSelected(p);
    setTab("code");
    setMobileTab("code");
  };

  const tabs: { id: Tab; label: string; icon: ReactNode; badge?: number }[] = [
    { id: "files", label: "Files", icon: Icon.folder },
    { id: "code", label: "Code", icon: Icon.file },
    { id: "changes", label: "Changes", icon: Icon.diff, badge: store.changes.length + store.approvals.length || undefined },
    { id: "log", label: "Log", icon: Icon.terminal },
  ];

  const tool = (t: Tab) => {
    switch (t) {
      case "files":
        return workspace ? (
          <FileExplorer workspace={workspace} onOpen={openFile} selected={selected} changed={changed} refreshKey={refreshKey} />
        ) : (
          <Empty text="Open a project to browse its files." action={<button onClick={onOpenProject} className="btn-secondary">Open project</button>} />
        );
      case "code":
        return selected ? <FileView path={selected} refreshKey={refreshKey} /> : <Empty text="Select a file to view it here." />;
      case "changes":
        return <ChangeReview approvals={store.approvals} changes={store.changes} report={store.report} busy={busy} onInspect={openFile} onRolledBack={store.refresh} />;
      case "log":
        return <ExecutionStream events={taskEvents.length ? taskEvents : store.events} />;
    }
  };

  const activity = (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-6">
        {locked ? (
          <Empty
            title="Cook is part of the Plus, Pro and Max plans"
            text={`Cook edits project files, runs tests and debugs code on this computer. ${lockedReason ?? ""}`}
            action={<button onClick={onUnlock} className="btn-cook">See plans</button>}
          />
        ) : !store.taskText ? (
          <Empty
            title="Describe a coding task"
            text={workspace
              ? "Infinity Bot will plan, read and edit files, and run checks in this project. Changes are checkpointed and can be undone."
              : "Open a project folder first. Cook only works inside folders you choose."}
            action={!workspace ? <button onClick={onOpenProject} className="btn-cook">Open project</button> : undefined}
          />
        ) : (
          <div className="mx-auto max-w-3xl space-y-4">
            <div className="flex justify-end">
              <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl bg-cook/10 px-4 py-2.5 text-[15px] leading-7 text-fg">{store.taskText}</p>
            </div>
            {store.approvals.length > 0 && (
              <section aria-label="Approval needed" className="rounded-xl border border-warn/40 bg-warn/5">
                <h3 className="border-b border-warn/30 px-3.5 py-2 text-xs font-semibold text-warn">Approval needed</h3>
                <ChangeReview approvals={store.approvals} changes={[]} report={null} busy={busy} onInspect={openFile} onRolledBack={store.refresh} />
              </section>
            )}
            <section aria-label="Task progress" className="card overflow-hidden rounded-xl">
              <TaskPanel taskText={store.taskText} plan={store.plan} events={taskEvents} report={store.report} busy={busy} />
            </section>
          </div>
        )}
      </div>
      <div className="border-t border-line bg-bg/80 px-3 pb-3 pt-3 sm:px-6">
        <div className="mx-auto max-w-3xl">{composer}</div>
      </div>
    </div>
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-2 text-xs sm:px-6">
        <span className="inline-flex items-center gap-1.5 font-medium text-cook">
          <span className="h-1.5 w-1.5 rounded-full bg-cook" aria-hidden />
          Cook
        </span>
        <button onClick={onOpenProject} className="inline-flex max-w-[50vw] items-center gap-1.5 truncate rounded-md px-2 py-1 text-fg/80 hover:bg-hover hover:text-fg"
          title={workspace ?? "Open a project"}>
          {Icon.folder}
          <span className="truncate">{workspace ? workspace.split(/[\\/]/).filter(Boolean).pop() : "No project"}</span>
        </button>
        {permission && <span className="rounded-md border border-line px-1.5 py-0.5 text-muted">permission: {permission}</span>}
        <span className="ml-auto inline-flex items-center gap-2 text-muted" role="status" aria-live="polite">
          {busy && <span className="h-3 w-3 animate-spin rounded-full border-2 border-cook/30 border-t-cook motion-reduce:animate-none" aria-hidden />}
          <span className="lowercase first-letter:uppercase">{store.label}</span>
        </span>
      </div>
      {/* Desktop: activity + tools side by side */}
      <div className="hidden min-h-0 flex-1 lg:flex">
        {activity}
        <aside aria-label="Project tools" className="flex w-[44%] max-w-[640px] min-w-[360px] flex-col border-l border-line bg-surface">
          <div role="tablist" aria-label="Project tools" className="flex gap-1 border-b border-line px-2 py-1.5">
            {tabs.map((t) => (
              <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}
                className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs transition ${
                  tab === t.id ? "bg-hover font-medium text-fg" : "text-muted hover:text-fg"
                }`}>
                {t.icon}
                {t.label}
                {t.badge ? <span className="rounded-full bg-cook/15 px-1.5 text-[10px] text-cook">{t.badge}</span> : null}
              </button>
            ))}
            {selected && tab === "code" && <span className="ml-auto self-center truncate px-2 font-mono text-[11px] text-muted">{selected}</span>}
          </div>
          <div className="min-h-0 flex-1 overflow-auto">{tool(tab)}</div>
        </aside>
      </div>
      {/* Tablet / phone: one view at a time */}
      <div className="flex min-h-0 flex-1 flex-col lg:hidden">
        <div role="tablist" aria-label="Cook views" className="scroll-thin flex gap-1 overflow-x-auto border-b border-line px-2 py-1.5">
          {[{ id: "activity" as const, label: "Activity", icon: Icon.list }, ...tabs].map((t) => (
            <button key={t.id} role="tab" aria-selected={mobileTab === t.id} onClick={() => setMobileTab(t.id)}
              className={`inline-flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs ${
                mobileTab === t.id ? "bg-hover font-medium text-fg" : "text-muted"
              }`}>
              {t.icon}
              {t.label}
            </button>
          ))}
        </div>
        {mobileTab === "activity" ? activity : <div className="min-h-0 flex-1 overflow-auto bg-surface">{tool(mobileTab)}</div>}
      </div>
    </div>
  );
}

function Empty({ title, text, action }: { title?: string; text: string; action?: ReactNode }) {
  return (
    <div className="mx-auto flex max-w-md flex-col items-center px-4 py-14 text-center">
      {title && <h2 className="text-base font-semibold text-fg">{title}</h2>}
      <p className="mt-1.5 text-sm leading-6 text-muted">{text}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
