import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, connectEvents } from "./api";
import { deriveCoreState, lastOf, statusLabel, type CoreInputs } from "./coreState";
import type { AgentEvent, ApprovalRequest, CoreState, Status, TaskReport } from "./types";

export interface FileChange {
  path: string;
  diff: string;
  step: number;
}

interface Store {
  status: Status | null;
  connected: boolean;
  events: AgentEvent[];
  approvals: ApprovalRequest[];
  changes: FileChange[];
  plan: string[];
  report: TaskReport | null;
  core: CoreState;
  label: string;
  taskText: string | null;
  listening: boolean;
  speaking: boolean;
  amplitude: React.MutableRefObject<number>;
  setListening: (v: boolean) => void;
  setSpeaking: (v: boolean) => void;
  refresh: () => Promise<void>;
  clearTaskView: () => void;
}

const Ctx = createContext<Store | null>(null);
const MAX_EVENTS = 3000;

export function StoreProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<Status | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [listening, setListening] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const [viewFrom, setViewFrom] = useState(0); // seq from which the current task view starts
  const amplitude = useRef(0);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.status());
    } catch {
      setStatus(null);
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const poll = setInterval(refresh, 5000);
    const stop = connectEvents(
      (batch) => {
        setEvents((prev) => {
          const next = prev.concat(batch);
          return next.length > MAX_EVENTS ? next.slice(next.length - MAX_EVENTS) : next;
        });
        if (batch.some((e) => ["task_start", "task_end", "approval_request", "approval_resolved", "model", "workspace", "report"].includes(e.type)))
          refresh();
      },
      setConnected,
    );
    return () => {
      clearInterval(poll);
      stop();
    };
  }, [refresh]);

  // Tick only while an outcome flash may be visible (no idle busy-loop).
  const lastEnd = lastOf(events, "task_end");
  useEffect(() => {
    if (!lastEnd) return;
    const remaining = (lastEnd.time as number) * 1000 + 4200 - Date.now();
    if (remaining <= 0) return;
    const t = setTimeout(() => setNow(Date.now()), remaining);
    setNow(Date.now());
    return () => clearTimeout(t);
  }, [lastEnd]);

  const derived = useMemo(() => {
    const taskEvents = events.filter((e) => e.seq > viewFrom);
    const start = lastOf(taskEvents, "task_start");
    const current = start ? taskEvents.filter((e) => e.seq >= start.seq) : [];
    const resolved = new Set(current.filter((e) => e.type === "approval_resolved").map((e) => e.id as string));
    const approvals = current
      .filter((e) => e.type === "approval_request" && !resolved.has(e.id as string))
      .map((e) => e as unknown as ApprovalRequest);
    const changes: FileChange[] = current
      .filter((e) => e.type === "tool_result" && typeof e.diff === "string" && e.diff)
      .map((e) => ({ path: (e.path as string) ?? "", diff: e.diff as string, step: e.step as number }));
    const planEv = lastOf(current, "plan");
    const reportEv = lastOf(current, "report");
    const stateEv = lastOf(current, "state");
    const endEv = lastOf(current, "task_end");
    const busy = status?.agent.busy ?? false;
    const inputs: CoreInputs = {
      connected: connected && status !== null,
      busy,
      awaitingApproval: approvals.length > 0,
      listening,
      speaking,
      lastAgentState: busy && stateEv ? (stateEv.state as string) : null,
      lastEnd: endEv ? { status: endEv.status as string, time: endEv.time } : null,
      now,
    };
    const core = deriveCoreState(inputs);
    return {
      approvals,
      changes,
      plan: (planEv?.plan as string[]) ?? [],
      report: (reportEv?.report as TaskReport) ?? null,
      core,
      label: statusLabel(inputs, core),
      taskText: start ? (start.task as string) : null,
    };
  }, [events, viewFrom, status, connected, listening, speaking, now]);

  const clearTaskView = useCallback(() => {
    setViewFrom(events.length ? events[events.length - 1].seq : 0);
  }, [events]);

  const value: Store = {
    status,
    connected: connected && status !== null,
    events,
    ...derived,
    listening,
    speaking,
    amplitude,
    setListening,
    setSpeaking,
    refresh,
    clearTaskView,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStore(): Store {
  const s = useContext(Ctx);
  if (!s) throw new Error("useStore outside StoreProvider");
  return s;
}
