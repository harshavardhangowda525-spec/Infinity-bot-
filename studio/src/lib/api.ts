import type { AgentEvent, Checkpoint, Mode, Permission, Project, RecentProject, Status, TreeNode } from "./types";

let token: string | null = null;

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) };
  if (init.method && init.method !== "GET") {
    headers["content-type"] = "application/json";
    if (!token) await ensureSession();
    headers["x-infinity-token"] = token ?? "";
  }
  const res = await fetch(path, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export async function ensureSession(): Promise<string> {
  if (token) return token;
  const s = await request<{ token: string }>("/api/session");
  token = s.token;
  return token;
}

export const api = {
  status: () => request<Status>("/api/status"),
  models: () => request<{ active: Status["model"]; checkpoints: Checkpoint[] }>("/api/models"),
  setModel: (checkpoint: string) =>
    request<Status["model"]>("/api/model", { method: "POST", body: JSON.stringify({ checkpoint }) }),
  projects: () => request<{ recent: RecentProject[]; current: string | null }>("/api/projects"),
  project: () => request<Project>("/api/project"),
  setWorkspace: (path: string, permission?: Permission) =>
    request<{ path: string; permission: Permission; project: Project }>("/api/workspace", {
      method: "POST",
      body: JSON.stringify({ path, permission }),
    }),
  tree: (path = ".", depth = 1) =>
    request<{ path: string; children: TreeNode[] }>(`/api/tree?path=${encodeURIComponent(path)}&depth=${depth}`),
  file: (path: string) =>
    request<{ path: string; content: string; size: number }>(`/api/file?path=${encodeURIComponent(path)}`),
  startTask: (task: string, mode: Mode) =>
    request("/api/tasks", { method: "POST", body: JSON.stringify({ task, mode }) }),
  cancel: () => request<{ cancelled: boolean }>("/api/tasks/cancel", { method: "POST" }),
  approve: (id: string, approve: boolean) =>
    request(`/api/approvals/${id}`, { method: "POST", body: JSON.stringify({ approve }) }),
  rollback: (session: string) =>
    request<{ restored: string[] }>("/api/rollback", { method: "POST", body: JSON.stringify({ session }) }),
};

/** Event stream with automatic reconnect; resumes from the last seen sequence number. */
export function connectEvents(
  onEvents: (events: AgentEvent[]) => void,
  onConnection: (connected: boolean) => void,
): () => void {
  let ws: WebSocket | null = null;
  let since = 0;
  let closed = false;
  let retry: ReturnType<typeof setTimeout> | undefined;

  const open = async () => {
    try {
      const t = await ensureSession();
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/api/events?token=${encodeURIComponent(t)}&since=${since}`);
      ws.onopen = () => onConnection(true);
      ws.onmessage = (m) => {
        const data = JSON.parse(m.data) as { events: AgentEvent[] };
        if (data.events.length) {
          since = data.events[data.events.length - 1].seq;
          onEvents(data.events);
        }
      };
      ws.onclose = () => {
        onConnection(false);
        if (!closed) retry = setTimeout(open, 1500);
      };
    } catch {
      onConnection(false);
      if (!closed) retry = setTimeout(open, 2000);
    }
  };
  open();
  return () => {
    closed = true;
    if (retry) clearTimeout(retry);
    ws?.close();
  };
}
