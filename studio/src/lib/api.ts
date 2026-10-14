import type {
  AdminOverview, AdminUser, Intelligence, AgentEvent, ApiKey, AuditEvent, BillingMe, Catalog, ChatResult, ChatTurn, Checkpoint, Conversation, ConversationSummary, CheckoutResult, HyperAdapter, Me, Mode, OpenModel,
  Permission, Plan, Project, RecentProject, Status, TreeNode,
} from "./types";

let token: string | null = null; // per-session CSRF token from /api/auth/me

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
  const res = await fetch(path, { ...init, headers, credentials: "same-origin" });
  if (res.status === 401) {
    token = null;
    window.dispatchEvent(new Event("infinity:unauthorized"));
  }
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
  const me = await request<Me>("/api/auth/me");
  token = me.csrf;
  return token;
}

export function clearSession() {
  token = null;
}

export const api = {
  status: () => request<Status>("/api/status"),
  models: () =>
    request<{ active: Status["model"]; checkpoints: Checkpoint[]; hyper_adapters?: HyperAdapter[]; open_models?: OpenModel[]; can_switch: boolean }>(
      "/api/models",
    ),
  /** value: a HYPER 5.5 checkpoint path, "hyper-adapter:<name>" (HYPER 5.5 — Fine-Tuned), or "open:<id>" */
  setModel: (value: string) =>
    request<Status["model"]>("/api/model", {
      method: "POST",
      body: JSON.stringify(
        value.startsWith("open:")
          ? { open_model: value.slice(5) }
          : value.startsWith("hyper-adapter:")
            ? { hyper_adapter: value.slice(14) }
            : { checkpoint: value },
      ),
    }),
  chat: (messages: ChatTurn[], conversation_id: number | null = null, max_new_tokens = 512) =>
    request<ChatResult>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ messages: messages.map(({ role, content }) => ({ role, content })), conversation_id, max_new_tokens }),
    }),
  chatCancel: () => request<{ cancelled: boolean }>("/api/chat/cancel", { method: "POST" }),
  conversations: () => request<{ conversations: ConversationSummary[] }>("/api/conversations"),
  conversation: (id: number) => request<Conversation>(`/api/conversations/${id}`),
  renameConversation: (id: number, title: string) =>
    request<ConversationSummary>(`/api/conversations/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  deleteConversation: (id: number) => request<{ deleted: boolean }>(`/api/conversations/${id}`, { method: "DELETE" }),
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

  intelligence: () => request<Intelligence>("/api/intelligence"),

  // ---- accounts
  setup: () => request<{ owner_exists: boolean; registration: boolean }>("/api/setup"),
  me: () => request<Me>("/api/auth/me"),
  login: async (email: string, password: string) => {
    token = null;
    const r = await fetch("/api/auth/login", {
      method: "POST", headers: { "content-type": "application/json" }, credentials: "same-origin",
      body: JSON.stringify({ email, password }),
    });
    if (!r.ok) throw new ApiError(r.status, (await r.json().catch(() => ({}))).detail ?? r.statusText);
  },
  register: async (email: string, password: string) => {
    const r = await fetch("/api/auth/register", {
      method: "POST", headers: { "content-type": "application/json" }, credentials: "same-origin",
      body: JSON.stringify({ email, password }),
    });
    if (!r.ok) throw new ApiError(r.status, (await r.json().catch(() => ({}))).detail ?? r.statusText);
  },
  logout: async () => {
    await request("/api/auth/logout", { method: "POST" });
    token = null;
  },

  // ---- billing
  plans: () => request<Catalog>("/api/plans"),
  billing: () => request<BillingMe>("/api/billing/me"),
  checkout: (plan: string, period: "monthly" | "yearly") =>
    request<CheckoutResult>("/api/billing/checkout", { method: "POST", body: JSON.stringify({ plan, period }) }),
  cancelSubscription: () => request("/api/billing/cancel", { method: "POST" }),

  // ---- keys
  keys: () => request<{ keys: ApiKey[] }>("/api/keys"),
  createKey: (name: string, rate_limit_per_min: number) =>
    request<{ key: ApiKey & { secret: string }; notice: string }>("/api/keys", { method: "POST", body: JSON.stringify({ name, rate_limit_per_min }) }),
  rotateKey: (id: number) => request<{ key: ApiKey & { secret: string } }>(`/api/keys/${id}/rotate`, { method: "POST" }),
  revokeKey: (id: number) => request<{ key: ApiKey }>(`/api/keys/${id}/revoke`, { method: "POST" }),

  // ---- admin (owner only; the server enforces it)
  adminOverview: () => request<AdminOverview>("/api/admin/overview"),
  adminUsers: (q = "") => request<{ users: AdminUser[] }>(`/api/admin/users?q=${encodeURIComponent(q)}`),
  adminStatus: (id: number, status: "active" | "suspended", reason: string) =>
    request(`/api/admin/users/${id}/status`, { method: "POST", body: JSON.stringify({ status, reason }) }),
  adminCredits: (id: number, delta: number, reason: string) =>
    request<{ balance: number }>(`/api/admin/users/${id}/credits`, { method: "POST", body: JSON.stringify({ delta, reason }) }),
  adminGrant: (id: number, plan: string, reason: string) =>
    request(`/api/admin/users/${id}/grant`, { method: "POST", body: JSON.stringify({ plan, reason }) }),
  adminRevokeSessions: (id: number, reason: string) =>
    request(`/api/admin/users/${id}/revoke-sessions`, { method: "POST", body: JSON.stringify({ reason }) }),
  adminRoots: (id: number, roots: string[], reason: string) =>
    request(`/api/admin/users/${id}/workspace-roots`, { method: "POST", body: JSON.stringify({ roots, reason }) }),
  adminPlan: (id: string, changes: Partial<Plan>, reason: string, confirm_price_change = false) =>
    request<{ plan: Plan }>(`/api/admin/plans/${id}`, { method: "PATCH", body: JSON.stringify({ changes, reason, confirm_price_change }) }),
  adminAudit: () => request<{ events: AuditEvent[]; chain: { ok: boolean; checked: number; broken_at?: number } }>("/api/admin/audit"),
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
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/api/events?since=${since}`); // authenticated by the session cookie
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
