export type Mode = "ask" | "plan" | "review" | "code" | "debug" | "build" | "autonomous";
/** Dock modes: "chat" is Normal chat (plain conversation); the rest are COOK agent modes. */
export type DockMode = Mode | "chat";
export type Permission = "read-only" | "ask" | "auto";

export interface ModelStatus {
  state: "ready" | "unavailable" | "loading";
  reason?: string;
  locality?: "local";
  product?: string;
  implementation?: string;
  run?: string;
  checkpoint?: string;
  step?: number;
  parameters?: number;
  context_length?: number;
  metrics?: Record<string, unknown>;
  /** "open-model" for Hugging Face models (never INFINITY HYPER 5.5); absent for HYPER 5.5. */
  family?: string;
  id?: string;
  base?: string;
  adapter?: string | null;
  device?: string;
  switch_error?: string;
}

export interface OpenModel {
  id: string;
  label: string;
  base: string;
  adapter: string | null;
  adapter_present: boolean | null;
  context_tokens: number;
  license?: string;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  model?: string;
}

export interface TaskSummary {
  task: string;
  mode: Mode;
  state: "running" | "finished" | "error";
  started_at: number;
  finished_at?: number;
  status?: string;
  reason?: string;
  error?: string;
}

export interface Status {
  product: string;
  model: ModelStatus;
  runtime: { connected: boolean; uptime_s: number };
  workspace: string | null;
  permission: Permission;
  agent: { busy: boolean; awaiting_approval: boolean; task: TaskSummary | null };
  approvals: { id: string }[];
  recent_errors: { time: number; message: string }[];
  last_seq: number;
}

export interface Checkpoint {
  path: string;
  run: string;
  step: number;
  parameters: number;
  metrics: Record<string, number>;
}

export interface Project {
  path: string;
  permission: Permission;
  name: string;
  languages: string[];
  frameworks: string[];
  package_managers: string[];
  entry_points: string[];
  test_commands: string[];
  instructions: string[];
  git: boolean;
  file_count: number;
  file_count_capped: boolean;
}

export interface RecentProject {
  path: string;
  name: string;
  exists: boolean;
  last_used: number;
  trust: Permission | null;
  last_task: { task: string; status: string } | null;
}

export interface TreeNode {
  name: string;
  path: string;
  type: "dir" | "file";
  sensitive?: boolean;
  size?: number;
  children?: TreeNode[];
}

export interface AgentEvent {
  seq: number;
  time: number;
  type: string;
  [key: string]: unknown;
}

export interface ApprovalRequest {
  id: string;
  tool: string;
  category: string;
  risk: string | null;
  summary: string;
  reasons: string[];
  diff: string | null;
  timeout_s: number;
}

export interface TaskReport {
  task_id: string;
  session_id: string;
  task: string;
  mode: Mode;
  status: string;
  summary: string;
  reason: string;
  plan: string[];
  modified_files: string[];
  checks: { command: string; exit_code: number | null; status: string; is_check: boolean }[];
  tokens_in: number;
  tokens_out: number;
  tokens_exact: boolean;
  elapsed_s: number;
  notes: string[];
}

export type CoreState = "idle" | "listening" | "thinking" | "executing" | "speaking" | "success" | "error";

export interface Plan {
  id: string;
  name: string;
  tagline: string;
  monthly_price_inr: number;
  paid: boolean;
  cook: boolean;
  api_access: boolean;
  monthly_token_credits: number | null;
  max_task_steps: number | null;
  requests_per_minute: number | null;
  priority: string;
  popular: boolean;
  features: string[];
  cta: string | null;
  accepting_new: boolean;
}

export interface Catalog {
  currency: "INR";
  yearly_billing: { enabled: boolean; note?: string };
  payment_provider: { configured: boolean; mode: string };
  plans: Plan[];
}

export interface Subscription {
  id: number;
  plan: string;
  status: string;
  billing_period: string;
  price_inr: number;
  source: string;
  current_period_start: number;
  current_period_end: number;
  cancel_at_period_end: number;
}

export interface Entitlement {
  plan: { id: string; name: string; monthly_price_inr: number; monthly_token_credits: number | null; max_task_steps: number | null };
  subscription: Subscription | null;
  balance: number;
  cook: boolean;
  cook_reason: string | null;
  api: boolean;
  owner_exempt: boolean;
}

export interface Account {
  id: number;
  email: string;
  role: "user" | "owner";
  status: string;
  workspace_roots: string[];
  created_at: number;
}

export interface Me {
  user: Account;
  csrf: string;
  entitlement: Entitlement;
}

export interface LedgerEntry {
  id: number;
  delta: number;
  kind: string;
  reason: string;
  ref: string | null;
  created_at: number;
}

export interface UsageSummary {
  days: number;
  tokens_in: number;
  tokens_out: number;
  requests: number;
  estimated_events: number;
  failed_events: number;
  daily: { day: number; tin: number; tout: number; n: number }[];
  by_source: { source: string; tin: number; tout: number; n: number }[];
}

export interface BillingMe {
  entitlement: Entitlement;
  ledger: LedgerEntry[];
  usage: UsageSummary;
  subscriptions: Subscription[];
}

export interface CheckoutResult {
  mode: "development" | string;
  activated: boolean;
  message: string;
  summary: { plan: string; plan_name: string; price_inr: number; currency: string; billing_period: string; monthly_token_credits: number | null };
}

export interface ApiKey {
  id: number;
  name: string;
  prefix: string;
  scopes: string[];
  status: "active" | "revoked";
  rate_limit_per_min: number;
  created_at: number;
  last_used_at: number | null;
  revoked_at: number | null;
  request_count: number;
  error_count: number;
  tokens_in: number;
  tokens_out: number;
}

export interface AdminUser {
  id: number;
  email: string;
  role: string;
  status: string;
  workspace_roots: string;
  created_at: number;
  last_login_at: number | null;
  last_active_at: number | null;
  plan: string;
  subscription_status: string | null;
  balance: number;
  usage_30d: number;
}

export interface AuditEvent {
  id: number;
  actor_id: number | null;
  action: string;
  target: string | null;
  success: number;
  reason: string | null;
  created_at: number;
}

export interface AdminOverview {
  users: { total: number; active_24h: number; suspended: number };
  subscriptions: Record<string, number>;
  active_by_plan: Record<string, number>;
  revenue_30d: { currency: string; gross_inr: number; refunds_inr: number; net_inr: number; verified_payments: number; refunds: number; failed_payments: number };
  usage_30d: UsageSummary;
  model: ModelStatus;
  tasks: (TaskSummary & { user_id: number; busy: boolean })[];
  security: AuditEvent[];
  audit_chain: { ok: boolean; checked: number; broken_at?: number };
  payment_provider: { configured: boolean; mode: string };
  ledger_duplicates: unknown[];
  database: { migrations: { version: number; applied_at: number }[]; path: string };
}

export interface BenchResult {
  id: number;
  created_at: number;
  suite: "lm" | "protocol" | "repair";
  checkpoint: string;
  run: string;
  step: number;
  model_sha256: string;
  config: Record<string, unknown>;
  metrics: Record<string, number | string | Record<string, unknown>> & {
    perplexity?: number; valid_action_rate?: number; tool_match_rate?: number; exact_action_rate?: number;
    verified_success_rate?: number; tasks?: number;
  };
  duration_s: number;
}

export interface Intelligence {
  product: string;
  model: ModelStatus & { tokenizer_fingerprint?: string; metrics?: Record<string, unknown> };
  promotion: { checkpoint: string; step: number; promoted_at: number; reason: string } | null;
  benchmarks: BenchResult[];
  index: ({ files: number; symbols: number; references: number; languages: Record<string, number>; parse_errors: Record<string, string> }) | { error: string } | null;
  memory: { experiences: number; project_notes: number; relevant_to_last_task: number };
  agent: { busy: boolean; awaiting_approval: boolean; task: TaskSummary | null };
  plan: string[] | null;
}
