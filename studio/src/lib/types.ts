export type Mode = "ask" | "plan" | "review" | "code" | "debug" | "build" | "autonomous";
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
  metrics?: Record<string, number>;
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
