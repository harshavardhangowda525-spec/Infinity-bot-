import type { AgentEvent, CoreState } from "./types";

export interface CoreInputs {
  connected: boolean;
  busy: boolean;
  awaitingApproval: boolean;
  listening: boolean;
  speaking: boolean;
  /** last "state" event emitted by the agent for the running task */
  lastAgentState: string | null;
  /** most recent task_end event, if any */
  lastEnd: { status: string; time: number } | null;
  now: number;
}

const OK = new Set(["completed_verified", "completed_unverified", "answered"]);
const BAD = new Set(["blocked", "partial", "error"]);
export const OUTCOME_FLASH_MS = 4000;

/**
 * Pure mapping from *real* runtime facts to the core's visual state.
 * Nothing here invents activity: with no task and no microphone/speech, it is idle.
 */
export function deriveCoreState(i: CoreInputs): CoreState {
  if (!i.connected) return "error";
  if (i.listening) return "listening";
  if (i.speaking) return "speaking";
  if (i.busy) {
    if (i.awaitingApproval) return "idle";
    if (i.lastAgentState && i.lastAgentState !== "thinking") return "executing";
    return "thinking";
  }
  if (i.lastEnd && i.now - i.lastEnd.time * 1000 < OUTCOME_FLASH_MS) {
    if (OK.has(i.lastEnd.status)) return "success";
    if (BAD.has(i.lastEnd.status)) return "error";
  }
  return "idle";
}

const AGENT_LABELS: Record<string, string> = {
  thinking: "THINKING",
  inspecting: "INSPECTING PROJECT",
  applying_changes: "APPLYING CHANGES",
  running_checks: "RUNNING TESTS",
  executing: "EXECUTING",
};

export function statusLabel(i: CoreInputs, core: CoreState): string {
  if (!i.connected) return "RUNTIME DISCONNECTED";
  if (core === "listening") return "LISTENING";
  if (core === "speaking") return "SPEAKING";
  if (i.busy && i.awaitingApproval) return "AWAITING YOUR APPROVAL";
  if (i.busy) return AGENT_LABELS[i.lastAgentState ?? "thinking"] ?? "WORKING";
  if (core === "success") return "COMPLETED";
  if (core === "error") return "NEEDS ATTENTION";
  return "IDLE";
}

export function lastOf(events: AgentEvent[], type: string): AgentEvent | undefined {
  for (let k = events.length - 1; k >= 0; k--) if (events[k].type === type) return events[k];
  return undefined;
}
