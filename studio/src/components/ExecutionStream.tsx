import { useEffect, useMemo, useRef, useState } from "react";
import type { AgentEvent } from "../lib/types";

interface Line {
  key: string;
  kind: "state" | "out" | "ok" | "err" | "info" | "warn";
  text: string;
}

/** Converts real agent events into terminal lines. Nothing is synthesised. */
export function eventsToLines(events: AgentEvent[]): Line[] {
  const lines: Line[] = [];
  for (const e of events) {
    const k = String(e.seq);
    switch (e.type) {
      case "task_start":
        lines.push({ key: k, kind: "info", text: `▶ task (${e.mode}): ${e.task}` });
        break;
      case "state":
        if (e.tool) lines.push({ key: k, kind: "state", text: `· step ${e.step} ${e.state} → ${e.tool} ${JSON.stringify(e.args ?? {}).slice(0, 160)}` });
        break;
      case "output":
        lines.push({ key: k, kind: "out", text: String(e.line).replace(/\n$/, "") });
        break;
      case "tool_result": {
        const first = String(e.observation ?? "").split("\n").slice(0, 2).join("  ");
        lines.push({ key: k, kind: e.ok ? "ok" : "err", text: `${e.ok ? "✓" : "✗"} ${first.slice(0, 220)}` });
        break;
      }
      case "invalid_output":
        lines.push({ key: k, kind: "warn", text: `! step ${e.step} invalid model output: ${e.error}` });
        break;
      case "approval_request":
        lines.push({ key: k, kind: "warn", text: `? approval needed: ${e.tool} — ${e.summary}` });
        break;
      case "approval_resolved":
        lines.push({ key: k, kind: e.approved ? "ok" : "warn", text: `${e.approved ? "✓ approved" : "✗ not approved"} (${e.reason})` });
        break;
      case "task_end":
        lines.push({
          key: k,
          kind: String(e.status).startsWith("completed") || e.status === "answered" ? "ok" : e.status === "cancelled" ? "info" : "err",
          text: `■ ${String(e.status).toUpperCase()} — ${e.reason}`,
        });
        break;
      case "rollback":
        lines.push({ key: k, kind: "info", text: `↺ rolled back: ${(e.restored as string[]).join(", ")}` });
        break;
      case "model":
        lines.push({ key: k, kind: e.error ? "err" : "info", text: e.error ? `model error: ${e.error}` : `model loaded: step ${(e.info as { step?: number })?.step}` });
        break;
    }
  }
  return lines;
}

const COLORS: Record<Line["kind"], string> = {
  state: "text-accent",
  out: "text-fg/80",
  ok: "text-ok",
  err: "text-bad",
  info: "text-cyan-glow",
  warn: "text-amber-diag",
};

export default function ExecutionStream({ events }: { events: AgentEvent[] }) {
  const lines = useMemo(() => eventsToLines(events), [events]);
  const box = useRef<HTMLDivElement>(null);
  const [follow, setFollow] = useState(true);
  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [lines, follow]);
  return (
    <div
      ref={box}
      onScroll={(e) => {
        const el = e.currentTarget;
        setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
      }}
      className="scroll-thin h-full overflow-auto p-3 font-mono text-[12px] leading-[1.55]"
      role="log"
      aria-live="polite"
      aria-label="Execution stream"
    >
      {lines.length === 0 ? (
        <p className="text-muted">No activity yet. Tool calls, command output and results appear here as they happen.</p>
      ) : (
        lines.map((l) => (
          <div key={l.key} className={`whitespace-pre-wrap break-words ${COLORS[l.kind]}`}>
            {l.text}
          </div>
        ))
      )}
    </div>
  );
}
