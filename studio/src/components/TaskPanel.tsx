import type { AgentEvent, TaskReport } from "../lib/types";

const STATUS_STYLE: Record<string, string> = {
  completed_verified: "text-ok border-ok/30",
  completed_unverified: "text-accent border-accent/30",
  answered: "text-accent border-accent/30",
  partial: "text-amber-diag border-amber-diag/40",
  blocked: "text-bad border-bad/30",
  cancelled: "text-fg/80 border-line",
  error: "text-bad border-bad/30",
};

const STATUS_TEXT: Record<string, string> = {
  completed_verified: "Completed · verified",
  completed_unverified: "Completed · not verified",
  answered: "Answered",
  partial: "Partially completed",
  blocked: "Blocked",
  cancelled: "Cancelled",
  error: "Error",
};

export default function TaskPanel({ taskText, plan, events, report, busy }: {
  taskText: string | null;
  plan: string[];
  events: AgentEvent[];
  report: TaskReport | null;
  busy: boolean;
}) {
  const steps = events.filter((e) => e.type === "tool_result");
  const lastState = [...events].reverse().find((e) => e.type === "state");
  const invalid = events.filter((e) => e.type === "invalid_output").length;
  const end = [...events].reverse().find((e) => e.type === "task_end");
  if (!taskText) return <p className="p-4 text-sm text-muted">No task yet. Describe one in the command dock.</p>;
  return (
    <div className="space-y-4 p-3.5 text-sm">
      <div>
        <h3 className="text-[11px] font-semibold tracking-[0.14em] text-muted">REQUEST</h3>
        <p className="mt-1 text-fg">{taskText}</p>
      </div>
      {busy && lastState && (
        <p className="text-xs text-cyan-glow">
          Step {String(lastState.step)} · {String(lastState.state).replace("_", " ")}
          {lastState.tool ? ` · ${lastState.tool}` : ""}
        </p>
      )}
      <div>
        <h3 className="text-[11px] font-semibold tracking-[0.14em] text-muted">PLAN</h3>
        {plan.length ? (
          <ol className="mt-1 list-decimal space-y-1 pl-5 text-fg/80">
            {plan.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ol>
        ) : (
          <p className="mt-1 text-xs text-muted">The model has not produced a plan{busy ? " yet" : ""}.</p>
        )}
      </div>
      <div>
        <h3 className="text-[11px] font-semibold tracking-[0.14em] text-muted">STEPS</h3>
        {steps.length ? (
          <ul className="mt-1 space-y-1">
            {steps.map((s) => (
              <li key={s.seq} className="flex items-center gap-2 text-xs">
                <span className={s.ok ? "text-ok" : "text-bad"}>{s.ok ? "✓" : "✗"}</span>
                <span className="font-mono text-fg/80">{String(s.tool)}</span>
                <span className="text-muted">step {String(s.step)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-1 text-xs text-muted">No tool calls{busy ? " yet" : ""}.</p>
        )}
        {invalid > 0 && <p className="mt-1 text-xs text-amber-diag">{invalid} invalid model output(s) rejected by the runtime.</p>}
      </div>
      {(report || end) && (
        <div className="space-y-2 rounded-xl border border-line p-3">
          {(() => {
            const status = report?.status ?? String(end?.status);
            return (
              <span className={`inline-block rounded border px-2 py-0.5 text-xs ${STATUS_STYLE[status] ?? ""}`}>{STATUS_TEXT[status] ?? status}</span>
            );
          })()}
          <p className="text-xs text-muted">{report?.reason ?? String(end?.reason ?? "")}</p>
          {report?.summary && <p className="whitespace-pre-wrap text-fg">{report.summary}</p>}
          {report && report.checks.length > 0 && (
            <ul className="space-y-1 text-xs">
              {report.checks.map((c, i) => (
                <li key={i} className="font-mono">
                  <span className={c.exit_code === 0 && c.status === "exited" ? "text-ok" : "text-bad"}>
                    [{c.exit_code === 0 && c.status === "exited" ? "pass" : "fail"}]
                  </span>{" "}
                  <span className="text-fg/80">{c.command}</span>
                </li>
              ))}
            </ul>
          )}
          {report && (
            <p className="text-[11px] text-muted">
              {report.elapsed_s.toFixed(1)}s · tokens in {report.tokens_in} / out {report.tokens_out}
              {report.tokens_exact ? " (exact)" : " (estimated)"}
            </p>
          )}
          {report?.notes.map((n) => (
            <p key={n} className="text-[11px] text-muted">
              · {n}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
