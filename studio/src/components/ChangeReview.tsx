import { useState } from "react";
import { api } from "../lib/api";
import type { FileChange } from "../lib/store";
import type { ApprovalRequest, TaskReport } from "../lib/types";
import { DiffView } from "./CodeViewer";

const RISK_STYLE: Record<string, string> = {
  safe: "text-slate-300 border-white/10",
  check: "text-sky-300 border-sky-300/30",
  medium: "text-amber-diag border-amber-diag/40",
  high: "text-red-300 border-red-300/40",
};

export function ApprovalCard({ req }: { req: ApprovalRequest }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const decide = async (approve: boolean) => {
    setBusy(true);
    try {
      await api.approve(req.id, approve);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };
  return (
    <div className="rounded-xl border border-amber-diag/30 bg-amber-diag/[0.06] p-3" role="alertdialog" aria-label={`Approval needed: ${req.tool}`}>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="font-semibold tracking-wider text-amber-diag">APPROVAL NEEDED</span>
        <span className="rounded border border-white/10 px-1.5 py-0.5 font-mono text-slate-300">{req.tool}</span>
        <span className="text-slate-400">{req.category}</span>
        {req.risk && <span className={`rounded border px-1.5 py-0.5 ${RISK_STYLE[req.risk] ?? ""}`}>risk: {req.risk}</span>}
      </div>
      <p className="mt-2 break-all font-mono text-[12.5px] text-slate-100">{req.summary}</p>
      {req.reasons.length > 0 && (
        <ul className="mt-1.5 list-disc pl-5 text-xs text-slate-400">
          {req.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      )}
      {req.diff && (
        <div className="scroll-thin mt-2 max-h-56 overflow-auto rounded-lg bg-black/30">
          <DiffView diff={req.diff} />
        </div>
      )}
      {error && <p className="mt-2 text-xs text-red-300">{error}</p>}
      <div className="mt-3 flex gap-2">
        <button disabled={busy} onClick={() => decide(true)} className="rounded-lg bg-cyan-glow/90 px-3 py-1.5 text-xs font-semibold text-obsidian hover:bg-cyan-glow disabled:opacity-40">
          Approve
        </button>
        <button disabled={busy} onClick={() => decide(false)} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-slate-200 hover:bg-white/5 disabled:opacity-40">
          Reject
        </button>
        <span className="ml-auto self-center text-[11px] text-slate-500">auto-rejects after {Math.round(req.timeout_s / 60)} min</span>
      </div>
    </div>
  );
}

export default function ChangeReview({ approvals, changes, report, busy, onInspect, onRolledBack }: {
  approvals: ApprovalRequest[];
  changes: FileChange[];
  report: TaskReport | null;
  busy: boolean;
  onInspect: (path: string) => void;
  onRolledBack: () => void;
}) {
  const [openIdx, setOpenIdx] = useState<number | null>(null);
  const [rollback, setRollback] = useState<{ state: "idle" | "confirm" | "working" | "done" | "error"; msg?: string }>({ state: "idle" });

  const doRollback = async () => {
    if (!report) return;
    setRollback({ state: "working" });
    try {
      const r = await api.rollback(report.session_id);
      setRollback({ state: "done", msg: `Restored ${r.restored.length} path(s)` });
      onRolledBack();
    } catch (e) {
      setRollback({ state: "error", msg: (e as Error).message });
    }
  };

  return (
    <div className="space-y-3 p-3">
      {approvals.map((a) => (
        <ApprovalCard key={a.id} req={a} />
      ))}
      {changes.length === 0 && approvals.length === 0 && <p className="text-sm text-slate-500">No file changes in this task.</p>}
      {changes.map((c, i) => (
        <div key={`${c.path}-${c.step}-${i}`} className="overflow-hidden rounded-xl border border-white/5">
          <div className="flex items-center gap-2 bg-white/[0.02] px-3 py-2 text-xs">
            <span className="font-mono text-slate-200">{c.path}</span>
            <span className="text-slate-500">step {c.step}</span>
            <span className="ml-auto flex gap-1">
              <button onClick={() => setOpenIdx(openIdx === i ? null : i)} className="rounded px-2 py-1 text-slate-300 hover:bg-white/5" aria-expanded={openIdx === i}>
                {openIdx === i ? "Hide diff" : "Diff"}
              </button>
              <button onClick={() => onInspect(c.path)} className="rounded px-2 py-1 text-slate-300 hover:bg-white/5">
                Inspect
              </button>
            </span>
          </div>
          {openIdx === i && (
            <div className="scroll-thin max-h-72 overflow-auto bg-black/30">
              <DiffView diff={c.diff} />
            </div>
          )}
        </div>
      ))}
      {report && report.modified_files.length > 0 && !busy && (
        <div className="rounded-xl border border-white/5 p-3 text-xs">
          {rollback.state === "confirm" ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-slate-300">Restore {report.modified_files.length} file(s) to their state before this task?</span>
              <button onClick={doRollback} className="rounded-lg bg-amber-diag/90 px-3 py-1.5 font-semibold text-obsidian">
                Restore
              </button>
              <button onClick={() => setRollback({ state: "idle" })} className="rounded-lg border border-white/10 px-3 py-1.5 text-slate-200">
                Cancel
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setRollback({ state: "confirm" })}
                disabled={rollback.state === "working" || rollback.state === "done"}
                className="rounded-lg border border-white/10 px-3 py-1.5 text-slate-200 hover:bg-white/5 disabled:opacity-40"
              >
                Undo all changes from this task
              </button>
              {rollback.msg && <span className={rollback.state === "error" ? "text-red-300" : "text-emerald-300"}>{rollback.msg}</span>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
