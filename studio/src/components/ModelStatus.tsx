import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { modelName } from "../lib/format";
import type { Status } from "../lib/types";

function Row({ label, value, tone = "text-fg" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex justify-between gap-6 py-1 text-xs">
      <span className="text-muted">{label}</span>
      <span className={`text-right ${tone}`}>{value}</span>
    </div>
  );
}

export function modelPillText(status: Status | null): { text: string; tone: string } {
  if (!status) return { text: "Runtime offline", tone: "text-bad" };
  const m = status.model;
  if (m.state === "loading") return { text: "Loading model…", tone: "text-fg/80" };
  if (m.state !== "ready") return { text: "No model · unavailable", tone: "text-amber-diag" };
  return { text: modelName(m), tone: "text-fg/80" };
}

/** Compact model pill that opens a status popover with real runtime facts only. */
export default function ModelStatus({ status, connected }: { status: Status | null; connected: boolean }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);
  const pill = modelPillText(connected ? status : null);
  const m = status?.model;
  const agent = status?.agent;
  const agentText = !agent ? "unknown" : agent.awaiting_approval ? "waiting for approval" : agent.busy ? "busy" : "idle";
  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-haspopup="dialog"
        className="flex max-w-[52vw] items-center gap-2 rounded-lg px-2.5 py-1.5 text-xs text-muted transition hover:bg-hover sm:max-w-[320px]"
      >
        <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${status?.model.state === "ready" && connected ? "bg-ok" : "bg-warn"}`} aria-hidden />
        <span className={`truncate ${pill.tone}`}>{pill.text}</span>
      </button>
      <AnimatePresence>
        {open && (
          <div className="absolute right-0 top-10 z-50 w-[min(360px,90vw)]">
          <motion.div
            role="dialog"
            aria-label="Model and agent status"
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="popover rounded-xl border border-line bg-surface p-4"
          >
            <h3 className="mb-2 text-xs font-semibold text-fg">Model</h3>
            <Row label="Model" value={m?.state === "ready" ? m.product ?? "" : "—"} />
            {m?.state === "ready" ? (
              <>
                <Row label="Implementation" value={m.implementation ?? ""} />
                <Row label="Locality" value="Local model connected" tone="text-ok" />
                {m.family === "open-model" ? (
                  <>
                    <Row label="Base model" value={m.base ?? ""} />
                    <Row label="Adapter" value={m.adapter ?? "none"} />
                  </>
                ) : m.variant === "fine-tuned" ? (
                  <>
                    <Row label="Original (frozen)" value={`INFINITY HYPER 5.5 · ${m.run} · step ${m.step}`} />
                    <Row label="Adapter (LoRA, not merged)" value={m.adapter_name ?? ""} />
                  </>
                ) : (
                  <Row label="Run · step" value={`${m.run} · ${m.step}`} />
                )}
                <Row label="Parameters" value={(m.parameters ?? 0).toLocaleString()} />
                <Row label="Context window" value={`${m.context_length} tokens`} />
                {m.metrics?.val_loss !== undefined && <Row label="Val loss (at save)" value={String(m.metrics.val_loss)} />}
              </>
            ) : (
              <Row label="Model" value={m?.state === "loading" ? "loading" : `Unavailable — ${m?.reason ?? "runtime offline"}`} tone="text-amber-diag" />
            )}
            {m?.switch_error && <Row label="Last switch failed" value={m.switch_error} tone="text-amber-diag" />}
            <div className="my-2 border-t border-line" />
            <Row label="Runtime" value={connected ? "connected" : "disconnected"} tone={connected ? "text-ok" : "text-bad"} />
            <Row label="Agent" value={agentText} tone={agent?.awaiting_approval ? "text-amber-diag" : "text-fg"} />
            <Row label="Task" value={agent?.task ? `${agent.task.state}${agent.task.status ? ` · ${agent.task.status}` : ""}` : "none"} />
            <Row label="Permission mode" value={status?.permission ?? "—"} />
            <Row label="Remote providers" value="none (disabled by design)" />
            {status && status.recent_errors.length > 0 && (
              <>
                <div className="my-2 border-t border-line" />
                <h4 className="mb-1 text-xs font-medium text-muted">Recent errors</h4>
                <ul className="scroll-thin max-h-28 space-y-1 overflow-auto text-[11px] text-bad">
                  {status.recent_errors
                    .slice()
                    .reverse()
                    .map((e) => (
                      <li key={e.time + e.message}>
                        {new Date(e.time * 1000).toLocaleTimeString()} — {e.message}
                      </li>
                    ))}
                </ul>
              </>
            )}
          </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
