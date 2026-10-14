import { useEffect, useRef } from "react";
import type { ChatTurn } from "../lib/types";

/** Normal chat transcript. Each reply is labelled with the model that actually produced it. */
export default function ChatPanel({ turns, waiting, onClear }: { turns: ChatTurn[]; waiting: boolean; onClear: () => void }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => end.current?.scrollIntoView({ block: "end" }), [turns.length, waiting]);
  return (
    <section aria-label="Chat" className="glass-strong w-full max-w-[760px] rounded-2xl p-3">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-[10px] font-semibold tracking-[0.18em] text-slate-500">CHAT</h2>
        <button onClick={onClear} className="text-[11px] text-slate-400 hover:text-white">
          Clear
        </button>
      </div>
      <ol className="scroll-thin max-h-[38vh] space-y-2 overflow-y-auto pr-1" aria-live="polite">
        {turns.map((t, i) => (
          <li key={i} className={`flex ${t.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[85%] whitespace-pre-wrap [overflow-wrap:anywhere] rounded-xl px-3 py-2 text-[14px] leading-relaxed ${
                t.role === "user" ? "bg-cyan-glow/10 text-slate-100" : "bg-white/[0.04] text-slate-200"
              }`}
            >
              {t.content || <span className="text-slate-500">(empty reply)</span>}
              {t.role === "assistant" && t.model && <div className="mt-1 text-[10px] tracking-wide text-slate-500">{t.model}</div>}
            </div>
          </li>
        ))}
        {waiting && (
          <li className="flex justify-start">
            <div className="rounded-xl bg-white/[0.04] px-3 py-2 text-[13px] text-slate-400">Thinking…</div>
          </li>
        )}
        <div ref={end} />
      </ol>
    </section>
  );
}
