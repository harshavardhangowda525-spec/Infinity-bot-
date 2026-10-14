import { useEffect, useRef, useState } from "react";
import { displayModel } from "../lib/format";
import type { ChatTurn } from "../lib/types";
import { Icon } from "./Icons";
import Markdown, { CopyButton } from "./Markdown";

/** Conversation layout: user turns as quiet bubbles, replies as plain text with Markdown (no oversized cards). */
export default function ChatThread({
  turns, waiting, onRegenerate, onEdit,
}: {
  turns: ChatTurn[];
  waiting: boolean;
  onRegenerate: () => void;
  onEdit: (index: number, text: string) => void;
}) {
  const end = useRef<HTMLDivElement>(null);
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  useEffect(() => end.current?.scrollIntoView?.({ block: "end", behavior: "smooth" }), [turns.length, waiting]);
  const lastAssistant = turns.map((t) => t.role).lastIndexOf("assistant");

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-6 pt-6 sm:px-6" aria-live="polite">
      <ol className="space-y-6">
        {turns.map((t, i) =>
          t.role === "user" ? (
            <li key={i} className="group fade-in flex flex-col items-end">
              {editing === i ? (
                <div className="w-full max-w-[85%] rounded-2xl border border-line bg-surface p-2">
                  <label className="sr-only" htmlFor={`edit-${i}`}>Edit message</label>
                  <textarea id={`edit-${i}`} value={draft} onChange={(e) => setDraft(e.target.value)} rows={3} autoFocus
                    className="block w-full resize-y bg-transparent p-2 text-[15px] text-fg focus:outline-none" />
                  <div className="flex justify-end gap-2 p-1">
                    <button onClick={() => setEditing(null)} className="rounded-lg px-3 py-1.5 text-sm text-muted hover:bg-hover hover:text-fg">Cancel</button>
                    <button disabled={!draft.trim() || waiting} onClick={() => { setEditing(null); onEdit(i, draft.trim()); }}
                      className="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent/90 disabled:opacity-40">
                      Save &amp; send
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl bg-hover px-4 py-2.5 text-[15px] leading-7 text-fg [overflow-wrap:anywhere]">
                    {t.content}
                  </div>
                  <div className="mt-1 flex gap-0.5 opacity-100 transition sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100">
                    <CopyButton text={t.content} />
                    <button onClick={() => { setEditing(i); setDraft(t.content); }} disabled={waiting} aria-label="Edit message" title="Edit and resend"
                      className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-xs text-muted hover:bg-hover hover:text-fg disabled:opacity-40">
                      {Icon.edit}<span className="sr-only sm:not-sr-only">Edit</span>
                    </button>
                  </div>
                </>
              )}
            </li>
          ) : (
            <li key={i} className="fade-in flex gap-3">
              <span aria-hidden className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-line bg-surface text-accent">
                {Icon.logo(16)}
              </span>
              <div className="min-w-0 flex-1">
                {t.content ? <Markdown text={t.content} /> : <p className="text-sm text-muted">(empty reply)</p>}
                <div className="mt-2 flex flex-wrap items-center gap-0.5 text-xs text-muted">
                  <CopyButton text={t.content} />
                  {i === lastAssistant && (
                    <button onClick={onRegenerate} disabled={waiting} aria-label="Regenerate response" title="Regenerate"
                      className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 hover:bg-hover hover:text-fg disabled:opacity-40">
                      {Icon.refresh}<span className="sr-only sm:not-sr-only">Regenerate</span>
                    </button>
                  )}
                  {t.stopped && <span className="ml-1 rounded bg-hover px-1.5 py-0.5">Stopped</span>}
                  {t.model && <span className="ml-auto truncate pl-2 text-[11px]">{displayModel(t.model)}</span>}
                </div>
              </div>
            </li>
          ),
        )}
        {waiting && (
          <li className="fade-in flex gap-3" aria-label="Generating">
            <span aria-hidden className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-line bg-surface text-accent">
              {Icon.logo(16)}
            </span>
            <span className="typing mt-3 inline-flex gap-1" role="status" aria-label="Infinity Bot is generating a reply">
              <span className="h-1.5 w-1.5 rounded-full bg-muted" />
              <span className="h-1.5 w-1.5 rounded-full bg-muted" />
              <span className="h-1.5 w-1.5 rounded-full bg-muted" />
            </span>
          </li>
        )}
      </ol>
      <div ref={end} />
    </div>
  );
}
