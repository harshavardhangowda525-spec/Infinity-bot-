import { useMemo, useState } from "react";
import { api } from "../lib/api";
import type { ConversationSummary } from "../lib/types";
import { Icon } from "../ui/Icons";

const when = (t: number) => {
  const d = new Date(t * 1000);
  const days = Math.floor((Date.now() - d.getTime()) / 86_400_000);
  return days === 0 ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : days < 7 ? `${days}d ago` : d.toLocaleDateString();
};

/** All saved Normal-chat conversations: search, open, rename, delete. */
export default function HistoryView({ conversations, onOpen, onChanged }: {
  conversations: ConversationSummary[];
  onOpen: (id: number) => void;
  onChanged: () => void;
}) {
  const [q, setQ] = useState("");
  const [renaming, setRenaming] = useState<number | null>(null);
  const [title, setTitle] = useState("");
  const [confirm, setConfirm] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const shown = useMemo(() => conversations.filter((c) => c.title.toLowerCase().includes(q.toLowerCase())), [conversations, q]);

  const act = async (fn: () => Promise<unknown>) => {
    setError(null);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
        <h1 className="text-xl font-semibold text-fg">Chat history</h1>
        <p className="mt-1 text-sm text-muted">Conversations are saved on this computer, in your account only.</p>
        <div className="relative mt-5">
          <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted">{Icon.search}</span>
          <label htmlFor="history-search" className="sr-only">Search conversations</label>
          <input id="history-search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search conversations" className="input pl-9" />
        </div>
        {error && <p role="alert" className="mt-3 text-sm text-bad">{error}</p>}
        <ul className="mt-4 divide-y divide-line overflow-hidden rounded-xl border border-line bg-surface">
          {shown.length === 0 && <li className="px-4 py-8 text-center text-sm text-muted">{conversations.length ? "No matches." : "No conversations yet. Start one from New chat."}</li>}
          {shown.map((c) => (
            <li key={c.id} className="group flex items-center gap-3 px-4 py-3">
              {renaming === c.id ? (
                <form className="flex flex-1 gap-2" onSubmit={(e) => { e.preventDefault(); act(() => api.renameConversation(c.id, title)).then(() => setRenaming(null)); }}>
                  <label htmlFor={`rename-${c.id}`} className="sr-only">Title</label>
                  <input id={`rename-${c.id}`} value={title} onChange={(e) => setTitle(e.target.value)} className="input py-1.5" autoFocus />
                  <button className="btn-primary py-1.5" disabled={!title.trim()}>Save</button>
                  <button type="button" onClick={() => setRenaming(null)} className="btn-secondary py-1.5">Cancel</button>
                </form>
              ) : (
                <>
                  <button onClick={() => onOpen(c.id)} className="min-w-0 flex-1 text-left">
                    <span className="block truncate text-sm font-medium text-fg">{c.title}</span>
                    <span className="text-xs text-muted">{c.messages} messages · {when(c.updated_at)}</span>
                  </button>
                  {confirm === c.id ? (
                    <span className="flex items-center gap-2 text-xs">
                      <span className="text-muted">Delete?</span>
                      <button onClick={() => act(() => api.deleteConversation(c.id)).then(() => setConfirm(null))} className="rounded-md bg-bad px-2 py-1 font-medium text-white">Delete</button>
                      <button onClick={() => setConfirm(null)} className="rounded-md px-2 py-1 text-muted hover:bg-hover">Keep</button>
                    </span>
                  ) : (
                    <span className="flex gap-0.5 opacity-100 transition sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100">
                      <button onClick={() => { setRenaming(c.id); setTitle(c.title); }} aria-label={`Rename ${c.title}`} className="rounded-md p-1.5 text-muted hover:bg-hover hover:text-fg">{Icon.edit}</button>
                      <button onClick={() => setConfirm(c.id)} aria-label={`Delete ${c.title}`} className="rounded-md p-1.5 text-muted hover:bg-hover hover:text-bad">{Icon.trash}</button>
                    </span>
                  )}
                </>
              )}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
