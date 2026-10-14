import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import type { Mode } from "../lib/types";
import { Icon } from "./Icons";

export type WorkMode = "normal" | "cook";

export const AGENT_MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "ask", label: "Ask", hint: "Explain code and answer questions — read-only" },
  { id: "plan", label: "Plan", hint: "Inspect and produce an implementation plan — read-only" },
  { id: "code", label: "Code", hint: "Implement changes and run checks" },
  { id: "debug", label: "Debug", hint: "Reproduce, find the root cause, fix, retest" },
  { id: "autonomous", label: "Autonomous", hint: "Plan, edit, run and iterate within limits" },
];

const MAX_ATTACH = 100_000;

type SpeechRec = {
  lang: string; interimResults: boolean; continuous: boolean;
  onresult: (e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void;
  onend: () => void; onerror: () => void; start: () => void; stop: () => void;
};
function speechCtor(): (new () => SpeechRec) | null {
  const w = window as unknown as { SpeechRecognition?: new () => SpeechRec; webkitSpeechRecognition?: new () => SpeechRec };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

/** Message composer for Normal chat and Cook. Enter sends, Shift+Enter adds a line, Esc stops a running reply. */
export default function Composer({
  mode, onMode, agentMode, onAgentMode, onSubmit, onStop, busy, disabledReason, placeholder, modelSlot,
  onAttachProject, workspaceName, size = "dock", autoFocus, prefill, onListening,
}: {
  mode: WorkMode;
  onMode: (m: WorkMode) => void;
  agentMode?: Mode;
  onAgentMode?: (m: Mode) => void;
  onSubmit: (text: string) => Promise<void> | void;
  onStop?: () => void;
  busy: boolean;
  disabledReason?: string | null;
  placeholder?: string;
  modelSlot?: ReactNode;
  onAttachProject?: () => void;
  workspaceName?: string | null;
  size?: "hero" | "dock";
  autoFocus?: boolean;
  prefill?: { text: string; n: number };
  onListening?: (v: boolean) => void;
}) {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [mic, setMic] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);
  const file = useRef<HTMLInputElement>(null);
  const rec = useRef<SpeechRec | null>(null);
  const SR = typeof window !== "undefined" ? speechCtor() : null;
  const cook = mode === "cook";

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, size === "hero" ? 260 : 220)}px`;
  }, [text, size]);

  useEffect(() => {
    if (prefill) {
      setText(prefill.text);
      ref.current?.focus();
    }
  }, [prefill]);

  useEffect(() => () => rec.current?.stop(), []);

  const blocked = !!disabledReason;
  const submit = async () => {
    const t = text.trim();
    if (!t || busy || blocked || sending) return;
    setSending(true);
    setNote(null);
    try {
      await onSubmit(t);
      setText("");
    } catch {
      /* the caller shows the error; keep the text so nothing is lost */
    } finally {
      setSending(false);
    }
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    } else if (e.key === "Escape" && busy && onStop) {
      e.preventDefault();
      onStop();
    }
  };

  const attach = () => (cook ? onAttachProject?.() : file.current?.click());
  const onFile = async (f: File | undefined) => {
    if (!f) return;
    if (f.size > MAX_ATTACH) {
      setNote(`${f.name} is larger than 100 KB; attach a smaller text file.`);
      return;
    }
    const body = await f.text();
    if (body.includes("\u0000")) {
      setNote(`${f.name} looks like a binary file; only text files can be attached.`);
      return;
    }
    const ext = f.name.split(".").pop() ?? "";
    setText((t) => `${t}${t ? "\n\n" : ""}${f.name}:\n\`\`\`${ext}\n${body.replace(/\n$/, "")}\n\`\`\``);
    setNote(`Attached ${f.name} as text.`);
    ref.current?.focus();
  };

  const toggleMic = () => {
    if (mic) {
      rec.current?.stop();
      return;
    }
    if (!SR) return;
    const r = new SR();
    r.lang = navigator.language || "en-US";
    r.interimResults = true;
    r.continuous = false;
    const base = text;
    r.onresult = (e) => setText(`${base}${base ? " " : ""}${Array.from(e.results).map((x) => x[0].transcript).join(" ")}`);
    r.onend = r.onerror = () => {
      setMic(false);
      onListening?.(false);
    };
    rec.current = r;
    r.start();
    setMic(true);
    onListening?.(true);
  };

  return (
    <div
      className={`w-full rounded-2xl border bg-surface shadow-card transition-colors ${
        cook ? "border-cook/35 focus-within:border-cook/60" : "border-line focus-within:border-accent/50"
      }`}
    >
      <label htmlFor="composer" className="sr-only">
        {cook ? "Coding task" : "Message"}
      </label>
      <textarea
        id="composer"
        ref={ref}
        rows={size === "hero" ? 3 : 1}
        value={text}
        autoFocus={autoFocus}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        placeholder={blocked ? disabledReason! : placeholder ?? (cook ? "Describe a coding task…" : "Message Infinity Bot…")}
        aria-describedby="composer-hint"
        className={`scroll-thin block w-full resize-none bg-transparent px-4 text-[15px] text-fg placeholder:text-muted focus:outline-none ${
          size === "hero" ? "min-h-[88px] pt-4" : "min-h-[52px] pt-3.5"
        }`}
      />
      <p id="composer-hint" className="sr-only">
        Enter to send, Shift+Enter for a new line, Escape to stop a running reply.
      </p>
      <div className="flex flex-wrap items-center gap-1.5 px-2.5 pb-2.5 pt-1">
        <button onClick={attach} title={cook ? "Open a project folder" : "Attach a text file"} aria-label={cook ? "Open project" : "Attach file"}
          className="inline-flex items-center gap-1.5 rounded-lg p-2 text-muted transition hover:bg-hover hover:text-fg">
          {cook ? Icon.folder : Icon.clip}
          {cook && <span className="max-w-[140px] truncate text-xs">{workspaceName ?? "Open project"}</span>}
        </button>
        <input ref={file} type="file" className="hidden" accept=".txt,.md,.py,.js,.jsx,.ts,.tsx,.json,.css,.html,.yml,.yaml,.toml,.sh,.csv,.sql,.java,.c,.cpp,.h,.go,.rs,.rb,.php,text/*"
          onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} />
        {SR && (
          <button onClick={toggleMic} aria-pressed={mic} title={mic ? "Stop voice input" : "Voice input"} aria-label="Voice input"
            className={`rounded-lg p-2 transition hover:bg-hover ${mic ? "text-accent" : "text-muted hover:text-fg"}`}>
            {Icon.mic}
          </button>
        )}
        <div className="ml-auto flex flex-wrap items-center justify-end gap-1.5">
          {modelSlot}
          {cook && onAgentMode && (
            <div className="relative inline-flex items-center">
              <label htmlFor="agent-mode" className="sr-only">Cook mode</label>
              <select id="agent-mode" value={agentMode} onChange={(e) => onAgentMode(e.target.value as Mode)}
                title={AGENT_MODES.find((m) => m.id === agentMode)?.hint}
                className="cursor-pointer appearance-none rounded-lg border border-line bg-surface py-1.5 pl-2.5 pr-7 text-xs text-fg/90 hover:bg-hover">
                {AGENT_MODES.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
              <span className="pointer-events-none absolute right-2 text-muted">{Icon.chevron}</span>
            </div>
          )}
          <div role="radiogroup" aria-label="Mode" className="inline-flex rounded-lg border border-line p-0.5 text-xs">
            {(["normal", "cook"] as const).map((m) => (
              <button key={m} role="radio" aria-checked={mode === m} onClick={() => onMode(m)}
                className={`rounded-md px-2.5 py-1 font-medium transition ${
                  mode === m ? (m === "cook" ? "bg-cook text-white" : "bg-accent text-white") : "text-muted hover:text-fg"
                }`}>
                {m === "normal" ? "Normal" : "Cook"}
              </button>
            ))}
          </div>
          {busy && onStop ? (
            <button onClick={onStop} title="Stop (Esc)" aria-label="Stop generating"
              className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-fg text-surface transition hover:opacity-90">
              {Icon.stop}
            </button>
          ) : (
            <button onClick={submit} disabled={!text.trim() || blocked || sending || busy} title="Send (Enter)" aria-label="Send"
              className={`inline-flex h-9 w-9 items-center justify-center rounded-lg text-white transition disabled:cursor-not-allowed disabled:opacity-35 ${
                cook ? "bg-cook hover:bg-cook/90" : "bg-accent hover:bg-accent/90"
              }`}>
              {Icon.send}
            </button>
          )}
        </div>
      </div>
      {note && <p className="px-4 pb-2 text-xs text-muted" role="status">{note}</p>}
    </div>
  );
}
