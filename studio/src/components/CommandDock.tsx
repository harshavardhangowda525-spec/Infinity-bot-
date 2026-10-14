import { motion } from "framer-motion";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { Checkpoint, DockMode, ModelStatus, OpenModel } from "../lib/types";

export const DOCK_MODES: { id: DockMode; label: string; hint: string }[] = [
  { id: "chat", label: "Chat", hint: "Normal chat — a plain conversation with the loaded model (no files, no tools)" },
  { id: "ask", label: "Ask", hint: "Explain code and answer questions — read-only" },
  { id: "plan", label: "Plan", hint: "Inspect and produce an implementation plan — read-only" },
  { id: "code", label: "Code", hint: "Implement changes and run checks" },
  { id: "debug", label: "Debug", hint: "Reproduce, find root cause, fix, retest" },
  { id: "autonomous", label: "Autonomous", hint: "Plan, edit, run and iterate within limits" },
];

type SpeechRec = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: (e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void;
  onend: () => void;
  onerror: () => void;
  start: () => void;
  stop: () => void;
};

function speechRecognitionCtor(): (new () => SpeechRec) | null {
  const w = window as unknown as { SpeechRecognition?: new () => SpeechRec; webkitSpeechRecognition?: new () => SpeechRec };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export default function CommandDock({
  mode,
  onMode,
  onSubmit,
  onCancel,
  busy,
  disabledReason,
  model,
  checkpoints,
  openModels = [],
  onSelectModel,
  onAttach,
  workspaceName,
  taskText,
  onListening,
  amplitude,
  placeholder,
}: {
  mode: DockMode;
  onMode: (m: DockMode) => void;
  onSubmit: (text: string) => Promise<void> | void;
  onCancel: () => void;
  busy: boolean;
  disabledReason: string | null;
  model: ModelStatus | undefined;
  checkpoints: Checkpoint[];
  openModels?: OpenModel[];
  onSelectModel: (value: string) => void;
  onAttach: () => void;
  workspaceName: string | null;
  taskText: string | null;
  onListening: (v: boolean) => void;
  amplitude: React.MutableRefObject<number>;
  placeholder?: string;
}) {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [micOn, setMicOn] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);
  const micCleanup = useRef<() => void>();
  const SR = speechRecognitionCtor();

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

  useEffect(() => () => micCleanup.current?.(), []);

  const submit = async () => {
    const t = text.trim();
    if (!t || busy || disabledReason || sending) return;
    setSending(true);
    try {
      await onSubmit(t);
      setText("");
    } finally {
      setSending(false);
    }
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  const toggleMic = async () => {
    if (micOn) {
      micCleanup.current?.();
      return;
    }
    if (!SR) return;
    let stream: MediaStream | null = null;
    let ctx: AudioContext | null = null;
    let raf = 0;
    const rec = new SR();
    rec.lang = navigator.language || "en-US";
    rec.interimResults = true;
    rec.continuous = false;
    const stop = () => {
      cancelAnimationFrame(raf);
      try {
        rec.stop();
      } catch {
        /* already stopped */
      }
      stream?.getTracks().forEach((t) => t.stop());
      ctx?.close();
      amplitude.current = 0;
      setMicOn(false);
      onListening(false);
      micCleanup.current = undefined;
    };
    micCleanup.current = stop;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      ctx = new AudioContext();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      ctx.createMediaStreamSource(stream).connect(analyser);
      const buf = new Uint8Array(analyser.fftSize);
      const loop = () => {
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (const v of buf) sum += ((v - 128) / 128) ** 2;
        amplitude.current = Math.min(1, Math.sqrt(sum / buf.length) * 4);
        raf = requestAnimationFrame(loop);
      };
      loop();
    } catch {
      stop();
      return;
    }
    rec.onresult = (e) => {
      const parts = Array.from(e.results).map((r) => r[0].transcript);
      setText(parts.join(" "));
    };
    rec.onend = stop;
    rec.onerror = stop;
    rec.start();
    setMicOn(true);
    onListening(true);
  };

  const blocked = !!disabledReason;

  return (
    <motion.div layout className="glass-strong w-full max-w-[760px] rounded-2xl p-3 shadow-glass" transition={{ type: "spring", stiffness: 260, damping: 30 }}>
      {busy && taskText && (
        <div className="mb-2 flex items-center gap-2 rounded-lg bg-white/[0.03] px-3 py-2 text-xs text-slate-300">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-glow/60 motion-reduce:animate-none" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-cyan-glow" />
          </span>
          <span className="truncate">Running: {taskText}</span>
        </div>
      )}
      <div className="flex items-end gap-2">
        <label htmlFor="command" className="sr-only">
          {mode === "chat" ? "Message for INFINITY BOT" : "Task for INFINITY BOT"}
        </label>
        <textarea
          id="command"
          ref={ref}
          rows={2}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          placeholder={blocked ? disabledReason! : placeholder ?? (mode === "chat" ? "Message INFINITY BOT…" : "What should I build today?")}
          aria-describedby="dock-hint"
          className="scroll-thin min-h-[52px] flex-1 resize-none rounded-xl border border-white/5 bg-black/30 px-3.5 py-3 text-[15px] text-slate-100 placeholder:text-slate-500 focus:border-cyan-glow/40 focus:outline-none"
        />
        <div className="flex flex-col gap-2">
          {busy ? (
            <button
              onClick={onCancel}
              title="Cancel the running task"
              className="h-11 rounded-xl border border-amber-diag/40 bg-amber-diag/10 px-4 text-sm font-medium text-amber-diag hover:bg-amber-diag/20"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!text.trim() || blocked || sending}
              title="Send (Enter) · new line (Shift+Enter)"
              aria-label="Send task"
              className="h-11 rounded-xl bg-gradient-to-r from-cyan-glow to-electric px-5 text-sm font-semibold text-obsidian shadow-glow transition disabled:cursor-not-allowed disabled:opacity-30 disabled:shadow-none"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden>
                <path d="M5 12h14M13 6l6 6-6 6" />
              </svg>
            </button>
          )}
        </div>
      </div>
      <p id="dock-hint" className="sr-only">
        Press Enter to send and Shift+Enter for a new line.
      </p>
      <div className="mt-2.5 flex flex-wrap items-center gap-2">
        <button
          onClick={onAttach}
          title="Open or switch project (Ctrl+O)"
          className="flex items-center gap-1.5 rounded-lg border border-white/5 px-2.5 py-1.5 text-xs text-slate-300 hover:border-cyan-glow/30 hover:text-white"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
            <path d="M21.4 11.1l-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5" />
          </svg>
          <span className="max-w-[140px] truncate">{workspaceName ?? "Attach project"}</span>
        </button>
        <label className="sr-only" htmlFor="model-select">
          Model
        </label>
        <select
          id="model-select"
          value={model?.state !== "ready" ? "" : model.family === "open-model" ? `open:${model.id}` : model.checkpoint}
          onChange={(e) => e.target.value && onSelectModel(e.target.value)}
          disabled={busy || (checkpoints.length === 0 && openModels.length === 0)}
          title="Model (INFINITY HYPER 5.5 checkpoints and open models)"
          className="max-w-[230px] truncate rounded-lg border border-white/5 bg-black/30 px-2 py-1.5 text-xs text-slate-300"
        >
          {model?.state !== "ready" && (
            <option value="">{checkpoints.length || openModels.length ? "Select model" : "No checkpoint trained"}</option>
          )}
          {checkpoints.map((c) => (
            <option key={c.path} value={c.path}>
              HYPER 5.5 · {c.run} · step {c.step}
            </option>
          ))}
          {openModels.length > 0 && (
            <optgroup label="Open models (not HYPER 5.5)">
              {openModels.map((o) => (
                <option key={o.id} value={`open:${o.id}`} disabled={o.adapter_present === false}>
                  {o.label}
                  {o.adapter ? (o.adapter_present ? " + INFINITY adapter" : " + adapter (not trained yet)") : ""}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        {SR && (
          <button
            onClick={toggleMic}
            aria-pressed={micOn}
            title={micOn ? "Stop voice input" : "Voice input"}
            className={`rounded-lg border px-2 py-1.5 text-xs ${micOn ? "border-cyan-glow/50 text-cyan-glow" : "border-white/5 text-slate-400 hover:text-white"}`}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <rect x="9" y="3" width="6" height="11" rx="3" />
              <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
            </svg>
            <span className="sr-only">Voice input</span>
          </button>
        )}
        <div role="radiogroup" aria-label="Mode" className="ml-auto flex flex-wrap gap-1">
          {DOCK_MODES.map((m) => (
            <button
              key={m.id}
              role="radio"
              aria-checked={mode === m.id}
              title={m.hint}
              onClick={() => onMode(m.id)}
              className={`rounded-lg px-2.5 py-1.5 text-xs transition ${
                mode === m.id ? "bg-cyan-glow/15 text-cyan-glow ring-1 ring-cyan-glow/40" : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
