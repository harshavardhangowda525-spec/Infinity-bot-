import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

export interface Settings {
  readAloud: boolean;
  view: "minimal" | "workspace";
}

const KEY = "infinity.settings";

export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { readAloud: false, view: "minimal", ...JSON.parse(raw) };
  } catch {
    /* storage unavailable */
  }
  return { readAloud: false, view: "minimal" };
}

export function saveSettings(s: Settings) {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* storage unavailable */
  }
}

export default function SettingsPopover({ settings, onChange }: { settings: Settings; onChange: (s: Settings) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const tts = typeof window !== "undefined" && "speechSynthesis" in window;
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
  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setOpen(!open)} aria-label="Settings" aria-expanded={open} className="glass rounded-xl p-2 text-slate-300 hover:text-white">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
          <path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" />
          <circle cx="16" cy="6" r="2" />
          <circle cx="10" cy="12" r="2" />
          <circle cx="18" cy="18" r="2" />
        </svg>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            role="dialog"
            aria-label="Settings"
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="glass-strong absolute right-0 top-11 z-50 w-72 rounded-2xl popover p-4 text-sm"
          >
            <h3 className="mb-3 text-[11px] font-semibold tracking-[0.14em] text-slate-400">SETTINGS</h3>
            <fieldset>
              <legend className="mb-1 text-xs text-slate-500">Interface</legend>
              <div className="grid grid-cols-2 gap-2">
                {(["minimal", "workspace"] as const).map((v) => (
                  <button
                    key={v}
                    aria-pressed={settings.view === v}
                    onClick={() => onChange({ ...settings, view: v })}
                    className={`rounded-lg border px-2 py-1.5 text-xs capitalize ${settings.view === v ? "border-cyan-glow/50 text-white" : "border-white/5 text-slate-400"}`}
                  >
                    {v}
                  </button>
                ))}
              </div>
            </fieldset>
            <label className={`mt-3 flex items-center gap-2 text-xs ${tts ? "text-slate-300" : "text-slate-600"}`}>
              <input type="checkbox" disabled={!tts} checked={settings.readAloud} onChange={(e) => onChange({ ...settings, readAloud: e.target.checked })} />
              Read task results aloud {tts ? "" : "(not supported in this browser)"}
            </label>
            <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
              Shortcuts: Enter send · Shift+Enter new line · Ctrl+O project · Ctrl+. workspace · Esc close
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
