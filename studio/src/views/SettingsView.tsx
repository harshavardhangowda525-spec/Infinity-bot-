import type { ReactNode } from "react";
import { modelName } from "../lib/format";
import type { Theme } from "../lib/theme";
import type { Me, ModelStatus } from "../lib/types";
import { Icon } from "../ui/Icons";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-xl border border-line bg-surface">
      <h2 className="border-b border-line px-4 py-3 text-sm font-semibold text-fg">{title}</h2>
      <div className="p-4">{children}</div>
    </section>
  );
}

export default function SettingsView({ me, theme, onTheme, readAloud, onReadAloud, model, onLogout, onPlans }: {
  me: Me;
  theme: Theme;
  onTheme: (t: Theme) => void;
  readAloud: boolean;
  onReadAloud: (v: boolean) => void;
  model: ModelStatus | undefined;
  onLogout: () => void;
  onPlans: () => void;
}) {
  const tts = typeof window !== "undefined" && "speechSynthesis" in window;
  const ent = me.entitlement;
  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto max-w-2xl space-y-5 px-4 py-8 sm:px-6">
        <h1 className="text-xl font-semibold text-fg">Settings</h1>
        <Section title="Appearance">
          <div role="radiogroup" aria-label="Theme" className="grid grid-cols-3 gap-2">
            {([["light", "Light", Icon.sun], ["dark", "Dark", Icon.moon], ["system", "System", Icon.monitor]] as const).map(([id, label, icon]) => (
              <button key={id} role="radio" aria-checked={theme === id} onClick={() => onTheme(id)}
                className={`flex flex-col items-center gap-1.5 rounded-lg border px-3 py-3 text-sm transition ${
                  theme === id ? "border-accent bg-accent/5 text-fg" : "border-line text-muted hover:bg-hover hover:text-fg"
                }`}>
                {icon}
                {label}
              </button>
            ))}
          </div>
        </Section>
        <Section title="Voice">
          <label className="flex items-start gap-3 text-sm">
            <input type="checkbox" checked={readAloud} disabled={!tts} onChange={(e) => onReadAloud(e.target.checked)} className="mt-0.5 h-4 w-4 accent-[rgb(var(--accent))]" />
            <span>
              <span className="block text-fg">Read replies aloud</span>
              <span className="text-muted">{tts ? "Uses your browser's speech voice. Off by default." : "Not supported by this browser."}</span>
            </span>
          </label>
        </Section>
        <Section title="Model">
          <p className="text-sm text-fg">{modelName(model)}</p>
          <p className="mt-1 text-xs text-muted">
            {model?.state === "ready" ? model.implementation : model?.reason ?? "Runtime offline"}
          </p>
          <p className="mt-2 text-xs text-muted">The owner chooses the model in the selector below the message box.</p>
        </Section>
        <Section title="Account">
          <dl className="grid grid-cols-[110px_1fr] gap-y-2 text-sm">
            <dt className="text-muted">Email</dt>
            <dd className="truncate text-fg">{me.user.email}</dd>
            <dt className="text-muted">Role</dt>
            <dd className="text-fg">{me.user.role === "owner" ? "Owner" : "User"}</dd>
            <dt className="text-muted">Plan</dt>
            <dd className="text-fg">{me.user.role === "owner" ? "Owner (usage not charged)" : ent.subscription ? ent.plan.name : "Free"}</dd>
          </dl>
          <div className="mt-4 flex flex-wrap gap-2">
            <button onClick={onPlans} className="btn-secondary">Plans &amp; billing</button>
            <button onClick={onLogout} className="btn-secondary">Sign out</button>
          </div>
        </Section>
      </div>
    </div>
  );
}
