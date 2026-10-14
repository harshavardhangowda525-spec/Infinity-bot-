export type View = "home" | "intelligence" | "tokens" | "plans" | "keys" | "admin";

const ITEMS: { id: View; label: string; ownerOnly?: boolean; icon: JSX.Element }[] = [
  {
    id: "home",
    label: "Home",
    icon: (
      <svg width="20" height="12" viewBox="0 0 64 32" aria-hidden>
        <path d="M32 16c-6-8-12-12-18-12a12 12 0 0 0 0 24c6 0 12-4 18-12s12-12 18-12a12 12 0 0 1 0 24c-6 0-12-4-18-12z" fill="none" stroke="currentColor" strokeWidth="4" />
      </svg>
    ),
  },
  {
    id: "intelligence",
    label: "Intelligence",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
        <path d="M4 19h16M7 16V9M12 16V5M17 16v-4" />
      </svg>
    ),
  },
  {
    id: "tokens",
    label: "Tokens",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v10M9 9.5h4.5a2 2 0 0 1 0 4H9" />
      </svg>
    ),
  },
  {
    id: "plans",
    label: "Plans",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
        <rect x="3" y="5" width="18" height="14" rx="2" />
        <path d="M3 10h18" />
      </svg>
    ),
  },
  {
    id: "keys",
    label: "API keys",
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
        <circle cx="8" cy="15" r="4" />
        <path d="M11 12l9-9M17 6l3 3" />
      </svg>
    ),
  },
  {
    id: "admin",
    label: "Admin",
    ownerOnly: true,
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
        <path d="M12 3l8 3v6c0 4.5-3.4 8.2-8 9-4.6-.8-8-4.5-8-9V6z" />
      </svg>
    ),
  },
];

/** Floating dock-style navigation (no sidebar). Admin appears only for the owner; the server enforces it regardless. */
export default function BottomNav({ view, onView, isOwner, onLogout, email }: {
  view: View;
  onView: (v: View) => void;
  isOwner: boolean;
  onLogout: () => void;
  email: string;
}) {
  return (
    <nav aria-label="Primary" className="glass flex items-center gap-0.5 rounded-2xl p-1">
      {ITEMS.filter((i) => !i.ownerOnly || isOwner).map((i) => (
        <button
          key={i.id}
          onClick={() => onView(i.id)}
          aria-current={view === i.id ? "page" : undefined}
          title={i.label}
          className={`flex items-center gap-1.5 rounded-xl px-2.5 py-2 text-xs transition ${
            view === i.id ? "bg-cyan-glow/15 text-cyan-glow" : "text-slate-400 hover:bg-white/5 hover:text-slate-100"
          }`}
        >
          {i.icon}
          <span className="hidden whitespace-nowrap md:inline">{i.label}</span>
        </button>
      ))}
      <span className="mx-1 hidden h-5 w-px bg-white/10 sm:block" />
      <button onClick={onLogout} title={`Sign out ${email}`} className="rounded-xl px-2.5 py-2 text-xs text-slate-400 hover:bg-white/5 hover:text-slate-100">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
          <path d="M15 4h4v16h-4M10 8l-4 4 4 4M6 12h10" />
        </svg>
        <span className="sr-only">Sign out</span>
      </button>
    </nav>
  );
}
