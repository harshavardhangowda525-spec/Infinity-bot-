import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Theme } from "../lib/theme";
import type { ConversationSummary } from "../lib/types";
import { Icon } from "./Icons";

export type Route =
  | { name: "chat"; id?: number }
  | { name: "history" | "cook" | "settings" | "plans" | "usage" | "keys" | "intelligence" | "admin" };

export function parseRoute(hash: string): Route {
  const h = hash.replace(/^#\/?/, "");
  const m = h.match(/^c\/(\d+)$/);
  if (m) return { name: "chat", id: Number(m[1]) };
  const names = ["history", "cook", "settings", "plans", "usage", "keys", "intelligence", "admin"] as const;
  const n = names.find((x) => x === h);
  return n ? { name: n } : { name: "chat" };
}

export const routeHash = (r: Route) => (r.name === "chat" ? (r.id ? `#/c/${r.id}` : "#/") : `#/${r.name}`);

function NavItem({ icon, label, active, onClick, collapsed, tone }: {
  icon: ReactNode; label: string; active?: boolean; onClick: () => void; collapsed: boolean; tone?: "cook";
}) {
  return (
    <button
      onClick={onClick}
      title={collapsed ? label : undefined}
      aria-current={active ? "page" : undefined}
      className={`flex h-9 w-full items-center gap-3 rounded-lg px-2.5 text-sm transition ${
        active ? "bg-hover font-medium text-fg" : "text-fg/75 hover:bg-hover hover:text-fg"
      } ${collapsed ? "justify-center px-0" : ""}`}
    >
      <span className={tone === "cook" ? "text-cook" : active ? "text-fg" : "text-muted"}>{icon}</span>
      {!collapsed && <span className="truncate">{label}</span>}
    </button>
  );
}

export default function Sidebar({
  route, go, collapsed, onToggle, conversations, onNewChat, onProjects, user, planLabel, isOwner, theme, onTheme, onLogout,
  mobileOpen, onMobileClose,
}: {
  route: Route;
  go: (r: Route) => void;
  collapsed: boolean;
  onToggle: () => void;
  conversations: ConversationSummary[];
  onNewChat: () => void;
  onProjects: () => void;
  user: { email: string };
  planLabel: string;
  isOwner: boolean;
  theme: Theme;
  onTheme: (t: Theme) => void;
  onLogout: () => void;
  mobileOpen: boolean;
  onMobileClose: () => void;
}) {
  const [menu, setMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!menu) return;
    const close = (e: MouseEvent) => !menuRef.current?.contains(e.target as Node) && setMenu(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setMenu(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [menu]);

  const nav = (r: Route) => {
    go(r);
    onMobileClose();
  };

  const body = (c: boolean) => (
    <div className="flex h-full flex-col">
      <div className={`flex h-14 items-center gap-2 px-3 ${c ? "justify-center" : ""}`}>
        {!c && (
          <button onClick={() => nav({ name: "chat" })} className="flex items-center gap-2 rounded-md px-1 py-1 text-fg">
            <span className="text-accent">{Icon.logo(26)}</span>
            <span className="text-[13px] font-semibold tracking-[0.12em]">INFINITY BOT</span>
          </button>
        )}
        <button onClick={mobileOpen ? onMobileClose : onToggle} aria-label={mobileOpen ? "Close menu" : c ? "Expand sidebar" : "Collapse sidebar"}
          title={c ? "Expand sidebar" : "Collapse sidebar"}
          className={`rounded-lg p-1.5 text-muted transition hover:bg-hover hover:text-fg ${c ? "" : "ml-auto"}`}>
          {mobileOpen ? Icon.close : Icon.sidebar}
        </button>
      </div>
      <div className="px-2.5">
        <button onClick={() => { onNewChat(); onMobileClose(); }} title={c ? "New chat" : undefined}
          className={`flex h-9 w-full items-center gap-2 rounded-lg border border-line bg-surface text-sm font-medium text-fg shadow-sm transition hover:bg-hover ${
            c ? "justify-center" : "px-3"
          }`}>
          {Icon.plus}
          {!c && "New chat"}
        </button>
      </div>
      <nav aria-label="Main" className="mt-3 space-y-0.5 px-2.5">
        <NavItem icon={Icon.history} label="Chat history" active={route.name === "history"} onClick={() => nav({ name: "history" })} collapsed={c} />
        <NavItem icon={Icon.cook} label="Cook" tone="cook" active={route.name === "cook"} onClick={() => nav({ name: "cook" })} collapsed={c} />
        <NavItem icon={Icon.folder} label="Projects" onClick={() => { onProjects(); onMobileClose(); }} collapsed={c} />
        <NavItem icon={Icon.settings} label="Settings" active={route.name === "settings"} onClick={() => nav({ name: "settings" })} collapsed={c} />
      </nav>
      {!c && (
        <div className="mt-5 flex min-h-0 flex-1 flex-col">
          <h2 className="px-5 pb-1 text-[11px] font-medium text-muted">Recent</h2>
          <ul className="scroll-thin min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2.5 pb-2">
            {conversations.length === 0 && <li className="px-2.5 py-1.5 text-xs text-muted">No conversations yet</li>}
            {conversations.slice(0, 30).map((cv) => {
              const active = route.name === "chat" && route.id === cv.id;
              return (
                <li key={cv.id}>
                  <button onClick={() => nav({ name: "chat", id: cv.id })} aria-current={active ? "page" : undefined} title={cv.title}
                    className={`block w-full truncate rounded-lg px-2.5 py-1.5 text-left text-[13px] transition ${
                      active ? "bg-hover text-fg" : "text-fg/70 hover:bg-hover hover:text-fg"
                    }`}>
                    {cv.title}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {c && <div className="flex-1" />}
      <div ref={menuRef} className="relative border-t border-line p-2.5">
        {menu && (
          <div role="menu" aria-label="Account" className={`popover absolute bottom-full z-50 mb-2 w-60 rounded-xl border border-line bg-surface p-1.5 text-sm ${c ? "left-2" : "left-2.5 right-2.5 w-auto"}`}>
            <p className="truncate px-2.5 pb-1.5 pt-1 text-xs text-muted">{user.email}</p>
            {([
              ["plans", "Plans & billing", Icon.card],
              ["usage", "Usage", Icon.chart],
              ["keys", "API keys", Icon.key],
              ["intelligence", "Model & benchmarks", Icon.gauge],
              ...(isOwner ? ([["admin", "Admin", Icon.shield]] as const) : []),
            ] as const).map(([name, label, icon]) => (
              <button key={name} role="menuitem" onClick={() => { setMenu(false); nav({ name } as Route); }}
                className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-fg/85 hover:bg-hover hover:text-fg">
                <span className="text-muted">{icon}</span>
                {label}
              </button>
            ))}
            <div className="my-1 border-t border-line" />
            <div className="flex items-center gap-1 px-1.5 py-1" role="group" aria-label="Theme">
              {([["light", Icon.sun], ["dark", Icon.moon], ["system", Icon.monitor]] as const).map(([t, icon]) => (
                <button key={t} onClick={() => onTheme(t)} aria-pressed={theme === t} title={`${t[0].toUpperCase()}${t.slice(1)} theme`}
                  className={`flex flex-1 items-center justify-center rounded-md py-1.5 ${theme === t ? "bg-hover text-fg" : "text-muted hover:text-fg"}`}>
                  {icon}
                  <span className="sr-only">{t} theme</span>
                </button>
              ))}
            </div>
            <div className="my-1 border-t border-line" />
            <button role="menuitem" onClick={onLogout} className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-fg/85 hover:bg-hover hover:text-fg">
              <span className="text-muted">{Icon.logout}</span>
              Sign out
            </button>
          </div>
        )}
        <button onClick={() => setMenu(!menu)} aria-expanded={menu} aria-haspopup="menu" title={c ? user.email : undefined}
          className={`flex w-full items-center gap-2.5 rounded-lg p-1.5 text-left transition hover:bg-hover ${c ? "justify-center" : ""}`}>
          <span aria-hidden className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent/10 text-sm font-semibold uppercase text-accent">
            {user.email[0]}
          </span>
          {!c && (
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[13px] font-medium text-fg">{user.email.split("@")[0]}</span>
              <span className="block truncate text-[11px] text-muted">{planLabel}</span>
            </span>
          )}
        </button>
      </div>
    </div>
  );

  return (
    <>
      <aside aria-label="Sidebar"
        className={`hidden shrink-0 border-r border-line bg-surface transition-[width] duration-200 ease-out md:block ${collapsed ? "w-[64px]" : "w-[260px]"}`}>
        {body(collapsed)}
      </aside>
      {mobileOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div className="absolute inset-0 bg-slate-900/40" onClick={onMobileClose} aria-hidden />
          <aside aria-label="Sidebar" className="fade-in absolute inset-y-0 left-0 w-[84vw] max-w-[300px] border-r border-line bg-surface shadow-pop">
            {body(false)}
          </aside>
        </div>
      )}
    </>
  );
}
