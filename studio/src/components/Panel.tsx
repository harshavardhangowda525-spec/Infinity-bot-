import { motion } from "framer-motion";
import type { ReactNode } from "react";

export default function Panel({
  title,
  icon,
  onClose,
  actions,
  children,
  className = "",
  bodyClass = "",
}: {
  title: string;
  icon?: ReactNode;
  onClose?: () => void;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <motion.section
      layout
      initial={{ opacity: 0, scale: 0.97, y: 8 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.97, y: 8 }}
      transition={{ type: "spring", stiffness: 260, damping: 28 }}
      className={`glass flex min-h-0 flex-col overflow-hidden rounded-2xl ${className}`}
      aria-label={title}
    >
      <header className="flex items-center gap-2 border-b border-white/5 px-3.5 py-2.5">
        {icon && <span className="text-cyan-glow/80">{icon}</span>}
        <h2 className="text-[11px] font-semibold tracking-[0.14em] text-slate-300">{title}</h2>
        <div className="ml-auto flex items-center gap-1">
          {actions}
          {onClose && (
            <button onClick={onClose} aria-label={`Close ${title}`} className="rounded-md p-1 text-slate-500 hover:bg-white/5 hover:text-slate-200">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M6 6l12 12M18 6L6 18" />
              </svg>
            </button>
          )}
        </div>
      </header>
      <div className={`scroll-thin min-h-0 flex-1 overflow-auto ${bodyClass}`}>{children}</div>
    </motion.section>
  );
}

export const Icon = {
  folder: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
    </svg>
  ),
  file: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
      <path d="M14 3v6h6" />
    </svg>
  ),
  code: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M8 6l-6 6 6 6M16 6l6 6-6 6" />
    </svg>
  ),
  terminal: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M4 17l6-5-6-5M12 19h8" />
    </svg>
  ),
  list: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01" />
    </svg>
  ),
  diff: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M12 3v12M6 9h12M6 21h12" />
    </svg>
  ),
};
