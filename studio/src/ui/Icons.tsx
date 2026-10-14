/** Consistent 1.75px line icons (24px grid). */
import type { ReactNode } from "react";

function I({ children, size = 18 }: { children: ReactNode; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {children}
    </svg>
  );
}

export const Icon = {
  logo: (size = 22) => (
    <svg width={size} height={size * 0.5} viewBox="0 0 64 32" aria-hidden>
      <path d="M32 16c-6-8-12-12-18-12a12 12 0 0 0 0 24c6 0 12-4 18-12s12-12 18-12a12 12 0 0 1 0 24c-6 0-12-4-18-12z"
        fill="none" stroke="currentColor" strokeWidth="4.5" strokeLinecap="round" />
    </svg>
  ),
  plus: <I><path d="M12 5v14M5 12h14" /></I>,
  history: <I><path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5M12 7v5l3 2" /></I>,
  cook: <I><path d="m8 9-4 3 4 3M16 9l4 3-4 3M13.5 6l-3 12" /></I>,
  folder: <I><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /></I>,
  settings: <I><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" /></I>,
  menu: <I><path d="M4 6h16M4 12h16M4 18h16" /></I>,
  sidebar: <I><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M9 4v16" /></I>,
  send: <I><path d="M12 19V5M5 12l7-7 7 7" /></I>,
  stop: <I><rect x="7" y="7" width="10" height="10" rx="1.5" /></I>,
  clip: <I><path d="m21.4 11.1-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5" /></I>,
  copy: <I><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h10" /></I>,
  check: <I><path d="m5 12 5 5L20 7" /></I>,
  refresh: <I><path d="M21 12a9 9 0 1 1-2.6-6.4L21 8" /><path d="M21 3v5h-5" /></I>,
  edit: <I><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 1 1 3 3L7 19l-4 1 1-4z" /></I>,
  chevron: <I size={14}><path d="m6 9 6 6 6-6" /></I>,
  close: <I><path d="M18 6 6 18M6 6l12 12" /></I>,
  sun: <I><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></I>,
  moon: <I><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></I>,
  monitor: <I><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></I>,
  logout: <I><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9" /></I>,
  card: <I><rect x="2" y="5" width="20" height="14" rx="2" /><path d="M2 10h20" /></I>,
  chart: <I><path d="M4 20h16M7 16V9M12 16V5M17 16v-4" /></I>,
  key: <I><circle cx="7.5" cy="15.5" r="3.5" /><path d="m10 13 9-9M15 8l3 3M18 5l2 2" /></I>,
  shield: <I><path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z" /></I>,
  gauge: <I><path d="M12 14l4-4" /><path d="M3.3 17a9 9 0 1 1 17.4 0" /></I>,
  trash: <I><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14" /></I>,
  mic: <I><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></I>,
  file: <I><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><path d="M14 3v6h6" /></I>,
  terminal: <I><path d="m4 17 6-5-6-5M12 19h8" /></I>,
  diff: <I><path d="M12 3v14M5 10h14M5 21h14" /></I>,
  list: <I><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" /></I>,
  search: <I><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></I>,
  dots: <I><circle cx="5" cy="12" r="1" /><circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /></I>,
};
