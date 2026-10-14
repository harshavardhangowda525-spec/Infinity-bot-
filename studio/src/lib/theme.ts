export type Theme = "system" | "light" | "dark";

const KEY = "infinity.theme";

export function loadTheme(): Theme {
  try {
    const t = localStorage.getItem(KEY);
    if (t === "light" || t === "dark" || t === "system") return t;
  } catch {
    /* storage unavailable */
  }
  return "system";
}

export function saveTheme(t: Theme) {
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* storage unavailable */
  }
}

export function isDark(t: Theme): boolean {
  if (t === "system") return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  return t === "dark";
}

/** Apply the theme to <html> and follow the OS setting while "system" is chosen. Returns a cleanup. */
export function applyTheme(t: Theme): () => void {
  const set = () => document.documentElement.classList.toggle("dark", isDark(t));
  set();
  if (t !== "system" || !window.matchMedia) return () => {};
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener?.("change", set);
  return () => mq.removeEventListener?.("change", set);
}
