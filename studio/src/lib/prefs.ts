export interface Prefs {
  readAloud: boolean;
  sidebarCollapsed: boolean;
}

const KEY = "infinity.prefs";
const DEFAULTS: Prefs = { readAloud: false, sidebarCollapsed: false };

export function loadPrefs(): Prefs {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...DEFAULTS, ...JSON.parse(raw) };
  } catch {
    /* storage unavailable */
  }
  return { ...DEFAULTS };
}

export function savePrefs(p: Prefs) {
  try {
    localStorage.setItem(KEY, JSON.stringify(p));
  } catch {
    /* storage unavailable */
  }
}
