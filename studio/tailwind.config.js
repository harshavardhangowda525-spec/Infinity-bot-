/** @type {import('tailwindcss').Config} */
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Semantic tokens (light/dark values live in src/index.css).
        bg: v("bg"),
        surface: v("surface"),
        "surface-2": v("surface-2"),
        fg: v("fg"),
        muted: v("muted"),
        line: v("line"),
        hover: v("hover"),
        accent: v("accent"),
        cook: v("cook"),
        ok: v("ok"),
        bad: v("bad"),
        warn: v("warn"),
        // Older names kept as aliases so existing views follow the theme.
        cyan: { glow: v("accent"), soft: v("accent") },
        electric: v("accent"),
        violet: { soft: v("cook") },
        amber: { diag: v("warn") },
        obsidian: { DEFAULT: v("surface"), 900: v("bg"), 800: v("surface"), 700: v("surface-2") },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgb(15 23 42 / 0.04), 0 4px 16px -6px rgb(15 23 42 / 0.08)",
        pop: "0 12px 32px -12px rgb(15 23 42 / 0.25)",
      },
      transitionDuration: { DEFAULT: "180ms" },
    },
  },
  plugins: [],
};
