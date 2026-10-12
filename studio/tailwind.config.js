/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        obsidian: { DEFAULT: "#04060b", 900: "#070a12", 800: "#0b101b", 700: "#111827" },
        glass: "rgba(16, 22, 34, 0.55)",
        cyan: { glow: "#5ee7ff", soft: "#9ff3ff" },
        electric: "#3b82f6",
        violet: { soft: "#8b7cf6" },
        amber: { diag: "#f5b84b" },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        glass: "0 10px 40px -10px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.05)",
        glow: "0 0 24px -4px rgba(94,231,255,0.45)",
      },
    },
  },
  plugins: [],
};
