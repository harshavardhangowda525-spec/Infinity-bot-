import { useMemo, useState } from "react";
import { day, num } from "../lib/format";
import type { UsageSummary } from "../lib/types";

// Validated with the dataviz palette checker on the dark surface #0b101b (all checks pass).
const IN = "#14a3c4";
const OUT = "#8b7cf6";
const W = 560;
const H = 180;
const PAD = { l: 44, r: 64, t: 12, b: 24 };

/** Daily tokens in/out over the last N days. Line chart, one axis, crosshair tooltip, legend, table view. */
export default function UsageChart({ usage }: { usage: UsageSummary }) {
  const [hover, setHover] = useState<number | null>(null);
  const [table, setTable] = useState(false);
  const pts = useMemo(() => {
    // Fill missing days with zero so the time axis is honest.
    const byDay = new Map(usage.daily.map((d) => [d.day, d]));
    const end = Math.floor(Date.now() / 1000 / 86400) * 86400;
    const out = [];
    for (let i = usage.days - 1; i >= 0; i--) {
      const t = end - i * 86400;
      const d = byDay.get(t);
      out.push({ day: t, tin: d?.tin ?? 0, tout: d?.tout ?? 0 });
    }
    return out;
  }, [usage]);
  const max = Math.max(1, ...pts.map((p) => Math.max(p.tin, p.tout)));
  const x = (i: number) => PAD.l + (i / Math.max(1, pts.length - 1)) * (W - PAD.l - PAD.r);
  const y = (v: number) => H - PAD.b - (v / max) * (H - PAD.t - PAD.b);
  const path = (k: "tin" | "tout") => pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p[k]).toFixed(1)}`).join("");
  const ticks = [0, 0.5, 1].map((f) => Math.round(max * f));
  const empty = usage.requests === 0;
  const last = pts[pts.length - 1];

  return (
    <figure className="m-0">
      <div className="mb-2 flex items-center gap-4 text-xs text-fg/80">
        <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4" style={{ background: IN }} />Tokens in</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 border-t-2 border-dashed" style={{ borderColor: OUT }} />Tokens out</span>
        <button onClick={() => setTable(!table)} className="ml-auto rounded border border-line px-2 py-0.5 text-[11px] text-muted hover:text-fg" aria-pressed={table}>
          {table ? "Chart" : "Table"}
        </button>
      </div>
      {empty ? (
        <p className="flex h-[180px] items-center justify-center rounded-xl border border-dashed border-line text-sm text-muted">
          No usage recorded in the last {usage.days} days.
        </p>
      ) : table ? (
        <div className="scroll-thin max-h-[220px] overflow-auto">
          <table className="w-full text-xs">
            <thead><tr className="text-muted"><th className="p-1 text-left font-normal">Day</th><th className="p-1 text-right font-normal">In</th><th className="p-1 text-right font-normal">Out</th></tr></thead>
            <tbody>
              {pts.filter((p) => p.tin || p.tout).map((p) => (
                <tr key={p.day} className="border-t border-line text-fg/80"><td className="p-1">{day(p.day)}</td><td className="p-1 text-right">{num(p.tin)}</td><td className="p-1 text-right">{num(p.tout)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="relative">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`Daily token usage, last ${usage.days} days`}
            onMouseLeave={() => setHover(null)}
            onMouseMove={(e) => {
              const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
              const px = ((e.clientX - r.left) / r.width) * W;
              const i = Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (pts.length - 1));
              setHover(Math.max(0, Math.min(pts.length - 1, i)));
            }}>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="rgba(148,163,184,0.12)" />
                <text x={PAD.l - 6} y={y(t) + 3} textAnchor="end" fontSize="10" fill="#8a97ab">{num(t)}</text>
              </g>
            ))}
            <text x={PAD.l} y={H - 6} fontSize="10" fill="#8a97ab">{day(pts[0].day)}</text>
            <text x={W - PAD.r} y={H - 6} fontSize="10" fill="#8a97ab" textAnchor="end">today</text>
            <path d={path("tin")} fill="none" stroke={IN} strokeWidth="2" strokeLinejoin="round" />
            <path d={path("tout")} fill="none" stroke={OUT} strokeWidth="2" strokeDasharray="5 3" strokeLinejoin="round" />
            {/* selective direct labels at the line ends */}
            <text x={W - PAD.r + 6} y={y(last.tin) + 3} fontSize="10" fill="#cbd5e1">In</text>
            <text x={W - PAD.r + 6} y={y(last.tout) + 3 + (Math.abs(y(last.tin) - y(last.tout)) < 10 ? 10 : 0)} fontSize="10" fill="#cbd5e1">Out</text>
            {hover !== null && (
              <g>
                <line x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={H - PAD.b} stroke="rgba(226,232,240,0.35)" />
                <circle cx={x(hover)} cy={y(pts[hover].tin)} r="4" fill={IN} stroke="#0b101b" strokeWidth="2" />
                <circle cx={x(hover)} cy={y(pts[hover].tout)} r="4" fill={OUT} stroke="#0b101b" strokeWidth="2" />
              </g>
            )}
          </svg>
          {hover !== null && (
            <div className="glass-strong popover pointer-events-none absolute top-0 rounded-lg px-2.5 py-1.5 text-[11px] text-fg"
              style={{ left: `${(x(hover) / W) * 100}%`, transform: `translateX(${hover > pts.length / 2 ? "-105%" : "5%"})` }}>
              <div className="text-muted">{day(pts[hover].day)}</div>
              <div>In: {num(pts[hover].tin)}</div>
              <div>Out: {num(pts[hover].tout)}</div>
            </div>
          )}
        </div>
      )}
      <figcaption className="mt-2 text-[11px] text-muted">
        {usage.estimated_events > 0
          ? `${usage.estimated_events} of ${usage.requests} events used estimated token counts (non-neural test model); the rest are exact tokenizer counts.`
          : "Exact counts from the INFINITY tokenizer."}
      </figcaption>
    </figure>
  );
}
