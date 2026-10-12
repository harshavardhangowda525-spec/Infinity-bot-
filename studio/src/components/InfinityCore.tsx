import { lazy, Suspense, useMemo, useSyncExternalStore } from "react";
import type { CoreState } from "../lib/types";

const InfinityCore3D = lazy(() => import("./InfinityCore3D"));

function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

function subscribeReducedMotion(cb: () => void) {
  const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
  mq?.addEventListener?.("change", cb);
  return () => mq?.removeEventListener?.("change", cb);
}
const getReducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

export function isLowPower(): boolean {
  const cores = navigator.hardwareConcurrency ?? 4;
  const mobile = /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 8;
  return mobile || cores <= 4 || mem <= 4;
}

const HALO: Record<CoreState, string> = {
  idle: "rgba(94,231,255,0.10)",
  listening: "rgba(94,231,255,0.18)",
  thinking: "rgba(120,110,255,0.20)",
  executing: "rgba(59,170,255,0.24)",
  speaking: "rgba(120,200,255,0.18)",
  success: "rgba(92,242,181,0.20)",
  error: "rgba(245,184,75,0.18)",
};

const FALLBACK_COLORS: Record<CoreState, string> = {
  idle: "#5ee7ff",
  listening: "#8ff1ff",
  thinking: "#8b9bff",
  executing: "#5ee7ff",
  speaking: "#a9f3ff",
  success: "#5cf2b5",
  error: "#f5b84b",
};

/** SVG fallback when WebGL is unavailable. Animates only while the agent is active. */
function FallbackCore({ state }: { state: CoreState }) {
  const active = state === "thinking" || state === "executing" || state === "listening" || state === "speaking";
  const color = FALLBACK_COLORS[state];
  const d = "M100 50c-15-20-30-30-45-30a30 30 0 0 0 0 60c15 0 30-10 45-30s30-30 45-30a30 30 0 0 1 0 60c-15 0-30-10-45-30z";
  return (
    <svg viewBox="0 0 200 100" className="h-full w-full" aria-hidden>
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <path d={d} fill="none" stroke={color} strokeOpacity={0.35} strokeWidth={6} />
      <path d={d} fill="none" stroke={color} strokeWidth={2} filter="url(#glow)" className={active ? "fallback-flow" : ""} />
    </svg>
  );
}

export default function InfinityCore({
  state,
  amplitude,
  onActivate,
  compact = false,
}: {
  state: CoreState;
  amplitude: React.MutableRefObject<number>;
  onActivate?: () => void;
  compact?: boolean;
}) {
  const hasWebGL = useMemo(webglAvailable, []);
  const lowPower = useMemo(isLowPower, []);
  const reducedMotion = useSyncExternalStore(subscribeReducedMotion, getReducedMotion);
  const label = `INFINITY core — ${state}. Open agent activity`;
  return (
    <button
      type="button"
      onClick={onActivate}
      aria-label={label}
      title="Open agent activity"
      data-core-state={state}
      className={`relative block focus-ring rounded-full transition-[width,height] duration-500 ${
        compact ? "h-[150px] w-[260px] sm:h-[180px] sm:w-[320px]" : "h-[230px] w-[340px] sm:h-[340px] sm:w-[560px] lg:h-[400px] lg:w-[680px]"
      }`}
    >
      <span aria-hidden className="core-halo pointer-events-none absolute inset-[-10%]" style={{ ["--halo" as string]: HALO[state] }} />
      {hasWebGL ? (
        <Suspense fallback={<FallbackCore state={state} />}>
          <InfinityCore3D state={state} amplitude={amplitude} lowPower={lowPower} reducedMotion={reducedMotion} />
        </Suspense>
      ) : (
        <FallbackCore state={state} />
      )}
    </button>
  );
}
