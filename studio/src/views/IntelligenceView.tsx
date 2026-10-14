import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { date, num } from "../lib/format";
import type { BenchResult, Intelligence } from "../lib/types";

function pct(v: unknown) {
  return typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—";
}

/** One row per checkpoint, latest result per suite. Values are copied from recorded benchmark runs. */
function latestByCheckpoint(results: BenchResult[]) {
  const rows = new Map<string, { run: string; step: number; checkpoint: string; suites: Record<string, BenchResult> }>();
  for (const r of results) {
    const row = rows.get(r.checkpoint) ?? { run: r.run, step: r.step, checkpoint: r.checkpoint, suites: {} };
    if (!row.suites[r.suite] || row.suites[r.suite].id < r.id) row.suites[r.suite] = r;
    rows.set(r.checkpoint, row);
  }
  return [...rows.values()].sort((a, b) => a.run.localeCompare(b.run) || a.step - b.step);
}

function Row({ k, v, tone }: { k: string; v: string; tone?: string }) {
  return (
    <div className="flex justify-between gap-4 py-1 text-xs">
      <span className="text-slate-500">{k}</span>
      <span className={`text-right ${tone ?? "text-slate-200"}`}>{v}</span>
    </div>
  );
}

export default function IntelligenceView() {
  const [data, setData] = useState<Intelligence | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.intelligence().then(setData, (e) => setError((e as Error).message));
  }, []);
  if (error) return <p className="p-8 text-center text-sm text-red-300">{error}</p>;
  if (!data) return <p className="p-8 text-center text-sm text-slate-500">Loading…</p>;
  const m = data.model;
  const lineage = m.metrics?.lineage as Record<string, unknown> | undefined;
  const rows = latestByCheckpoint(data.benchmarks);

  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto grid max-w-6xl gap-4 px-4 pb-32 pt-16 sm:px-6 lg:grid-cols-3">
        <section className="glass rounded-2xl p-5" aria-labelledby="model-h">
          <h2 id="model-h" className="text-[11px] font-semibold tracking-[0.2em] text-slate-400">{m.state === "ready" && m.product ? m.product : "INFINITY HYPER 5.5"}</h2>
          {m.state === "ready" ? (
            <>
              <Row k="Implementation" v={m.implementation ?? ""} />
              {m.family === "open-model" ? <Row k="Base · adapter" v={`${m.base} · ${m.adapter ?? "none"}`} /> : <Row k="Run · step" v={`${m.run} · ${m.step}`} />}
              <Row k="Parameters" v={num(m.parameters ?? 0)} />
              <Row k="Context window" v={`${m.context_length} tokens`} />
              <Row k="Tokenizer" v={String(m.tokenizer_fingerprint ?? "")} />
              {lineage ? (
                <>
                  <Row k="Training" v={String(lineage.method)} />
                  <Row k="Fine-tuned from" v={`step ${lineage.parent_step}`} />
                  <Row k="SFT examples" v={num(Number(lineage.train_examples))} />
                </>
              ) : (
                <Row k="Training" v="pretraining only" />
              )}
            </>
          ) : (
            <Row k="Status" v={`Unavailable — ${m.reason}`} tone="text-amber-diag" />
          )}
          <div className="my-3 border-t border-white/5" />
          <Row k="Promoted model" v={data.promotion ? `step ${data.promotion.step}` : "none (newest checkpoint is loaded)"} />
          {data.promotion && <Row k="Promoted" v={`${date(data.promotion.promoted_at)} — ${data.promotion.reason}`} />}
        </section>

        <section className="glass rounded-2xl p-5" aria-labelledby="ctx-h">
          <h2 id="ctx-h" className="text-[11px] font-semibold tracking-[0.2em] text-slate-400">PROJECT CONTEXT</h2>
          {data.index === null ? (
            <p className="mt-2 text-xs text-slate-500">No project open. The code index is built when you open one.</p>
          ) : "error" in data.index ? (
            <p className="mt-2 text-xs text-red-300">{String(data.index.error)}</p>
          ) : (
            <>
              <Row k="Indexed files" v={num(data.index.files)} />
              <Row k="Symbols" v={num(data.index.symbols)} />
              <Row k="References" v={num(data.index.references)} />
              <Row k="Languages" v={Object.entries(data.index.languages).map(([k, v]) => `${k} ${v}`).join(" · ") || "—"} />
              {Object.keys(data.index.parse_errors).length > 0 && <Row k="Parse errors" v={String(Object.keys(data.index.parse_errors).length)} tone="text-amber-diag" />}
            </>
          )}
          <div className="my-3 border-t border-white/5" />
          <h3 className="text-[11px] font-semibold tracking-[0.2em] text-slate-400">MEMORY</h3>
          <Row k="Recorded experiences" v={num(data.memory.experiences)} />
          <Row k="Project notes" v={num(data.memory.project_notes)} />
          <Row k="Relevant to last task" v={num(data.memory.relevant_to_last_task)} />
        </section>

        <section className="glass rounded-2xl p-5" aria-labelledby="agent-h">
          <h2 id="agent-h" className="text-[11px] font-semibold tracking-[0.2em] text-slate-400">AGENT</h2>
          <Row k="State" v={data.agent.awaiting_approval ? "waiting for approval" : data.agent.busy ? "busy" : "idle"} />
          <Row k="Last task" v={data.agent.task ? `${data.agent.task.state}${data.agent.task.status ? ` · ${data.agent.task.status}` : ""}` : "none"} />
          {data.plan && data.plan.length > 0 ? (
            <ol className="mt-2 list-decimal space-y-0.5 pl-5 text-xs text-slate-300">{data.plan.map((p, i) => <li key={i}>{p}</li>)}</ol>
          ) : (
            <p className="mt-2 text-xs text-slate-500">No plan recorded for the last task.</p>
          )}
        </section>

        <section className="glass rounded-2xl p-5 lg:col-span-3" aria-labelledby="bench-h">
          <h2 id="bench-h" className="text-[11px] font-semibold tracking-[0.2em] text-slate-400">BENCHMARK HISTORY</h2>
          <p className="mt-1 text-xs text-slate-500">
            Held-out evaluation, recorded with <code className="font-mono">infinity bench run</code>. Repair = real agent runs on unseen bug tasks,
            judged by the runtime (tests must pass). These synthetic tasks measure narrow tool-use skill, not general coding ability.
          </p>
          {rows.length === 0 ? (
            <p className="mt-4 text-sm text-slate-500">No benchmark results recorded yet.</p>
          ) : (
            <div className="scroll-thin mt-3 overflow-x-auto">
              <table className="w-full min-w-[760px] text-left text-sm">
                <thead className="text-xs text-slate-500">
                  <tr className="border-b border-white/5">
                    {["Checkpoint", "Test perplexity", "Valid actions", "Right tool", "Exact action", "Repair success", "Recorded"].map((h) => (
                      <th key={h} className="p-2.5 font-normal">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const lm = r.suites.lm?.metrics;
                    const pr = r.suites.protocol?.metrics;
                    const rp = r.suites.repair?.metrics;
                    const latest = Math.max(...Object.values(r.suites).map((s) => s.created_at));
                    return (
                      <tr key={r.checkpoint} className="border-b border-white/5 last:border-0 text-slate-200">
                        <td className="p-2.5"><div>{r.run}</div><div className="text-[11px] text-slate-500">step {r.step}</div></td>
                        <td className="p-2.5 font-mono">{lm ? String(lm.perplexity) : "—"}</td>
                        <td className="p-2.5 font-mono">{pct(pr?.valid_action_rate)}</td>
                        <td className="p-2.5 font-mono">{pct(pr?.tool_match_rate)}</td>
                        <td className="p-2.5 font-mono">{pct(pr?.exact_action_rate)}</td>
                        <td className="p-2.5 font-mono">{rp ? `${pct(rp.verified_success_rate)} of ${rp.tasks}` : "—"}</td>
                        <td className="p-2.5 text-[11px] text-slate-500">{date(latest)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
