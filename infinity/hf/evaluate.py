"""Held-out evaluation for any INFINITY ModelAdapter (open model, open model + adapter, or HYPER 5.5).

Suites (training/evaluation/test_cases.jsonl): instruction, factual, reasoning, uncertainty
(regex checks), coding and error_handling (generated code runs against unit tests in
``sandbox.run_tests``). The optional ``repair`` suite is the live agent benchmark from
``infinity bench`` (real runtime, fresh temporary workspaces).

Only measured numbers are reported: pass rates, latency, tokens/s and peak process memory.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from ..agent.models import Message
from .chat import system_prompt_for
from .sandbox import extract_code, run_tests


def load_cases(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def check_text(case: dict, text: str) -> tuple[bool, str | None]:
    t = text.strip()
    if "fullmatch" in case and not re.fullmatch(case["fullmatch"], t, flags=re.I | re.S):
        return False, "reply does not match the required format"
    if "any" in case and not any(re.search(p, t, flags=re.I) for p in case["any"]):
        return False, "expected content not found"
    if "none" in case:
        for p in case["none"]:
            if re.search(p, t, flags=re.I):
                return False, f"forbidden content found: {p}"
    return True, None


def _rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:
        return None


def run_suite(model, cases: list[dict], max_new_tokens: int = 384, log=print, include_repair: int = 0) -> dict:
    model.temperature = 0.0  # greedy decoding: reproducible
    info = model.info()
    system = system_prompt_for(info)
    rows = []
    peak = _rss_mb()
    for c in cases:
        t0 = time.time()
        comp = model.complete([Message("system", system), Message("user", c["prompt"])], max_new_tokens=max_new_tokens)
        dt = time.time() - t0
        if "tests" in c:
            res = run_tests(extract_code(comp.text), c["tests"])
            ok, why = res["passed"], res["error"]
        else:
            ok, why = check_text(c, comp.text)
        rss = _rss_mb()
        peak = max(peak or 0, rss or 0) or None
        rows.append({"id": c["id"], "suite": c["suite"], "passed": ok, "reason": why, "latency_s": round(dt, 2),
                     "tokens_out": comp.tokens_out, "reply": comp.text[:1500]})
        log(f"  {'PASS' if ok else 'FAIL'} {c['suite']:<15} {c['id']:<18} {dt:5.1f}s" + (f"  ({why})" if why else ""))
    suites: dict[str, dict] = {}
    for r in rows:
        s = suites.setdefault(r["suite"], {"passed": 0, "total": 0})
        s["total"] += 1
        s["passed"] += int(r["passed"])
    for s in suites.values():
        s["rate"] = round(s["passed"] / s["total"], 3)
    lat = [r["latency_s"] for r in rows]
    toks = sum(r["tokens_out"] for r in rows)
    report = {
        "created_unix": int(time.time()), "model": info,
        "suites": suites,
        "overall": {"passed": sum(r["passed"] for r in rows), "total": len(rows),
                    "rate": round(sum(r["passed"] for r in rows) / max(1, len(rows)), 3)},
        "latency_s": {"mean": round(statistics.mean(lat), 2) if lat else None,
                      "median": round(statistics.median(lat), 2) if lat else None},
        "tokens_per_s": round(toks / max(1e-9, sum(lat)), 2),
        "peak_rss_mb": peak, "decoding": "greedy", "max_new_tokens": max_new_tokens, "cases": rows,
    }
    if include_repair:
        from ..bench.runner import bench_repair

        log(f"  running {include_repair} live repair tasks through the agent runtime…")
        metrics, cfg = bench_repair(model, n=include_repair)
        report["repair"] = {"metrics": metrics, "config": cfg}
    return report


def to_markdown(r: dict, title: str) -> str:
    m = r["model"]
    lines = [f"# {title}", "", f"- Model: **{m.get('product')}** ({m.get('implementation')})",
             f"- Base: `{m.get('base', m.get('checkpoint', ''))}` · adapter: `{m.get('adapter')}`",
             f"- Decoding: {r['decoding']}, max {r['max_new_tokens']} new tokens",
             f"- Latency mean {r['latency_s']['mean']} s (median {r['latency_s']['median']} s) · "
             f"{r['tokens_per_s']} tokens/s · peak memory {r['peak_rss_mb']} MB", "",
             "| Suite | Passed | Rate |", "|---|---|---|"]
    for k, s in sorted(r["suites"].items()):
        lines.append(f"| {k} | {s['passed']}/{s['total']} | {s['rate']:.0%} |")
    lines.append(f"| **overall** | {r['overall']['passed']}/{r['overall']['total']} | {r['overall']['rate']:.0%} |")
    if "repair" in r:
        rm = r["repair"]["metrics"]
        lines += ["", f"Live repair (agent runtime): {rm['verified_success_rate']:.0%} verified of {rm['tasks']} · outcomes {rm['outcomes']}"]
    fails = [c for c in r["cases"] if not c["passed"]]
    if fails:
        lines += ["", "## Failures", ""] + [f"- `{c['id']}` ({c['suite']}): {c['reason']}" for c in fails]
    return "\n".join(lines) + "\n"


def compare(a: dict, b: dict, label_a: str = "A", label_b: str = "B") -> str:
    lines = [f"# Comparison: {label_a} → {label_b}", "", f"| Suite | {label_a} | {label_b} | Δ |", "|---|---|---|---|"]
    for k in sorted(set(a["suites"]) | set(b["suites"])):
        ra = a["suites"].get(k, {}).get("rate")
        rb = b["suites"].get(k, {}).get("rate")
        d = f"{(rb - ra) * 100:+.0f} pts" if ra is not None and rb is not None else "n/a"
        lines.append(f"| {k} | {ra if ra is None else f'{ra:.0%}'} | {rb if rb is None else f'{rb:.0%}'} | {d} |")
    lines.append(f"| **overall** | {a['overall']['rate']:.0%} | {b['overall']['rate']:.0%} | "
                 f"{(b['overall']['rate'] - a['overall']['rate']) * 100:+.0f} pts |")
    if "repair" in a and "repair" in b:
        lines.append(f"| repair (agent) | {a['repair']['metrics']['verified_success_rate']:.0%} | "
                     f"{b['repair']['metrics']['verified_success_rate']:.0%} | |")
    pa = {c["id"]: c["passed"] for c in a["cases"]}
    reg = [c["id"] for c in b["cases"] if pa.get(c["id"]) and not c["passed"]]
    fixed = [c["id"] for c in b["cases"] if pa.get(c["id"]) is False and c["passed"]]
    lines += ["", f"Regressions ({len(reg)}): {', '.join(reg) or 'none'}", f"Newly passing ({len(fixed)}): {', '.join(fixed) or 'none'}",
              "", f"Latency mean: {a['latency_s']['mean']} s → {b['latency_s']['mean']} s"]
    return "\n".join(lines) + "\n"
