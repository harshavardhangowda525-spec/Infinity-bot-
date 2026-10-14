"""INFINITY EVALUATOR: benchmark suites with recorded, reproducible results.

Suites (all on held-out data the model is never trained on):

- ``lm``       : next-token loss / perplexity on the pretraining test split.
- ``protocol`` : greedy completion of held-out eval transcripts. Measures the
                 valid-action rate, the tool-match rate and the exact-action rate
                 against the expert.
- ``repair``   : end-to-end. The real Agent with the neural model fixes
                 held-out bug tasks in fresh temporary workspaces. The runtime's
                 verdict decides success (completed_verified).

Every run is stored in ``runs/benchmarks.db`` with the timestamp, checkpoint
path, step, model sha256, suite configuration and metrics. Nothing is estimated.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from ..agent import Agent, Limits, Mode, PermissionMode, PermissionPolicy
from ..agent.models import NeuralCoreModel
from ..agent.protocol import ProtocolError, parse_action
from ..core.inference import SamplingConfig
from ..sft.finetune import agent_new_tokens
from ..sft.generate import materialise
from ..sft.tasks import make_tasks

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL, suite TEXT NOT NULL,
  checkpoint TEXT NOT NULL, run TEXT, step INTEGER, model_sha256 TEXT, config TEXT NOT NULL,
  metrics TEXT NOT NULL, duration_s REAL NOT NULL);
"""


class BenchStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def add(self, suite: str, info: dict, config: dict, metrics: dict, duration: float) -> int:
        cur = self.db.execute(
            "INSERT INTO results (created_at, suite, checkpoint, run, step, model_sha256, config, metrics, duration_s) VALUES (?,?,?,?,?,?,?,?,?)",
            (time.time(), suite, info["checkpoint"], info.get("run"), info.get("step"), info.get("model_sha256"),
             json.dumps(config, sort_keys=True), json.dumps(metrics), round(duration, 2)))
        self.db.commit()
        return int(cur.lastrowid)

    def list(self, checkpoint: str | None = None, limit: int = 200) -> list[dict]:
        sql = "SELECT * FROM results" + (" WHERE checkpoint = ?" if checkpoint else "") + " ORDER BY id DESC LIMIT ?"
        rows = self.db.execute(sql, ((checkpoint, limit) if checkpoint else (limit,))).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d["config"])
            d["metrics"] = json.loads(d["metrics"])
            out.append(d)
        return out


def _info(model: NeuralCoreModel) -> dict:
    eng = model.engine
    info = {"checkpoint": str(eng.path), "run": eng.run_cfg.name, "step": eng.manifest["step"],
            "model_sha256": eng.manifest["sha256"]["model.pt"], "context_length": eng.run_cfg.model.context_length}
    if eng.adapter_dir is not None:  # INFINITY HYPER 5.5 — Fine-Tuned: recorded separately from the original
        info.update(checkpoint=f"{eng.path}::{eng.adapter_dir}", run=f"{eng.run_cfg.name}+lora:{eng.adapter_dir.name}",
                    adapter_sha256=eng.adapter_conf["adapter_sha256"], product=eng.adapter_conf["name"])
    return info


@torch.no_grad()
def bench_lm(model: NeuralCoreModel, data_dir: Path, windows: int = 64, seed: int = 7) -> tuple[dict, dict]:
    eng = model.engine
    prepared = json.loads((data_dir / "prepared.json").read_text())
    if prepared["tokenizer"]["fingerprint"] != eng.tokenizer.fingerprint():
        raise ValueError("test split was tokenized with a different tokenizer")
    ctx = min(256, eng.run_cfg.model.context_length)
    losses = []
    if prepared.get("format") == "sharded-v1":
        from ..core.data_large import ShardedTokenDataset

        ds = ShardedTokenDataset(data_dir, prepared["splits"]["test"], ctx)
        gen = torch.Generator().manual_seed(seed)
        for _ in range(windows):
            x, y = ds.batch(1, gen)
            losses.append(eng.model(x, y)[1].item())
    else:
        data = np.memmap(data_dir / "test.bin", dtype=np.dtype(prepared["splits"]["test"]["dtype"]), mode="r")
        g = np.random.default_rng(seed)
        for s in g.integers(0, len(data) - ctx - 1, size=windows):
            x = torch.from_numpy(np.asarray(data[s : s + ctx], dtype=np.int64))[None]
            y = torch.from_numpy(np.asarray(data[s + 1 : s + 1 + ctx], dtype=np.int64))[None]
            losses.append(eng.model(x, y)[1].item())
    mean = float(np.mean(losses))
    return {"test_loss": round(mean, 4), "perplexity": round(math.exp(mean), 2)}, {"windows": windows, "window_tokens": ctx, "seed": seed}


@torch.no_grad()
def bench_protocol(model: NeuralCoreModel, eval_jsonl: Path, limit: int = 200) -> tuple[dict, dict]:
    eng = model.engine
    ctx = eng.run_cfg.model.context_length
    rows = [json.loads(l) for l in open(eval_jsonl)][:limit]
    valid = tool_ok = exact = 0
    per_tool: dict[str, list[int]] = {}
    for r in rows:
        text = eng.generate(r["prompt"], SamplingConfig(max_new_tokens=agent_new_tokens(ctx), temperature=0), allow_special=True)
        target = json.loads(r["completion"].removesuffix("<|end|>"))
        try:
            a = parse_action(text)
            valid += 1
            pred_tool = a.tool if a.kind == "tool" else "final"
            hit = pred_tool == r["tool"]
            tool_ok += hit
            if a.kind == "tool" and hit and a.args == target.get("args"):
                exact += 1
            elif a.kind == "final" and hit:
                exact += 1
        except ProtocolError:
            hit = False
        per_tool.setdefault(r["tool"], []).append(int(hit))
    n = max(1, len(rows))
    return ({"examples": len(rows), "valid_action_rate": round(valid / n, 4), "tool_match_rate": round(tool_ok / n, 4),
             "exact_action_rate": round(exact / n, 4),
             "tool_match_by_step_tool": {k: round(sum(v) / len(v), 4) for k, v in per_tool.items()}},
            {"eval_file_sha256": hashlib.sha256(eval_jsonl.read_bytes()).hexdigest(), "limit": limit, "decoding": "greedy"})


def bench_repair(model: NeuralCoreModel, n: int = 30, seed: int = 1337, max_steps: int = 8) -> tuple[dict, dict]:
    model.temperature = 0.0
    tasks = make_tasks("eval", n, seed)
    outcomes: dict[str, int] = {}
    by_family: dict[str, list[int]] = {}
    steps, tokens = [], []
    with tempfile.TemporaryDirectory(prefix="infinity-bench-") as tmp:
        for t in tasks:
            ws = materialise(t, Path(tmp))
            report = Agent(model, ws, PermissionPolicy(PermissionMode.AUTO, approver=lambda r: False), Mode.DEBUG,
                           limits=Limits(max_steps=max_steps, command_timeout=60)).run(t.description)
            outcomes[report.status] = outcomes.get(report.status, 0) + 1
            ok = int(report.status == "completed_verified")
            by_family.setdefault(t.family, []).append(ok)
            steps.append(len(report.steps))
            tokens.append(report.tokens_in + report.tokens_out)
    return ({"tasks": n, "verified_success_rate": round(outcomes.get("completed_verified", 0) / max(1, n), 4),
             "outcomes": outcomes, "success_by_family": {k: round(sum(v) / len(v), 4) for k, v in by_family.items()},
             "mean_tool_steps": round(float(np.mean(steps)), 2), "mean_tokens": round(float(np.mean(tokens)), 1)},
            {"split": "eval", "seed": seed, "max_steps": max_steps, "decoding": "greedy"})


def run(checkpoint: str, suites: list[str], store_path: str | Path, data_dir: str | Path = "runs/data",
        sft_dir: str | Path = "runs/sft", n_repair: int = 30, n_protocol: int = 200, repair_seed: int = 1337,
        log=print, adapter: str | None = None) -> list[dict]:
    model = NeuralCoreModel(checkpoint, temperature=0.0, adapter=adapter)
    info = _info(model)
    store = BenchStore(store_path)
    results = []
    for suite in suites:
        t0 = time.time()
        if suite == "lm":
            metrics, cfg = bench_lm(model, Path(data_dir))
        elif suite == "protocol":
            metrics, cfg = bench_protocol(model, Path(sft_dir) / "eval.jsonl", n_protocol)
        elif suite == "repair":
            metrics, cfg = bench_repair(model, n_repair, seed=repair_seed)
        else:
            raise ValueError(f"unknown suite {suite!r}")
        dur = time.time() - t0
        if "adapter_sha256" in info:
            cfg = {**cfg, "adapter_sha256": info["adapter_sha256"]}
        rid = store.add(suite, info, cfg, metrics, dur)
        log(f"[{suite}] {json.dumps(metrics)} ({dur:.1f}s, result #{rid})")
        results.append({"id": rid, "suite": suite, "metrics": metrics, "config": cfg, **info})
    return results
