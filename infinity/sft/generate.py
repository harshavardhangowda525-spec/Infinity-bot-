"""Generate tool-use SFT data by replaying expert actions through the REAL runtime.

For each synthetic task the expert action sequence is fed to the actual
``Agent`` (real file tools, real ``python3 -m unittest`` runs, real
permission checks). The exact prompts the agent builds at every step are
captured, so the training prompts are identical to what the neural model sees
at inference time (same compact prompt, same context fitting, same chat
template). Only transcripts the runtime judges ``completed_verified`` are kept.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from ..agent import Agent, Limits, Mode, PermissionMode, PermissionPolicy
from ..agent.models import Completion, Message, ModelAdapter, render_chat
from .tasks import TEST_COMMAND, BugTask, expert_actions, make_tasks

MAX_LEADING_FAILURES = 5  # stop early instead of silently writing an empty dataset


class RecordingModel(ModelAdapter):
    """Replays expert actions while counting tokens with the real tokenizer at the target context length."""

    name = "expert-replay"

    def __init__(self, actions: list[dict], tokenizer, context_tokens: int) -> None:
        self.actions = [json.dumps(a, separators=(",", ":")) for a in actions]
        self.tok = tokenizer
        self.ctx = context_tokens
        self.records: list[tuple[list[Message], str]] = []

    def info(self) -> dict:
        return {"product": "INFINITY HYPER 5.5", "implementation": "expert replay (data generation only)"}

    @property
    def max_context_tokens(self) -> int:
        return self.ctx

    def count_tokens(self, text: str) -> tuple[int, bool]:
        return len(self.tok.encode(text)), True

    def complete(self, messages: Sequence[Message], max_new_tokens: int = 512, cancel=None) -> Completion:
        text = self.actions[len(self.records)]
        self.records.append((list(messages), text))
        return Completion(text, 0, 0, True)


def materialise(task: BugTask, root: Path) -> Path:
    ws = root / task.id
    ws.mkdir(parents=True)
    for name, content in task.files.items():
        (ws / name).write_text(content, encoding="utf-8")
    return ws


def run_expert(task: BugTask, tokenizer, context_tokens: int, scratch: Path) -> tuple[list[dict], str]:
    ws = materialise(task, scratch)
    model = RecordingModel(expert_actions(task), tokenizer, context_tokens)
    agent = Agent(model, ws, PermissionPolicy(PermissionMode.AUTO, approver=lambda r: False), Mode.DEBUG,
                  limits=Limits(max_steps=8, command_timeout=60))
    report = agent.run(task.description)
    examples = []
    for step, (msgs, completion) in enumerate(model.records):
        examples.append({
            "task_id": task.id, "family": task.family, "split": task.split, "step": step,
            "prompt": render_chat(msgs), "completion": completion + "<|end|>",
            "tool": json.loads(completion).get("tool", "final"),
        })
    shutil.rmtree(ws, ignore_errors=True)
    return examples, report.status


def generate(out_dir: str | Path, tokenizer, context_tokens: int, n_train: int, n_eval: int, seed: int = 1337, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"context_tokens": context_tokens, "seed": seed, "splits": {}}
    with tempfile.TemporaryDirectory(prefix="infinity-sft-") as tmp:
        for split, n in (("train", n_train), ("eval", n_eval)):
            tasks = make_tasks(split, n, seed)
            kept = rejected = 0
            path = out / f"{split}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for i, task in enumerate(tasks):
                    examples, status = run_expert(task, tokenizer, context_tokens, Path(tmp))
                    if status != "completed_verified":
                        rejected += 1
                        if kept == 0 and rejected >= MAX_LEADING_FAILURES:
                            raise RuntimeError(
                                f"the first {rejected} expert runs all ended as {status!r}, so no training data "
                                f"would be produced. The task tests could not pass on this machine: check that "
                                f"`{TEST_COMMAND}` runs in a terminal (on Windows the interpreter running INFINITY "
                                f"is used for python/python3).")
                        continue
                    kept += 1
                    for ex in examples:
                        f.write(json.dumps(ex) + "\n")
                    if (i + 1) % 100 == 0:
                        log(f"  {split}: {i + 1}/{n} tasks")
            (out / f"{split}_tasks.json").write_text(json.dumps([t.__dict__ for t in tasks]))
            summary["splits"][split] = {"tasks": n, "kept": kept, "rejected": rejected, "examples": kept * len(expert_actions(tasks[0])),
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    summary["tokenizer_fingerprint"] = tokenizer.fingerprint()
    (out / "sft_manifest.json").write_text(json.dumps(summary, indent=2))
    return summary
