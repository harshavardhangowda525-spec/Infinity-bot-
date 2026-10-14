"""Conversational SFT datasets for open models: schema, validation, dedup, leak checks, tokenization.

One JSON object per line:

    {"id": "chat-greet-001",                      # optional; derived from content if missing
     "domain": "chat" | "coding" | "tool",
     "license": "Project-Original",               # required: where the example may come from
     "source": "datasets/chat/seed.jsonl",        # optional provenance note
     "loss_on": "all_assistant" | "last_assistant",
     "messages": [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}, ...]}

The loss covers assistant tokens only (``loss_on`` chooses all assistant turns or just
the last one). The chat template is the base model's own, applied through
``tokenizer.apply_chat_template``, the same path ``HFModel`` uses at inference.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..agent.redact import looks_secret
from .chat import to_hf_messages

DOMAINS = ("chat", "coding", "tool")
ROLES = ("system", "user", "assistant", "tool")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def content_key(ex: dict) -> str:
    """Duplicate key: normalised role+content of every message."""
    blob = json.dumps([[m["role"], _norm(m["content"])] for m in ex["messages"]])
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def prompt_key(ex: dict) -> str:
    """Leak key: normalised first user message."""
    first = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
    return hashlib.sha256(_norm(first).encode()).hexdigest()[:20]


def validate_example(ex) -> list[str]:
    errs: list[str] = []
    if not isinstance(ex, dict):
        return ["not a JSON object"]
    if ex.get("domain") not in DOMAINS:
        errs.append(f"domain must be one of {DOMAINS}")
    if not isinstance(ex.get("license"), str) or not ex["license"].strip():
        errs.append("license is required")
    if ex.get("loss_on", "all_assistant") not in ("all_assistant", "last_assistant"):
        errs.append("loss_on must be all_assistant or last_assistant")
    msgs = ex.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return errs + ["messages must be a non-empty list"]
    prev = None
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or set(m) - {"role", "content"} or "role" not in m or "content" not in m:
            errs.append(f"message {i}: needs exactly role and content")
            continue
        role, content = m["role"], m["content"]
        if role not in ROLES:
            errs.append(f"message {i}: unknown role {role!r}")
        if not isinstance(content, str) or not content.strip():
            errs.append(f"message {i}: empty content")
            continue
        if "�" in content or "\x00" in content:
            errs.append(f"message {i}: corrupted text (replacement character or NUL)")
        if looks_secret(content):
            errs.append(f"message {i}: looks like it contains a secret")
        if role == "system" and i != 0:
            errs.append(f"message {i}: system message must come first")
        if role == "assistant" and prev not in ("user", "tool"):
            errs.append(f"message {i}: assistant must follow a user or tool message")
        prev = role
    if not any(isinstance(m, dict) and m.get("role") == "user" for m in msgs):
        errs.append("needs at least one user message")
    if isinstance(msgs[-1], dict) and msgs[-1].get("role") != "assistant":
        errs.append("last message must be from the assistant")
    return errs


@dataclass
class LoadResult:
    examples: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)


def load_jsonl(path: Path) -> LoadResult:
    res = LoadResult()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
        except json.JSONDecodeError as e:
            res.failures.append({"file": str(path), "line": n, "errors": [f"invalid JSON: {e.msg}"]})
            continue
        errs = validate_example(ex)
        if errs:
            res.failures.append({"file": str(path), "line": n, "errors": errs})
            continue
        ex.setdefault("loss_on", "all_assistant")
        ex.setdefault("source", str(path))
        ex.setdefault("id", f"{ex['domain']}-{content_key(ex)}")
        res.examples.append(ex)
    return res


def dedupe(examples: list[dict]) -> tuple[list[dict], int]:
    seen, out = set(), []
    for ex in examples:
        k = content_key(ex)
        if k not in seen:
            seen.add(k)
            out.append(ex)
    return out, len(examples) - len(out)


def remove_leaks(train: list[dict], held_out_prompts: set[str]) -> tuple[list[dict], int]:
    """Drop training examples whose first user message also appears in a held-out set."""
    kept = [ex for ex in train if prompt_key(ex) not in held_out_prompts]
    return kept, len(train) - len(kept)


def domain_counts(examples: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for ex in examples:
        out[ex["domain"]] = out.get(ex["domain"], 0) + 1
    return out


def write_jsonl(path: Path, examples: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ tokenization
class TemplateError(ValueError):
    pass


def encode_example(tokenizer, ex: dict, max_len: int) -> tuple[list[int], list[int]] | None:
    """Token ids + labels (-100 outside the trained assistant turns). None if longer than max_len."""
    msgs = to_hf_messages(ex["messages"])
    full = tokenizer.apply_chat_template(msgs, tokenize=False)
    targets = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    if ex.get("loss_on") == "last_assistant":
        targets = targets[-1:]
    spans = []
    for i in targets:
        start_text = tokenizer.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)
        end_text = tokenizer.apply_chat_template(msgs[: i + 1], tokenize=False)
        if not (full.startswith(end_text) and end_text.startswith(start_text)):
            raise TemplateError("chat template is not prefix-stable; assistant spans cannot be located safely")
        spans.append((len(start_text), len(end_text)))
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) > max_len:
        return None
    labels = [-100] * len(ids)
    for t, (a, b) in enumerate(enc["offset_mapping"]):
        if any(s <= a < e for s, e in spans) and b > a:
            labels[t] = ids[t]
    if all(x == -100 for x in labels):
        return None
    return ids, labels


# ------------------------------------------------------------- tool transcripts
class _TokCounter:
    def __init__(self, tok) -> None:
        self.tok = tok

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)


def tool_transcripts(tokenizer, context_tokens: int, n_tasks: int, split: str, seed: int,
                     exclude_descriptions: set[str] | None = None) -> tuple[list[dict], dict]:
    """Expert bug-fix transcripts replayed through the REAL agent runtime (as ``infinity sft
    generate`` does), stored as messages so the open model's own chat template is used.

    Only transcripts the runtime judges ``completed_verified`` are kept. Each agent step
    becomes one example whose target is the expert action (``loss_on=last_assistant``).
    """
    import tempfile

    from ..agent import Agent, Limits, Mode, PermissionMode, PermissionPolicy
    from ..sft.generate import RecordingModel, materialise
    from ..sft.tasks import expert_actions, make_tasks

    tasks = make_tasks(split, n_tasks, seed)
    exclude = exclude_descriptions or set()
    out, kept, rejected, skipped = [], 0, 0, 0
    with tempfile.TemporaryDirectory(prefix="infinity-hf-tool-") as tmp:
        for task in tasks:
            if task.description in exclude:
                skipped += 1
                continue
            ws = materialise(task, Path(tmp))
            model = RecordingModel(expert_actions(task), _TokCounter(tokenizer), context_tokens)
            agent = Agent(model, ws, PermissionPolicy(PermissionMode.AUTO, approver=lambda r: False), Mode.DEBUG,
                          limits=Limits(max_steps=8, command_timeout=60))
            report = agent.run(task.description)
            if report.status != "completed_verified":
                rejected += 1
                if kept == 0 and rejected >= 5:
                    raise RuntimeError("the first expert runs all failed; the task tests cannot run on this machine "
                                       "(check that `python -m unittest` works)")
                continue
            kept += 1
            for step, (msgs, completion) in enumerate(model.records):
                out.append({
                    "id": f"tool-{task.id}-s{step}", "domain": "tool", "license": "Project-Original",
                    "source": f"infinity expert replay ({split} split, seed {seed})", "loss_on": "last_assistant",
                    "messages": [{"role": m.role, "content": m.content} for m in msgs] +
                                [{"role": "assistant", "content": completion}],
                })
    return out, {"tasks": n_tasks, "kept": kept, "rejected": rejected, "skipped_overlap": skipped,
                 "examples": len(out)}
