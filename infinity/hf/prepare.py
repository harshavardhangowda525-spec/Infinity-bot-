"""Build training/data/{train,val}.jsonl plus a dataset report from the configured sources."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .chat import chat_system_prompt
from .data import (
    domain_counts, dedupe, encode_example, load_jsonl, prompt_key, remove_leaks, tool_transcripts, write_jsonl,
)


def _load_dirs(root: Path, dirs: list[str], domain: str | None) -> tuple[list[dict], list[dict]]:
    examples, failures = [], []
    for d in dirs:
        for f in sorted((root / d).glob("*.jsonl")):
            res = load_jsonl(f)
            failures += res.failures
            for ex in res.examples:
                if domain and ex["domain"] != domain:
                    failures.append({"file": str(f), "line": None, "errors": [
                        f"example {ex['id']} has domain {ex['domain']!r} but sits in a {domain} folder"]})
                    continue
                examples.append(ex)
    return examples, failures


def test_prompts(path: Path) -> set[str]:
    """Prompt keys of the held-out evaluation set (never trained on)."""
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            keys.add(prompt_key({"messages": [{"role": "user", "content": case["prompt"]}]}))
    return keys


def adapter_label(cfg: dict) -> str:
    """The product label the bot shows when this adapter is loaded (see HFModel.info)."""
    return cfg["model_label"] + " + INFINITY adapter"


def prepare(cfg: dict, root: Path, tokenizer=None, log=print) -> dict:
    d = cfg["data"]
    t0 = time.time()
    report: dict = {"created_unix": int(t0), "base_model": cfg["base_model"], "train_domains": d["train_domains"]}

    chat, f1 = _load_dirs(root, d["chat_dirs"], "chat")
    coding, f2 = _load_dirs(root, d["coding_dirs"], "coding")
    val, f3 = _load_dirs(root, d["validation_dirs"], None)
    failures = f1 + f2 + f3

    tool_stats: dict = {}
    tool_train: list[dict] = []
    if "tool" in d["train_domains"] and d["tool_tasks_train"] > 0:
        if tokenizer is None:
            raise ValueError("tool transcripts need the base model's tokenizer (to fit prompts to the context)")
        from ..sft.tasks import make_tasks

        # The live-repair benchmark uses make_tasks("eval", n, 1337); keep those tasks out of every split.
        bench = {t.description for t in make_tasks("eval", 30, 1337)}
        log(f"replaying {d['tool_tasks_train']} train + {d['tool_tasks_val']} validation tool tasks through the runtime…")
        tool_train, tool_stats["train"] = tool_transcripts(tokenizer, cfg["context_tokens"], d["tool_tasks_train"],
                                                           "train", d["tool_seed"])
        tool_val, tool_stats["val"] = tool_transcripts(tokenizer, cfg["context_tokens"], d["tool_tasks_val"],
                                                       "eval", d["tool_val_seed"], exclude_descriptions=bench)
        val += tool_val

    # Chat/coding examples get the same system prompt the bot's Normal chat mode sends.
    label = adapter_label(cfg)
    for ex in chat + coding + val:
        if ex["domain"] != "tool" and ex["messages"][0]["role"] != "system":
            ex["messages"] = [{"role": "system", "content": chat_system_prompt(label)}] + ex["messages"]

    train = [ex for ex in chat + coding + tool_train if ex["domain"] in d["train_domains"]]
    train, dup_train = dedupe(train)
    val, dup_val = dedupe(val)
    held_out = {prompt_key(ex) for ex in val} | test_prompts(root / d["test_cases"])
    train, leaks = remove_leaks(train, held_out)

    too_long = {"train": 0, "val": 0}
    template_errors = 0
    if tokenizer is not None:
        from .data import TemplateError

        for name, rows in (("train", train), ("val", val)):
            kept = []
            for ex in rows:
                try:
                    enc = encode_example(tokenizer, ex, d["max_seq_len"])
                except TemplateError:
                    template_errors += 1
                    continue
                if enc is None:
                    too_long[name] += 1
                    continue
                ex["n_tokens"] = len(enc[0])
                kept.append(ex)
            if name == "train":
                train = kept
            else:
                val = kept

    out = root / d["out_dir"]
    report.update({
        "train": {"examples": len(train), "by_domain": domain_counts(train),
                  "sha256": write_jsonl(out / "train.jsonl", train)},
        "val": {"examples": len(val), "by_domain": domain_counts(val), "sha256": write_jsonl(out / "val.jsonl", val)},
        "validation_failures": failures,
        "duplicates_removed": {"train": dup_train, "val": dup_val},
        "leaks_removed_from_train": leaks,
        "too_long_dropped": too_long,
        "template_errors": template_errors,
        "max_seq_len": d["max_seq_len"],
        "tool_transcripts": tool_stats,
        "tokens": {k: sum(ex.get("n_tokens", 0) for ex in rows) for k, rows in (("train", train), ("val", val))},
        "seconds": round(time.time() - t0, 1),
    })
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.md").write_text(report_markdown(report), encoding="utf-8")
    return report


def report_markdown(r: dict) -> str:
    lines = ["# Dataset report", "", f"Base model tokenizer: `{r['base_model']}` · max length {r['max_seq_len']} tokens", "",
             "| Split | Examples | By domain | Tokens |", "|---|---|---|---|"]
    for s in ("train", "val"):
        lines.append(f"| {s} | {r[s]['examples']} | {r[s]['by_domain']} | {r['tokens'][s]:,} |")
    lines += ["", f"- Duplicates removed: {r['duplicates_removed']}",
              f"- Training examples removed because their prompt is in validation/test: {r['leaks_removed_from_train']}",
              f"- Dropped for exceeding max length: {r['too_long_dropped']}",
              f"- Chat-template errors: {r['template_errors']}",
              f"- Tool transcripts: {r['tool_transcripts'] or 'not generated'}",
              f"- Validation failures: {len(r['validation_failures'])}"]
    for f in r["validation_failures"][:50]:
        lines.append(f"  - `{f['file']}` line {f['line']}: {'; '.join(f['errors'])}")
    return "\n".join(lines) + "\n"
