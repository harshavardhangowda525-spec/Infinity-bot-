# INFINITY BOT · INFINITY HYPER 5.5

A local autonomous coding agent (**INFINITY BOT**) built around an original
language model (**INFINITY HYPER 5.5**). The tokenizer, transformer, trainer
and inference engine are all written and trained from scratch in this
repository. No pretrained weights and no external AI APIs are used.

> **Honest status (v0.1, first increment).** The model pipeline works end to
> end: data → tokenizer → training → checkpoint → reload → generation. The
> agent runtime (tools, permissions, memory, bounded loop, verification) works
> and is tested. The current INFINITY HYPER 5.5 checkpoint is a **tiny CPU-trained
> prototype**. It produces code-flavoured text but **cannot yet drive the agent**:
> it does not follow the action protocol, so the runtime reports such tasks as
> `blocked` instead of pretending. Making the model useful for agent work
> needs far more data and compute (see `docs/FEASIBILITY.md`). The web
> dashboard, COOK mode, subscriptions and admin center are later increments;
> see `docs/FEATURES.md`.

## Layout

```
infinity/
  core/        tokenizer.py  model.py  trainer.py  inference.py  checkpoint.py  data.py  config.py
  agent/       loop.py (brain)  protocol.py  permissions.py  memory.py  checkpoints.py  redact.py
               models/ (adapter interface)   tools/ (fs, shell, git)
  audit.py     cli.py
configs/       hyper-5.5-tiny.json  hyper-5.5-small.json
data/          manifest.json (licensed sources)  PROVENANCE.md  corpus/
docs/          ARCHITECTURE.md  FEASIBILITY.md  FEATURES.md  PERMISSIONS.md  TROUBLESHOOTING.md
tests/         tokenizer, model, training pipeline, permissions/sandbox, tools, agent loop
```

## Install

Requires Python ≥ 3.11 and PyTorch (CPU is enough for the tiny model).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy pytest     # or: uv pip install torch numpy pytest
pip install -e .                   # provides the `infinity` command (or use `python -m infinity`)
```

No API keys are needed, and none are read. See `.env.example`.

## Train INFINITY HYPER 5.5 from scratch

```bash
infinity audit                                   # measured hardware report + recommended config
infinity data prepare --vocab 4096               # licence filter, secret filter, dedup, split, BPE, tokenize
infinity train --config configs/hyper-5.5-tiny.json --data runs/data
#   Ctrl+C saves a checkpoint; re-running resumes from the latest one.
infinity checkpoints --run runs/hyper-5.5-tiny   # list + sha256-verify checkpoints
infinity generate --checkpoint runs/hyper-5.5-tiny --prompt "def read_config(path):" --seed 1
```

Training writes `runs/<name>/train_log.jsonl`. It contains only measured
values (loss, validation loss, learning rate, gradient norm, elapsed time).

## Run the agent

```bash
infinity agent --workspace ~/code/myproject --model neural:runs/hyper-5.5-tiny \
               --mode code --permission ask --task "Fix the failing test in test_math.py"
```

- Modes: `ask`, `plan`, `review` (read-only), and `code`, `debug`, `build`, `autonomous`.
- Permissions: `read-only` (the default for untrusted workspaces), `ask`, `auto`. See `docs/PERMISSIONS.md`.
- Every change is checkpointed: `infinity rollback --workspace ... --session <id>`.
- Memory: `infinity memory show | export FILE | delete-session ID | delete-note ID | wipe`.

`--model scripted:file.json` replays a fixed list of actions. It is a test
double for exercising the runtime, and reports itself as such.

## Tests

```bash
pytest
```

## Docs

- `docs/ARCHITECTURE.md`: system design, permission matrix, outcome judging
- `docs/FEASIBILITY.md`: hardware report, model config, data strategy, compute budget, plan
- `docs/FEATURES.md`: implemented / partial / not implemented
- `docs/PERMISSIONS.md`, `docs/TROUBLESHOOTING.md`
