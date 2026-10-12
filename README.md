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
> needs far more data and compute (see `docs/FEASIBILITY.md`). The
> **INFINITY STUDIO** dashboard (3D infinity core, command dock, floating
> workspace panels) runs on top of the real agent. Accounts, plans, the credit
> ledger, COOK gating, agent API keys and an owner admin console are in place.
> Payments run in development mode. See `docs/FEATURES.md`.

## Layout

```
infinity/
  core/        tokenizer.py  model.py  trainer.py  inference.py  checkpoint.py  data.py  config.py
  agent/       loop.py (brain)  protocol.py  permissions.py  memory.py  checkpoints.py  redact.py
               models/ (adapter interface)   tools/ (fs, shell, git)
  server/      app.py (FastAPI, local only)  runtime.py (per-user runtimes, shared model)
  platform/    auth.py  billing.py (plans, subscriptions, payments, ledger)  keys.py  audit.py  db.py (migrations)
  audit.py     cli.py
studio/        INFINITY STUDIO — React + TypeScript + Tailwind + Framer Motion + React Three Fiber
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

## Make it smarter (tool-use fine-tuning and benchmarks)

```bash
infinity sft generate --out runs/sft-v2            # expert transcripts replayed through the real runtime
infinity sft train --base runs/hyper-5.5-tiny --data runs/sft-v2 --out runs/my-sft --lm-mix 0.25
infinity bench run --checkpoint runs/my-sft --sft runs/sft-v2      # lm, protocol, live repair (recorded)
infinity model promote --checkpoint runs/my-sft/checkpoints/step_001000 --reason "..."   # explicit; needs results
```

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

## INFINITY STUDIO (dashboard and platform)

```bash
pip install fastapi uvicorn websockets          # backend extras (or: pip install -e ".[studio]")
cd studio && npm install && npm run build && cd ..
infinity admin create-owner --email you@example.com   # once; you type the password (no default account exists)
infinity studio                                       # http://127.0.0.1:8765, then sign in
```

- **Security:** the server binds to **127.0.0.1 only** and rejects foreign Host and Origin headers.
  - Accounts use scrypt-hashed passwords and HttpOnly SameSite=Strict session cookies.
  - Every state-changing request needs that session's CSRF token.
  - Logins are throttled after repeated failures.
- **Plans:** PLUS ₹500, PRO ₹1,000, MAX ₹2,000 per month, all from one source (`configs/plans.json`, seeded into the database).
  - Token allowances, step limits and rate limits start **unconfigured**. Set them in **Admin → Plans**; every change is audited.
  - Yearly billing shows as unavailable.
- **Payments:** development mode. Checkout never charges and never activates anything.
  - A subscription activates only from a payment notification signed with `INFINITY_BILLING_WEBHOOK_SECRET` and matching the plan price (`POST /api/billing/webhook`, HMAC-SHA256 in `X-Infinity-Signature`).
  - Or from an audited owner grant: Admin → Users, or `infinity admin grant-plan`.
- **COOK** (agent tasks on this machine) is checked on the server on every request.
  - It needs an active PLUS, PRO or MAX plan with credits, and a folder inside a workspace root the owner authorised for that account.
  - The owner is exempt: their usage is recorded but not charged.
- **Credits:** an append-only ledger with idempotency keys. Actual token usage (exact counts from the INFINITY tokenizer) is deducted after each COOK task and API call.
- **Agent API keys:** create, rotate and revoke them under **API keys**. Each key is shown once.
  - Agents call `POST /api/v1/generate` with `Authorization: Bearer inf_…`. Calls are rate limited and metered.
- **Admin console** (owner only): accounts, subscriptions, verified revenue (refunds subtracted, failed payments excluded), token usage, security events, plan configuration, and a hash-chained audit log.
- **Frontend development:** run `npm run dev` in `studio/`.
- **Shortcuts:** Enter to send · Shift+Enter for a new line · Ctrl+O project · Ctrl+. workspace.

## Tests

```bash
pytest                      # Python: tokenizer, model, training, permissions, tools, agent, server
cd studio && npm test       # UI: core-state mapping, command dock, execution stream
```

## Docs

- `docs/ARCHITECTURE.md`: system design, permission matrix, outcome judging
- `docs/FEASIBILITY.md`: hardware report, model config, data strategy, compute budget, plan
- `docs/FEATURES.md`: implemented / partial / not implemented
- `docs/TRAINING_REPORT.md`, `docs/FINETUNE_REPORT.md`: measured training and fine-tuning results
- `docs/PERMISSIONS.md`, `docs/TROUBLESHOOTING.md`
