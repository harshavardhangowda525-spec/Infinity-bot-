# INFINITY BOT — Architecture

INFINITY BOT is two separable systems:

1. **INFINITY HYPER 5.5**: the language model. It has its own tokenizer,
   transformer implementation, trainer, checkpoints and inference engine, all
   written in this repository and trained from scratch. No pretrained weights
   are used.
2. **INFINITY BRAIN**: the agent runtime. It handles tools, permissions,
   memory, the bounded loop and verification. The model only *proposes*
   actions; the runtime executes them and decides what counts as success.

```
                ┌──────────────────────── INFINITY BRAIN (infinity/agent) ───────────────────────┐
  task ───────▶ │ Agent.run()                                                                     │
                │   system prompt ← mode rules + tool signatures + INFINITY.md/AGENTS.md          │
                │                 + project notes + relevant past experiences (Memory)            │
                │   loop (≤ max_steps, ≤ max_seconds, ≤ max_tokens, cancellable):                 │
                │     ModelAdapter.complete() ──▶ protocol.parse_action()  (strict JSON)          │
                │        │ invalid → error fed back; N in a row → BLOCKED                         │
                │        ▼                                                                        │
                │     ToolRegistry.execute() ──▶ PermissionPolicy (allow / ask human / deny)      │
                │        │                       Workspace.resolve() (no traversal / symlinks)    │
                │        │                       classify_command() (SAFE…BLOCKED)                │
                │        ▼                       CheckpointStore.snapshot() before any change     │
                │     real ToolResult (stdout/stderr/exit code/diff) → redacted → back to model   │
                │   final → _judge(): VERIFIED only if a check passed after the last edit         │
                └──────────────┬───────────────────────────────────────────────┬──────────────────┘
                               │ ModelAdapter interface                         │ SQLite
                ┌──────────────▼──────────────┐                    ┌────────────▼────────────┐
                │ NeuralCoreModel             │                    │ Memory (~/.infinity)    │
                │  → core.inference           │                    │ sessions, messages,     │
                │  → InfinityNeuralCore       │                    │ tasks, tool_calls,      │
                │  → BPETokenizer             │                    │ project_notes, prefs,   │
                │ ScriptedModel (tests only)  │                    │ trusted ws, experiences │
                └─────────────────────────────┘                    └─────────────────────────┘
```

## Modules

| Module | Path | Responsibility |
|---|---|---|
| INFINITY TOKENIZER | `infinity/core/tokenizer.py` | Byte-level BPE. Training, encode/decode, special tokens, versioned JSON with fingerprint |
| INFINITY NEURAL CORE | `infinity/core/model.py` | Decoder-only transformer: RoPE, RMSNorm pre-norm, causal multi-head attention (explicit math path plus a fused-kernel path, tested equal), SwiGLU, tied embeddings, KV cache |
| Config / identity | `infinity/core/config.py`, `configs/*.json` | `ModelConfig`, `TrainConfig`, `RunConfig`; product identity `INFINITY HYPER 5.5` |
| INFINITY TRAINER | `infinity/core/trainer.py` | AdamW, warmup and cosine decay, gradient accumulation, clipping, bf16 autocast, validation, JSONL log, resume, graceful interrupt |
| Data pipeline | `infinity/core/data.py`, `data/manifest.json` | Licence allow-list, secret filter, PII scrub, dedup, per-document splits, tokenizer trained on train split only, provenance report |
| Checkpoints | `infinity/core/checkpoint.py` | Atomic save, never overwrite, sha256 manifest, `weights_only` loading, `LATEST` pointer |
| INFINITY INFERENCE | `infinity/core/inference.py` | Verified load, streaming generation, temperature / top-k / top-p / repetition penalty, cancellation, sliding window |
| Model adapters | `infinity/agent/models/` | `ModelAdapter` interface: `NeuralCoreModel` (exact token counts), `ScriptedModel` (deterministic test double) |
| Action protocol | `infinity/agent/protocol.py` | Strict single-JSON-object action format (the model has no native tool calling) |
| Permissions | `infinity/agent/permissions.py` | Workspace scoping, sensitive files, command risk classifier, policy matrix |
| Tools | `infinity/agent/tools/` | fs (list/find/search/read/write/edit/move/mkdir/delete), shell (run/start/output/stop), git (status/diff/log) |
| Checkpoints (files) | `infinity/agent/checkpoints.py` | Pre-change snapshots per session; `infinity rollback` |
| Memory | `infinity/agent/memory.py` | SQLite memory with inspect/export/delete/wipe; refuses secrets |
| Redaction | `infinity/agent/redact.py` | Removes secrets from logs, memory and model context; scrubs child-process environments |
| Brain | `infinity/agent/loop.py` | Modes, limits, the loop, outcome judging, events for UIs |
| CLI | `infinity/cli.py` | `audit`, `data prepare`, `train`, `generate`, `checkpoints`, `agent`, `rollback`, `memory` |

## Modes

| Mode | Tools | Policy |
|---|---|---|
| ask, plan, review | read-only tools, plus `run_command` only for SAFE commands | forced `read-only`, whatever the configured policy |
| code, debug, build, autonomous | all tools | configured policy (`read-only` / `ask` / `auto`) |

## Permission matrix

| | read-only | ask | auto |
|---|---|---|---|
| read tools | allow | allow | allow |
| write / edit / move / mkdir | deny | **ask** | allow |
| delete | deny | **ask** | **ask** |
| SAFE command (ls, git status…) | allow | allow | allow |
| CHECK command (pytest, npm test, tsc…) | deny | **ask** | allow |
| MEDIUM (installs, scripts, shell syntax, unknown programs) | deny | **ask** | **ask** |
| HIGH (rm, sudo, curl, git push/reset, deploy, paths outside workspace, secret files) | deny | **ask** | **ask** |
| BLOCKED (rm -rf /, curl \| sh, mkfs, fork bomb, force-push to main…) | deny | deny | deny |

Unfamiliar workspaces run **read-only** until you trust them
(`infinity memory trust <path> ask|auto`). If the approver raises an error,
that counts as **no**.

## Outcome judging (anti-fabrication)

The model can only claim `completed`, `partial` or `blocked`. The runtime then
decides the reported status:

- `completed_verified`: a CHECK command exited 0 **after** the last file change.
- `completed_unverified`: no check ran after the last change.
- `partial`: the last check after the last change failed, or the step / time /
  token budget ran out after files changed.
- `blocked`: the model reported a blocker, the model was unavailable, it
  produced repeated invalid output, or the budget ran out with no changes.
- `answered`: ask, plan or review modes (there is nothing to verify).
- `cancelled`: the user cancelled.

## Product identity vs implementation

Every place that shows the model reports `ModelAdapter.info()`, which contains
both `product` (`INFINITY HYPER 5.5`) and `implementation` (for example
`infinity-neural-core (original weights, trained in this project)` or
`scripted (test double — not a language model)`). The two are never blurred.

## Platform layer (`infinity/platform`, served by `infinity/server/app.py`)

```
browser ──cookie+CSRF──▶ FastAPI ──▶ Auth (scrypt, sessions, throttling)
                              │ ──▶ Billing.entitlement(user)  ← plans (DB) + subscription + ledger balance
                              │        └─ COOK / API allowed?  (402 with a reason if not)
                              │ ──▶ RuntimeHub.get(user) ──▶ per-user Runtime ──▶ Agent (shared INFINITY HYPER 5.5 model)
                              │        └─ on task end: usage_events + ledger charge (idempotency key = task id)
agents ──Bearer inf_…──▶ /api/v1/generate ──▶ Keys.authenticate → rate limit → entitlement → model → meter + charge
provider ──HMAC──▶ /api/billing/webhook ──▶ verify signature → match amount/currency to plan → activate + allocate
owner ──▶ /api/admin/* (owner-only, reasons required) ──▶ Audit (hash chain; DB triggers forbid UPDATE/DELETE)
```

| Table | Purpose |
|---|---|
| `users`, `auth_sessions`, `login_attempts` | accounts, hashed session tokens with a CSRF token each, throttling |
| `plans` | plan catalogue (single source after seeding) |
| `subscriptions`, `payments` | lifecycle, verified payments, refunds and failures |
| `ledger` | append-only credits (allocation, usage, adjustment, refund); unique idempotency keys |
| `usage_events` | per-request token counts (exact/estimated flag, source, key) |
| `api_keys` | sha256 of each key, prefix, scopes, rate limit, counters |
| `audit_log` | hash-chained privileged actions |

Schema changes are numbered migrations (`infinity/platform/db.py`).
