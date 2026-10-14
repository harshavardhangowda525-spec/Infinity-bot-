# Feature checklist

Legend: ✅ implemented and tested · 🟡 partially implemented · ⬜ not implemented.
"Tested" means automated tests and/or the manual runs recorded in this repository.

## INFINITY HYPER 5.5 (original model)

| Feature | Status | Notes |
|---|---|---|
| Byte-level BPE tokenizer: train, encode/decode, special tokens, versioning | ✅ | `infinity/core/tokenizer.py`, `tests/test_tokenizer.py` |
| Transformer: RoPE, RMSNorm, causal MHA, SwiGLU, tied embeddings, KV cache | ✅ | causality, cache equivalence and overfitting tests |
| Trainer: AdamW, warmup+cosine, grad accumulation, clipping, bf16, resume, interrupt | ✅ | `tests/test_training_pipeline.py` |
| Atomic, sha256-verified checkpoints; never overwrite; safe loading | ✅ | corruption and incompleteness tests |
| Inference: streaming, sampling controls, cancellation | ✅ | |
| Licensed data pipeline: allow-list, secret/PII filter, dedup, per-document splits, provenance | ✅ | `data/manifest.json`, `data/PROVENANCE.md` |
| Trained prototype checkpoint (5.8M params, 6.1M tokens) | ✅ | `docs/TRAINING_REPORT.md`. Produces code-shaped text only |
| Model good enough to drive the agent | ⬜ | tool-use SFT reached 100% valid actions but 0/30 live repairs; see `docs/FINETUNE_REPORT.md` |
| Tool-use fine-tuning pipeline (runtime-verified expert data, masked loss, context extension, LM rehearsal, lineage) | ✅ | `infinity sft generate/train`, `tests/test_sft_bench.py` |
| **INFINITY HYPER 5.5 — Fine-Tuned**: LoRA adapters on the frozen original. sha256-pinned to the base, never merged, base verified unchanged after training, selectable next to the original | ✅ | `infinity sft train --lora`, `bench run --adapter`, `tests/test_lora.py` |
| Benchmark suite: lm perplexity, protocol validity, live repair; recorded with checkpoint hash | ✅ | `infinity bench run/list`; Intelligence view |
| Explicit model promotion gated on recorded benchmarks | ✅ | `infinity model promote` (nothing promoted yet) |
| Hardware audit and model-size recommendation | ✅ | `infinity audit` |
| 100M / 1B / 7B configs (GQA, gradient checkpointing, exact parameter formula) | ✅ | 7B = 7,110,660,096 parameters; not trained (needs a GPU cluster) |
| Sizing tool (memory, FLOPs, GPU-hours, optional cost) | ✅ | `infinity size` |
| Large-corpus pipeline: streaming clean/dedupe, hash splits, sampled tokenizer, parallel sharded tokenisation | ✅ | `infinity data prepare-large` |
| Licence-filtered Hugging Face Parquet fetcher with provenance | 🟡 | filtering tested locally; download untested here (huggingface.co blocked) |
| Multi-GPU training: DDP + FSDP2, bf16/fp16, resume across world sizes, time budget | 🟡 | verified with 2 CPU processes (gloo); NCCL/GPU runs untested here |
| Kaggle notebook for the 100M model | ✅ | smoke-tested cell by cell; real Kaggle GPU run not done |

## Open model option (not HYPER 5.5)

| Feature | Status | Notes |
|---|---|---|
| Open Hugging Face model as a selectable, honestly labelled model (Qwen2.5-Coder 0.5B) | 🟡 | `infinity/hf/model.py`, `configs/open-models.json`. Tested with a tiny offline model; the real Qwen download is untested here |
| Switch with fallback; owner's choice persisted | ✅ | `tests/test_hf.py` |
| LoRA SFT pipeline: data validation, dedupe, leak checks, assistant-only loss, dry-run, smoke, resume, OOM handling | ✅ | `training/scripts/*.py`. Pipeline tested with the tiny model only |
| Evaluation suite (24 held-out cases + live repair) with sandboxed code tests | ✅ | `training/scripts/evaluate.py`. HYPER 5.5 tiny measured 0/24 |
| Trained Qwen adapter and baseline-vs-adapter results | ⬜ | to be run on the owner's PC |

## INFINITY BRAIN (agent runtime)

| Feature | Status | Notes |
|---|---|---|
| Bounded loop: steps, time, tokens, cancellation, invalid-output limit | ✅ | `tests/test_agent_loop.py` |
| Strict JSON action protocol with validation | ✅ | |
| Runtime-judged outcomes (verified / unverified / partial / blocked / answered / cancelled) | ✅ | |
| Modes: ask, plan, review (read-only); code, debug, build, autonomous | ✅ | build and autonomous share the generic loop (no special spec workflow yet) |
| Filesystem tools: list, find, search, read, write, edit (multi-edit), move, mkdir, delete | ✅ | stale-edit detection, syntax checks, diffs |
| Terminal: run with streaming output, timeouts, process-group kill; background processes | ✅ | |
| Git status / diff / log | ✅ | mutating git goes through risk-classified `run_command` |
| Permission policy (read-only / ask / auto), approvals, command risk classes | ✅ | `docs/PERMISSIONS.md`, sandbox test proves blocked commands never spawn |
| Workspace scoping, symlink escape and secrets-file protection | ✅ | |
| Secret redaction; scrubbed child environment | ✅ | |
| File checkpoints and rollback | ✅ | `infinity rollback` and the dashboard "Undo" button |
| SQLite memory: sessions, tasks, tool calls, project notes, preferences, trust, experiences | ✅ | inspect, export, delete, wipe; refuses secrets |
| Experience retrieval before tasks | 🟡 | keyword overlap; no embeddings |
| Project detection (language, framework, package manager, test commands) | ✅ | `infinity/agent/project.py` |
| Project-wide code index (symbols, references, imports; incremental) | ✅ | `find_symbol`, `find_references`, `code_overview`; Python via ast, JS/TS via regex |
| Browser automation / API testing tools | ⬜ | |
| Blender integration | ⬜ | Blender is not installed in the dev environment |
| JARVIS / ASTON / other agent integrations | ⬜ | none present in this repository |

## INFINITY STUDIO dashboard (image 1)

| Feature | Status | Notes |
|---|---|---|
| Local backend: status, models, workspace, tree, file preview, tasks, approvals, cancel, rollback, event WebSocket | ✅ | `tests/test_server.py` (8 tests) |
| Loopback-only bind, Host/Origin guard, session token on state changes | ✅ | |
| 3D infinity core (React Three Fiber): glass ribbons, particles, light pulse | ✅ | lazy-loaded; low-power tier; SVG fallback without WebGL |
| Core states from real data: idle, thinking, executing, success, error | ✅ | `deriveCoreState` unit tests; observed in a browser run |
| Listening (mic amplitude plus speech-to-text) | 🟡 | only where the browser supports SpeechRecognition; not tested headlessly |
| Speaking (read results aloud) | 🟡 | opt-in setting; pulses on real speech boundary events; not tested headlessly |
| Reduced motion, hidden-tab pause, keyboard focus, ARIA labels | ✅ | |
| Top bar: wordmark, real model pill, connection indicator, settings | ✅ | |
| Model status popover (product vs implementation, runtime, agent, permission, errors) | ✅ | |
| Command dock: Enter / Shift+Enter, modes, real checkpoint selector, attach project, Stop | ✅ | component tests |
| Floating panels: Task, Change review (approve/reject, diffs, undo), Files, Code (highlighted), Execution stream | ✅ | no permanent sidebar; panels open on demand |
| Project launcher: open folder with permission, recent projects, detection, last task | ✅ | |
| Minimal ↔ workspace view switch; responsive desktop / tablet / mobile | ✅ | screenshots at 1440×900 and 390×844 |
| Live website preview panel | ⬜ | |
| Intelligence panel with benchmark history | ✅ | Intelligence view in the floating navigation |

## Platform (image 2)

| Feature | Status | Notes |
|---|---|---|
| Accounts: register, sign in, sign out, scrypt hashing, session cookies, CSRF, login throttling, idle and absolute expiry | ✅ | `infinity/platform/auth.py`, `tests/test_platform.py`, `tests/test_server.py` |
| Owner account created explicitly from the CLI; no default credentials | ✅ | `infinity admin create-owner` |
| Owner/user roles enforced on every admin endpoint | ✅ | |
| Plans PLUS ₹500 / PRO ₹1,000 / MAX ₹2,000 from one data source; INR; yearly disabled | ✅ | `configs/plans.json` seeds the database |
| Allowances / step limits / rate limits configurable by owner; null = not configured | ✅ | never invented |
| Subscription page: hero, toggle, 3 cards with Pro highlight, comparison table, confirmation dialog, footer | ✅ | footer policies say "not published yet" (no text was provided) |
| Checkout: development-mode message, no activation | ✅ | |
| Payment provider integration (Razorpay/Stripe checkout) | ⬜ | you chose dev mode; the generic signed webhook is ready for an adapter |
| Signed payment webhook: amount/currency check, idempotent events, refunds, failed payments | ✅ | |
| Subscription expiry and cancel-at-period-end | ✅ | expiry is checked lazily on access; no background job |
| Credit ledger: append-only, idempotent, atomic; monthly allocation | ✅ | concurrency test |
| Automatic monthly renewal / re-allocation without a new payment event | ⬜ | each paid period needs a verified payment event |
| COOK gating on the server (direct API calls cannot bypass it) | ✅ | 402 with a clear reason; upgrade overlay in the UI |
| Workspace roots per user (COOK on the host is limited to owner-authorised folders) | ✅ | |
| Per-user agent runtimes (no cross-user events, files or tasks) | ✅ | |
| Usage metering: exact tokens for COOK and API; estimated only for the scripted test model (labelled) | ✅ | |
| Token screen: plan, remaining credits, low-credit warning, usage chart (+table), ledger | ✅ | |
| Agent API keys: create (shown once), rotate, revoke, per-key rate limit, usage, last used | ✅ | |
| `/api/v1/generate` for agents, metered and rate limited | ✅ | |
| Owner admin console: overview, users (grant, credits, suspend, sign out, workspaces), plans editor, audit log | ✅ | |
| Hash-chained, append-only audit log with verification | ✅ | |
| MFA, trusted devices | ⬜ | |
| Normal (chat) mode | ✅ | `/api/chat` and the Chat mode in the dock. Every account can use it; usage is metered, and charged only on paid plans. Useful only with an open model: HYPER 5.5 tiny still produces nonsense |
| Admin: service restart, backups and restore, log search, config centre, notifications | ⬜ | |
| JARVIS / ASTON / FRIDAY / DARWIN / ROBIN registry with heartbeats | 🟡 | they can be issued API keys; no heartbeat or registry service exists |
