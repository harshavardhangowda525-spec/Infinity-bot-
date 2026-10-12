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
| Model good enough to drive the agent | ⬜ | needs a large corpus, GPU training and tool-use fine-tuning |
| Instruction / tool-use fine-tuning pipeline | ⬜ | Phase 6 |
| Benchmark suite (coding, tool use, retrieval…) | ⬜ | held-out validation loss only so far |
| Hardware audit and model-size recommendation | ✅ | `infinity audit` |

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
| Project-wide index (symbols, call sites) | ⬜ | search and read tools only |
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
| Intelligence panel with benchmark history | ⬜ | no benchmarks exist yet |

## Platform (image 2 and earlier specs)

| Feature | Status |
|---|---|
| Accounts / authentication | ⬜ |
| Subscriptions (Plus ₹500, Pro ₹1,000, Max ₹2,000), credit ledger, plan gating, COOK gating | ⬜ |
| Normal chat mode | ⬜ |
| Token / API-key platform, agent registry | ⬜ |
| Owner admin command center | ⬜ |
| Payment provider (dev mode only, as decided) | ⬜ |
