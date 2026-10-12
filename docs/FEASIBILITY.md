# INFINITY HYPER 5.5 — Feasibility report

All numbers below were measured in the development environment (a cloud
container) on 2026-10-12 unless marked *estimate*. Run `infinity audit` to
measure your own machine.

## 1. Hardware capability report (development environment)

| Resource | Measured |
|---|---|
| CPU | 4 cores, x86_64 |
| RAM | 15.7 GB, no swap |
| GPU / accelerator | **none** |
| Free disk | ~22 GB (ephemeral container) |
| Python | 3.13.16 |
| PyTorch | 2.14.1 (PyPI build; CUDA unused, CPU execution) |
| Node / npm / git / docker | 22.22.0 / 10.9.4 / 2.43.0 / 29.8.2 |
| Blender | not installed |

## 2. Initial model configuration (`configs/hyper-5.5-tiny.json`)

| Setting | Value |
|---|---|
| Architecture | decoder-only transformer: RoPE, RMSNorm pre-norm, SwiGLU, tied embeddings |
| Layers / d_model / heads / d_ff | 6 / 256 / 8 / 688 |
| Context length | 256 tokens |
| Vocabulary | 4,096 (byte-level BPE, trained in-project) |
| Parameters | **5,795,072** (measured) |
| Optimizer | AdamW (β = 0.9, 0.95; weight decay 0.1), warmup 100 steps, cosine decay to 10% |
| Batch | 16 × 256 = 4,096 tokens/step |

**Measured throughput on this CPU:** ~0.62 s/step, about **6,600 tokens/s**
(20-step benchmark, including one validation pass).

A scale-up config, `configs/hyper-5.5-small.json`, is included: 12 layers,
d_model 768, 1,024 context, 16k vocabulary, *estimate* ~100M parameters. It
is meant for a GPU.

## 3. Training-data strategy

1. **Smoke corpus (now).** Text written for this project, plus CPython
   standard-library modules (PSF-2.0, read locally and not redistributed):
   153 documents, **1.04M training tokens** and 87k validation tokens after
   filtering. This proves the pipeline. It is far too small for a capable model.
2. **Pretraining corpus (next).** Permissively licensed source code and docs
   (MIT / Apache-2.0 / BSD / CC0), each source declared in `data/manifest.json`
   with origin and licence. The licence allow-list, secret filter, PII scrub and
   dedup are already enforced. Target: 1–10B tokens for a ~100M model.
3. **Instruction and tool-use tuning (Phase 6).** Curated examples of the JSON
   action protocol (`infinity/agent/protocol.py`): task → tool calls with real
   observations → final report. Sources: synthetic tasks run against real
   repositories by the runtime itself, with user-approved transcripts only,
   filtered for secrets. Evaluation tasks are held out and never trained on.

## 4. Compute and storage budget

Training cost ≈ 6 × parameters × tokens FLOPs.

| Config | Params | Tokens | FLOPs (*estimate*) | This CPU (measured rate) | One 24 GB consumer GPU (*estimate*, ~30% util.) |
|---|---|---|---|---|---|
| tiny (this run) | 5.8M | 6.1M | 2.1e14 | **~16 min** | < 1 min |
| tiny, Chinchilla-optimal | 5.8M | 116M | 4e15 | ~5 h | ~2 min |
| small | ~100M | 2B | 1.2e18 | months (impractical) | ~1–2 days |
| ~1B | ~1B | 20B | 1.2e20 | — | ~months; needs a multi-GPU cluster |

Storage: a checkpoint is ~70 MB for tiny (weights plus AdamW state). For small
it would be ~1.2 GB *estimate*. The tokenized corpus takes 2 bytes per token
(uint16).

**Honest expectation.** The tiny model learns code-shaped text: indentation,
keywords and common identifiers. It will not plan, follow the action protocol
or fix bugs. Useful agent behaviour in an original model needs, at minimum,
the *small* configuration on billions of tokens plus instruction tuning. That
means GPU hardware. Until then the agent runtime reports the model's invalid
actions as `blocked` instead of faking success.

## 5. Module-by-module plan

| Phase | Module | Status |
|---|---|---|
| 1 | Environment audit, feasibility (this document) | done |
| 2 | INFINITY TOKENIZER, INFINITY NEURAL CORE | done, tested |
| 3 | Smoke training: train → checkpoint → reload → generate | done (see the training results in `docs/FEATURES.md`) |
| 4 | Licensed dataset pipeline with provenance | pipeline done; large corpus not yet collected |
| 5 | Training, checkpointing, inference, evaluation | training and inference done; held-out loss done; benchmark suite pending |
| 6 | Instruction / tool-use fine-tuning | not started (needs a dataset and GPU) |
| 7 | INFINITY BRAIN, tools, memory | done, tested |
| 8 | INFINITY CODER repository workflows (indexing, symbol search) | partial (search and read tools; no index yet) |
| 9 | INFINITY STUDIO dashboard | home screen and workspace done (see `FEATURES.md`); Normal/COOK, subscriptions and admin not started |
| 10 | Integration, docs, optimisation | ongoing |
