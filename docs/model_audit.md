# INFINITY HYPER 5.5: model audit (Phase 1)

Date: 2026-10-14. Audited commit: `42e060f` on `claude/infinity-bot-platform-lgt431`.
Method: read-only inspection of the repository, the checkpoint files and the build
environment. No weights, data or settings were changed.

Each fact is labelled by how it was verified:

- **verified**: read from code, files or command output.
- **reported**: taken from the owner's own logs or messages.
- **unverified**: could not be checked from here.

## 1. What the model is

| Question | Answer | Evidence |
|---|---|---|
| Pretrained Hugging Face model? | **No** | No `transformers` import anywhere in `infinity/`. No HF `config.json` or `model.safetensors` (verified) |
| Local Transformers checkpoint? | **No** | Checkpoints are `model.pt` (a PyTorch `state_dict`), `config.json` (INFINITY format), `tokenizer.json` (INFINITY BPE) and `manifest.json` (sha256 values) (verified) |
| Fine-tuned model or adapter? | **No adapters.** Full fine-tunes exist as separate runs | `runs/hyper-5.5-tiny-sft*` are full-weight SFT runs with lineage to the base (verified) |
| Custom model? | **Yes. Complete and original** | `infinity/core/model.py` (`InfinityNeuralCore`), trained from random initialisation by `infinity/core/trainer.py` (verified) |
| Weights complete? | **Yes, for the tiny run** | `runs/hyper-5.5-tiny/checkpoints/step_001500/model.pt`, 23.2 MB; sha256 `6d07aa07…420d` matches the manifest (verified) |

### Architecture (verified, `infinity/core/model.py`)

- **Type:** decoder-only transformer.
- **Embeddings:** token embedding, tied to the output head.
- **Each block:** pre-norm RMSNorm → causal self-attention (fused `qkv` projection, no biases, optional grouped-query attention) → RMSNorm → SwiGLU MLP (`gate`/`up`/`down`, no biases).
- **Positions:** RoPE with interleaved pairs (`x[..., 0::2]`, `x[..., 1::2]`), θ = 10000.
- **Also:** KV cache for generation; optional gradient checkpointing.

### Configurations

| Config | Params | Layers | d_model | Heads (KV) | Context | Vocab | Trained? |
|---|---|---|---|---|---|---|---|
| `hyper-5.5-tiny` | **5,795,072** (manifest) | 6 | 256 | 8 (8) | 256 | 4,096 | **Yes**: 1,500 steps, 6.14M tokens, val loss 3.62 (verified) |
| `hyper-5.5-small` | 97.5M | see config | | | 1,024 | 16,384 | No (Kaggle notebook prepared) |
| `hyper-5.5-1b` | 1.03B | | | | | | No |
| `hyper-5.5-7b` | 7,110,660,096 | | | GQA | | | No |

- **Tokenizer:** original byte-level BPE (`infinity/core/tokenizer.py`), 4,096 tokens, fingerprint `7e487f94e3434858`. Not a Hugging Face `tokenizers` file (verified).
- **Chat template** (`render_chat` in `infinity/agent/models/base.py`): `<|role|>content<|end|>…<|assistant|>` (verified).
- **Precision:** training supports fp32, bf16 and fp16 with loss scaling. The tiny run used fp32 on CPU (verified).
- **Licence:**
  - All model code and weights are original to this repository. No third-party weights are involved (verified).
  - **The repository has no LICENSE file** (verified). The owner should choose one before publishing weights.
  - Training data licences are listed in `data/manifest.json` and `data/PROVENANCE.md` (verified).

### Measured capability (from `runs/benchmarks.db` and `docs/FINETUNE_REPORT.md`, verified)

| Model | Perplexity (test) | Valid JSON actions | Exact actions | Live repair tasks |
|---|---|---|---|---|
| base tiny, step 1500 | 40.15 | n/a | n/a | n/a |
| SFT v2, step 1000 | 44.19 | 100% | 67% | **0 / 30** |

**Conclusion:** the model produces code-shaped text. It cannot hold a conversation or solve tasks. **No checkpoint has been promoted.**

## 2. Hardware

### The owner's computer (the target for training)

| Item | Status |
|---|---|
| OS | Windows 10 (reported) |
| CPU / RAM / disk | **unverified** |
| GPU | **unverified**. No GPU use was reported |
| Measured training speed | **3.3 s per step** for the tiny model (batch 16 × 256 tokens), about 1,240 tokens/s. This matches CPU-only training (reported: the owner's training log, step 210 at 801 s) |

To verify, the owner should run `infinity audit` (read-only) and share the output.
That command reports CPU, RAM, disk and CUDA availability, plus a recommended config.

### This build environment (where the audit ran; not the training target)

- 4 CPU cores, 15 GB RAM, about 21 GB free disk, **no GPU** (`nvidia-smi` is absent).
- Python 3.13.16 and PyTorch 2.14.1.
- `transformers`, `peft`, `trl`, `datasets` and `accelerate` are **not installed** (verified).
- huggingface.co was unreachable from here in earlier sessions.

## 3. Is the requested Hugging Face / PEFT / TRL workflow feasible?

**Not as written. The Phase 3 blocker condition applies.**

### Blocker 1: the model is not a Transformers model
PEFT, TRL and `AutoModelForCausalLM` cannot load `InfinityNeuralCore` checkpoints.

- The architecture maps closely onto Hugging Face's `LlamaForCausalLM` (RMSNorm, SwiGLU, RoPE, GQA, no biases, tied embeddings).
- A converter would need to:
  1. split the fused `qkv` weight;
  2. permute the q/k rows from interleaved RoPE to Hugging Face's half-rotation layout (the same permutation Meta's conversion script uses);
  3. rename the tensors;
  4. convert the BPE tokenizer to a `tokenizers` JSON.
- **None of this exists yet, and it is unverified.** Its correctness would have to be proven by a logit-equivalence test against the original model.

### Blocker 2: LoRA has no purpose at this size
LoRA and QLoRA cut memory for models with billions of parameters.

- At 5.8M parameters, full fine-tuning costs less than LoRA's overhead and fits in a few hundred MB of RAM.
- Full fine-tuning already exists: `infinity sft train`, with masked loss, LM rehearsal, lineage, and evaluation in `infinity bench`.
- QLoRA additionally needs a CUDA GPU and `bitsandbytes`. No GPU was reported on the target machine.

### Blocker 3: capacity, not tooling, is the limit
- No tooling (PEFT, TRL or the existing pipeline) can give a 5.8M-parameter model conversational or general coding ability.
- The measured 0/30 repair result after fine-tuning is the evidence.
- Changing the training library does not change this.

## 4. What already exists from the requested phases

| Requested | Status in this repository |
|---|---|
| Dataset pipeline with licence allow-list, validation, dedup and leak-free splits | Exists for pretraining (`infinity data prepare`, `prepare-large`) and SFT (`infinity sft generate`, runtime-verified). There is no general chat dataset |
| Training script with smoke mode, resume, checkpoints, seeds and logging | Exists (`infinity train`, `infinity sft train`, `dist_train.py`). Atomic sha256 checkpoints |
| Evaluation suite with baseline preserved and machine-readable results | Exists (`infinity bench run`: lm, protocol, live repair → `runs/benchmarks.db`). There is no chat or factual-QA suite |
| Original model never overwritten; every experiment reversible | Exists. Runs never overwrite, and promotion is explicit (`infinity model promote`) |
| Normal chat mode | **Does not exist.** It was deliberately not built because the model cannot converse |
| Cook mode with tools | Exists: list, find, search, read, write and edit with diffs, tests, git diff, approvals, rollback, workspace scoping, secret redaction, audit log |
| Persistent memory | Exists: SQLite preferences, project notes, experiences and session history, with inspect, export, delete and wipe. Retrieval is keyword-based |
| Restricted execution of generated code | Partly. Risk classes, approvals, a scrubbed environment and timeouts exist. There is no OS-level sandbox (container or job object) |

## 5. Options (owner decision required. Nothing below has been started)

| Option | What it gives | Honest limit |
|---|---|---|
| **A. Stay on the INFINITY stack** (current path) | Kaggle 100M pretraining, then SFT and bench with the existing tools. 100% original model | 100M still won't chat or build websites (see `docs/FEASIBILITY.md`) |
| **B. Write an HF exporter** for HYPER 5.5 | Makes HYPER 5.5 weights loadable in Transformers, PEFT and TRL, verified by a logit-equivalence test | Changes tooling only, not intelligence |
| **C. Fine-tune an open pretrained model with LoRA** | Real conversational and coding gains at ₹0 | **That model would not be INFINITY HYPER 5.5.** It needs the owner's explicit approval, a licence check, and honest labelling in the UI. CPU-only LoRA is slow and limited to sub-1B models |

Per the instructions, training setup has stopped at this point. No dependencies were
installed, and no training workspace was created.

## 6. Decision (2026-10-14): option C

The owner chose option C with **Qwen2.5-Coder 0.5B Instruct** as the base. What was built:

- An open model plus an optional LoRA adapter, selectable next to INFINITY HYPER 5.5.
- HYPER 5.5 is unchanged and remains the default and the fallback.
- The open model is always labelled by its real name.

Licence: Qwen2.5-Coder 0.5B Instruct is published under Apache-2.0. **Re-check the model
card before use.** It could not be fetched from the build environment.

See `docs/training_guide.md`, `docs/dataset_guide.md`, `docs/evaluation_report.md` and `docs/cook_security.md`.

## 7. Preservation requirement (2026-10-14)

The owner required INFINITY HYPER 5.5 to stay intact, with fine-tuning done only through
separate adapters. The trained version is called **INFINITY HYPER 5.5 — Fine-Tuned**.

- **A (new).** LoRA for the original model.
  - Code: `infinity/core/lora.py` and `infinity/sft/lora_train.py`.
  - Train with `infinity sft train --lora`.
  - Adapters go to `adapters/infinity-hyper-5-5/` and are pinned to the base checkpoint's sha256.
  - They are never merged, and are verified before and after training.
- **B (kept).** The Qwen adapter keeps an honest name, "Qwen2.5-Coder 0.5B Instruct — Fine-Tuned".
  - It lives in `adapters/open-models/`.
  - It is no longer merged at load time, and its base files are sha256-checked before and after training.
  - It is not called "INFINITY HYPER 5.5 — Fine-Tuned", because that would credit HYPER 5.5 with Qwen's work.

Details: `docs/training_guide.md` §0.
