# Training guide: adapters for INFINITY HYPER 5.5 and the open model

## 0. Model identity and preservation

There are three models. Each has its own name, folder and label.

| Name shown everywhere | What it is | Files |
|---|---|---|
| **INFINITY HYPER 5.5** | The original model, trained from scratch | `runs/hyper-5.5-*/checkpoints/` (**read-only**) |
| **INFINITY HYPER 5.5 — Fine-Tuned** | The original, frozen, with a LoRA adapter applied on top at load time | `adapters/infinity-hyper-5-5/<name>/` (adapter only) |
| **Qwen2.5-Coder 0.5B Instruct — Fine-Tuned** | Open model + its own LoRA adapter (option C). **Not** HYPER 5.5 | `adapters/open-models/<name>/` |

Guarantees, enforced in code and covered by `tests/test_lora.py` and `tests/test_hf.py`:

1. **Only LoRA adapters train.** Every base weight is frozen. QLoRA would need an NVIDIA GPU and is not used.
2. **Base files are never written.**
   - Each base checkpoint is fingerprinted with sha256 before and after training.
   - If anything changed, the run **fails** and exports nothing.
   - In memory, the frozen weights are also compared bit for bit.
3. **Adapters go in their own folders**, as shown above.
   - Checkpoints go to `training/checkpoints/<name>/` and logs to `training/logs/<name>.jsonl`.
   - An existing adapter is never overwritten; a timestamped name is used instead.
4. **Adapters load on top of the original at inference.**
   - An INFINITY adapter records the exact base checkpoint sha256.
   - It refuses to load on any other checkpoint, and refuses if the adapter file itself is corrupted.
5. **The original stays selectable as baseline and rollback.**
   - It's always in the model selector, under "INFINITY HYPER 5.5 (original)".
   - It's also the automatic fallback when an adapter fails to load.
6. **No merging.**
   - INFINITY adapters are never merged.
   - Open-model adapters are also kept separate by default (`merge_adapter=False`). The app reports `adapter_merged: false`.
7. **Each run writes a verification record:**
   - the training log's `start` event (with base hashes) and `end` event (`base_unchanged_after_training: true`);
   - the adapter's config or manifest.
8. **Labels:**
   - Config, logs, manifests, benchmark records and the UI name the base as "original, read-only" and the output as "… — Fine-Tuned".
   - Benchmarks of a fine-tuned model are stored under the run `<run>+lora:<name>`, never as the original.

## A. INFINITY HYPER 5.5 — Fine-Tuned (LoRA on your own model)

```bat
infinity sft generate --out runs/sft-v2 --context 512
infinity sft train --lora --base runs/hyper-5.5-tiny --data runs/sft-v2 --name hyper-5.5-finetuned --context 512 --steps 1000 --rank 16 --alpha 32 --lm-mix 0.25
infinity bench run --checkpoint runs/hyper-5.5-tiny --adapter adapters/infinity-hyper-5-5/hyper-5.5-finetuned --suites lm,protocol,repair --sft runs/sft-v2
python training/scripts/evaluate.py --model "neural:runs/hyper-5.5-tiny::adapters/infinity-hyper-5-5/hyper-5.5-finetuned" --label hyper-finetuned
```

- In the dashboard, choose "HYPER 5.5 — Fine-Tuned · hyper-5.5-finetuned" in the model selector.
- Choose a "HYPER 5.5 · …" entry to go back to the original.
- **Limit:** this is still the 5.8M-parameter model. Expect the format and tool-protocol
  results to improve, not general chat or coding. See `docs/evaluation_report.md`.

## B. Open model: Qwen2.5-Coder 0.5B + LoRA (option C)

This is a **separate** open model, default **Qwen2.5-Coder 0.5B Instruct** (Apache-2.0),
with a LoRA adapter you train on your own computer.

- **INFINITY HYPER 5.5 is not involved.** Its code and weights are untouched.
- **The open model is always shown by its real name**: "Qwen2.5-Coder 0.5B Instruct", or
  "Qwen2.5-Coder 0.5B Instruct — Fine-Tuned" with the adapter.
  - Its system prompt tells it not to claim to be HYPER 5.5.
  - It is never called "INFINITY HYPER 5.5 — Fine-Tuned", because that name belongs to adapters trained on your own model.
- **Every experiment can be undone.**
  - Base weights stay in the Hugging Face cache and are sha256-checked before and after training.
  - Adapters live in `adapters/open-models/`.

### B1. Setup (once)

Use the virtual environment you already use for INFINITY BOT. It already contains PyTorch.

```bat
cd C:\Users\Home\Infinity-bot-
.venv\Scripts\activate.bat
git pull origin claude/infinity-bot-platform-lgt431
pip install -r training/requirements-training.txt
```

This installs `transformers==5.19.0`, `peft==0.21.2`, `safetensors`, `pyyaml` and `psutil`.
`peft` also pulls in `accelerate`. TRL and `datasets` are **not** needed, because the
training loop is plain PyTorch.

**Disk:** the first run downloads the base model, about 1 GB, into
`%USERPROFILE%\.cache\huggingface`.

**Memory:** about 8 GB of RAM should be enough for the 0.5B model with the default
settings, and 16 GB is more comfortable. This is an estimate; the training script prints
its real memory use every step.

### B2. Commands

All commands run from the project root.

| Step | Command |
|---|---|
| Prepare data (validate, dedupe, leak-check, tool transcripts, report) | `python training/scripts/prepare_data.py --config training/configs/default.yaml` |
| Validate config + data without loading weights | `python training/scripts/train.py --config training/configs/default.yaml --dry-run` |
| Smoke test (2 steps, temporary folder, reload check) | `python training/scripts/train.py --config training/configs/default.yaml --smoke` |
| **Evaluate the baseline first** | `python training/scripts/evaluate.py --model hf:Qwen/Qwen2.5-Coder-0.5B-Instruct --label baseline` |
| Start training (asks for confirmation) | `python training/scripts/train.py --config training/configs/default.yaml` |
| Resume after Ctrl+C or a crash | `python training/scripts/train.py --config training/configs/default.yaml --resume` |
| Evaluate the trained adapter | `python training/scripts/evaluate.py --model "hf:Qwen/Qwen2.5-Coder-0.5B-Instruct::adapters/open-models/qwen-coder-0.5b-combined" --label combined` |
| Compare | `python training/scripts/evaluate.py --compare baseline combined` |
| Add the live agent benchmark (slow) | add `--repair 10` to an evaluate command |

- **Progress:** training prints loss, seconds per step and an **ETA** after every step, so
  you can tell it is running.
- **Ctrl+C** saves a checkpoint. Run the same command with `--resume` to continue.
- **Changing `--steps`** changes the learning-rate schedule. Keep it the same when resuming.

### Separate vs combined adapters (Phase 6 experiment)

```bat
python training/scripts/prepare_data.py --config training/configs/chat.yaml
python training/scripts/train.py --config training/configs/chat.yaml
python training/scripts/prepare_data.py --config training/configs/coding.yaml
python training/scripts/train.py --config training/configs/coding.yaml
```

- Each config writes to its own run folder: `qwen-coder-0.5b-chat` and `qwen-coder-0.5b-coding`.
- Evaluate each one with its own `--label`, then use `--compare`.
- **Prepare the data again before each config.** All three configs share `training/data/`.
- Don't assume one strategy is better. Only the evaluation results decide.

### B3. Use it in the bot

1. `configs/open-models.json` lists two entries: the plain base model, and the base +
   `adapters/open-models/qwen-coder-0.5b-combined`.
   - The adapter entry is greyed out in the model selector until that folder exists.
   - If you train under another `run_name`, add an entry for it.
2. `infinity studio` → sign in as the owner → choose the model in the selector under the command box.
   - The first load downloads the base model.
   - The owner's choice is saved in `runs/SELECTED_MODEL.json` and restored on restart.
3. **Chat** mode is a plain conversation. **Ask, Plan, Code, Debug** and **Autonomous**
   run the COOK agent with the selected model.

**Fallback:** if a switch fails (packages missing, download blocked, broken adapter), the
current model stays loaded. If nothing was loaded, INFINITY HYPER 5.5 is loaded instead.
The error appears in the model popover under "Last switch failed".

**Revert to the original model:** choose a `HYPER 5.5 · …` entry in the selector. To
remove the open-model option completely, delete its entries from
`configs/open-models.json`. Adapters can be deleted from `adapters/` at any time; nothing
else depends on them.

## What was actually tested (in the build environment: Linux, CPU, no internet to huggingface.co)

| Command / feature | Tested? | How |
|---|---|---|
| `prepare_data.py`, `train.py --dry-run`, `--smoke`, full run, `--resume` | ✅ | With the tiny random test model (`training/configs/tiny-test.yaml`) and `tests/test_hf.py` |
| Adapter export, reload equivalence, never-overwrite, OOM handling | ✅ | `tests/test_hf.py` |
| `evaluate.py` (open model, open model + adapter, HYPER 5.5) and `--compare` | ✅ | Tiny model + HYPER 5.5 tiny checkpoint |
| Evaluation sandbox (timeout, blocked network/process/outside writes) | ✅ | `tests/test_hf.py` (Linux) |
| Bot: model selector, switching, fallback, Normal chat, labels | ✅ | `tests/test_hf.py`, `tests/test_lora.py`, vitest, and a Chromium run against the tiny model |
| HYPER 5.5 LoRA training, base sha256 before/after, wrong-base and corruption refusal, switching and rollback | ✅ | `tests/test_lora.py`, plus a real run on the HYPER 5.5 tiny checkpoint (see `docs/evaluation_report.md`) |
| **Downloading and running the real Qwen2.5-Coder 0.5B** | ❌ | huggingface.co is blocked here. Your PC will be the first real run |
| **Real LoRA training and its speed on your PC** | ❌ | not run. The script prints the measured speed |
| Windows paths of these scripts | ❌ | written to be portable (pathlib, UTF-8), not run on Windows |

Offline pipeline check you can run yourself, with no download:

```bat
python training/scripts/make_tiny_model.py
python training/scripts/prepare_data.py --config training/configs/tiny-test.yaml
python training/scripts/train.py --config training/configs/tiny-test.yaml --smoke
```
