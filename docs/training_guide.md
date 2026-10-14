# Training guide: open model + LoRA adapter (option C)

This adds a **second, separate brain** to INFINITY BOT: an open Hugging Face model
(default: **Qwen2.5-Coder 0.5B Instruct**, Apache-2.0), optionally improved with a LoRA
adapter that you train on your own computer.

- **INFINITY HYPER 5.5 is not changed.** Its code, weights and training are untouched.
- **The open model is always shown by its real name.** The bot labels it
  "Qwen2.5-Coder 0.5B Instruct" or "… + INFINITY adapter", and its system prompt tells it
  not to claim to be HYPER 5.5.
- **Every experiment can be undone.**
  - Base weights stay in the Hugging Face cache and are never modified.
  - Adapters, checkpoints and logs live in their own folders.
  - Existing exports and results are never overwritten.

## 1. Setup (once)

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

## 2. Commands

All commands run from the project root.

| Step | Command |
|---|---|
| Prepare data (validate, dedupe, leak-check, tool transcripts, report) | `python training/scripts/prepare_data.py --config training/configs/default.yaml` |
| Validate config + data without loading weights | `python training/scripts/train.py --config training/configs/default.yaml --dry-run` |
| Smoke test (2 steps, temporary folder, reload check) | `python training/scripts/train.py --config training/configs/default.yaml --smoke` |
| **Evaluate the baseline first** | `python training/scripts/evaluate.py --model hf:Qwen/Qwen2.5-Coder-0.5B-Instruct --label baseline` |
| Start training (asks for confirmation) | `python training/scripts/train.py --config training/configs/default.yaml` |
| Resume after Ctrl+C or a crash | `python training/scripts/train.py --config training/configs/default.yaml --resume` |
| Evaluate the trained adapter | `python training/scripts/evaluate.py --model "hf:Qwen/Qwen2.5-Coder-0.5B-Instruct::adapters/infinity-hyper-5-5/qwen-coder-0.5b-combined" --label combined` |
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

## 3. Use it in the bot

1. `configs/open-models.json` lists two entries: the plain base model, and the base +
   `adapters/infinity-hyper-5-5/qwen-coder-0.5b-combined`.
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

## 4. What was actually tested (in the build environment: Linux, CPU, no internet to huggingface.co)

| Command / feature | Tested? | How |
|---|---|---|
| `prepare_data.py`, `train.py --dry-run`, `--smoke`, full run, `--resume` | ✅ | With the tiny random test model (`training/configs/tiny-test.yaml`) and `tests/test_hf.py` |
| Adapter export, reload equivalence, never-overwrite, OOM handling | ✅ | `tests/test_hf.py` |
| `evaluate.py` (open model, open model + adapter, HYPER 5.5) and `--compare` | ✅ | Tiny model + HYPER 5.5 tiny checkpoint |
| Evaluation sandbox (timeout, blocked network/process/outside writes) | ✅ | `tests/test_hf.py` (Linux) |
| Bot: model selector, switching, fallback, Normal chat, labels | ✅ | `tests/test_hf.py`, vitest, and a Chromium run against the tiny model |
| **Downloading and running the real Qwen2.5-Coder 0.5B** | ❌ | huggingface.co is blocked here. Your PC will be the first real run |
| **Real LoRA training and its speed on your PC** | ❌ | not run. The script prints the measured speed |
| Windows paths of these scripts | ❌ | written to be portable (pathlib, UTF-8), not run on Windows |

Offline pipeline check you can run yourself, with no download:

```bat
python training/scripts/make_tiny_model.py
python training/scripts/prepare_data.py --config training/configs/tiny-test.yaml
python training/scripts/train.py --config training/configs/tiny-test.yaml --smoke
```
