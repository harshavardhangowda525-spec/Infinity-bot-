# Troubleshooting

**`No module named torch`.** Install PyTorch: `pip install torch` (or `uv pip install torch`).
On Linux, the PyPI wheel bundles CUDA libraries (several GB). If
download.pytorch.org is reachable, the CPU-only wheel is much smaller:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

**`INFINITY HYPER 5.5 is not available: could not load checkpoint`.** Train first
(`infinity train`), or point `--model neural:` at a run directory or a
`step_XXXXXX` directory. Checkpoints that fail sha256 verification are
refused on purpose. Use an earlier step or retrain.

**Agent reports `blocked: model produced N consecutive invalid actions`.** This is
expected with the tiny prototype checkpoint. It has not learned the JSON
action protocol. The runtime refuses to guess what the model meant.
Instruction tuning on tool-use examples (Phase 6) is what fixes this.

**`context window too small` note.** The tiny model has a 256-token context.
The system prompt alone is longer, so it gets truncated. Larger configs
(`configs/hyper-5.5-small.json`, 1024 tokens) fix this, but they need more compute.

**`permission denied: not permitted in read-only mode`.** The workspace isn't
trusted. Pass `--permission ask`, or run `infinity memory trust <path> ask`.

**`read X before modifying it` / `stale edit`.** The agent must read a file in
the same session, and the file must be unchanged since, before editing it. This is by design.

**git errors about `GIT_CONFIG_KEY_0`.** These come from older builds. Current builds drop all
`GIT_CONFIG_*` variables together from child environments.

**Tests flaky after quick edits (stale `.pyc`).** Agent commands run with
`PYTHONDONTWRITEBYTECODE=1`. If you see stale imports in your own shell, delete `__pycache__/`.

**Training is slow.** On CPU, throughput is measured and logged. Reduce
`d_model`, `n_layers` or `context_length`, or train on a CUDA machine (`device: auto` picks it up).
`precision: bf16` helps on GPUs that support it.

**Undo agent changes.** Run `infinity rollback --workspace <path>` to list sessions, then
`infinity rollback --workspace <path> --session <id>`.

**Windows.** Commands run without a shell when possible. Shell syntax (`&&`, `|`, `>`)
and cmd built-ins (`dir`, `type`, `del`…) go through `cmd /d /s /c`. Timeouts and Stop
use `taskkill /T` to end the whole process tree. `python` / `python3` fall back to the
interpreter running INFINITY when they are missing or are only the Microsoft Store
placeholder (Windows virtual environments have no `python3`). These code paths are
unit-tested on Linux by forcing the Windows branch (`tests/test_windows.py`). They have
not yet been run on a real Windows machine in CI.

**`infinity sft generate` stops with "no training data would be produced".** The first
expert runs all failed, which means the task tests could not run on this machine. Check
that `python3 -m unittest -q` (or `python -m unittest -q` on Windows) works in a terminal.

## Open models (Qwen + LoRA)

This file is also the `troubleshooting.md` the training plan asked for. A second file
differing only in letter case would clash on Windows.

**"open models need transformers + peft".** Run `pip install -r training/requirements-training.txt` in the INFINITY venv.

**The model switch fails with a download or connection error.** The first load downloads the
base model from huggingface.co (about 1 GB). Check your internet connection. To work
offline, download the model folder once and set `"base"` in `configs/open-models.json`
to that folder (relative to the project root). The current model stays loaded when a switch fails.

**The adapter entry is greyed out.** Its folder under `adapters/infinity-hyper-5-5/` doesn't
exist yet. Train it first, or point `"adapter"` at the folder you trained.

**Training is slow.** On a CPU each step can take tens of seconds; the script prints the
measured speed and an ETA. Lower `training.max_steps`, `data.tool_tasks_train` or
`data.max_seq_len` in the YAML config. Ctrl+C saves a checkpoint; `--resume` continues.

**"out of memory".** Nothing is deleted. Lower `data.max_seq_len` (for example 1024) or keep
`batch_size: 1`, close other programs, then rerun with `--resume`.

**"chat template is not prefix-stable".** The base model's chat template rewrites earlier
turns (some reasoning models do this), so assistant tokens cannot be located safely. Use a
model whose template only appends (Qwen2.5 does).

**The evaluate script refuses to run: "already exists".** Saved results are never overwritten. Choose a new `--label`.
