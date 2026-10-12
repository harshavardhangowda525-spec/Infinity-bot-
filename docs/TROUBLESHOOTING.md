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
