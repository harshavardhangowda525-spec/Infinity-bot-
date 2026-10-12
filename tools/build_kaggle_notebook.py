"""Build notebooks/kaggle_train_100m.ipynb (run: python tools/build_kaggle_notebook.py).

Every code cell is plain Python, so the same cells can be smoke-tested locally with
INFINITY_NOTEBOOK_SMOKE=1 (tiny model, local corpus, no downloads).
"""

import json
from pathlib import Path

CELLS = [
    ("md", """# INFINITY HYPER 5.5 — train the 100M model on Kaggle (free GPU)

This notebook trains the **`hyper-5.5-small`** configuration (97.5M parameters) **from scratch** with the
INFINITY BOT code, on permissively licensed source code. No pretrained weights are used.

**Before you run**
1. Kaggle → *Settings* (right panel) → **Accelerator: GPU T4 x2** (or P100) and **Internet: On**.
2. Click **Run All**. One session trains for up to ~11 hours, then saves a checkpoint and stops cleanly.

**Continuing in the next session (the full run needs about 2 sessions on T4 x2):**
*Save Version* (Quick Save is fine). In a new session, choose *Add Input* → *Your Work* → this notebook's
output, then *Run All* again. Cell 3 restores the previous run and training resumes where it stopped.

**At the end**, download `infinity-hyper-5.5-small.tar.gz` from the *Output* tab, extract it inside your
INFINITY BOT folder (it creates `runs/hyper-5.5-small/` and `runs/data-small/`), then run:
```
infinity bench run --checkpoint runs/hyper-5.5-small --suites lm --data runs/data-small
infinity model promote --checkpoint runs/hyper-5.5-small --reason "trained on Kaggle"
```
Licences: rows are kept only if their per-file licence is MIT, Apache-2.0, BSD, ISC or CC0. A provenance file
is written next to the data. Review the dataset's card and terms before using the model commercially."""),
    ("code", """# 1. Settings
import json, os, shutil, subprocess, sys
from pathlib import Path

SMOKE = os.environ.get("INFINITY_NOTEBOOK_SMOKE") == "1"   # local test mode: tiny model, no downloads
REPO_URL = "https://github.com/harshavardhangowda525-spec/Infinity-bot-.git"
BRANCH = "claude/infinity-bot-platform-lgt431"
WORK = Path(os.environ.get("INFINITY_WORK", "/kaggle/working"))
REPO = Path(os.environ["INFINITY_REPO"]) if SMOKE else WORK / "Infinity-bot-"
DATA_MB = 3 if SMOKE else 1500           # licensed code to download (MB of text)
RUN = WORK / "runs" / "hyper-5.5-small"
DATA = WORK / "runs" / "data-small"

def sh(*cmd, cwd=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)

print("smoke test" if SMOKE else "Kaggle run", "| work dir:", WORK)"""),
    ("code", """# 2. Get the INFINITY BOT code and check the GPU
if not SMOKE:
    if not REPO.exists():
        sh("git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO)
    sh(sys.executable, "-m", "pip", "install", "-q", "-e", REPO)
import torch
GPUS = torch.cuda.device_count()
print("GPUs:", GPUS, [torch.cuda.get_device_name(i) for i in range(GPUS)])
if not SMOKE and GPUS == 0:
    raise SystemExit("No GPU: Settings -> Accelerator -> GPU T4 x2")
BF16 = GPUS > 0 and torch.cuda.is_bf16_supported()
PRECISION = "fp32" if GPUS == 0 else ("bf16" if BF16 else "fp16")   # T4 / P100 -> fp16 with loss scaling
print("precision:", PRECISION)"""),
    ("code", """# 3. Restore a previous session's run (if you attached this notebook's earlier output as input)
for inp in Path("/kaggle/input").glob("*/runs") if Path("/kaggle/input").exists() else []:
    for name in ("hyper-5.5-small", "data-small"):
        src, dst = inp / name, WORK / "runs" / name
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
            print("restored", dst)
print("existing checkpoints:", sorted(p.name for p in (RUN / "checkpoints").glob("step_*")) if RUN.exists() else "none")"""),
    ("code", """# 4. Download permissively licensed code (skipped if data is already prepared)
HF_DIR = WORK / "hf" / "github-code"
if not (DATA / "prepared.json").exists() and not SMOKE:
    sh(sys.executable, "-m", "infinity", "data", "fetch-hf", "--dataset", "codeparrot/github-code-clean",
       "--out", HF_DIR, "--max-mb", DATA_MB, "--languages", "Python,JavaScript,TypeScript,HTML,CSS",
       "--licenses", "mit,apache-2.0,bsd-3-clause,bsd-2-clause,isc,cc0-1.0", cwd=REPO)
    print(open(HF_DIR / "PROVENANCE.json").read())"""),
    ("code", """# 5. Manifest: project text + Python standard library + downloaded licensed code
manifest = {"sources": [
    {"name": "infinity-original", "license": "Project-Original", "include": [str(REPO / "data/corpus/**/*.md")]},
    {"name": "python-stdlib", "license": "PSF-2.0", "include": ["{stdlib}/*.py"]},
]}
if HF_DIR.exists():
    manifest["sources"].append({"name": "github-code-permissive", "type": "jsonl", "include": [str(HF_DIR / "*.jsonl")],
                                "text_field": "text", "license_field": "license",
                                "origin": "codeparrot/github-code-clean, per-file licence filtered"})
(WORK / "data").mkdir(parents=True, exist_ok=True)
(WORK / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))"""),
    ("code", """# 6. Clean, deduplicate, split, train the tokenizer and tokenize in parallel (skipped if already done)
if not (DATA / "prepared.json").exists():
    sh(sys.executable, "-m", "infinity", "data", "prepare-large", "--manifest", WORK / "data" / "manifest.json",
       "--out", DATA, "--vocab", 512 if SMOKE else 16384, "--sample-mb", 1 if SMOKE else 50, "--shard-mb", 1 if SMOKE else 64,
       "--workers", 2 if SMOKE else max(1, (os.cpu_count() or 2) - 1), cwd=REPO)
prepared = json.loads((DATA / "prepared.json").read_text())
print({k: v["tokens"] for k, v in prepared["splits"].items()}, prepared["cleaning"])"""),
    ("code", """# 7. Training config (from configs/hyper-5.5-small.json, adapted to this machine)
cfg = json.loads((REPO / "configs" / "hyper-5.5-small.json").read_text())
cfg["output_dir"] = str(RUN)
cfg["model"]["grad_checkpoint"] = True
cfg["train"].update({"precision": PRECISION, "time_budget_hours": 0.02 if SMOKE else 11.0,
                     "batch_size": 8, "grad_accum_steps": 16 // max(1, GPUS), "max_steps": 15000,
                     "eval_interval": 500, "checkpoint_interval": 500, "strategy": "ddp" if GPUS > 1 else "single"})
if SMOKE:  # tiny model so the whole notebook runs on a CPU in seconds
    cfg["model"].update({"d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 128, "context_length": 64})
    cfg["train"].update({"batch_size": 2, "grad_accum_steps": 1, "max_steps": 4, "eval_interval": 2,
                         "eval_batches": 2, "checkpoint_interval": 2, "warmup_steps": 1})
CFG = WORK / "hyper-5.5-small.kaggle.json"
CFG.write_text(json.dumps(cfg, indent=2))
tokens_per_step = cfg["train"]["batch_size"] * cfg["train"]["grad_accum_steps"] * max(1, GPUS) * cfg["model"]["context_length"]
print(f"{tokens_per_step:,} tokens/step x {cfg['train']['max_steps']:,} steps = {tokens_per_step * cfg['train']['max_steps']:,} tokens")"""),
    ("code", """# 8. Train (stops cleanly at the time budget; run the notebook again in a new session to continue)
if GPUS > 1:
    sh(sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", GPUS, "-m", "infinity.core.dist_train",
       "--config", CFG, "--data", DATA, "--out", RUN, "--strategy", "ddp", cwd=REPO)
else:
    sh(sys.executable, "-m", "infinity", "train", "--config", CFG, "--data", DATA, "--out", RUN, cwd=REPO)"""),
    ("code", """# 9. Measure (held-out test split) and look at a sample
sh(sys.executable, "-m", "infinity", "bench", "run", "--checkpoint", RUN, "--suites", "lm", "--data", DATA,
   "--store", WORK / "runs" / "benchmarks.db", cwd=REPO)
sh(sys.executable, "-m", "infinity", "generate", "--checkpoint", RUN, "--prompt", "def read_json(path):",
   "--max-tokens", 80, "--seed", 1, cwd=REPO)"""),
    ("code", """# 10. Package the latest checkpoint + tokenizer for download (Output tab)
import tarfile
ckpts = sorted((RUN / "checkpoints").glob("step_*"))
latest = ckpts[-1]
out = WORK / "infinity-hyper-5.5-small.tar.gz"
with tarfile.open(out, "w:gz") as tar:
    tar.add(latest, arcname=f"runs/hyper-5.5-small/checkpoints/{latest.name}")
    (WORK / "LATEST").write_text(latest.name)
    tar.add(WORK / "LATEST", arcname="runs/hyper-5.5-small/checkpoints/LATEST")
    for f in ("tokenizer.json", "prepared.json"):
        tar.add(DATA / f, arcname=f"runs/data-small/{f}")
    for f in sorted(DATA.glob("test_*.bin")):
        tar.add(f, arcname=f"runs/data-small/{f.name}")
print("download:", out, f"({out.stat().st_size / 1e6:.0f} MB)", "| checkpoint:", latest.name)"""),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines})
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}, "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_train_100m.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=1) + "\n")
    print("wrote", out)
