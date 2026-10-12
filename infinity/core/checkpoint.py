"""Checkpoint management with integrity verification.

Layout of one checkpoint directory::

    step_000600/
      model.pt        weights only (state_dict), loaded with weights_only=True
      trainer.pt      optimizer + RNG + step, for exact resume
      config.json     RunConfig used to build the model
      tokenizer.json  the tokenizer this checkpoint was trained with
      manifest.json   sha256 of every file above + metrics + identity

Checkpoints are written to a temporary directory and atomically renamed, so
an interrupted save never leaves a half-written "valid" checkpoint. A
``LATEST`` file in the checkpoints folder names the newest complete one.
Existing checkpoints are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch

from .config import MODEL_FAMILY, MODEL_VERSION, RunConfig
from .model import InfinityNeuralCore
from .tokenizer import BPETokenizer

CHECKPOINT_FILES = ("model.pt", "trainer.pt", "config.json", "tokenizer.json")


class CheckpointError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(
    ckpt_root: Path,
    step: int,
    model: InfinityNeuralCore,
    optimizer: torch.optim.Optimizer | None,
    run_cfg: RunConfig,
    tokenizer: BPETokenizer,
    metrics: dict,
    rng_state: dict | None = None,
) -> Path:
    ckpt_root = Path(ckpt_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    final = ckpt_root / f"step_{step:06d}"
    if final.exists():
        raise CheckpointError(f"refusing to overwrite existing checkpoint {final}")
    tmp = ckpt_root / f".tmp_step_{step:06d}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    torch.save(model.state_dict(), tmp / "model.pt")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "rng": rng_state or {},
        },
        tmp / "trainer.pt",
    )
    (tmp / "config.json").write_text(json.dumps(run_cfg.to_dict(), indent=2))
    tokenizer.save(tmp / "tokenizer.json")
    manifest = {
        "identity": {"family": MODEL_FAMILY, "version": MODEL_VERSION, "run": run_cfg.name},
        "step": step,
        "created_unix": int(time.time()),
        "parameters": model.num_parameters(),
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "metrics": metrics,
        "sha256": {name: sha256_file(tmp / name) for name in CHECKPOINT_FILES},
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, final)
    latest_tmp = ckpt_root / ".LATEST.tmp"
    latest_tmp.write_text(final.name)
    os.replace(latest_tmp, ckpt_root / "LATEST")
    return final


def verify_checkpoint(path: Path) -> dict:
    path = Path(path)
    mpath = path / "manifest.json"
    if not mpath.exists():
        raise CheckpointError(f"{path} has no manifest.json (incomplete checkpoint?)")
    manifest = json.loads(mpath.read_text())
    for name, digest in manifest["sha256"].items():
        f = path / name
        if not f.exists():
            raise CheckpointError(f"{path}: missing {name}")
        if sha256_file(f) != digest:
            raise CheckpointError(f"{path}: integrity check failed for {name}")
    return manifest


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept a checkpoint dir, a checkpoints root (uses LATEST), or a run dir."""
    p = Path(path)
    if (p / "manifest.json").exists():
        return p
    for root in (p, p / "checkpoints"):
        latest = root / "LATEST"
        if latest.exists():
            return root / latest.read_text().strip()
    raise CheckpointError(f"no checkpoint found at {p}")


def list_checkpoints(ckpt_root: str | Path) -> list[dict]:
    root = Path(ckpt_root)
    out = []
    for d in sorted(root.glob("step_*")):
        m = d / "manifest.json"
        if m.exists():
            info = json.loads(m.read_text())
            out.append({"path": str(d), "step": info["step"], "metrics": info.get("metrics", {})})
    return out


def load_checkpoint(path: str | Path, device: str = "cpu", verify: bool = True):
    path = resolve_checkpoint(path)
    manifest = verify_checkpoint(path) if verify else json.loads((path / "manifest.json").read_text())
    run_cfg = RunConfig.load(path / "config.json")
    tokenizer = BPETokenizer.load(path / "tokenizer.json")
    if tokenizer.fingerprint() != manifest["tokenizer_fingerprint"]:
        raise CheckpointError("tokenizer does not match the checkpoint manifest")
    if tokenizer.vocab_size != run_cfg.model.vocab_size:
        raise CheckpointError("tokenizer vocab size does not match model config")
    model = InfinityNeuralCore(run_cfg.model)
    state = torch.load(path / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    trainer_state = torch.load(path / "trainer.pt", map_location=device, weights_only=True)
    return model, tokenizer, run_cfg, trainer_state, manifest, path
