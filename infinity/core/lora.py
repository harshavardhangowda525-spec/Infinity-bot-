"""LoRA adapters for the original INFINITY HYPER 5.5 model (``InfinityNeuralCore``).

The base weights are frozen and never modified, saved or merged. An adapter is a separate
folder (``adapters/infinity-hyper-5-5/<name>/``) holding only the low-rank matrices plus a
config that pins the exact base checkpoint by sha256. At inference the adapter is applied
on top of the untouched base. The result is called **INFINITY HYPER 5.5 — Fine-Tuned**,
while the base keeps the name **INFINITY HYPER 5.5**.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
import torch.nn as nn

from .checkpoint import load_checkpoint, sha256_file, verify_checkpoint

FORMAT = "infinity-lora-v1"
ORIGINAL_NAME = "INFINITY HYPER 5.5"
FINETUNED_NAME = "INFINITY HYPER 5.5 — Fine-Tuned"
TARGETS = ("qkv", "proj", "gate", "up", "down")  # attention and MLP projections; not the tied embedding/head


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: tuple[str, ...] = field(default_factory=lambda: TARGETS)


class LoRALinear(nn.Module):
    """y = base(x) + (dropout(x) Aᵀ Bᵀ) · alpha/r, with ``base`` frozen. B starts at zero, so a fresh
    adapter leaves the model's outputs exactly unchanged."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.drop(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scale


def apply_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Freeze every base parameter, then wrap the target linear layers. Returns the wrapped module names."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in cfg.targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, cfg.r, cfg.alpha, cfg.dropout))
                wrapped.append(f"{name}.{child_name}" if name else child_name)
    if not wrapped:
        raise ValueError(f"no layers named {cfg.targets} found")
    return wrapped


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if ".lora_" in k}


def base_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if ".lora_" not in k}


def base_fingerprint(checkpoint_dir: Path) -> dict[str, str]:
    """sha256 of every file in the base checkpoint folder (used to prove it is unchanged)."""
    return {p.name: sha256_file(p) for p in sorted(Path(checkpoint_dir).iterdir()) if p.is_file()}


def _rel(path: Path, root: Path | None) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())) if root else str(path)
    except ValueError:
        return str(path)


def save_adapter(out: Path, model: nn.Module, cfg: LoRAConfig, base_path: Path, base_manifest: dict,
                 context_length: int, extra: dict | None = None, root: Path | None = None) -> Path:
    """Atomically write an adapter folder. An existing folder is never overwritten (a timestamped name is used)."""
    out = Path(out)
    if out.exists():
        out = out.with_name(f"{out.name}-{time.strftime('%Y%m%d-%H%M%S')}")
    tmp = out.with_name(f".tmp-{out.name}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    torch.save(adapter_state(model), tmp / "adapter.pt")
    conf = {
        "format": FORMAT, "name": FINETUNED_NAME, "base_model": ORIGINAL_NAME,
        "base_checkpoint": _rel(Path(base_path), root), "base_step": base_manifest["step"],
        "base_model_sha256": base_manifest["sha256"]["model.pt"],
        "base_tokenizer_fingerprint": base_manifest["tokenizer_fingerprint"],
        "context_length": context_length, "lora": {**asdict(cfg), "targets": list(cfg.targets)},
        "adapter_sha256": sha256_file(tmp / "adapter.pt"), "created_unix": int(time.time()),
        "merged_into_base": False, **(extra or {}),
    }
    (tmp / "adapter_config.json").write_text(json.dumps(conf, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    return out


class AdapterError(ValueError):
    pass


def read_adapter_config(adapter_dir: Path) -> dict:
    p = Path(adapter_dir) / "adapter_config.json"
    if not p.exists():
        raise AdapterError(f"{adapter_dir} has no adapter_config.json")
    conf = json.loads(p.read_text(encoding="utf-8"))
    if conf.get("format") != FORMAT:
        raise AdapterError(f"{adapter_dir} is not an INFINITY HYPER 5.5 adapter (format {conf.get('format')!r})")
    return conf


def load_with_adapter(base: str | Path, adapter_dir: str | Path, device: str = "cpu"):
    """Original checkpoint (sha256-verified) + adapter on top, never merged.

    Returns the same tuple as ``load_checkpoint`` with the adapted model and an updated run config.
    Refuses an adapter trained on a different base checkpoint.
    """
    from .model import InfinityNeuralCore

    adapter_dir = Path(adapter_dir)
    conf = read_adapter_config(adapter_dir)
    model, tok, run_cfg, trainer, manifest, path = load_checkpoint(base, device="cpu")
    if manifest["sha256"]["model.pt"] != conf["base_model_sha256"]:
        raise AdapterError(f"adapter {adapter_dir.name} was trained on a different INFINITY HYPER 5.5 checkpoint "
                           f"(sha256 {conf['base_model_sha256'][:12]}…, this one is {manifest['sha256']['model.pt'][:12]}…)")
    if sha256_file(adapter_dir / "adapter.pt") != conf["adapter_sha256"]:
        raise AdapterError(f"adapter {adapter_dir.name} is corrupted (sha256 mismatch)")
    ctx = int(conf["context_length"])
    if ctx != run_cfg.model.context_length:  # rotary tables are recomputed; weights are identical
        mcfg = replace(run_cfg.model, context_length=ctx)
        resized = InfinityNeuralCore(mcfg)
        resized.load_state_dict(model.state_dict())
        model = resized
        run_cfg = replace(run_cfg, model=mcfg)
    lc = conf["lora"]
    apply_lora(model, LoRAConfig(r=lc["r"], alpha=lc["alpha"], dropout=0.0, targets=tuple(lc["targets"])))
    state = torch.load(adapter_dir / "adapter.pt", map_location="cpu", weights_only=True)
    missing = set(adapter_state(model)) ^ set(state)
    if missing:
        raise AdapterError(f"adapter keys do not match the model: {sorted(missing)[:4]}")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, tok, run_cfg, trainer, manifest, path, conf


def verify_unchanged(checkpoint_dir: Path, before: dict[str, str]) -> None:
    """Raise if any base checkpoint file changed (also re-checks the manifest hashes)."""
    after = base_fingerprint(checkpoint_dir)
    if after != before:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise RuntimeError(f"ORIGINAL MODEL FILES CHANGED during training: {changed}")
    verify_checkpoint(checkpoint_dir)
