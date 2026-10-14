"""LoRA fine-tuning of the ORIGINAL INFINITY HYPER 5.5 checkpoint → "INFINITY HYPER 5.5 — Fine-Tuned".

Same data, masked loss and LM rehearsal as ``finetune`` (full fine-tuning), but:
- every base weight is frozen and only low-rank adapter matrices train;
- output goes to ``adapters/infinity-hyper-5-5/<name>/`` (and intermediate adapters to
  ``training/checkpoints/<name>/step_*``), never into the base checkpoint;
- the base checkpoint's files are sha256-fingerprinted before and after training, and the
  run fails loudly if anything changed.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch

from ..core.checkpoint import load_checkpoint, sha256_file
from ..core.config import TrainConfig
from ..core.lora import (
    FINETUNED_NAME, ORIGINAL_NAME, LoRAConfig, apply_lora, base_fingerprint, save_adapter, verify_unchanged,
)
from ..core.model import InfinityNeuralCore
from ..core.trainer import lr_at, pick_device
from .finetune import SFTData, eval_loss


def lora_finetune(base: str, data_dir: str | Path, name: str, root: str | Path = ".", steps: int = 600,
                  context: int = 512, batch_size: int = 16, lr: float = 2e-3, warmup: int = 30, seed: int = 1337,
                  eval_every: int = 200, r: int = 8, alpha: float = 16.0, dropout: float = 0.0,
                  lm_mix: float = 0.0, lm_data: str | Path | None = None, log=print, stop_flag=None) -> dict:
    from dataclasses import replace

    root, data_dir = Path(root), Path(data_dir)
    adapter_dir = root / "adapters" / "infinity-hyper-5-5" / name
    ckpt_dir = root / "training" / "checkpoints" / name
    log_path = root / "training" / "logs" / f"{name}.jsonl"
    device = pick_device("auto")
    torch.manual_seed(seed)

    base_model, tok, base_cfg, _, base_manifest, base_path = load_checkpoint(base, device="cpu")
    before = base_fingerprint(base_path)
    mcfg = replace(base_cfg.model, context_length=context)
    model = InfinityNeuralCore(mcfg)
    model.load_state_dict(base_model.state_dict())
    del base_model
    lcfg = LoRAConfig(r=r, alpha=alpha, dropout=dropout)
    wrapped = apply_lora(model, lcfg)
    model.to(device)
    frozen_copy = {k: v.detach().clone() for k, v in model.state_dict().items() if ".lora_" not in k}
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())

    train = SFTData(data_dir / "train.jsonl", tok, context)
    evald = SFTData(data_dir / "eval.jsonl", tok, context)
    tc = TrainConfig(batch_size=batch_size, max_steps=steps, learning_rate=lr, warmup_steps=warmup, min_lr_ratio=0.1,
                     weight_decay=0.0, eval_interval=eval_every, checkpoint_interval=steps, seed=seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    lm_ds = None
    if lm_mix > 0:
        from ..core.data import TokenDataset

        prepared = json.loads((Path(lm_data) / "prepared.json").read_text())
        if prepared["tokenizer"]["fingerprint"] != tok.fingerprint():
            raise ValueError("pretraining data uses a different tokenizer")
        lm_ds = TokenDataset(Path(lm_data) / "train.bin", prepared["splits"]["train"]["dtype"], min(256, context))
        lm_gen = torch.Generator().manual_seed(seed)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = {"event": "start", "base_model": ORIGINAL_NAME, "base_role": "original, read-only (frozen)",
             "base_checkpoint": str(base_path), "base_model_sha256": before.get("model.pt"),
             "output": FINETUNED_NAME, "adapter_dir": str(adapter_dir), "method": "LoRA (not merged)",
             "lora": {"r": r, "alpha": alpha, "targets": list(lcfg.targets), "layers": len(wrapped)},
             "trainable_params": n_train, "total_params": n_total, "examples": len(train.items),
             "dropped_examples": train.dropped, "context_length": context, "device": device}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(start) + "\n")
    log(f"{FINETUNED_NAME}: LoRA on frozen {ORIGINAL_NAME} ({base_path.name}) · trainable {n_train:,} of {n_total:,} "
        f"· {len(train.items)} examples · device {device}")
    lineage = {"train_data_sha256": sha256_file(data_dir / "train.jsonl"), "train_examples": len(train.items),
               "method": "LoRA supervised fine-tuning (masked loss), base frozen", "lm_mix": lm_mix}
    eval_before = eval_loss(model, evald, batch_size, device)
    log(f"eval loss before (= original model): {eval_before:.4f}")

    rng = random.Random(seed)
    order: list[int] = []
    t0 = time.time()
    model.train()
    last: dict = {}
    for step in range(steps):
        if stop_flag is not None and stop_flag.is_set():
            break
        if not order:
            order = list(range(len(train.items)))
            rng.shuffle(order)
        x, y = train.batch([order.pop() for _ in range(min(batch_size, len(order)))])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        _, loss = model(x.to(device), y.to(device))
        if lm_ds is not None and rng.random() < lm_mix:
            lx, ly = lm_ds.batch(batch_size, lm_gen)
            loss = loss + model(lx.to(device), ly.to(device))[1]
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        rec = {"step": step + 1, "loss": round(loss.item(), 5), "lr": lr_at(step, tc), "elapsed_s": round(time.time() - t0, 2)}
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            rec["eval_loss"] = round(eval_loss(model, evald, batch_size, device), 5)
            log(f"step {step + 1:>5} loss {rec['loss']:.4f} eval {rec['eval_loss']:.4f} {rec['elapsed_s']:.0f}s")
            if step + 1 < steps:
                save_adapter(ckpt_dir / f"step_{step + 1:06d}", model, lcfg, base_path, base_manifest, context,
                             {"step": step + 1, "metrics": rec}, root=root)
        elif (step + 1) % 25 == 0:
            log(f"step {step + 1:>5} loss {rec['loss']:.4f} {rec['elapsed_s']:.0f}s")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        last = rec

    # The frozen base must be bit-identical in memory and on disk.
    now = {k: v for k, v in model.state_dict().items() if ".lora_" not in k}
    if any(not torch.equal(frozen_copy[k], now[k]) for k in frozen_copy):
        raise RuntimeError("base weights changed in memory during LoRA training")
    verify_unchanged(base_path, before)
    eval_after = eval_loss(model, evald, batch_size, device)
    out = save_adapter(adapter_dir, model.cpu(), lcfg, base_path, base_manifest, context, {
        "trained_steps": last.get("step", 0), "eval_loss_before": round(eval_before, 5),
        "eval_loss_after": round(eval_after, 5), "lineage": lineage, "base_files_sha256": before,
        "base_unchanged_after_training": True}, root=root)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "end", "adapter": str(out), "base_unchanged_after_training": True,
                            "eval_loss_before": round(eval_before, 5), "eval_loss_after": round(eval_after, 5)}) + "\n")
    log(f"original model verified unchanged ({len(before)} files, sha256) · adapter saved: {out}")
    return {"adapter": str(out), "name": FINETUNED_NAME, "base_checkpoint": str(base_path),
            "eval_loss_before": eval_before, "eval_loss_after": eval_after, "steps": last.get("step", 0),
            "base_unchanged": True, "wall_time_s": round(time.time() - t0, 1)}
