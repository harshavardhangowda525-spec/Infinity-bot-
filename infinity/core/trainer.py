"""INFINITY TRAINER — next-token pretraining loop.

Features: AdamW with decoupled weight decay (no decay on norms/embeddings
biases), linear warmup + cosine decay, gradient accumulation, gradient
clipping, optional bf16 autocast, periodic validation, JSONL logs,
atomic checkpoints and exact resume (model, optimizer, RNG, step).

Every number written to the log is measured during the run; nothing is
estimated or synthesised.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from .checkpoint import list_checkpoints, load_checkpoint, resolve_checkpoint, save_checkpoint
from .config import RunConfig
from .data import TokenDataset
from .model import InfinityNeuralCore
from .tokenizer import BPETokenizer


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    return min_lr + 0.5 * (cfg.learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


def build_optimizer(model: InfinityNeuralCore, cfg, device: str) -> torch.optim.AdamW:
    decay, no_decay = [], []
    seen = set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=device == "cuda")


@torch.no_grad()
def evaluate(model, dataset: TokenDataset, batches: int, batch_size: int, device: str, seed: int, autocast) -> float:
    model.eval()
    g = torch.Generator().manual_seed(seed)  # fixed windows -> comparable across steps
    total = 0.0
    for _ in range(batches):
        x, y = dataset.batch(batch_size, g)
        with autocast():
            _, loss = model(x.to(device), y.to(device))
        total += loss.item()
    model.train()
    return total / batches


class Trainer:
    def __init__(self, run_cfg: RunConfig, data_dir: str | Path, out_dir: str | Path | None = None, log=print):
        self.cfg = run_cfg
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir or run_cfg.output_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.log = log
        prepared = json.loads((self.data_dir / "prepared.json").read_text())
        self.tokenizer = BPETokenizer.load(self.data_dir / "tokenizer.json")
        if self.tokenizer.fingerprint() != prepared["tokenizer"]["fingerprint"]:
            raise RuntimeError("prepared data was tokenized with a different tokenizer")
        if run_cfg.model.vocab_size != self.tokenizer.vocab_size:
            self.log(f"model.vocab_size {run_cfg.model.vocab_size} -> {self.tokenizer.vocab_size} (from tokenizer)")
            run_cfg.model.vocab_size = self.tokenizer.vocab_size
        ctx = run_cfg.model.context_length
        self.train_ds = TokenDataset(self.data_dir / "train.bin", prepared["splits"]["train"]["dtype"], ctx)
        self.val_ds = TokenDataset(self.data_dir / "val.bin", prepared["splits"]["val"]["dtype"], ctx)
        self.prepared = prepared

    def train(self, resume: bool = True, max_steps: int | None = None, stop_flag=None) -> dict:
        tc = self.cfg.train
        if max_steps is not None:
            tc.max_steps = max_steps
        if tc.num_threads:
            torch.set_num_threads(tc.num_threads)
        device = pick_device(tc.device)
        torch.manual_seed(tc.seed)

        model = InfinityNeuralCore(self.cfg.model).to(device)
        optimizer = build_optimizer(model, tc, device)
        gen = torch.Generator().manual_seed(tc.seed)
        start_step = 0

        if resume and list_checkpoints(self.ckpt_dir):
            path = resolve_checkpoint(self.ckpt_dir)
            m, _, _, tstate, manifest, path = load_checkpoint(path, device=device)
            model.load_state_dict(m.state_dict())
            if tstate.get("optimizer"):
                optimizer.load_state_dict(tstate["optimizer"])
            if "data_gen" in tstate.get("rng", {}):
                gen.set_state(tstate["rng"]["data_gen"])
            if "torch" in tstate.get("rng", {}):
                torch.set_rng_state(tstate["rng"]["torch"])
            start_step = int(tstate["step"])
            self.log(f"resumed from {path} (step {start_step})")

        use_bf16 = tc.precision == "bf16"
        autocast = (lambda: torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)) if use_bf16 else nullcontext

        self.out_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.out_dir / "train_log.jsonl"
        (self.out_dir / "run_config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2))
        n_params = model.num_parameters()
        self.log(
            f"INFINITY HYPER 5.5 [{self.cfg.name}] params={n_params:,} device={device} "
            f"train_tokens={len(self.train_ds):,} val_tokens={len(self.val_ds):,}"
        )

        model.train()
        last = {}
        best_val = None
        t0 = time.time()
        tokens_per_step = tc.batch_size * tc.grad_accum_steps * self.cfg.model.context_length
        step = start_step
        interrupted = False
        while step < tc.max_steps:
            if stop_flag is not None and stop_flag.is_set():
                interrupted = True
                break
            lr = lr_at(step, tc)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _ in range(tc.grad_accum_steps):
                x, y = self.train_ds.batch(tc.batch_size, gen)
                with autocast():
                    _, loss = model(x.to(device), y.to(device))
                (loss / tc.grad_accum_steps).backward()
                loss_acc += loss.item() / tc.grad_accum_steps
            if not math.isfinite(loss_acc):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss_acc}")
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
            optimizer.step()
            step += 1
            dt = time.time() - t0
            rec = {"step": step, "loss": round(loss_acc, 5), "lr": lr, "grad_norm": round(gnorm, 4),
                   "elapsed_s": round(dt, 2), "tokens_seen": (step - start_step) * tokens_per_step}
            if step % tc.eval_interval == 0 or step == tc.max_steps:
                rec["val_loss"] = round(evaluate(model, self.val_ds, tc.eval_batches, tc.batch_size, device, tc.seed + 1, autocast), 5)
                best_val = rec["val_loss"] if best_val is None else min(best_val, rec["val_loss"])
                self.log(f"step {step:>6} loss {loss_acc:.4f} val {rec['val_loss']:.4f} lr {lr:.2e} {dt:.0f}s")
            elif step % 10 == 0:
                self.log(f"step {step:>6} loss {loss_acc:.4f} lr {lr:.2e} {dt:.0f}s")
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            last = rec
            if step % tc.checkpoint_interval == 0 or step == tc.max_steps:
                self._save(step, model, optimizer, gen, last)

        if interrupted and step > start_step and not (self.ckpt_dir / f"step_{step:06d}").exists():
            self._save(step, model, optimizer, gen, last)
        return {"final_step": step, "start_step": start_step, "last": last, "best_val_loss": best_val,
                "parameters": n_params, "device": device, "interrupted": interrupted,
                "wall_time_s": round(time.time() - t0, 2)}

    def _save(self, step, model, optimizer, gen, metrics):
        rng = {"data_gen": gen.get_state(), "torch": torch.get_rng_state()}
        path = save_checkpoint(self.ckpt_dir, step, model, optimizer, self.cfg, self.tokenizer, metrics, rng)
        self.log(f"checkpoint saved: {path}")
