"""Multi-GPU / multi-node pretraining for INFINITY HYPER 5.5 (launch with torchrun).

    torchrun --nproc_per_node 8 -m infinity.core.dist_train --config configs/hyper-5.5-1b.json --data runs/data-large
    # multi-node: add --nnodes N --node_rank R --rdzv_endpoint HOST:PORT

Strategies (TrainConfig.strategy):
- ``ddp``  : every GPU holds the full model; gradients are averaged. Good up to ~1B on 80 GB GPUs.
- ``fsdp`` : FSDP2 (``fully_shard``): parameters, gradients and optimizer state are sharded
             across GPUs, one transformer block per unit. Needed for 7B. Each rank first
             builds the full model on its device before sharding, so a 7B run needs
             enough memory per GPU for the bf16 weights during start-up.

Checkpoints are gathered to rank 0 in the standard INFINITY format (sha256-verified,
loadable by the inference engine), including the full optimizer state, so runs resume
exactly, even on a different number of GPUs. Every rank samples different data windows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .checkpoint import list_checkpoints, load_checkpoint, resolve_checkpoint, save_checkpoint_state
from .config import RunConfig
from .model import Block, InfinityNeuralCore
from .tokenizer import BPETokenizer
from .trainer import load_datasets, lr_at, make_autocast


def _setup():
    if "RANK" not in os.environ:  # plain `python -m`: behave as a 1-process group
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
    cuda = torch.cuda.is_available()
    dist.init_process_group(backend="nccl" if cuda else "gloo")
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    if cuda:
        torch.cuda.set_device(local)
    return rank, world, local, (f"cuda:{local}" if cuda else "cpu")


def _wrap(model: InfinityNeuralCore, strategy: str, precision: str, device: str, local: int, world: int):
    cuda = device.startswith("cuda")
    model.to(device)
    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        return DDP(model, device_ids=[local] if cuda else None)
    if strategy == "fsdp":
        # FSDP2: parameters (and therefore gradients and optimizer state) are DTensors sharded over the mesh.
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        if precision == "fp16":
            raise ValueError("use bf16 (or fp32) with FSDP; fp16 loss scaling is supported for single/ddp runs")
        mesh = init_device_mesh("cuda" if cuda else "cpu", (world,))
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if precision == "bf16" else MixedPrecisionPolicy()
        for block in model.blocks:  # one FSDP unit per transformer block
            fully_shard(block, mesh=mesh, mp_policy=mp)
        fully_shard(model, mesh=mesh, mp_policy=mp)
        return model
    raise ValueError(f"strategy must be ddp or fsdp, got {strategy!r}")


def _full_states(wrapped, optimizer):
    """Full (unsharded) model + optimizer state, keyed by parameter name, offloaded to CPU."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    msd, osd = get_state_dict(wrapped, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    msd = {k.removeprefix("module."): v for k, v in msd.items()}
    return msd, {"format": "fqn-v1", "state": osd}


def _load_optimizer(wrapped, optimizer, saved: dict, log) -> None:
    if not (isinstance(saved, dict) and saved.get("format") == "fqn-v1"):
        log("checkpoint optimizer state comes from single-process training; starting fresh optimizer moments")
        return
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict

    set_state_dict(wrapped, optimizer, model_state_dict={}, optim_state_dict=saved["state"],
                   options=StateDictOptions(full_state_dict=True, strict=False))


def _optimizer(wrapped, tc) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for _, p in wrapped.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": tc.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=tc.learning_rate, betas=(tc.beta1, tc.beta2))


def _mean(x: float, device: str) -> float:
    t = torch.tensor([x], device=device)
    dist.all_reduce(t)
    return t.item() / dist.get_world_size()


def train(config: str, data_dir: str, out_dir: str | None = None, max_steps: int | None = None, strategy: str | None = None) -> dict:
    rank, world, local, device = _setup()
    log = (lambda m: print(m, flush=True)) if rank == 0 else (lambda m: None)
    try:
        cfg = RunConfig.load(config)
        tc = cfg.train
        strategy = strategy or (tc.strategy if tc.strategy in ("ddp", "fsdp") else "ddp")
        if max_steps:
            tc.max_steps = max_steps
        data_dir = Path(data_dir)
        out = Path(out_dir or cfg.output_dir)
        ckpt_dir = out / "checkpoints"
        prepared = json.loads((data_dir / "prepared.json").read_text())
        tok = BPETokenizer.load(data_dir / "tokenizer.json")
        if tok.fingerprint() != prepared["tokenizer"]["fingerprint"]:
            raise RuntimeError("prepared data was tokenized with a different tokenizer")
        cfg.model.vocab_size = tok.vocab_size
        train_ds, val_ds = load_datasets(data_dir, prepared, cfg.model.context_length)

        torch.manual_seed(tc.seed)  # identical initialisation on every rank
        model = InfinityNeuralCore(cfg.model)
        start, opt_state, rng_state = 0, None, None
        if list_checkpoints(ckpt_dir):
            m, _, _, tstate, _, path = load_checkpoint(resolve_checkpoint(ckpt_dir), device="cpu")
            model.load_state_dict(m.state_dict())
            start, opt_state, rng_state = int(tstate["step"]), tstate.get("optimizer"), tstate.get("rng", {})
            log(f"resumed from {path} (step {start}) on {world} process(es)")
        n_params = model.num_parameters()
        wrapped = _wrap(model, strategy, tc.precision, device, local, world)
        opt = _optimizer(wrapped, tc)
        if opt_state:
            _load_optimizer(wrapped, opt, opt_state, log)
        gen = torch.Generator().manual_seed(tc.seed + 1000 * rank + start)
        autocast = make_autocast(tc.precision, device) if strategy == "ddp" else (lambda: torch.autocast("cpu", enabled=False))
        scaler = torch.amp.GradScaler("cuda", enabled=tc.precision == "fp16" and strategy == "ddp")
        deadline = time.time() + tc.time_budget_hours * 3600 if tc.time_budget_hours else None
        tokens_per_step = tc.batch_size * tc.grad_accum_steps * cfg.model.context_length * world
        log(f"INFINITY HYPER 5.5 [{cfg.name}] {n_params:,} params · {strategy} on {world} x {device.split(':')[0]} · "
            f"{tokens_per_step:,} tokens/step")
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            (out / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        t0, step, last = time.time(), start, {}

        def save(step_: int, metrics: dict) -> None:
            model_state, optim_state = _full_states(wrapped, opt)
            if rank == 0:
                path = save_checkpoint_state(ckpt_dir, step_, model_state, n_params, optim_state, cfg, tok,
                                             {**metrics, "world_size": world, "strategy": strategy})
                log(f"checkpoint saved: {path}")
            dist.barrier()

        wrapped.train()
        while step < tc.max_steps:
            stop = torch.tensor([1.0 if deadline and time.time() > deadline else 0.0], device=device)
            dist.all_reduce(stop)  # every rank stops together
            if stop.item() > 0:
                log("time budget reached: saving and stopping (rerun to resume)")
                break
            lr = lr_at(step, tc)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _ in range(tc.grad_accum_steps):
                x, y = train_ds.batch(tc.batch_size, gen)
                with autocast():
                    _, loss = wrapped(x.to(device), y.to(device))
                scaler.scale(loss / tc.grad_accum_steps).backward()
                loss_acc += loss.item() / tc.grad_accum_steps
            if not math.isfinite(loss_acc):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), tc.grad_clip)
            if hasattr(gnorm, "full_tensor"):  # DTensor under FSDP2
                gnorm = gnorm.full_tensor()
            scaler.step(opt)
            scaler.update()
            step += 1
            loss_mean = _mean(loss_acc, device)
            last = {"step": step, "loss": round(loss_mean, 5), "lr": lr, "grad_norm": round(float(gnorm), 4),
                    "elapsed_s": round(time.time() - t0, 2), "tokens_seen": (step - start) * tokens_per_step}
            if step % tc.eval_interval == 0 or step == tc.max_steps:
                wrapped.eval()
                with torch.no_grad():
                    vg = torch.Generator().manual_seed(tc.seed + 7 + rank)
                    vl = 0.0
                    for _ in range(tc.eval_batches):
                        x, y = val_ds.batch(tc.batch_size, vg)
                        with autocast():
                            vl += wrapped(x.to(device), y.to(device))[1].item()
                last["val_loss"] = round(_mean(vl / tc.eval_batches, device), 5)
                wrapped.train()
                log(f"step {step:>7} loss {loss_mean:.4f} val {last['val_loss']:.4f} lr {lr:.2e} {last['elapsed_s']:.0f}s")
            elif step % 10 == 0:
                log(f"step {step:>7} loss {loss_mean:.4f} lr {lr:.2e} {last['elapsed_s']:.0f}s")
            if rank == 0:
                with open(out / "train_log.jsonl", "a") as f:
                    f.write(json.dumps(last) + "\n")
            if step % tc.checkpoint_interval == 0 or step == tc.max_steps:
                save(step, last)
        if step > start and not (ckpt_dir / f"step_{step:06d}").exists():
            save(step, last)
        return {"final_step": step, "start_step": start, "world_size": world, "strategy": strategy, "last": last}
    finally:
        dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="infinity.core.dist_train")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--strategy", choices=["ddp", "fsdp"], default=None)
    a = ap.parse_args(argv)
    r = train(a.config, a.data, a.out, a.steps, a.strategy)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
