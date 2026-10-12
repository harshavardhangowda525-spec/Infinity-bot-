"""Supervised fine-tuning of an INFINITY HYPER 5.5 checkpoint on tool-use transcripts.

- Loss is computed only on completion tokens (the model's action + <|end|>);
  prompt tokens are masked with -100.
- Prompt truncation mirrors ``InferenceEngine.stream`` exactly: BOS + prompt,
  keep the last ``context - new_tokens`` tokens.
- The context window can be extended (RoPE positions are recomputed; rotary
  tables are not stored in checkpoints).
- The result is saved as a NEW run with lineage (parent checkpoint + sha256,
  dataset sha256). The parent is never modified. Promotion is a separate,
  explicit step (``infinity model promote``).
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..core.checkpoint import load_checkpoint, save_checkpoint, sha256_file
from ..core.config import RunConfig, TrainConfig
from ..core.model import InfinityNeuralCore
from ..core.trainer import build_optimizer, lr_at, pick_device


def agent_new_tokens(context: int) -> int:
    """Must match Agent.new_tokens for compact-prompt models."""
    return max(64, context // 3)


def encode_example(tok, prompt: str, completion: str, context: int) -> tuple[list[int], list[int]] | None:
    budget = context - min(agent_new_tokens(context), context // 2)
    p = tok.encode(prompt, allow_special=True, add_bos=True)[-budget:]
    c = tok.encode(completion, allow_special=True)
    if len(c) > context - len(p) or len(c) > agent_new_tokens(context):
        return None
    ids = p + c
    labels = [-100] * len(p) + c
    return ids, labels


class SFTData:
    def __init__(self, path: Path, tok, context: int) -> None:
        self.items: list[tuple[list[int], list[int]]] = []
        self.dropped = 0
        for line in open(path):
            ex = json.loads(line)
            enc = encode_example(tok, ex["prompt"], ex["completion"], context)
            if enc is None:
                self.dropped += 1
            else:
                self.items.append(enc)
        if not self.items:
            raise ValueError(f"no usable examples in {path}")
        self.pad = tok.pad_id

    def batch(self, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.items[i] for i in idx]
        L = max(len(s[0]) for s in seqs) - 1
        x = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        y = torch.full((len(seqs), L), -100, dtype=torch.long)
        for r, (ids, labels) in enumerate(seqs):
            n = len(ids) - 1
            x[r, :n] = torch.tensor(ids[:-1])
            y[r, :n] = torch.tensor(labels[1:])
        return x, y


@torch.no_grad()
def eval_loss(model, data: SFTData, batch_size: int, device: str, limit: int = 400) -> float:
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, min(len(data.items), limit), batch_size):
        idx = list(range(s, min(s + batch_size, len(data.items), limit)))
        x, y = data.batch(idx)
        _, loss = model(x.to(device), y.to(device))
        tot += loss.item() * len(idx)
        n += len(idx)
    model.train()
    return tot / max(1, n)


def finetune(base: str, data_dir: str | Path, out_dir: str | Path, steps: int = 1200, context: int = 512, batch_size: int = 16,
             lr: float = 1e-3, warmup: int = 50, seed: int = 1337, eval_every: int = 200, log=print, stop_flag=None,
             save_every_eval: bool = True) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    if (out_dir / "checkpoints").exists() and any((out_dir / "checkpoints").glob("step_*")):
        raise FileExistsError(f"{out_dir} already has checkpoints; choose a new output directory (runs are never overwritten)")
    device = pick_device("auto")
    torch.manual_seed(seed)
    base_model, tok, base_cfg, _, base_manifest, base_path = load_checkpoint(base, device="cpu")
    mcfg = replace(base_cfg.model, context_length=context)
    model = InfinityNeuralCore(mcfg)
    model.load_state_dict(base_model.state_dict())  # rotary tables are recomputed for the new context
    model.to(device)
    train = SFTData(data_dir / "train.jsonl", tok, context)
    evald = SFTData(data_dir / "eval.jsonl", tok, context)
    tc = TrainConfig(batch_size=batch_size, max_steps=steps, learning_rate=lr, warmup_steps=warmup, min_lr_ratio=0.1,
                     weight_decay=0.0, eval_interval=eval_every, checkpoint_interval=steps, seed=seed)
    opt = build_optimizer(model, tc, device)
    run_cfg = RunConfig(name=f"{base_cfg.name}-sft", model=mcfg, train=tc, data_manifest=str(data_dir / "sft_manifest.json"),
                        tokenizer_vocab_size=tok.vocab_size, output_dir=str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    rng = random.Random(seed)
    order: list[int] = []
    lineage = {"parent_checkpoint": str(base_path), "parent_step": base_manifest["step"],
               "parent_model_sha256": base_manifest["sha256"]["model.pt"],
               "train_data_sha256": sha256_file(data_dir / "train.jsonl"), "train_examples": len(train.items),
               "dropped_examples": train.dropped, "context_length": context, "method": "supervised fine-tuning (masked loss)"}
    log(f"SFT from {base_path} (step {base_manifest['step']}) ctx {base_cfg.model.context_length}->{context} "
        f"examples {len(train.items)} (dropped {train.dropped}) device {device}")
    before = eval_loss(model, evald, batch_size, device)
    log(f"eval loss before: {before:.4f}")
    t0 = time.time()
    model.train()
    last: dict = {}
    for step in range(steps):
        if stop_flag is not None and stop_flag.is_set():
            break
        if not order:
            order = list(range(len(train.items)))
            rng.shuffle(order)
        idx = [order.pop() for _ in range(min(batch_size, len(order)))]
        x, y = train.batch(idx)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        _, loss = model(x.to(device), y.to(device))
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        rec = {"step": step + 1, "loss": round(loss.item(), 5), "lr": lr_at(step, tc), "elapsed_s": round(time.time() - t0, 2)}
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            rec["eval_loss"] = round(eval_loss(model, evald, batch_size, device), 5)
            log(f"step {step + 1:>5} loss {rec['loss']:.4f} eval {rec['eval_loss']:.4f} {rec['elapsed_s']:.0f}s")
            if save_every_eval and step + 1 < steps:
                # Keep intermediate checkpoints so the best one can be chosen by benchmark (eval loss != task success).
                save_checkpoint(out_dir / "checkpoints", step + 1, model, None, run_cfg, tok,
                                {**rec, "eval_loss_before": round(before, 5), "lineage": lineage})
        elif (step + 1) % 25 == 0:
            log(f"step {step + 1:>5} loss {rec['loss']:.4f} {rec['elapsed_s']:.0f}s")
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        last = rec
    final_eval = eval_loss(model, evald, batch_size, device)
    metrics = {**last, "eval_loss_before": round(before, 5), "eval_loss": round(final_eval, 5), "lineage": lineage}
    path = save_checkpoint(out_dir / "checkpoints", last.get("step", 0), model.cpu(), None, run_cfg, tok, metrics)
    log(f"saved {path}")
    return {"checkpoint": str(path), "eval_loss_before": before, "eval_loss_after": final_eval, "steps": last.get("step", 0),
            "examples": len(train.items), "wall_time_s": round(time.time() - t0, 1)}
