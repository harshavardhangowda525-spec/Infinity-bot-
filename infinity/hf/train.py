"""LoRA supervised fine-tuning for open Hugging Face models (plain PyTorch loop + PEFT).

A plain loop is used instead of ``Trainer``/TRL for three reasons: their APIs change between
releases, the loop is short enough to audit, and it gives exact control over resume,
atomic checkpoints and out-of-memory handling.

Modes:
- ``dry_run``: validate the config and data, tokenize, and print the plan. Loads the tokenizer only.
- ``smoke``: 2 optimizer steps on 4 examples into a temporary folder, then reload the
  saved adapter and check that it reproduces the in-memory model's logits.
- full run: as configured. Ctrl+C saves a checkpoint; ``--resume`` continues from it.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

from .data import encode_example


class TrainingOOM(RuntimeError):
    pass


def _is_oom(e: BaseException) -> bool:
    import torch

    if isinstance(e, torch.OutOfMemoryError):
        return True
    msg = str(e).lower()
    return isinstance(e, (RuntimeError, MemoryError)) and (
        "out of memory" in msg or "can't allocate" in msg or "not enough memory" in msg or isinstance(e, MemoryError))


def _rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def lr_at(step: int, cfg: dict) -> float:
    t = cfg["training"]
    base, warm, total = float(t["learning_rate"]), int(t["warmup_steps"]), int(t["max_steps"])
    if step < warm:
        return base * (step + 1) / max(1, warm)
    progress = min(1.0, (step - warm) / max(1, total - warm))
    floor = base * float(t["min_lr_ratio"])
    return floor + (base - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def resolve_device_precision(cfg: dict) -> tuple[str, str]:
    import torch

    t = cfg["training"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if t["device"] == "auto" else t["device"]
    prec = t["precision"]
    if prec == "auto":
        prec = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "fp32"
    if device == "cpu" and prec == "fp16":
        raise ValueError("fp16 training needs a CUDA GPU; use fp32 (or bf16) on CPU")
    return device, prec


def encode_split(tokenizer, rows: list[dict], max_len: int) -> list[tuple[list[int], list[int]]]:
    out = []
    for ex in rows:
        enc = encode_example(tokenizer, ex, max_len)
        if enc is not None:
            out.append(enc)
    return out


def _paths(cfg: dict, root: Path) -> dict[str, Path]:
    t, run = cfg["training"], cfg["run_name"]
    return {"ckpt": root / t["checkpoint_dir"] / run, "log": root / t["log_dir"] / f"{run}.jsonl",
            "adapter": root / t["adapter_dir"] / run, "data": root / cfg["data"]["out_dir"]}


def plan(cfg: dict, root: Path, tokenizer, log=print) -> dict:
    """Dry run: everything except loading model weights."""
    p = _paths(cfg, root)
    for name in ("train.jsonl", "val.jsonl"):
        if not (p["data"] / name).exists():
            raise FileNotFoundError(f"{p['data'] / name} missing — run training/scripts/prepare_data.py first")
    t = cfg["training"]
    max_len = cfg["data"]["max_seq_len"]
    train = encode_split(tokenizer, _read_jsonl(p["data"] / "train.jsonl"), max_len)
    val = encode_split(tokenizer, _read_jsonl(p["data"] / "val.jsonl"), max_len)
    if not train:
        raise ValueError("no usable training examples after tokenization")
    lens = sorted(len(ids) for ids, _ in train)
    seqs = t["max_steps"] * t["batch_size"] * t["grad_accum_steps"]
    summary = {
        "base_model": cfg["base_model"], "run_name": cfg["run_name"],
        "train_examples": len(train), "val_examples": len(val),
        "tokens_per_example": {"min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1]},
        "trained_tokens": sum(sum(1 for x in lab if x != -100) for _, lab in train),
        "sequences_processed": seqs, "epochs": round(seqs / len(train), 2),
        "checkpoints_to": str(p["ckpt"]), "adapter_to": str(p["adapter"]), "log_to": str(p["log"]),
        "resume_available": (p["ckpt"] / "LATEST").exists(),
    }
    for k, v in summary.items():
        log(f"  {k}: {v}")
    return summary


def _save_checkpoint(model, optimizer, scaler, step: int, cfg: dict, ckpt_root: Path, metrics: dict) -> Path:
    import torch

    final = ckpt_root / f"step_{step:06d}"
    tmp = ckpt_root / f".tmp_step_{step:06d}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp)
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                "rng_torch": torch.get_rng_state(), "rng_py": random.getstate()}, tmp / "trainer_state.pt")
    (tmp / "state.json").write_text(json.dumps({"step": step, "run_name": cfg["run_name"], "metrics": metrics,
                                                "base_model": cfg["base_model"]}, indent=2))
    if final.exists():  # never overwrite an existing checkpoint
        shutil.rmtree(tmp)
        return final
    os.replace(tmp, final)
    (ckpt_root / "LATEST.tmp").write_text(final.name)
    os.replace(ckpt_root / "LATEST.tmp", ckpt_root / "LATEST")
    return final


def base_files(base_path: str) -> Path | None:
    """Folder holding the base model's files: a local folder, or the Hugging Face cache snapshot."""
    p = Path(base_path)
    if p.is_dir():
        return p
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=base_path, local_files_only=True))
    except Exception:
        return None


def fingerprint_files(folder: Path | None) -> dict[str, str]:
    """sha256 of every file (symlinks followed) in the base model folder."""
    import hashlib

    if folder is None:
        return {}
    out = {}
    for f in sorted(folder.rglob("*")):
        if f.is_file():
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[str(f.relative_to(folder))] = h.hexdigest()
    return out


def _versions() -> dict:
    import peft
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}


def train(cfg: dict, root: Path, resume: bool = False, smoke: bool = False, log=print) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .model import _resolve

    t = cfg["training"]
    tmpdir = None
    if smoke:
        tmpdir = tempfile.mkdtemp(prefix="infinity-smoke-")
        cfg = json.loads(json.dumps(cfg))
        cfg["training"].update(max_steps=2, eval_interval=1, eval_examples=2, checkpoint_interval=1, warmup_steps=1,
                               checkpoint_dir=str(Path(tmpdir) / "ckpt"), log_dir=str(Path(tmpdir) / "logs"),
                               adapter_dir=str(Path(tmpdir) / "adapter"))
        t = cfg["training"]
    p = _paths(cfg, root)
    seed = int(t["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device, prec = resolve_device_precision(cfg)
    base_path = _resolve(cfg["base_model"], root)

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_len = cfg["data"]["max_seq_len"]
    train_rows = _read_jsonl(p["data"] / "train.jsonl")
    val_rows = _read_jsonl(p["data"] / "val.jsonl")
    if smoke:
        train_rows, val_rows = train_rows[:4], val_rows[:2]
    train_set = encode_split(tokenizer, train_rows, max_len)
    val_set = encode_split(tokenizer, val_rows, max_len)[: t["eval_examples"]]
    if not train_set:
        raise ValueError("no usable training examples")

    dtype = torch.bfloat16 if prec == "bf16" else torch.float32
    log(f"loading {cfg['base_model']} on {device} ({prec})…")
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype)
    if t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.enable_input_require_grads()
    lc = cfg["lora"]
    try:
        model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                                 target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    except ValueError as e:
        raise ValueError(f"LoRA target modules {lc['target_modules']} do not match this architecture: {e}")
    model.to(device)
    trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    total = sum(p_.numel() for p_ in model.parameters())
    log(f"trainable LoRA parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")
    base_dir = base_files(base_path)
    base_before = fingerprint_files(base_dir)
    finetuned_name = cfg["model_label"] + " — Fine-Tuned"
    log(f"base model (original, read-only): {cfg['base_model']} · {len(base_before)} files fingerprinted"
        + ("" if base_before else " — WARNING: base files not found locally, cannot verify"))

    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(t["learning_rate"]), weight_decay=float(t["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda") if prec == "fp16" else None
    step = 0
    if resume and (p["ckpt"] / "LATEST").exists():
        ck = p["ckpt"] / (p["ckpt"] / "LATEST").read_text().strip()
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(ck / "adapter_model.safetensors"), device=device))
        st = torch.load(ck / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(st["optimizer"])
        if scaler and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        torch.set_rng_state(st["rng_torch"])
        random.setstate(st["rng_py"])
        step = json.loads((ck / "state.json").read_text())["step"]
        log(f"resumed from {ck} (step {step})")
    elif resume:
        log("no checkpoint to resume from; starting at step 0")

    pad = tokenizer.pad_token_id
    bs, accum = t["batch_size"], t["grad_accum_steps"]
    n = len(train_set)

    def example_at(i: int):
        epoch, pos = divmod(i, n)
        order = list(range(n))
        random.Random(seed + epoch).shuffle(order)
        return train_set[order[pos]]

    def collate(items):
        L = max(len(ids) for ids, _ in items)
        ids = torch.full((len(items), L), pad, dtype=torch.long)
        lab = torch.full((len(items), L), -100, dtype=torch.long)
        att = torch.zeros((len(items), L), dtype=torch.long)
        for r, (a, b) in enumerate(items):
            ids[r, : len(a)] = torch.tensor(a)
            lab[r, : len(b)] = torch.tensor(b)
            att[r, : len(a)] = 1
        return ids.to(device), lab.to(device), att.to(device)

    def autocast():
        return torch.autocast("cuda", dtype=torch.float16) if prec == "fp16" else torch.autocast(device, enabled=False)

    @torch.no_grad()
    def evaluate() -> float | None:
        if not val_set:
            return None
        model.eval()
        losses = []
        for item in val_set:
            ids, lab, att = collate([item])
            with autocast():
                losses.append(model(input_ids=ids, attention_mask=att, labels=lab).loss.float().item())
        model.train()
        return round(sum(losses) / len(losses), 4)

    p["log"].parent.mkdir(parents=True, exist_ok=True)
    p["ckpt"].mkdir(parents=True, exist_ok=True)
    logf = open(p["log"], "a", encoding="utf-8")
    logf.write(json.dumps({"event": "start", "base_model": cfg["base_model"], "base_role": "original, read-only (frozen)",
                           "output": finetuned_name, "method": "LoRA (adapter saved separately, not merged)",
                           "base_files_sha256": base_before, "resume_from_step": step}) + "\n")
    model.train()
    t0 = time.time()
    start_step = step
    last: dict = {}
    eval_loss = None
    try:
        while step < t["max_steps"]:
            lr = lr_at(step, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr
            loss_sum = 0.0
            try:
                for micro in range(accum):
                    base = (step * accum + micro) * bs
                    ids, lab, att = collate([example_at(base + k) for k in range(bs)])
                    with autocast():
                        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / accum
                    (scaler.scale(loss) if scaler else loss).backward()
                    loss_sum += loss.float().item()
                if scaler:
                    scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(params, float(t["grad_clip"])).item()
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            except BaseException as e:
                if not isinstance(e, KeyboardInterrupt) and _is_oom(e):
                    raise TrainingOOM(
                        f"out of memory at step {step + 1}. Nothing was deleted; the last checkpoint is kept. "
                        f"Lower data.max_seq_len (now {max_len}) or training.batch_size (now {bs}), keep "
                        f"gradient_checkpointing on, close other programs, then rerun with --resume.") from e
                raise
            step += 1
            elapsed = time.time() - t0
            per = elapsed / (step - start_step)
            last = {"step": step, "loss": round(loss_sum, 4), "lr": lr, "grad_norm": round(gn, 4),
                    "elapsed_s": round(elapsed, 1), "s_per_step": round(per, 2),
                    "eta_s": round(per * (t["max_steps"] - step)), "rss_mb": _rss_mb()}
            if step % t["eval_interval"] == 0 or step == t["max_steps"]:
                eval_loss = evaluate()
                last["eval_loss"] = eval_loss
            logf.write(json.dumps(last) + "\n")
            logf.flush()
            eta = f"{last['eta_s'] // 60}m{last['eta_s'] % 60:02d}s"
            log(f"step {step}/{t['max_steps']} loss {last['loss']:.4f}"
                + (f" eval {eval_loss:.4f}" if "eval_loss" in last and eval_loss is not None else "")
                + f" · {per:.1f}s/step · ETA {eta}")
            if step % t["checkpoint_interval"] == 0 or step == t["max_steps"]:
                ck = _save_checkpoint(model, optimizer, scaler, step, cfg, p["ckpt"], last)
                log(f"checkpoint saved: {ck}")
    except KeyboardInterrupt:
        if step > start_step:
            ck = _save_checkpoint(model, optimizer, scaler, step, cfg, p["ckpt"], last)
            log(f"interrupted — checkpoint saved: {ck}. Rerun with --resume to continue.")
        logf.close()
        raise
    base_after = fingerprint_files(base_dir)
    if base_after != base_before:
        logf.close()
        raise RuntimeError(f"ORIGINAL BASE MODEL FILES CHANGED during training: "
                           f"{sorted(k for k in set(base_before) | set(base_after) if base_before.get(k) != base_after.get(k))}")
    logf.write(json.dumps({"event": "end", "base_unchanged_after_training": bool(base_before), "steps": step}) + "\n")
    logf.close()

    # Export the final adapter (atomic; an existing export is never overwritten).
    out = p["adapter"]
    if out.exists():
        out = out.with_name(f"{out.name}-{time.strftime('%Y%m%d-%H%M%S')}")
    tmp = out.with_name(f".tmp-{out.name}")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    manifest = {"name": finetuned_name, "base_model": cfg["base_model"], "base_role": "original, read-only",
                "model_label": cfg["model_label"], "run_name": cfg["run_name"], "merged_into_base": False,
                "base_files_sha256": base_before, "base_unchanged_after_training": bool(base_before),
                "steps": step, "final": last, "config": cfg, "versions": _versions(),
                "data_report": str(p["data"] / "report.json"),
                "note": "LoRA adapter for an open model. It is NOT INFINITY HYPER 5.5 and not HYPER 5.5 — Fine-Tuned."}
    (tmp / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, out)
    log(f"adapter exported: {out}")
    result = {"adapter": str(out), "name": finetuned_name, "steps": step, "final": last, "device": device,
              "precision": prec, "base_unchanged": bool(base_before)}
    log(f"original base model verified unchanged ({len(base_before)} files, sha256)" if base_before
        else "base model files could not be located for verification")

    if smoke:  # reload the export and check it reproduces the in-memory model
        from peft import PeftModel

        ids, _, att = collate([train_set[0]])
        model.eval()
        with torch.no_grad():
            want = model(input_ids=ids, attention_mask=att).logits.float()
            fresh = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype)
            base_logits = fresh.to(device).eval()(input_ids=ids, attention_mask=att).logits.float()
            got = PeftModel.from_pretrained(fresh, str(out)).to(device).eval()(input_ids=ids, attention_mask=att).logits.float()
        diff = (want - got).abs().max().item()
        moved = (want - base_logits).abs().max().item()
        result.update(reload_max_logit_diff=diff, adapter_effect_max_logit=moved)
        if diff > 1e-3:
            raise AssertionError(f"reloaded adapter differs from the trained model (max |Δlogit| = {diff})")
        if moved == 0.0:
            raise AssertionError("the trained adapter has no effect on the outputs")
        log(f"smoke test OK: trained 2 steps, saved, reloaded (reload |Δlogit| {diff:.2e}; "
            f"adapter changes logits by up to {moved:.2e})")
        shutil.rmtree(tmpdir, ignore_errors=True)
    return result
