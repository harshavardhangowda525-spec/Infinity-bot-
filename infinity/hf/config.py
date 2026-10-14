"""YAML training configs with ``extends:`` inheritance (deep merge)."""

from __future__ import annotations

from pathlib import Path

REQUIRED = {
    "base_model": str, "model_label": str, "context_tokens": int, "run_name": str,
}
DEFAULTS = {
    "data": {"chat_dirs": ["datasets/chat"], "coding_dirs": ["datasets/coding"],
             "validation_dirs": ["datasets/validation"], "test_cases": "training/evaluation/test_cases.jsonl",
             "tool_tasks_train": 120, "tool_tasks_val": 10, "tool_seed": 2026, "tool_val_seed": 4242,
             "out_dir": "training/data", "max_seq_len": 2048, "train_domains": ["chat", "coding", "tool"]},
    "training": {"seed": 1337, "max_steps": 100, "batch_size": 1, "grad_accum_steps": 4, "learning_rate": 2e-4,
                 "warmup_steps": 10, "min_lr_ratio": 0.1, "weight_decay": 0.0, "grad_clip": 1.0,
                 "eval_interval": 25, "eval_examples": 20, "checkpoint_interval": 25,
                 "gradient_checkpointing": True, "precision": "auto", "device": "auto",
                 "checkpoint_dir": "training/checkpoints", "log_dir": "training/logs",
                 "adapter_dir": "adapters/infinity-hyper-5-5"},
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
}


class ConfigError(ValueError):
    pass


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    import yaml

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "extends" in raw:
        parent = load_config(path.parent / raw.pop("extends"))
        raw = _merge(parent, raw)
    cfg = _merge(DEFAULTS, raw)
    if overrides:
        cfg = _merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    for k, t in REQUIRED.items():
        if not isinstance(cfg.get(k), t):
            raise ConfigError(f"{k} is required and must be {t.__name__}")
    t = cfg["training"]
    for k in ("max_steps", "batch_size", "grad_accum_steps", "eval_interval", "checkpoint_interval"):
        if not isinstance(t[k], int) or t[k] < 1:
            raise ConfigError(f"training.{k} must be a positive integer")
    if not (0 < float(t["learning_rate"]) < 1):
        raise ConfigError("training.learning_rate must be between 0 and 1")
    if t["precision"] not in ("auto", "fp32", "bf16", "fp16"):
        raise ConfigError("training.precision must be auto, fp32, bf16 or fp16")
    bad = set(cfg["data"]["train_domains"]) - {"chat", "coding", "tool"}
    if bad or not cfg["data"]["train_domains"]:
        raise ConfigError(f"data.train_domains must be a non-empty subset of chat, coding, tool (got {bad or 'none'})")
    if cfg["lora"]["r"] < 1 or not cfg["lora"]["target_modules"]:
        raise ConfigError("lora.r must be >= 1 and lora.target_modules non-empty")
    if cfg["data"]["max_seq_len"] > cfg["context_tokens"]:
        raise ConfigError("data.max_seq_len cannot exceed context_tokens")
