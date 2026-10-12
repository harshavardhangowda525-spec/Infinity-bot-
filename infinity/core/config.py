"""Model and training configuration for INFINITY HYPER 5.5."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

MODEL_FAMILY = "INFINITY HYPER"
MODEL_VERSION = "5.5"


@dataclass
class ModelConfig:
    """Architecture hyper-parameters for the INFINITY NEURAL CORE."""

    vocab_size: int = 2048
    context_length: int = 256
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    d_ff: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension must be even for rotary embeddings")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 600
    learning_rate: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval: int = 200
    seed: int = 1337
    precision: str = "fp32"  # fp32 | bf16 (autocast)
    device: str = "auto"
    num_threads: int = 0  # 0 = torch default


@dataclass
class RunConfig:
    """Top-level config file: identity + model + training + data locations."""

    name: str = "hyper-5.5-tiny"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_manifest: str = "data/manifest.json"
    tokenizer_vocab_size: int = 2048
    output_dir: str = "runs/hyper-5.5-tiny"

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        raw = json.loads(Path(path).read_text())
        model = _build(ModelConfig, raw.pop("model", {}))
        train = _build(TrainConfig, raw.pop("train", {}))
        raw.pop("identity", None)
        return cls(model=model, train=train, **_known(cls, raw))

    def to_dict(self) -> dict:
        return {
            "identity": {"family": MODEL_FAMILY, "version": MODEL_VERSION},
            "name": self.name,
            "model": asdict(self.model),
            "train": asdict(self.train),
            "data_manifest": self.data_manifest,
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "output_dir": self.output_dir,
        }


def _known(cls, raw: dict) -> dict:
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return raw


def _build(cls, raw: dict):
    return cls(**_known(cls, dict(raw)))
