"""Model adapters: the agent runtime talks to models only through this interface.

The product identity is always INFINITY HYPER 5.5; ``info()`` reports which
implementation is actually serving it, so the UI and logs never blur the two.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

PRODUCT = "INFINITY HYPER 5.5"


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str


@dataclass
class Completion:
    text: str
    tokens_in: int
    tokens_out: int
    tokens_exact: bool


class ModelUnavailable(RuntimeError):
    pass


def render_chat(messages: Sequence[Message]) -> str:
    """The one chat template used by INFINITY HYPER 5.5 for inference AND fine-tuning data."""
    return "".join(f"<|{m.role}|>{m.content}<|end|>" for m in messages) + "<|assistant|>"


class ModelAdapter:
    name = "base"
    native_tool_calling = False

    def info(self) -> dict:
        raise NotImplementedError

    @property
    def max_context_tokens(self) -> int:
        raise NotImplementedError

    def count_tokens(self, text: str) -> tuple[int, bool]:
        """Return (count, exact). Adapters without a tokenizer estimate."""
        return max(1, len(text) // 4), False

    def complete(self, messages: Sequence[Message], max_new_tokens: int = 512,
                 cancel: threading.Event | None = None) -> Completion:
        raise NotImplementedError


class ScriptedModel(ModelAdapter):
    """Deterministic stand-in for tests and demos. Never used silently: it
    reports itself as 'scripted' everywhere the active model is shown."""

    name = "scripted"

    def __init__(self, script: list[str] | Callable[[Sequence[Message]], str], context_tokens: int = 32768) -> None:
        self._script = script
        self._i = 0
        self._ctx = context_tokens
        self.calls: list[list[Message]] = []

    def info(self) -> dict:
        return {"product": PRODUCT, "implementation": "scripted (test double — not a language model)",
                "context_tokens": self._ctx}

    @property
    def max_context_tokens(self) -> int:
        return self._ctx

    def complete(self, messages, max_new_tokens=512, cancel=None) -> Completion:
        self.calls.append(list(messages))
        if callable(self._script):
            text = self._script(messages)
        else:
            if self._i >= len(self._script):
                raise ModelUnavailable("scripted model has no more responses")
            text = self._script[self._i]
            self._i += 1
        tin = sum(self.count_tokens(m.content)[0] for m in messages)
        return Completion(text, tin, self.count_tokens(text)[0], False)

    @classmethod
    def from_file(cls, path: str | Path) -> "ScriptedModel":
        data = json.loads(Path(path).read_text())
        return cls([d if isinstance(d, str) else json.dumps(d) for d in data])


class NeuralCoreModel(ModelAdapter):
    """INFINITY HYPER 5.5 served by the original INFINITY NEURAL CORE weights."""

    name = "infinity-neural-core"

    def __init__(self, checkpoint: str, device: str = "cpu", temperature: float = 0.7, seed: int | None = None,
                 adapter: str | None = None) -> None:
        try:
            from ...core.inference import InferenceEngine, SamplingConfig
        except ImportError as e:  # torch missing
            raise ModelUnavailable(f"PyTorch is required for the neural core: {e}")
        try:
            self.engine = InferenceEngine(checkpoint, device=device, adapter=adapter)
        except Exception as e:
            what = f"adapter {adapter!r} on checkpoint {checkpoint!r}" if adapter else f"checkpoint {checkpoint!r}"
            raise ModelUnavailable(f"could not load {what}: {e}")
        self._Sampling = SamplingConfig
        self.temperature = temperature
        self.seed = seed

    def info(self) -> dict:
        return self.engine.info()

    @property
    def max_context_tokens(self) -> int:
        return self.engine.run_cfg.model.context_length

    def count_tokens(self, text: str) -> tuple[int, bool]:
        return len(self.engine.tokenizer.encode(text)), True

    def render(self, messages: Sequence[Message]) -> str:
        return render_chat(messages)

    def complete(self, messages, max_new_tokens=512, cancel=None) -> Completion:
        prompt = self.render(messages)
        cfg = self._Sampling(max_new_tokens=max_new_tokens, temperature=self.temperature, seed=self.seed)
        text = self.engine.generate(prompt, cfg, cancel=cancel)
        tin = min(len(self.engine.tokenizer.encode(prompt, allow_special=True)) + 1, self.max_context_tokens)
        return Completion(text, tin, len(self.engine.tokenizer.encode(text)), True)


def load_model(spec: str) -> ModelAdapter:
    """``neural:<checkpoint>[::<adapter folder>]`` (INFINITY HYPER 5.5, optionally Fine-Tuned),
    ``scripted:<json list file>``, or ``hf:<HF id or folder>[::<adapter folder>]`` for an open model."""
    kind, _, arg = spec.partition(":")
    if kind == "neural":
        ckpt, _, adapter = arg.partition("::")
        return NeuralCoreModel(ckpt, adapter=adapter or None)
    if kind == "scripted":
        return ScriptedModel.from_file(arg)
    if kind == "hf":
        from ...hf.model import HFModel

        base, _, adapter = arg.partition("::")
        return HFModel(base, adapter or None, base_dir=Path.cwd())
    raise ValueError(f"unknown model spec {spec!r}; use neural:<checkpoint>, hf:<model>[::<adapter>] or scripted:<file>")
