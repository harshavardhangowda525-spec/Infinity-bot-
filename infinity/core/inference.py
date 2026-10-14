"""INFINITY INFERENCE — loads verified checkpoints and generates text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .config import MODEL_FAMILY, MODEL_VERSION


@dataclass
class SamplingConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    seed: int | None = None


def sample_next(logits: torch.Tensor, cfg: SamplingConfig, history: list[int], generator) -> int:
    logits = logits.float().clone()
    if cfg.repetition_penalty != 1.0 and history:
        idx = torch.tensor(sorted(set(history)), dtype=torch.long)
        vals = logits[idx]
        logits[idx] = torch.where(vals > 0, vals / cfg.repetition_penalty, vals * cfg.repetition_penalty)
    if cfg.temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / cfg.temperature
    if cfg.top_k and cfg.top_k < logits.numel():
        kth = torch.topk(logits, cfg.top_k).values[-1]
        logits[logits < kth] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    if 0 < cfg.top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        cut = cum - sorted_p > cfg.top_p
        sorted_p[cut] = 0.0
        probs = torch.zeros_like(probs).scatter(0, sorted_i, sorted_p)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


class InferenceEngine:
    def __init__(self, checkpoint: str, device: str = "cpu", adapter: str | None = None) -> None:
        self.adapter_dir = None
        self.adapter_conf = None
        if adapter:  # original weights + LoRA adapter on top (never merged)
            from pathlib import Path

            from .lora import load_with_adapter

            model, tok, run_cfg, _, manifest, path, conf = load_with_adapter(checkpoint, adapter, device=device)
            self.adapter_dir, self.adapter_conf = Path(adapter), conf
        else:
            model, tok, run_cfg, _, manifest, path = load_checkpoint(checkpoint, device=device)
        model.eval()
        self.model = model
        self.tokenizer = tok
        self.run_cfg = run_cfg
        self.manifest = manifest
        self.path = path
        self.device = device

    def info(self) -> dict:
        if self.adapter_conf is not None:
            c = self.adapter_conf
            return {
                "product": c["name"], "variant": "fine-tuned",
                "implementation": "infinity-neural-core original weights (frozen) + LoRA adapter (not merged)",
                "run": self.run_cfg.name, "checkpoint": str(self.path), "base_checkpoint": str(self.path),
                "adapter": str(self.adapter_dir), "adapter_name": self.adapter_dir.name,
                "step": self.manifest["step"], "parameters": self.manifest["parameters"],
                "adapter_parameters": sum(p.numel() for n, p in self.model.named_parameters() if ".lora_" in n),
                "context_length": self.run_cfg.model.context_length,
                "tokenizer_fingerprint": self.manifest["tokenizer_fingerprint"],
                "metrics": {k: c.get(k) for k in ("trained_steps", "eval_loss_before", "eval_loss_after")},
            }
        return {
            "product": f"{MODEL_FAMILY} {MODEL_VERSION}",
            "variant": "original",
            "implementation": "infinity-neural-core (original weights, trained in this project)",
            "run": self.run_cfg.name,
            "checkpoint": str(self.path),
            "step": self.manifest["step"],
            "parameters": self.manifest["parameters"],
            "context_length": self.run_cfg.model.context_length,
            "tokenizer_fingerprint": self.manifest["tokenizer_fingerprint"],
            "metrics": self.manifest.get("metrics", {}),
        }

    @torch.no_grad()
    def stream(self, prompt: str, cfg: SamplingConfig | None = None, allow_special: bool = True,
               stop_ids: set[int] | None = None, cancel=None) -> Iterator[str]:
        cfg = cfg or SamplingConfig()
        tok = self.tokenizer
        ctx = self.run_cfg.model.context_length
        ids = tok.encode(prompt, allow_special=allow_special, add_bos=True)
        stop_ids = stop_ids if stop_ids is not None else {tok.eos_id, tok.token_id("<|end|>")}
        gen = torch.Generator().manual_seed(cfg.seed) if cfg.seed is not None else None
        # Leave room to generate: keep the most recent part of the prompt.
        budget = max(1, ctx - min(cfg.max_new_tokens, ctx // 2))
        ids = ids[-budget:]
        history = list(ids)
        caches = self.model.new_caches()
        logits, _ = self.model(torch.tensor([ids], device=self.device), caches=caches)
        pending = bytearray()
        produced: list[int] = []
        for _ in range(cfg.max_new_tokens):
            if cancel is not None and cancel.is_set():
                break
            nxt = sample_next(logits[0, -1].cpu(), cfg, history, gen)
            if nxt in stop_ids:
                break
            produced.append(nxt)
            history.append(nxt)
            # Emit only complete UTF-8 sequences.
            if nxt in tok.vocab:
                pending.extend(tok.vocab[nxt])
                try:
                    text = pending.decode("utf-8")
                    pending.clear()
                    yield text
                except UnicodeDecodeError:
                    if len(pending) > 8:
                        yield pending.decode("utf-8", errors="replace")
                        pending.clear()
            else:
                yield tok.decode([nxt])
            if len(history) >= ctx:
                # Slide the window: recompute the cache on the most recent tokens.
                history_window = history[-(ctx // 2):]
                caches = self.model.new_caches()
                logits, _ = self.model(torch.tensor([history_window], device=self.device), caches=caches)
                history = list(history_window)
            else:
                logits, _ = self.model(torch.tensor([[nxt]], device=self.device), caches=caches)
        if pending:
            yield pending.decode("utf-8", errors="replace")

    def generate(self, prompt: str, cfg: SamplingConfig | None = None, **kw) -> str:
        return "".join(self.stream(prompt, cfg, **kw))
