"""``HFModel``: an open Hugging Face causal LM (optionally + a PEFT/LoRA adapter) as a ModelAdapter.

It plugs into the same agent runtime as INFINITY HYPER 5.5, but reports its real
identity in ``info()``. It never claims to be HYPER 5.5.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Sequence

from ..agent.models.base import Completion, ModelAdapter, ModelUnavailable
from .chat import to_hf_messages


def _resolve(ref: str, base_dir: Path | None) -> str:
    """Relative local paths resolve against the project; anything else is a Hugging Face model id."""
    p = Path(ref)
    if not p.is_absolute() and base_dir is not None and (base_dir / p).exists():
        return str(base_dir / p)
    return ref


class HFModel(ModelAdapter):
    name = "hf"

    def __init__(self, base: str, adapter: str | None = None, label: str | None = None,
                 context_tokens: int = 4096, device: str = "auto", dtype: str = "auto",
                 temperature: float = 0.2, seed: int | None = None, base_dir: Path | None = None,
                 merge_adapter: bool = False, model_id: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ModelUnavailable(
                f"open models need transformers + peft (pip install -r training/requirements-training.txt): {e}")
        self.torch = torch
        self.base_ref = base
        self.adapter_ref = adapter
        self.label = label or base
        self.model_id = model_id
        self.temperature = temperature
        self.seed = seed
        self._lock = threading.Lock()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if dtype == "auto":
            dt = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        else:
            dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
        base_path = _resolve(base, base_dir)
        t0 = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_path)
            model = AutoModelForCausalLM.from_pretrained(base_path, dtype=dt)
        except Exception as e:
            raise ModelUnavailable(f"could not load open model {base!r}: {type(e).__name__}: {e}")
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ModelUnavailable(f"peft is required to load adapter {adapter!r}: {e}")
            try:
                model = PeftModel.from_pretrained(model, _resolve(adapter, base_dir))
                if merge_adapter:  # in memory only; never written back. Off by default (adapter stays separate)
                    model = model.merge_and_unload()
            except Exception as e:
                raise ModelUnavailable(f"could not load adapter {adapter!r}: {type(e).__name__}: {e}")
        self.merged = bool(adapter and merge_adapter)
        self.model = model.to(self.device).eval()
        self.load_seconds = round(time.time() - t0, 1)
        limit = getattr(self.model.config, "max_position_embeddings", context_tokens) or context_tokens
        self._ctx = int(min(context_tokens, limit))
        self.parameters = sum(p.numel() for p in self.model.parameters())
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------- interface
    def info(self) -> dict:
        product = self.label + (" — Fine-Tuned" if self.adapter_ref else "")
        return {
            "product": product,
            "family": "open-model",
            "variant": "fine-tuned" if self.adapter_ref else "original",
            "adapter_merged": self.merged,
            "id": self.model_id,
            "implementation": "Hugging Face Transformers — open model, not INFINITY HYPER 5.5",
            "base": self.base_ref,
            "adapter": self.adapter_ref,
            "parameters": self.parameters,
            "context_length": self._ctx,
            "device": self.device,
            "load_seconds": self.load_seconds,
        }

    @property
    def max_context_tokens(self) -> int:
        return self._ctx

    def count_tokens(self, text: str) -> tuple[int, bool]:
        return len(self.tokenizer.encode(text, add_special_tokens=False)), True

    def render(self, messages: Sequence) -> str:
        return self.tokenizer.apply_chat_template(to_hf_messages(messages), tokenize=False, add_generation_prompt=True)

    def complete(self, messages, max_new_tokens: int = 512, cancel: threading.Event | None = None) -> Completion:
        torch = self.torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return bool(cancel is not None and cancel.is_set())

        prompt = self.render(messages)
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        max_new_tokens = max(1, min(max_new_tokens, self._ctx - 16))
        budget = self._ctx - max_new_tokens
        if ids.shape[1] > budget:  # keep the most recent context
            ids = ids[:, -budget:]
        ids = ids.to(self.device)
        kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tokenizer.pad_token_id,
                  "stopping_criteria": StoppingCriteriaList([_Cancel()])}
        if self.temperature and self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with self._lock, torch.no_grad():
            if self.seed is not None:
                torch.manual_seed(self.seed)
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), **kwargs)
        new = out[0, ids.shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return Completion(text.strip(), int(ids.shape[1]), int(new.shape[0]), True)
