"""Shared helpers for the training scripts (run them from the project root)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def tokenizer_for(cfg):
    from transformers import AutoTokenizer

    from infinity.hf.model import _resolve

    return AutoTokenizer.from_pretrained(_resolve(cfg["base_model"], ROOT))
