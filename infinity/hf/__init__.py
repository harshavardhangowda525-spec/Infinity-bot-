"""Open Hugging Face models (e.g. Qwen2.5-Coder) inside INFINITY BOT, with optional LoRA adapters.

These models are NOT INFINITY HYPER 5.5. Everything here labels them by their real
name, so the UI, logs and reports never present an open model as the original one.
``transformers`` and ``peft`` are optional dependencies (training/requirements-training.txt).
"""

from .chat import CHAT_SYSTEM_PROMPT, TOOL_PREFIX, chat_system_prompt, system_prompt_for, to_hf_messages

__all__ = ["CHAT_SYSTEM_PROMPT", "TOOL_PREFIX", "chat_system_prompt", "system_prompt_for", "to_hf_messages"]
