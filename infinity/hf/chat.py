"""Message mapping shared by inference AND training data, so both see identical prompts."""

from __future__ import annotations

from typing import Iterable

# Tool observations are sent as user turns with this prefix. Chat templates differ in
# how (or whether) they support a "tool" role; a prefixed user turn works everywhere.
TOOL_PREFIX = "[tool result]\n"

CHAT_SYSTEM_PROMPT = (
    "You are the chat assistant inside INFINITY BOT, running the open-source model {label}. "
    "You are not INFINITY HYPER 5.5; if asked which model you are, say {label}. "
    "Be helpful, clear and concise. If you are not sure about something, say so instead of guessing."
)


def chat_system_prompt(label: str) -> str:
    return CHAT_SYSTEM_PROMPT.format(label=label)


def to_hf_messages(messages: Iterable) -> list[dict]:
    """INFINITY ``Message``/dict list -> chat-template messages (system/user/assistant only)."""
    out: list[dict] = []
    for m in messages:
        role = m["role"] if isinstance(m, dict) else m.role
        content = m["content"] if isinstance(m, dict) else m.content
        if role == "tool":
            role, content = "user", TOOL_PREFIX + content
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported role {role!r}")
        out.append({"role": role, "content": content})
    return out


HYPER_SYSTEM_PROMPT = (
    "You are INFINITY HYPER 5.5, the original model of INFINITY BOT. "
    "Be helpful and concise. If you are not sure about something, say so."
)


def system_prompt_for(info: dict) -> str:
    """Normal-chat system prompt matching the model that is actually loaded."""
    if info.get("family") == "open-model":
        return chat_system_prompt(info.get("product") or "an open model")
    return HYPER_SYSTEM_PROMPT
