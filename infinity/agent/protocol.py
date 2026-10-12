"""Structured-action protocol between the model and the runtime.

The INFINITY NEURAL CORE has no native tool-calling API, so the runtime uses a
strict text protocol. Each model turn must contain exactly one JSON object:

    {"thought": "...", "plan": ["..."], "tool": "read_file", "args": {"path": "app.py"}}
or
    {"thought": "...", "final": {"status": "completed", "summary": "..."}}

Anything else is rejected with a precise error that is fed back to the model.
The runtime — not the model — decides whether a task counts as *verified*.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

FINAL_STATUSES = {"completed", "partial", "blocked"}


@dataclass
class Action:
    kind: str  # "tool" | "final"
    thought: str = ""
    plan: list[str] = field(default_factory=list)
    tool: str = ""
    args: dict = field(default_factory=dict)
    status: str = ""
    summary: str = ""


class ProtocolError(ValueError):
    pass


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _extract_json_object(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    start = text.find("{")
    if start < 0:
        raise ProtocolError("no JSON object found; reply with one JSON object")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ProtocolError("unterminated JSON object")


def parse_action(text: str) -> Action:
    raw = _extract_json_object(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"invalid JSON: {e.msg} at position {e.pos}")
    if not isinstance(obj, dict):
        raise ProtocolError("top-level value must be an object")
    thought = obj.get("thought", "")
    plan = obj.get("plan", [])
    if not isinstance(thought, str):
        raise ProtocolError("'thought' must be a string")
    if not isinstance(plan, list) or not all(isinstance(p, str) for p in plan):
        raise ProtocolError("'plan' must be a list of strings")
    has_tool = "tool" in obj
    has_final = "final" in obj
    if has_tool == has_final:
        raise ProtocolError("include exactly one of 'tool' or 'final'")
    if has_tool:
        tool = obj["tool"]
        args = obj.get("args", {})
        if not isinstance(tool, str) or not tool:
            raise ProtocolError("'tool' must be a non-empty string")
        if not isinstance(args, dict):
            raise ProtocolError("'args' must be an object")
        return Action("tool", thought, plan, tool=tool, args=args)
    final = obj["final"]
    if not isinstance(final, dict):
        raise ProtocolError("'final' must be an object with 'status' and 'summary'")
    status = final.get("status")
    summary = final.get("summary", "")
    if status not in FINAL_STATUSES:
        raise ProtocolError(f"final.status must be one of {sorted(FINAL_STATUSES)}")
    if not isinstance(summary, str) or not summary.strip():
        raise ProtocolError("final.summary must be a non-empty string")
    return Action("final", thought, plan, status=status, summary=summary)
