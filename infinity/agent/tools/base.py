"""Tool registry, structured results, and argument validation."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..checkpoints import CheckpointStore
from ..permissions import ApprovalRequest, Category, PermissionPolicy, Workspace
from ..redact import redact


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str | None = None
    data: dict = field(default_factory=dict)
    # filled by the registry
    tool: str = ""
    decision: str = ""
    duration_ms: int = 0

    def for_model(self, limit: int = 4000) -> str:
        head = f"[{self.tool}] {'ok' if self.ok else 'error'}"
        body = self.output if self.ok else (self.error or "") + ("\n" + self.output if self.output else "")
        body = redact(body)
        if len(body) > limit:
            keep = limit // 2
            body = body[:keep] + f"\n... [{len(body) - limit} chars truncated] ...\n" + body[-keep:]
        return f"{head}\n{body}".rstrip()


@dataclass
class ToolContext:
    workspace: Workspace
    policy: PermissionPolicy
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    cancel: threading.Event = field(default_factory=threading.Event)
    on_output: Callable[[str], None] | None = None  # streaming hook for long commands
    command_timeout: float = 120.0
    max_output_chars: int = 20000
    # state the runtime uses for stale-edit detection and verification tracking
    read_hashes: dict[str, str] = field(default_factory=dict)
    modified: list[str] = field(default_factory=list)
    step: int = 0
    last_edit_step: int = -1
    checks: list[dict] = field(default_factory=list)
    _checkpoints: CheckpointStore | None = None
    _index: object | None = None

    @property
    def checkpoints(self) -> CheckpointStore:
        if self._checkpoints is None:
            self._checkpoints = CheckpointStore(self.workspace.root, self.session_id)
        return self._checkpoints

    @property
    def code_index(self):
        if self._index is None:
            from ..index import CodeIndex

            self._index = CodeIndex(self.workspace.root)
        return self._index

    def note_modified(self, rel: str) -> None:
        if rel not in self.modified:
            self.modified.append(rel)
        self.last_edit_step = self.step


ArgSpec = dict[str, tuple[type | tuple[type, ...], bool, str]]  # name -> (type, required, description)


@dataclass
class Tool:
    name: str
    description: str
    category: Category
    args: ArgSpec
    fn: Callable[..., ToolResult]
    read_only: bool = False

    def signature(self) -> str:
        parts = []
        for name, (typ, req, desc) in self.args.items():
            tname = typ.__name__ if isinstance(typ, type) else "|".join(t.__name__ for t in typ)
            parts.append(f'"{name}": {tname}{"" if req else "?"}  # {desc}')
        return f"{self.name}({', '.join(parts)}) — {self.description}"


class ToolError(Exception):
    pass


def validate_args(tool: Tool, args: Any) -> dict:
    if not isinstance(args, dict):
        raise ToolError("args must be a JSON object")
    unknown = set(args) - set(tool.args)
    if unknown:
        raise ToolError(f"unknown argument(s) for {tool.name}: {sorted(unknown)}")
    out = {}
    for name, (typ, required, _) in tool.args.items():
        if name not in args or args[name] is None:
            if required:
                raise ToolError(f"missing required argument '{name}' for {tool.name}")
            continue
        val = args[name]
        types = typ if isinstance(typ, tuple) else (typ,)
        if isinstance(val, bool) and bool not in types:
            ok = False  # JSON true/false is not an int
        else:
            ok = isinstance(val, types) or (float in types and isinstance(val, int))
        if not ok:
            raise ToolError(f"argument '{name}' for {tool.name} must be {typ}")
        out[name] = val
    return out


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def names(self, read_only: bool = False) -> list[str]:
        return [n for n, t in self.tools.items() if t.read_only or not read_only]

    def execute(self, name: str, args: Any, ctx: ToolContext) -> ToolResult:
        start = time.monotonic()
        tool = self.tools.get(name)
        if tool is None:
            res = ToolResult(False, error=f"unknown tool: {name}")
        else:
            try:
                clean = validate_args(tool, args)
                res = tool.fn(ctx, **clean)
            except ToolError as e:
                res = ToolResult(False, error=str(e))
            except PermissionError as e:
                res = ToolResult(False, error=f"permission denied: {e}", decision="deny")
            except Exception as e:  # tools must never crash the agent loop
                res = ToolResult(False, error=f"{type(e).__name__}: {e}")
        res.tool = name
        res.duration_ms = int((time.monotonic() - start) * 1000)
        return res


def require(ctx: ToolContext, tool: str, category: Category, summary: str, **kw) -> str:
    """Ask the permission policy; raise PermissionError when not allowed."""
    ok, why = ctx.policy.authorize(ApprovalRequest(tool=tool, category=category, summary=summary, **kw))
    if not ok:
        raise PermissionError(why)
    return why
