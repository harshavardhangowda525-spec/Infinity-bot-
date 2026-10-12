"""Code-intelligence tools backed by the project index (read-only)."""

from __future__ import annotations

from ..permissions import Category
from .base import Tool, ToolContext, ToolRegistry, ToolResult


def _index(ctx: ToolContext):
    idx = ctx.code_index
    idx.update()  # incremental: only changed files are re-parsed
    return idx


def find_symbol(ctx: ToolContext, name: str, limit: int = 20) -> ToolResult:
    rows = _index(ctx).find_symbol(name, max(1, min(limit, 100)))
    if not rows:
        return ToolResult(True, f"no symbol matching {name!r}", data={"matches": []})
    lines = [f"{r['path']}:{r['line']}  {r['kind']:<9} {r['signature'] or r['name']}" + (f"  (in {r['parent']})" if r["parent"] else "") for r in rows]
    return ToolResult(True, "\n".join(lines), data={"matches": rows})


def find_references(ctx: ToolContext, name: str, limit: int = 50) -> ToolResult:
    rows = _index(ctx).references(name, max(1, min(limit, 300)))
    if not rows:
        return ToolResult(True, f"no references to {name!r}", data={"matches": []})
    return ToolResult(True, "\n".join(f"{r['path']}:{r['line']}" for r in rows), data={"matches": rows})


def code_overview(ctx: ToolContext) -> ToolResult:
    idx = _index(ctx)
    s = idx.stats()
    langs = ", ".join(f"{k}: {v}" for k, v in sorted(s["languages"].items()))
    out = f"files {s['files']} ({langs}) · symbols {s['symbols']} · references {s['references']}"
    if s["parse_errors"]:
        out += "\nparse errors:\n" + "\n".join(f"  {k}: {v}" for k, v in list(s["parse_errors"].items())[:10])
    return ToolResult(True, out, data=s)


def register(reg: ToolRegistry) -> None:
    R = Category.READ
    reg.register(Tool("find_symbol", "Find where a function/class/variable is defined (project index).", R,
                      {"name": (str, True, "symbol name or fragment"), "limit": (int, False, "max results")}, find_symbol, True))
    reg.register(Tool("find_references", "Find lines that use a symbol (project index).", R,
                      {"name": (str, True, "exact symbol name"), "limit": (int, False, "max results")}, find_references, True))
    reg.register(Tool("code_overview", "Index statistics: files, languages, symbols, parse errors.", R, {}, code_overview, True))
