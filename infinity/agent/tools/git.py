"""Read-only git inspection tools (status, diff, log).

Mutating git operations go through ``run_command`` so they are risk-classified
(e.g. ``git push`` / ``git reset`` are HIGH and always need approval).
"""

from __future__ import annotations

import subprocess

from ..permissions import Category
from ..redact import redact, scrubbed_env
from .base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult


def _git(ctx: ToolContext, *args: str, limit: int = 30000) -> ToolResult:
    try:
        cp = subprocess.run(
            ["git", "-c", "core.pager=cat", "-c", "color.ui=false", *args],
            cwd=ctx.workspace.root, env=scrubbed_env(), capture_output=True, text=True,
            errors="replace", timeout=30,
        )
    except FileNotFoundError:
        raise ToolError("git is not installed")
    except subprocess.TimeoutExpired:
        raise ToolError("git timed out")
    if cp.returncode != 0:
        return ToolResult(False, redact(cp.stdout), error=redact(cp.stderr.strip()) or f"git exited {cp.returncode}")
    out = redact(cp.stdout)
    if len(out) > limit:
        out = out[:limit] + f"\n... [{len(out) - limit} chars truncated]"
    return ToolResult(True, out or "(no output)")


def git_status(ctx: ToolContext) -> ToolResult:
    return _git(ctx, "status", "--short", "--branch")


def git_diff(ctx: ToolContext, path: str = "", staged: bool = False) -> ToolResult:
    args = ["diff"] + (["--cached"] if staged else [])
    if path:
        args += ["--", str(ctx.workspace.rel(ctx.workspace.resolve(path)))]
    return _git(ctx, *args)


def git_log(ctx: ToolContext, n: int = 10) -> ToolResult:
    return _git(ctx, "log", f"-{max(1, min(n, 100))}", "--oneline", "--decorate")


def register(reg: ToolRegistry) -> None:
    R = Category.READ
    reg.register(Tool("git_status", "Show git status.", R, {}, git_status, True))
    reg.register(Tool("git_diff", "Show git diff (working tree or staged).", R,
                      {"path": (str, False, "limit to path"), "staged": (bool, False, "staged changes")}, git_diff, True))
    reg.register(Tool("git_log", "Show recent commits.", R, {"n": (int, False, "count")}, git_log, True))
