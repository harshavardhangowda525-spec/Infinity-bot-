"""Filesystem tools: list, find, search, read, write, edit, move, mkdir, delete.

Edits are guarded against stale writes: a file must have been read by the
agent (and unchanged on disk since) before it can be overwritten or edited.
Every modification is checkpointed first and returns a unified diff.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from ..permissions import BINARY_EXTS, IGNORED_DIRS, PROTECTED_DIRS, Category, is_sensitive_path
from .base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult, require

MAX_READ_BYTES = 400_000
MAX_SEARCH_FILE_BYTES = 1_000_000


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_readable(ctx: ToolContext, path: Path) -> None:
    rel = ctx.workspace.rel(path)
    if is_sensitive_path(rel) and not ctx.policy.allow_sensitive_reads:
        raise PermissionError(f"{rel} looks like a secrets file; reading it requires explicit authorization")


def _check_writable(ctx: ToolContext, path: Path) -> None:
    rel = ctx.workspace.rel(path)
    if rel.parts and rel.parts[0] in PROTECTED_DIRS:
        raise PermissionError(f"{rel} is inside a protected directory")
    if is_sensitive_path(rel):
        raise PermissionError(f"{rel} looks like a secrets file; the agent does not write secrets files")


def _walk(ctx: ToolContext, start: Path, max_files: int = 20000):
    count = 0
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info"))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if p.is_symlink():
                try:
                    ctx.workspace.resolve(p)
                except PermissionError:
                    continue
            yield p
            count += 1
            if count >= max_files:
                return


def _is_text_candidate(p: Path) -> bool:
    return p.suffix.lower() not in BINARY_EXTS


def _diff(rel: str, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=f"a/{rel}", tofile=f"b/{rel}")
    )


def _syntax_check(path: Path, text: str) -> str | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".py":
            compile(text, str(path), "exec")
        elif suffix == ".json":
            json.loads(text)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    except json.JSONDecodeError as e:
        return f"JSONDecodeError: {e.msg} (line {e.lineno})"
    return None


# ------------------------------------------------------------------ read tools
def list_dir(ctx: ToolContext, path: str = ".", depth: int = 2) -> ToolResult:
    root = ctx.workspace.resolve(path)
    if not root.is_dir():
        raise ToolError(f"not a directory: {path}")
    depth = max(1, min(depth, 6))
    lines: list[str] = []

    def walk(d: Path, level: int) -> None:
        try:
            entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as e:
            lines.append("  " * level + f"<unreadable: {e}>")
            return
        for e in entries:
            if e.name in IGNORED_DIRS:
                continue
            if len(lines) >= 1000:
                return
            if e.is_dir():
                lines.append("  " * level + e.name + "/")
                if level + 1 < depth:
                    walk(e, level + 1)
            else:
                lines.append("  " * level + e.name)

    walk(root, 0)
    return ToolResult(True, "\n".join(lines) or "(empty)", data={"entries": len(lines)})


def find_files(ctx: ToolContext, pattern: str, path: str = ".") -> ToolResult:
    root = ctx.workspace.resolve(path)
    matches = []
    for p in _walk(ctx, root):
        rel = str(ctx.workspace.rel(p))
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p.name, pattern):
            matches.append(rel)
            if len(matches) >= 500:
                break
    return ToolResult(True, "\n".join(matches) or "(no matches)", data={"matches": matches})


def search_text(ctx: ToolContext, query: str, path: str = ".", glob: str = "*", regex: bool = False,
                max_results: int = 100) -> ToolResult:
    root = ctx.workspace.resolve(path)
    try:
        rx = re.compile(query if regex else re.escape(query))
    except re.error as e:
        raise ToolError(f"invalid regex: {e}")
    results: list[str] = []
    files = [root] if root.is_file() else _walk(ctx, root)
    for p in files:
        rel = ctx.workspace.rel(p)
        if not (fnmatch.fnmatch(p.name, glob) or fnmatch.fnmatch(str(rel), glob)):
            continue
        if is_sensitive_path(rel) or not _is_text_candidate(p):
            continue
        try:
            if p.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                results.append(f"{rel}:{i}: {line.strip()[:200]}")
                if len(results) >= max_results:
                    return ToolResult(True, "\n".join(results) + "\n(result limit reached)", data={"count": len(results)})
    return ToolResult(True, "\n".join(results) or "(no matches)", data={"count": len(results)})


def read_file(ctx: ToolContext, path: str, start_line: int = 1, end_line: int = 0) -> ToolResult:
    p = ctx.workspace.resolve(path)
    if not p.is_file():
        raise ToolError(f"not a file: {path}")
    _check_readable(ctx, p)
    if p.stat().st_size > MAX_READ_BYTES and start_line == 1 and end_line == 0:
        raise ToolError(f"{path} is large ({p.stat().st_size} bytes); read a line range")
    raw = p.read_bytes()
    if b"\x00" in raw[:8192]:
        raise ToolError(f"{path} appears to be binary")
    text = raw.decode("utf-8", errors="replace")
    rel = str(ctx.workspace.rel(p))
    ctx.read_hashes[rel] = _hash(raw)
    lines = text.splitlines()
    start = max(1, start_line)
    end = len(lines) if end_line <= 0 else min(end_line, len(lines))
    width = len(str(end))
    body = "\n".join(f"{i:>{width}}| {lines[i - 1]}" for i in range(start, end + 1))
    return ToolResult(True, body or "(empty file)", data={"path": rel, "lines": len(lines), "range": [start, end]})


# ----------------------------------------------------------------- write tools
def _ensure_fresh(ctx: ToolContext, p: Path, rel: str) -> str:
    current = p.read_bytes()
    known = ctx.read_hashes.get(rel)
    if known is None:
        raise ToolError(f"read {rel} before modifying it")
    if known != _hash(current):
        raise ToolError(f"{rel} changed on disk since it was read; read it again before editing (stale edit)")
    return current.decode("utf-8", errors="replace")


def _commit_write(ctx: ToolContext, p: Path, rel: str, new_text: str) -> None:
    ctx.checkpoints.snapshot(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.infinity-tmp")
    tmp.write_text(new_text, encoding="utf-8")
    if p.exists():
        shutil.copymode(p, tmp)
    os.replace(tmp, p)
    ctx.read_hashes[rel] = _hash(new_text.encode("utf-8"))
    ctx.note_modified(rel)


def write_file(ctx: ToolContext, path: str, content: str) -> ToolResult:
    p = ctx.workspace.resolve(path)
    _check_writable(ctx, p)
    rel = str(ctx.workspace.rel(p))
    if p.is_dir():
        raise ToolError(f"{rel} is a directory")
    old = _ensure_fresh(ctx, p, rel) if p.exists() else ""
    diff = _diff(rel, old, content)
    action = "overwrite" if p.exists() else "create"
    require(ctx, "write_file", Category.WRITE, f"{action} {rel} ({len(content)} chars)", diff=diff)
    _commit_write(ctx, p, rel, content)
    warn = _syntax_check(p, content)
    out = f"{action}d {rel}" + (f"\nWARNING syntax check failed: {warn}" if warn else "")
    return ToolResult(True, out, data={"path": rel, "action": action, "diff": diff, "syntax_error": warn})


def edit_file(ctx: ToolContext, path: str, old: str = "", new: str = "", replace_all: bool = False,
              edits: list | None = None) -> ToolResult:
    p = ctx.workspace.resolve(path)
    _check_writable(ctx, p)
    rel = str(ctx.workspace.rel(p))
    if not p.is_file():
        raise ToolError(f"not a file: {rel} (use write_file to create it)")
    text = _ensure_fresh(ctx, p, rel)
    pairs = edits if edits is not None else [{"old": old, "new": new, "replace_all": replace_all}]
    updated = text
    for i, e in enumerate(pairs):
        if not isinstance(e, dict) or not isinstance(e.get("old"), str) or not isinstance(e.get("new"), str):
            raise ToolError(f"edit #{i} must be an object with string 'old' and 'new'")
        o, n = e["old"], e["new"]
        if o == "":
            raise ToolError(f"edit #{i}: 'old' must not be empty")
        count = updated.count(o)
        if count == 0:
            raise ToolError(f"edit #{i}: text to replace was not found in {rel}")
        if count > 1 and not e.get("replace_all"):
            raise ToolError(f"edit #{i}: text occurs {count} times in {rel}; add context or set replace_all")
        updated = updated.replace(o, n) if e.get("replace_all") else updated.replace(o, n, 1)
    if updated == text:
        return ToolResult(True, "no changes", data={"path": rel, "diff": ""})
    diff = _diff(rel, text, updated)
    require(ctx, "edit_file", Category.WRITE, f"edit {rel}", diff=diff)
    _commit_write(ctx, p, rel, updated)
    warn = _syntax_check(p, updated)
    out = diff + (f"\nWARNING syntax check failed: {warn}" if warn else "")
    return ToolResult(True, out, data={"path": rel, "diff": diff, "syntax_error": warn})


def make_dir(ctx: ToolContext, path: str) -> ToolResult:
    p = ctx.workspace.resolve(path)
    _check_writable(ctx, p)
    rel = str(ctx.workspace.rel(p))
    if p.exists():
        return ToolResult(True, f"{rel} already exists")
    require(ctx, "make_dir", Category.WRITE, f"create directory {rel}")
    ctx.checkpoints.snapshot(p)
    p.mkdir(parents=True)
    ctx.note_modified(rel)
    return ToolResult(True, f"created {rel}/")


def move_path(ctx: ToolContext, src: str, dst: str) -> ToolResult:
    s = ctx.workspace.resolve(src)
    d = ctx.workspace.resolve(dst)
    _check_writable(ctx, s)
    _check_writable(ctx, d)
    if not s.exists():
        raise ToolError(f"source does not exist: {src}")
    if d.exists():
        raise ToolError(f"destination already exists: {dst}")
    rs, rd = str(ctx.workspace.rel(s)), str(ctx.workspace.rel(d))
    require(ctx, "move_path", Category.WRITE, f"move {rs} -> {rd}")
    ctx.checkpoints.snapshot(s)
    ctx.checkpoints.snapshot(d)
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(s), str(d))
    ctx.note_modified(rs)
    ctx.note_modified(rd)
    return ToolResult(True, f"moved {rs} -> {rd}")


def delete_path(ctx: ToolContext, path: str, recursive: bool = False) -> ToolResult:
    p = ctx.workspace.resolve(path)
    if p == ctx.workspace.root:
        raise PermissionError("refusing to delete the workspace root")
    _check_writable(ctx, p)
    rel = str(ctx.workspace.rel(p))
    if not p.exists() and not p.is_symlink():
        raise ToolError(f"does not exist: {rel}")
    if p.is_dir() and not recursive:
        raise ToolError(f"{rel} is a directory; set recursive=true to delete it")
    require(ctx, "delete_path", Category.DESTRUCTIVE, f"delete {rel}{'/ (recursive)' if p.is_dir() else ''}")
    ctx.checkpoints.snapshot(p)
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    else:
        p.unlink()
    ctx.note_modified(rel)
    return ToolResult(True, f"deleted {rel} (recoverable via checkpoint session {ctx.session_id})")


def register(reg: ToolRegistry) -> None:
    R, W, D = Category.READ, Category.WRITE, Category.DESTRUCTIVE
    reg.register(Tool("list_dir", "List a directory tree (ignores dependency/build folders).", R,
                      {"path": (str, False, "directory, default ."), "depth": (int, False, "1-6, default 2")}, list_dir, True))
    reg.register(Tool("find_files", "Find files by glob pattern.", R,
                      {"pattern": (str, True, "glob, e.g. *.py or src/**/*.ts"), "path": (str, False, "start dir")}, find_files, True))
    reg.register(Tool("search_text", "Search file contents.", R,
                      {"query": (str, True, "text or regex"), "path": (str, False, "dir or file"),
                       "glob": (str, False, "file filter"), "regex": (bool, False, "treat query as regex"),
                       "max_results": (int, False, "default 100")}, search_text, True))
    reg.register(Tool("read_file", "Read a file (optionally a line range). Required before editing.", R,
                      {"path": (str, True, "file"), "start_line": (int, False, "1-based"), "end_line": (int, False, "inclusive, 0=end")},
                      read_file, True))
    reg.register(Tool("write_file", "Create a file, or overwrite one you have read.", W,
                      {"path": (str, True, "file"), "content": (str, True, "full file content")}, write_file))
    reg.register(Tool("edit_file", "Replace exact text in a file you have read.", W,
                      {"path": (str, True, "file"), "old": (str, False, "exact text to replace"), "new": (str, False, "replacement"),
                       "replace_all": (bool, False, "replace every occurrence"),
                       "edits": (list, False, "list of {old,new,replace_all} applied in order")}, edit_file))
    reg.register(Tool("make_dir", "Create a directory.", W, {"path": (str, True, "directory")}, make_dir))
    reg.register(Tool("move_path", "Move or rename a file/directory.", W,
                      {"src": (str, True, "source"), "dst": (str, True, "destination (must not exist)")}, move_path))
    reg.register(Tool("delete_path", "Delete a file/directory (needs approval; recoverable).", D,
                      {"path": (str, True, "path"), "recursive": (bool, False, "required for directories")}, delete_path))
