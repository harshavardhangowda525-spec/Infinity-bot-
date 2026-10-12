"""Project code index: files, symbol definitions and references.

Stored in ``<workspace>/.infinity/index.db`` (SQLite) and updated
incrementally: only files whose size/mtime changed are re-parsed.

- Python: parsed with ``ast`` (functions, async functions, classes, methods,
  module-level assignments, imports; references = Name/Attribute loads).
- JavaScript/TypeScript: regex extraction of functions, classes, arrow
  functions, interfaces/types and imports (no full parser; documented).
- Other text files: indexed for path search only.

Secrets files, ignored directories, binaries and huge files are skipped.
"""

from __future__ import annotations

import ast
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

from .permissions import BINARY_EXTS, IGNORED_DIRS, is_sensitive_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime REAL, lang TEXT, indexed_at REAL);
CREATE TABLE IF NOT EXISTS symbols (
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL,
  line INTEGER NOT NULL, end_line INTEGER, parent TEXT, signature TEXT);
CREATE TABLE IF NOT EXISTS refs (name TEXT NOT NULL, path TEXT NOT NULL, line INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS imports (path TEXT NOT NULL, module TEXT NOT NULL, line INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS idx_sym_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_sym_path ON symbols(path);
CREATE INDEX IF NOT EXISTS idx_refs_name ON refs(name);
CREATE INDEX IF NOT EXISTS idx_refs_path ON refs(path);
"""
LANGS = {".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".ts": "typescript", ".tsx": "typescript"}
TEXT_EXTS = {".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".cfg", ".ini", ".html", ".css", ".sh", ".sql", ".rs", ".go", ".java", ".kt", ".dart", ".c", ".h", ".cpp"}
MAX_BYTES = 500_000
MAX_FILES = 20_000

JS_PATTERNS = [
    ("function", re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*\(([^)]*)\)")),
    ("class", re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)")),
    ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>")),
    ("interface", re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)")),
    ("type", re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*=")),
    ("variable", re.compile(r"^\s*export\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=")),
]
JS_IMPORT = re.compile(r"""^\s*import\s+(?:[^'"]*from\s+)?['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\)""")
IDENT = re.compile(r"[A-Za-z_$][\w$]*")
JS_KEYWORDS = set("if else for while return const let var function class new this import export from default async await try catch finally throw switch case break continue typeof instanceof in of null undefined true false void delete yield extends super static get set interface type enum implements public private protected readonly".split())


def _py_extract(text: str):
    tree = ast.parse(text)
    symbols, refs, imports = [], [], []

    def sig(node) -> str:
        try:
            return f"{node.name}({ast.unparse(node.args)})"
        except Exception:
            return node.name

    def visit(node, parent: str | None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if parent and parent[0].isupper() else "function"
                symbols.append((child.name, kind, child.lineno, getattr(child, "end_lineno", None), parent, sig(child)))
                visit(child, child.name)
            elif isinstance(child, ast.ClassDef):
                symbols.append((child.name, "class", child.lineno, getattr(child, "end_lineno", None), parent, f"class {child.name}"))
                visit(child, child.name)
            elif isinstance(child, (ast.Assign, ast.AnnAssign)) and parent is None:
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        symbols.append((t.id, "variable", child.lineno, getattr(child, "end_lineno", None), None, t.id))
                visit(child, parent)
            elif isinstance(child, ast.Import):
                imports.extend((a.name, child.lineno) for a in child.names)
            elif isinstance(child, ast.ImportFrom):
                imports.append(("." * child.level + (child.module or ""), child.lineno))
                refs.extend((a.name, child.lineno) for a in child.names if a.name != "*")  # imported names are uses
            else:
                visit(child, parent)

    visit(tree, None)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            refs.append((node.id, node.lineno))
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            refs.append((node.attr, node.lineno))
    return symbols, refs, imports


def _js_extract(text: str):
    symbols, refs, imports = [], [], []
    for i, line in enumerate(text.splitlines(), 1):
        for kind, pat in JS_PATTERNS:
            m = pat.match(line)
            if m:
                name = m.group(1)
                signature = f"{name}({m.group(2)})" if kind == "function" and m.lastindex and m.lastindex >= 2 else line.strip()[:120]
                symbols.append((name, kind, i, None, None, signature))
                break
        im = JS_IMPORT.search(line)
        if im:
            imports.append((im.group(1) or im.group(2), i))
        stripped = re.sub(r"(['\"`]).*?\1", "", line.split("//", 1)[0])
        for tok in IDENT.findall(stripped):
            if tok not in JS_KEYWORDS:
                refs.append((tok, i))
    return symbols, refs, imports


class CodeIndex:
    def __init__(self, root: str | Path, db_path: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        path = Path(db_path) if db_path else self.root / ".infinity" / "index.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.errors: dict[str, str] = {}

    # ------------------------------------------------------------- update
    def _candidates(self):
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info"))
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                rel = p.relative_to(self.root)
                ext = p.suffix.lower()
                if ext in BINARY_EXTS or is_sensitive_path(rel) or p.is_symlink():
                    continue
                if ext not in LANGS and ext not in TEXT_EXTS:
                    continue
                yield p, rel
                count += 1
                if count >= MAX_FILES:
                    return

    def update(self) -> dict:
        """Incrementally re-index changed files; drop deleted ones."""
        t0 = time.monotonic()
        seen, changed = set(), 0
        with self.lock:
            known = {r["path"]: (r["size"], r["mtime"]) for r in self.db.execute("SELECT path, size, mtime FROM files")}
            for p, rel in self._candidates():
                key = str(rel)
                seen.add(key)
                st = p.stat()
                if st.st_size > MAX_BYTES:
                    continue
                if known.get(key) == (st.st_size, st.st_mtime):
                    continue
                self._index_file(p, key, st)
                changed += 1
            removed = [k for k in known if k not in seen]
            for k in removed:
                self._drop(k)
            self.db.commit()
        return {"changed": changed, "removed": len(removed), "files": len(seen), "seconds": round(time.monotonic() - t0, 3)}

    def _drop(self, key: str) -> None:
        for table in ("files", "symbols", "refs", "imports"):
            self.db.execute(f"DELETE FROM {table} WHERE path = ?", (key,))

    def _index_file(self, p: Path, key: str, st) -> None:
        self._drop(key)
        lang = LANGS.get(p.suffix.lower(), "text")
        self.db.execute("INSERT INTO files VALUES (?,?,?,?,?)", (key, st.st_size, st.st_mtime, lang, time.time()))
        if lang == "text":
            return
        try:
            text = p.read_text(encoding="utf-8")
            symbols, refs, imports = _py_extract(text) if lang == "python" else _js_extract(text)
            self.errors.pop(key, None)
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            self.errors[key] = f"{type(e).__name__}: {e}"
            return
        self.db.executemany("INSERT INTO symbols (name, kind, path, line, end_line, parent, signature) VALUES (?,?,?,?,?,?,?)",
                            [(n, k, key, l, e, par, s) for n, k, l, e, par, s in symbols])
        self.db.executemany("INSERT INTO refs VALUES (?,?,?)", [(n, key, l) for n, l in set(refs)])
        self.db.executemany("INSERT INTO imports VALUES (?,?,?)", [(key, m, l) for m, l in imports])

    # -------------------------------------------------------------- query
    def find_symbol(self, query: str, limit: int = 30) -> list[dict]:
        with self.lock:
            exact = [dict(r) for r in self.db.execute(
                "SELECT name, kind, path, line, end_line, parent, signature FROM symbols WHERE name = ? ORDER BY path, line LIMIT ?", (query, limit))]
            if len(exact) >= limit:
                return exact
            fuzzy = [dict(r) for r in self.db.execute(
                "SELECT name, kind, path, line, end_line, parent, signature FROM symbols WHERE name LIKE ? AND name != ? ORDER BY length(name), path LIMIT ?",
                (f"%{query}%", query, limit - len(exact)))]
            return exact + fuzzy

    def references(self, name: str, limit: int = 100) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT path, line FROM refs WHERE name = ? ORDER BY path, line LIMIT ?", (name, limit))]

    def importers(self, module_fragment: str, limit: int = 50) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT path, module, line FROM imports WHERE module LIKE ? ORDER BY path LIMIT ?",
                                                     (f"%{module_fragment}%", limit))]

    def relevant(self, task: str, limit: int = 8) -> list[dict]:
        """Rank symbols/files by overlap with identifiers mentioned in the task text."""
        words = {w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", task)}
        lowered = {w.lower() for w in words}
        scores: dict[tuple, float] = {}
        with self.lock:
            for w in words:
                for r in self.db.execute("SELECT name, kind, path, line, signature FROM symbols WHERE name = ? LIMIT 20", (w,)):
                    k = (r["path"], r["name"], r["line"], r["kind"], r["signature"])
                    scores[k] = scores.get(k, 0) + 3
            for r in self.db.execute("SELECT path FROM files"):
                stem = Path(r["path"]).stem.lower()
                if stem in lowered or any(part.lower() in lowered for part in Path(r["path"]).parts):
                    k = (r["path"], "", 0, "file", r["path"])
                    scores[k] = scores.get(k, 0) + 2
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][2]))
        return [{"path": k[0], "name": k[1], "line": k[2], "kind": k[3], "signature": k[4], "score": s} for k, s in ranked[:limit]]

    def stats(self) -> dict:
        with self.lock:
            q = lambda sql: self.db.execute(sql).fetchone()[0]
            langs = {r[0]: r[1] for r in self.db.execute("SELECT lang, COUNT(*) FROM files GROUP BY lang")}
            return {"files": q("SELECT COUNT(*) FROM files"), "symbols": q("SELECT COUNT(*) FROM symbols"),
                    "references": q("SELECT COUNT(*) FROM refs"), "languages": langs, "parse_errors": dict(self.errors)}

    def close(self) -> None:
        self.db.close()
