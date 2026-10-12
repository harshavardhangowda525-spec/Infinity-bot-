"""Recoverable file checkpoints.

Before the agent modifies, moves, or deletes a file for the first time in a
session, the original bytes (or the fact that it did not exist) are copied
into ``<workspace>/.infinity/checkpoints/<session>/``. ``restore`` puts every
touched path back exactly as it was before the session started.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path


class CheckpointStore:
    def __init__(self, workspace_root: Path, session_id: str) -> None:
        self.root = Path(workspace_root)
        self.dir = self.root / ".infinity" / "checkpoints" / session_id
        self.index_path = self.dir / "index.json"
        self.index: dict[str, dict] = {}
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text())

    def _save_index(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index, indent=1))
        tmp.replace(self.index_path)

    def snapshot(self, path: Path) -> None:
        """Record the pre-session state of ``path`` (first touch only)."""
        rel = str(path.relative_to(self.root))
        if rel in self.index:
            return
        entry: dict = {"time": time.time()}
        if path.is_file():
            blob = self.dir / "blobs" / hashlib.sha256(rel.encode()).hexdigest()
            blob.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, blob)
            entry.update(kind="file", blob=str(blob.relative_to(self.dir)))
        elif path.is_dir():
            blob = self.dir / "trees" / hashlib.sha256(rel.encode()).hexdigest()
            shutil.copytree(path, blob, symlinks=True)
            entry.update(kind="dir", blob=str(blob.relative_to(self.dir)))
        else:
            entry.update(kind="absent")
        self.index[rel] = entry
        self._save_index()

    def touched(self) -> list[str]:
        return sorted(self.index)

    def restore(self) -> list[str]:
        restored = []
        # Shallowest paths first: a directory snapshot may have been taken after
        # a file inside it was already modified, so the file's own (earlier)
        # original must be applied afterwards and win.
        for rel in sorted(self.index, key=lambda r: (len(Path(r).parts), r)):
            entry = self.index[rel]
            target = self.root / rel
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
            if entry["kind"] == "file":
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.dir / entry["blob"], target)
            elif entry["kind"] == "dir":
                shutil.copytree(self.dir / entry["blob"], target, symlinks=True)
            restored.append(rel)
        return restored


def list_sessions(workspace_root: Path) -> list[dict]:
    base = Path(workspace_root) / ".infinity" / "checkpoints"
    out = []
    if base.exists():
        for d in sorted(base.iterdir()):
            idx = d / "index.json"
            if idx.exists():
                out.append({"session": d.name, "files": len(json.loads(idx.read_text()))})
    return out
