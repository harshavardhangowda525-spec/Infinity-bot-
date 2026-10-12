"""INFINITY data pipeline: provenance-tracked corpus -> filtered, deduped,
split, tokenized uint16/uint32 token files.

Every source must be declared in a manifest with its license. Files whose
license is not in ``ALLOWED_LICENSES`` are rejected, documents containing
likely secrets are dropped, and exact duplicates are removed before
splitting. A provenance report is written next to the prepared data.
"""

from __future__ import annotations

import glob
import hashlib
import json
import random
import re
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tokenizer import BPETokenizer

ALLOWED_LICENSES = {
    "MIT",
    "ISC",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "PSF-2.0",
    "CC0-1.0",
    "CC-BY-4.0",
    "Public-Domain",
    "Project-Original",  # written for this repository, MIT with the repo
}

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),  # GitHub tokens
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),  # common API key prefix
    re.compile(r"\bxox[abpr]-[A-Za-z0-9\-]{10,}\b"),  # Slack
    re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
]
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def contains_secret(text: str) -> bool:
    return any(p.search(text) for p in SECRET_PATTERNS)


def scrub_pii(text: str) -> str:
    return EMAIL_RE.sub("<email>", text)


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(lines).strip("\n") + "\n"


@dataclass
class Document:
    source: str
    path: str
    license: str
    text: str


def _expand(pattern: str, base: Path) -> list[Path]:
    pattern = pattern.replace("{stdlib}", sysconfig.get_paths()["stdlib"])
    p = Path(pattern)
    if not p.is_absolute():
        pattern = str(base / pattern)
    return sorted(Path(x) for x in glob.glob(pattern, recursive=True) if Path(x).is_file())


def load_documents(manifest_path: str | Path, max_file_bytes: int = 200_000) -> tuple[list[Document], dict]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent.parent  # manifest lives in <repo>/data/
    docs: list[Document] = []
    report: dict = {"sources": [], "rejected": []}
    for src in manifest["sources"]:
        lic = src["license"]
        entry = {"name": src["name"], "license": lic, "origin": src.get("origin", ""), "files": 0, "bytes": 0}
        if lic not in ALLOWED_LICENSES:
            report["rejected"].append({"name": src["name"], "reason": f"license {lic!r} not allowed"})
            continue
        limit = src.get("max_files")
        for pattern in src["include"]:
            for path in _expand(pattern, base):
                if limit is not None and entry["files"] >= limit:
                    break
                if path.stat().st_size > max_file_bytes:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                docs.append(Document(src["name"], str(path), lic, text))
                entry["files"] += 1
                entry["bytes"] += len(text.encode())
        report["sources"].append(entry)
    return docs, report


def clean_and_dedupe(docs: list[Document], min_chars: int = 40) -> tuple[list[Document], dict]:
    seen: set[str] = set()
    kept: list[Document] = []
    stats = {"input": len(docs), "secret_dropped": 0, "duplicate_dropped": 0, "too_short": 0}
    for d in docs:
        text = normalize(d.text)
        if len(text) < min_chars:
            stats["too_short"] += 1
            continue
        if contains_secret(text):
            stats["secret_dropped"] += 1
            continue
        text = scrub_pii(text)
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            stats["duplicate_dropped"] += 1
            continue
        seen.add(h)
        kept.append(Document(d.source, d.path, d.license, text))
    stats["kept"] = len(kept)
    return kept, stats


def split_documents(docs: list[Document], seed: int, val_frac: float = 0.05, test_frac: float = 0.05):
    """Document-level split so no file leaks across splits."""
    order = sorted(docs, key=lambda d: hashlib.sha256(d.path.encode()).hexdigest())
    rng = random.Random(seed)
    rng.shuffle(order)
    n = len(order)
    n_val = max(1, int(n * val_frac)) if n >= 3 else 0
    n_test = max(1, int(n * test_frac)) if n >= 3 else 0
    return {
        "test": order[:n_test],
        "val": order[n_test : n_test + n_val],
        "train": order[n_test + n_val :],
    }


def tokenize_split(docs: list[Document], tok: BPETokenizer) -> np.ndarray:
    dtype = np.uint16 if tok.vocab_size <= 65535 else np.uint32
    ids: list[int] = []
    for d in docs:
        ids.extend(tok.encode(d.text, add_bos=True, add_eos=True))
    return np.asarray(ids, dtype=dtype)


def prepare(manifest: str | Path, out_dir: str | Path, vocab_size: int, seed: int, verbose: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs, provenance = load_documents(manifest)
    docs, clean_stats = clean_and_dedupe(docs)
    if not docs:
        raise RuntimeError("no documents survived filtering; check the manifest")
    splits = split_documents(docs, seed)
    if verbose:
        print(f"documents: {clean_stats}")
        print("training tokenizer on train split only ...")
    tok = BPETokenizer.train((d.text for d in splits["train"]), vocab_size=vocab_size, verbose=verbose)
    tok.save(out / "tokenizer.json")
    summary = {
        "tokenizer": {"vocab_size": tok.vocab_size, "fingerprint": tok.fingerprint(), "merges": len(tok.merges)},
        "cleaning": clean_stats,
        "provenance": provenance,
        "splits": {},
        "python": sys.version.split()[0],
    }
    for name, split_docs in splits.items():
        arr = tokenize_split(split_docs, tok)
        arr.tofile(out / f"{name}.bin")
        summary["splits"][name] = {
            "documents": len(split_docs),
            "tokens": int(arr.size),
            "dtype": str(arr.dtype),
            "bytes": int(sum(len(d.text.encode()) for d in split_docs)),
        }
    (out / "prepared.json").write_text(json.dumps(summary, indent=2))
    (out / "files.json").write_text(
        json.dumps({k: [{"path": d.path, "license": d.license, "source": d.source} for d in v] for k, v in splits.items()}, indent=1)
    )
    return summary


class TokenDataset:
    """Random fixed-length windows from a flat token file (memory-mapped)."""

    def __init__(self, path: str | Path, dtype: str, context_length: int) -> None:
        self.data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        self.context_length = context_length
        if len(self.data) <= context_length + 1:
            raise ValueError(f"{path}: {len(self.data)} tokens is too few for context {context_length}")

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, generator):
        import torch

        hi = len(self.data) - self.context_length - 1
        starts = torch.randint(0, hi, (batch_size,), generator=generator).tolist()
        x = np.stack([self.data[s : s + self.context_length] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1 : s + 1 + self.context_length] for s in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)
