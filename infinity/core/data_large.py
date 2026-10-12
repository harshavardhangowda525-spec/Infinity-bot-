"""Large-corpus data pipeline (hundreds of MB to TB scale).

Pass 1 (streaming, bounded memory): read documents from manifest sources
(files and/or license-tagged JSONL), normalise, filter secrets, scrub e-mail
addresses, deduplicate by sha256, assign a split by hashing the document id
(stable, no file leaks across splits) and write cleaned text into raw JSONL
shards. A capped sample of the train split is kept for tokenizer training.

Pass 2: train the BPE tokenizer on the sample only, then tokenize the raw shards
in parallel worker processes into ``{split}_{k:05d}.bin`` token shards.

``fetch_hf_parquet`` downloads a public Hugging Face dataset's Parquet files and
keeps only rows whose per-file licence is permissive, recording provenance.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import sysconfig
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np

from .data import ALLOWED_LICENSES, contains_secret, normalize, scrub_pii
from .tokenizer import BPETokenizer

# Licence spellings used by public code datasets -> our allow-list names.
LICENSE_ALIASES = {
    "mit": "MIT", "apache-2.0": "Apache-2.0", "bsd-2-clause": "BSD-2-Clause", "bsd-3-clause": "BSD-3-Clause",
    "cc0-1.0": "CC0-1.0", "isc": "ISC", "unlicense": "Public-Domain", "psf-2.0": "PSF-2.0", "cc-by-4.0": "CC-BY-4.0",
}


def canonical_license(name: str | None) -> str | None:
    if not name:
        return None
    if name in ALLOWED_LICENSES:
        return name
    return LICENSE_ALIASES.get(name.strip().lower())


def _iter_source(src: dict, base: Path) -> Iterator[tuple[str, str, str]]:
    """Yield (doc_id, license, text) for one manifest source."""
    kind = src.get("type", "files")
    for pattern in src["include"]:
        pattern = pattern.replace("{stdlib}", sysconfig.get_paths()["stdlib"])
        full = pattern if Path(pattern).is_absolute() else str(base / pattern)
        for path in sorted(glob.glob(full, recursive=True)):
            p = Path(path)
            if not p.is_file():
                continue
            if kind == "files":
                lic = canonical_license(src["license"])
                if lic is None:
                    return
                try:
                    yield str(p), lic, p.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
            elif kind == "jsonl":
                tf, lf = src.get("text_field", "text"), src.get("license_field", "license")
                with open(p, encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        lic = canonical_license(rec.get(lf))
                        text = rec.get(tf)
                        if lic is None or not isinstance(text, str):
                            continue
                        yield rec.get("id") or f"{p}:{i}", lic, text
            else:
                raise ValueError(f"unknown source type {kind!r}")


def _split_for(doc_id: str, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"


def _encode_shard(args: tuple[str, str, str]) -> tuple[str, int]:
    raw_path, tok_path, out_path = args
    tok = BPETokenizer.load(tok_path)
    dtype = np.uint16 if tok.vocab_size <= 65535 else np.uint32
    ids: list[int] = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            ids.extend(tok.encode(json.loads(line)["text"], add_bos=True, add_eos=True))
    np.asarray(ids, dtype=dtype).tofile(out_path)
    return out_path, len(ids)


def prepare_large(manifest: str | Path, out_dir: str | Path, vocab_size: int, seed: int = 1337,
                  val_frac: float = 0.005, test_frac: float = 0.005, shard_bytes: int = 64_000_000,
                  tokenizer_sample_bytes: int = 50_000_000, workers: int = 0, max_doc_bytes: int = 1_000_000,
                  log=print) -> dict:
    manifest = Path(manifest)
    cfg = json.loads(manifest.read_text())
    base = manifest.parent.parent
    out = Path(out_dir)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    seen: set[bytes] = set()
    stats = {"input": 0, "secret_dropped": 0, "duplicate_dropped": 0, "too_short": 0, "too_large": 0, "kept": 0}
    provenance: dict[str, dict] = {}
    writers: dict[str, tuple] = {}
    raw_shards: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    sample: list[str] = []
    sample_bytes = 0
    rng = random.Random(seed)

    def writer(split: str):
        f, size, path = writers.get(split, (None, 0, None))
        if f is None or size >= shard_bytes:
            if f is not None:
                f.close()
            path = raw_dir / f"{split}_{len(raw_shards[split]):05d}.jsonl"
            raw_shards[split].append(str(path))
            f, size = open(path, "w", encoding="utf-8"), 0
            writers[split] = (f, size, path)
        return writers[split]

    for src in cfg["sources"]:
        lic_name = src.get("license", "per-record")
        entry = provenance.setdefault(src["name"], {"license": lic_name, "origin": src.get("origin", ""), "documents": 0, "bytes": 0})
        for doc_id, lic, text in _iter_source(src, base):
            stats["input"] += 1
            if len(text) > max_doc_bytes:
                stats["too_large"] += 1
                continue
            text = normalize(text)
            if len(text) < 40:
                stats["too_short"] += 1
                continue
            if contains_secret(text):
                stats["secret_dropped"] += 1
                continue
            text = scrub_pii(text)
            h = hashlib.sha256(text.encode()).digest()[:16]
            if h in seen:
                stats["duplicate_dropped"] += 1
                continue
            seen.add(h)
            split = _split_for(doc_id, val_frac, test_frac)
            f, size, _ = writer(split)
            line = json.dumps({"id": doc_id, "license": lic, "text": text}) + "\n"
            f.write(line)
            writers[split] = (f, size + len(line), writers[split][2])
            stats["kept"] += 1
            entry["documents"] += 1
            entry["bytes"] += len(text)
            if split == "train":
                if sample_bytes < tokenizer_sample_bytes:
                    sample.append(text)
                    sample_bytes += len(text)
                elif rng.random() < 0.01:  # light reservoir refresh so the sample is not only the first files
                    sample[rng.randrange(len(sample))] = text
            if stats["input"] % 50_000 == 0:
                log(f"  scanned {stats['input']:,} documents, kept {stats['kept']:,}")
    for f, _, _ in writers.values():
        f.close()
    if not raw_shards["train"]:
        raise RuntimeError("no training documents survived filtering")
    for split in ("val", "test"):  # tiny corpora: borrow one train shard's worth so eval sets exist
        if not raw_shards[split]:
            raw_shards[split] = [raw_shards["train"][-1]]
    log(f"training tokenizer on {sample_bytes:,} bytes sampled from the train split ...")
    tok = BPETokenizer.train(sample, vocab_size=vocab_size)
    tok_path = out / "tokenizer.json"
    tok.save(tok_path)
    jobs = []
    for split, paths in raw_shards.items():
        for i, rp in enumerate(paths):
            jobs.append((rp, str(tok_path), str(out / f"{split}_{i:05d}.bin")))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    log(f"tokenizing {len(jobs)} shards with {workers} worker(s) ...")
    results: dict[str, int] = {}
    if workers == 1:
        for j in jobs:
            path, n = _encode_shard(j)
            results[path] = n
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for path, n in ex.map(_encode_shard, jobs):
                results[path] = n
    dtype = "uint16" if tok.vocab_size <= 65535 else "uint32"
    splits = {}
    for split in raw_shards:
        shards = [{"path": Path(p).name, "tokens": n} for p, n in sorted(results.items()) if Path(p).name.startswith(split + "_")]
        splits[split] = {"shards": shards, "tokens": sum(s["tokens"] for s in shards), "dtype": dtype}
    summary = {"format": "sharded-v1", "tokenizer": {"vocab_size": tok.vocab_size, "fingerprint": tok.fingerprint(),
               "merges": len(tok.merges), "sample_bytes": sample_bytes},
               "cleaning": stats, "provenance": provenance, "splits": splits}
    (out / "prepared.json").write_text(json.dumps(summary, indent=2))
    return summary


class ShardedTokenDataset:
    """Random windows across many token shards, sampled proportionally to shard size."""

    def __init__(self, data_dir: str | Path, split_info: dict, context_length: int) -> None:
        self.context_length = context_length
        self.shards = []
        for s in split_info["shards"]:
            if s["tokens"] > context_length + 1:
                self.shards.append(np.memmap(Path(data_dir) / s["path"], dtype=np.dtype(split_info["dtype"]), mode="r"))
        if not self.shards:
            raise ValueError("no shard is longer than the context length")
        sizes = np.array([len(m) - context_length - 1 for m in self.shards], dtype=np.float64)
        self.weights = sizes / sizes.sum()
        self.total = int(sum(len(m) for m in self.shards))

    def __len__(self) -> int:
        return self.total

    def batch(self, batch_size: int, generator):
        import torch

        picks = torch.multinomial(torch.from_numpy(self.weights), batch_size, replacement=True, generator=generator).tolist()
        xs, ys = [], []
        for k in picks:
            m = self.shards[k]
            s = int(torch.randint(0, len(m) - self.context_length - 1, (1,), generator=generator))
            xs.append(np.asarray(m[s : s + self.context_length], dtype=np.int64))
            ys.append(np.asarray(m[s + 1 : s + 1 + self.context_length], dtype=np.int64))
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def fetch_hf_parquet(dataset: str, out_dir: str | Path, licenses: list[str], languages: list[str] | None = None,
                     max_bytes: int = 2_000_000_000, text_field: str = "code", license_field: str = "license",
                     language_field: str = "language", files: list[str] | None = None, log=print) -> dict:
    """Download a public HF dataset's Parquet files and keep only permissively licensed rows."""
    import pyarrow.parquet as pq

    allowed = {l.lower() for l in licenses}
    for l in allowed:
        if canonical_license(l) is None:
            raise ValueError(f"licence {l!r} is not on the permissive allow-list")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if files is None:
        from huggingface_hub import HfApi, hf_hub_download

        names = sorted(f for f in HfApi().list_repo_files(dataset, repo_type="dataset") if f.endswith(".parquet"))
        files = [hf_hub_download(dataset, n, repo_type="dataset") for n in names]  # downloaded lazily one by one
    kept_bytes = rows_kept = rows_seen = 0
    by_license: dict[str, int] = {}
    shard, size, shard_i = None, 0, 0
    for path in files:
        table = pq.read_table(path)
        cols = table.column_names
        for rec in table.to_pylist():
            rows_seen += 1
            lic = str(rec.get(license_field, "")).lower()
            if lic not in allowed:
                continue
            if languages and language_field in cols and rec.get(language_field) not in languages:
                continue
            text = rec.get(text_field)
            if not isinstance(text, str):
                continue
            if shard is None or size > 100_000_000:
                if shard:
                    shard.close()
                shard = open(out / f"part_{shard_i:05d}.jsonl", "w", encoding="utf-8")
                shard_i += 1
                size = 0
            rid = f"{dataset}:{rec.get('repo_name', '')}/{rec.get('path', rows_seen)}"
            line = json.dumps({"id": rid, "license": lic, "text": text}) + "\n"
            shard.write(line)
            size += len(line)
            kept_bytes += len(text)
            rows_kept += 1
            by_license[lic] = by_license.get(lic, 0) + 1
            if kept_bytes >= max_bytes:
                break
        log(f"  {Path(path).name}: kept {rows_kept:,} rows, {kept_bytes / 1e6:,.0f} MB so far")
        if kept_bytes >= max_bytes:
            break
    if shard:
        shard.close()
    prov = {"dataset": f"https://huggingface.co/datasets/{dataset}", "licenses_kept": by_license, "rows_seen": rows_seen,
            "rows_kept": rows_kept, "bytes_kept": kept_bytes, "languages": languages,
            "note": "Per-file licences as labelled by the dataset; review the dataset card and terms before use."}
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))
    return prov
