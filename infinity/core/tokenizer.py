"""INFINITY TOKENIZER — original byte-level BPE.

Design:
- Text is UTF-8 encoded; the base vocabulary is the 256 byte values, so any
  input (code, emoji, binary-ish text) round-trips without an <unk> token.
- A regex pre-tokenizer splits text into chunks (words, numbers, punctuation
  runs, whitespace) so merges never cross those boundaries. Leading single
  spaces stay attached to the following word, and runs of indentation are kept
  together, which suits source code.
- Merges are learned with the classic BPE procedure, using incremental pair
  counts so training stays practical in pure Python.
- Special tokens are reserved ids after the learned vocabulary and are only
  produced when explicitly requested (``allow_special=True``), so text that
  merely *contains* "<|eos|>" cannot inject control tokens.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

TOKENIZER_FORMAT = "infinity-bpe-v1"

SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|end|>",
)

# Letters (incl. underscore, unicode), digits in groups of up to 3, runs of
# other symbols, newlines with their following indentation, other whitespace.
PRETOKENIZE_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)(?![^\W\d])"
    r"| ?[^\W\d]+"
    r"| ?\d{1,3}"
    r"| ?[^\s\w]+"
    r"|\n[ \t]*"
    r"|[ \t]+(?=[ \t])"
    r"|[ \t]+"
    r"|\s"
)
_PRETOKENIZE_RE = re.compile(PRETOKENIZE_PATTERN)


def pretokenize(text: str) -> list[str]:
    chunks = _PRETOKENIZE_RE.findall(text)
    # The pattern is exhaustive; guard against silent data loss anyway.
    if sum(len(c) for c in chunks) != len(text):
        raise AssertionError("pre-tokenizer dropped characters")
    return chunks


@dataclass
class BPETokenizer:
    merges: list[tuple[int, int]] = field(default_factory=list)
    special_tokens: tuple[str, ...] = SPECIAL_TOKENS

    def __post_init__(self) -> None:
        self._rebuild()

    # ------------------------------------------------------------------ setup
    def _rebuild(self) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.ranks: dict[tuple[int, int], int] = {}
        for rank, (a, b) in enumerate(self.merges):
            new_id = 256 + rank
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            self.ranks[(a, b)] = rank
        base = 256 + len(self.merges)
        self.special_to_id = {tok: base + i for i, tok in enumerate(self.special_tokens)}
        self.id_to_special = {i: t for t, i in self.special_to_id.items()}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    def token_id(self, special: str) -> int:
        return self.special_to_id[special]

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    # --------------------------------------------------------------- training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        special_tokens: tuple[str, ...] = SPECIAL_TOKENS,
        verbose: bool = False,
    ) -> "BPETokenizer":
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be >= {256 + len(special_tokens)}")

        chunk_counts: Counter[str] = Counter()
        for text in texts:
            chunk_counts.update(pretokenize(text))

        words: list[list[int]] = []
        freqs: list[int] = []
        for chunk, count in chunk_counts.items():
            words.append(list(chunk.encode("utf-8")))
            freqs.append(count)

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            f = freqs[wi]
            for pair in zip(w, w[1:]):
                pair_counts[pair] += f
                pair_where[pair].add(wi)

        merges: list[tuple[int, int]] = []
        for step in range(n_merges):
            if not pair_counts:
                break
            # Deterministic tie-break: highest count, then smallest pair.
            best, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if best_count < min_frequency:
                break
            new_id = 256 + len(merges)
            merges.append(best)
            for wi in list(pair_where.get(best, ())):
                w = words[wi]
                f = freqs[wi]
                for pair in zip(w, w[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_where[pair].discard(wi)
                merged: list[int] = []
                i = 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == best[0] and w[i + 1] == best[1]:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(w[i])
                        i += 1
                words[wi] = merged
                for pair in zip(merged, merged[1:]):
                    pair_counts[pair] += f
                    pair_where[pair].add(wi)
            pair_where.pop(best, None)
            if verbose and (step + 1) % 250 == 0:
                print(f"  merge {step + 1}/{n_merges} count={best_count}")
        return cls(merges=merges, special_tokens=special_tokens)

    # --------------------------------------------------------------- encoding
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best_rank = None
            best_i = -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            ids[best_i : best_i + 2] = [256 + best_rank]
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in pretokenize(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = False, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = [self.bos_id] if add_bos else []
        if allow_special and self._special_re is not None:
            for part in self._special_re.split(text):
                if not part:
                    continue
                if part in self.special_to_id:
                    ids.append(self.special_to_id[part])
                else:
                    ids.extend(self._encode_ordinary(part))
        else:
            ids.extend(self._encode_ordinary(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        buf = bytearray()
        parts: list[str] = []
        for i in ids:
            i = int(i)
            if i in self.vocab:
                buf.extend(self.vocab[i])
            elif i in self.id_to_special:
                if not skip_special:
                    parts.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                    parts.append(self.id_to_special[i])
            else:
                raise ValueError(f"token id {i} out of range (vocab_size={self.vocab_size})")
        parts.append(buf.decode("utf-8", errors="replace"))
        return "".join(parts)

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "format": TOKENIZER_FORMAT,
            "pretokenize_pattern": PRETOKENIZE_PATTERN,
            "special_tokens": list(self.special_tokens),
            "merges": [list(m) for m in self.merges],
        }

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data["fingerprint"] = self.fingerprint()
        data["vocab_size"] = self.vocab_size
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        if data.get("format") != TOKENIZER_FORMAT:
            raise ValueError(f"unsupported tokenizer format: {data.get('format')!r}")
        if data.get("pretokenize_pattern") != PRETOKENIZE_PATTERN:
            raise ValueError("tokenizer was trained with a different pre-tokenizer")
        tok = cls(merges=[tuple(m) for m in data["merges"]], special_tokens=tuple(data["special_tokens"]))
        if "fingerprint" in data and data["fingerprint"] != tok.fingerprint():
            raise ValueError("tokenizer fingerprint mismatch (file corrupted or edited)")
        return tok
