"""Secret redaction for logs, memory, and model context."""

from __future__ import annotations

import os
import re

_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "[REDACTED PRIVATE KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED AWS KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "[REDACTED GITHUB TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "[REDACTED API KEY]"),
    (re.compile(r"\bxox[abpr]-[A-Za-z0-9\-]{10,}\b"), "[REDACTED SLACK TOKEN]"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "[REDACTED JWT]"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{16,}"), r"\1 [REDACTED]"),
    (
        re.compile(r"(?i)\b([A-Z0-9_]*(?:password|passwd|secret|api[_-]?key|token|private[_-]?key)[A-Z0-9_]*)(\s*[:=]\s*)(['\"]?)[^\s'\"]{6,}\3"),
        r"\1\2\3[REDACTED]\3",
    ),
    (re.compile(r"(?i)(://[^:/\s]+:)[^@/\s]+(@)"), r"\1[REDACTED]\2"),  # credentials in URLs
]

SENSITIVE_ENV_RE = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION)")


def _env_secret_values() -> list[str]:
    vals = []
    for k, v in os.environ.items():
        if SENSITIVE_ENV_RE.search(k) and v and len(v) >= 8:
            vals.append(v)
    return sorted(vals, key=len, reverse=True)


def redact(text: str) -> str:
    if not text:
        return text
    for value in _env_secret_values():
        text = text.replace(value, "[REDACTED ENV]")
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def looks_secret(text: str) -> bool:
    return redact(text) != text


def scrubbed_env(extra: dict | None = None) -> dict:
    """Environment for child processes with credentials removed."""
    # GIT_CONFIG_COUNT/KEY_n/VALUE_n must be dropped as a group (they can carry
    # auth headers, and a partial set makes git refuse to run).
    env = {
        k: v for k, v in os.environ.items()
        if not SENSITIVE_ENV_RE.search(k) and not k.startswith("GIT_CONFIG_")
    }
    if extra:
        env.update(extra)
    return env
