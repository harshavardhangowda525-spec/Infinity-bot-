"""Permission policy: workspace scoping, sensitive files, command risk.

Nothing in here trusts the model. Every tool call is checked against the
policy *before* it runs, and the decision (allow / ask / deny) is recorded.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class PermissionMode(str, Enum):
    READ_ONLY = "read-only"  # default for unfamiliar workspaces
    ASK = "ask"  # writes and non-trivial commands need approval
    AUTO = "auto"  # writes allowed; risky commands still need approval


class Category(str, Enum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    EXECUTE = "execute"


class Risk(str, Enum):
    SAFE = "safe"  # read-only inspection
    CHECK = "check"  # tests / builds / linters: run project code, expected
    MEDIUM = "medium"  # installs, scripts, unknown programs, shell syntax
    HIGH = "high"  # deletes, network transfer, privilege, VCS rewrite, deploy
    BLOCKED = "blocked"  # never run


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


IGNORED_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", ".nuxt",
    "target", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".idea", ".gradle",
    "coverage", ".turbo", ".cache", ".infinity", "runs",
}
SENSITIVE_NAME_PATTERNS = [
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*", "id_ecdsa*",
    ".npmrc", ".pypirc", ".netrc", ".git-credentials", "credentials", "credentials.*",
    "secrets.*", "*.keystore", "service-account*.json",
]
SENSITIVE_ALLOWED = {".env.example", ".env.sample", ".env.template"}
SENSITIVE_DIRS = {".ssh", ".aws", ".gnupg", ".kube", ".docker"}
PROTECTED_DIRS = {".infinity", ".git"}  # agent may not write here directly
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".tar", ".7z", ".so",
    ".dylib", ".dll", ".exe", ".bin", ".pt", ".pth", ".onnx", ".safetensors", ".woff", ".woff2",
    ".ttf", ".mp3", ".mp4", ".mov", ".blend", ".glb", ".sqlite", ".db", ".pyc", ".class", ".jar",
}


def is_sensitive_path(rel: Path) -> bool:
    if any(part in SENSITIVE_DIRS for part in rel.parts):
        return True
    name = rel.name
    if name in SENSITIVE_ALLOWED:
        return False
    return any(fnmatch.fnmatch(name, pat) for pat in SENSITIVE_NAME_PATTERNS)


class PathError(PermissionError):
    pass


@dataclass
class Workspace:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"workspace does not exist: {self.root}")

    def resolve(self, path: str | Path) -> Path:
        """Resolve a user/model-supplied path, refusing anything outside root.

        ``Path.resolve`` follows symlinks, so a link pointing outside the
        workspace is rejected as well as ``..`` traversal and absolute paths.
        """
        if path is None or str(path).strip() == "":
            raise PathError("empty path")
        raw = str(path)
        if "\x00" in raw:
            raise PathError("path contains NUL byte")
        p = Path(raw).expanduser()
        candidate = (p if p.is_absolute() else self.root / p).resolve()
        if candidate != self.root and not candidate.is_relative_to(self.root):
            raise PathError(f"path escapes workspace: {raw}")
        return candidate

    def rel(self, path: Path) -> Path:
        return path.relative_to(self.root) if path != self.root else Path(".")


# --------------------------------------------------------------- commands
SAFE_PROGRAMS = {
    "ls", "pwd", "cat", "head", "tail", "wc", "grep", "rg", "echo", "which", "file", "stat",
    "tree", "du", "df", "date", "whoami", "uname", "env", "printenv", "diff", "sort", "uniq", "cut",
    "basename", "dirname", "realpath", "true", "false", "test",
}
SAFE_GIT = {"status", "diff", "log", "show", "branch", "rev-parse", "ls-files", "blame", "remote", "tag", "describe"}
CHECK_PATTERNS = [
    r"^(python3?|py) -m (pytest|unittest|mypy|ruff|flake8|pylint|black --check|compileall|py_compile)\b",
    r"^(pytest|mypy|ruff|flake8|pylint|tsc|eslint|prettier --check|vitest|jest|mocha)\b",
    r"^(npm|pnpm|yarn|bun) (run )?(test|build|lint|typecheck|check)\b",
    r"^npx (tsc|eslint|vitest|jest|prettier --check)\b",
    r"^(cargo) (test|build|check|clippy)\b",
    r"^go (test|build|vet)\b",
    r"^make (test|check|build|lint)?\s*$",
    r"^(flutter|dart) (test|analyze)\b",
    r"^(node|python3?|deno|bun) --version$",
    r"^(gradle|\./gradlew|mvn) (test|build|check)\b",
]
INSTALL_PATTERNS = [
    r"^(npm|pnpm|yarn|bun) (install|i|add|ci)\b",
    r"^(pip3?|python3? -m pip|uv pip|poetry|pipenv) (install|add|sync)\b",
    r"^(cargo|go|gem|composer|brew|apt|apt-get|dnf|yum|pacman) (install|add|get)\b",
]
HIGH_PROGRAMS = {
    "rm", "rmdir", "sudo", "su", "doas", "chmod", "chown", "chgrp", "kill", "killall", "pkill",
    "ssh", "scp", "rsync", "sftp", "ftp", "curl", "wget", "nc", "ncat", "telnet", "dd", "mount",
    "umount", "crontab", "systemctl", "service", "launchctl", "docker", "podman", "kubectl",
    "terraform", "vercel", "netlify", "firebase", "heroku", "flyctl", "aws", "gcloud", "az",
    "truncate", "shred", "ngrok", "cloudflared", "eval", "exec", "xargs",
}
HIGH_GIT = {"push", "reset", "clean", "rebase", "filter-branch", "checkout", "restore", "rm", "gc", "reflog", "stash", "config"}
BLOCKED_PATTERNS = [
    (r"\brm\s+(-[a-zA-Z]*\s+)*(/|~|\$HOME|/\*|\.\.?/?)(\s|$)", "recursive delete of root/home/parent"),
    (r":\(\)\s*\{.*\};\s*:", "fork bomb"),
    (r"\bmkfs(\.\w+)?\b", "filesystem format"),
    (r"\bdd\b.*\bof=/dev/", "raw device write"),
    (r">\s*/dev/(sd|nvme|hd|disk)", "raw device write"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "power control"),
    (r"(curl|wget)[^|]*\|\s*(sudo\s+)?(ba|z|da)?sh\b", "pipe remote script to shell"),
    (r"(curl|wget)[^|]*\|\s*python", "pipe remote script to interpreter"),
    (r"\bchmod\s+(-R\s+)?[0-7]*777\s+/", "world-writable system path"),
    (r"\bgit\s+push\b.*(--force|-f)\b.*\b(main|master|production|prod)\b", "force-push to protected branch"),
]
SHELL_META = re.compile(r"[|;&<>`]|\$\(|\$\{|\n")


@dataclass
class CommandAssessment:
    command: str
    risk: Risk
    reasons: list[str] = field(default_factory=list)
    needs_shell: bool = False
    argv: list[str] = field(default_factory=list)
    is_check: bool = False


def classify_command(command: str, workspace: Workspace | None = None) -> CommandAssessment:
    cmd = command.strip()
    reasons: list[str] = []
    if not cmd:
        return CommandAssessment(command, Risk.BLOCKED, ["empty command"])
    for pat, why in BLOCKED_PATTERNS:
        if re.search(pat, cmd):
            return CommandAssessment(command, Risk.BLOCKED, [why])

    needs_shell = bool(SHELL_META.search(cmd))
    try:
        argv = shlex.split(cmd)
    except ValueError as e:
        return CommandAssessment(command, Risk.BLOCKED, [f"unparseable command: {e}"])
    if not argv:
        return CommandAssessment(command, Risk.BLOCKED, ["empty command"])

    # Assess every sub-command of a compound shell line; the line takes the max risk.
    segments = _split_segments(cmd) if needs_shell else [cmd]
    order = [Risk.SAFE, Risk.CHECK, Risk.MEDIUM, Risk.HIGH, Risk.BLOCKED]
    worst = Risk.SAFE
    all_check = True
    for seg in segments:
        r, why, is_check = _classify_simple(seg)
        reasons.extend(why)
        all_check = all_check and is_check
        if order.index(r) > order.index(worst):
            worst = r
    if needs_shell and order.index(worst) < order.index(Risk.MEDIUM):
        if re.search(r"(^|[^>])>{1,2}\s*[^&\s]", cmd):
            worst = Risk.MEDIUM
            reasons.append("shell redirection writes files")
        elif not re.fullmatch(r"[^;&<>`$]*(\|\s*(head|tail|grep|wc|sort|uniq|cat)\b[^;&<>`$|]*)+", cmd):
            worst = Risk.MEDIUM
            reasons.append("uses shell syntax")

    if workspace is not None:
        for tok in argv:
            if tok.startswith("-"):
                tok = tok.split("=", 1)[-1] if "=" in tok else ""
            if not tok:
                continue
            if tok.startswith(("/", "~", "..")):
                try:
                    workspace.resolve(tok)
                except PathError:
                    if not tok.startswith(("/dev/null", "/tmp")):
                        reasons.append(f"argument outside workspace: {tok}")
                        if order.index(worst) < order.index(Risk.HIGH):
                            worst = Risk.HIGH
            if is_sensitive_path(Path(tok)):
                reasons.append(f"touches sensitive file: {tok}")
                if order.index(worst) < order.index(Risk.HIGH):
                    worst = Risk.HIGH
    return CommandAssessment(command, worst, reasons, needs_shell, argv, is_check=all_check and worst == Risk.CHECK)


def _split_segments(cmd: str) -> list[str]:
    parts = re.split(r"\|\||&&|[|;&\n]", cmd)
    return [p.strip() for p in parts if p.strip()]


def _classify_simple(seg: str) -> tuple[Risk, list[str], bool]:
    try:
        argv = shlex.split(seg)
    except ValueError:
        return Risk.HIGH, [f"unparseable segment: {seg}"], False
    # Strip leading VAR=value assignments.
    while argv and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]):
        argv = argv[1:]
    if not argv:
        return Risk.SAFE, [], False
    prog = Path(argv[0]).name
    norm = " ".join([prog] + argv[1:])
    for pat in CHECK_PATTERNS:
        if re.search(pat, norm):
            return Risk.CHECK, [], True
    for pat in INSTALL_PATTERNS:
        if re.search(pat, norm):
            return Risk.MEDIUM, ["installs dependencies (runs third-party install scripts)"], False
    if prog == "git":
        sub = next((a for a in argv[1:] if not a.startswith("-")), "")
        if sub in SAFE_GIT and not (sub == "branch" and any(a in ("-D", "-d", "--delete", "-m", "-M") for a in argv)):
            return Risk.SAFE, [], False
        if sub in HIGH_GIT:
            return Risk.HIGH, [f"git {sub} can discard work or publish changes"], False
        return Risk.MEDIUM, [f"git {sub} modifies the repository"], False
    if prog == "find":
        if any(a in ("-delete", "-exec", "-execdir", "-ok") for a in argv):
            return Risk.HIGH, ["find with -delete/-exec"], False
        return Risk.SAFE, [], False
    if prog in ("sed", "awk", "perl") and any(a.startswith("-i") for a in argv):
        return Risk.MEDIUM, [f"{prog} -i edits files in place"], False
    if prog in SAFE_PROGRAMS:
        return Risk.SAFE, [], False
    if prog in HIGH_PROGRAMS:
        return Risk.HIGH, [f"{prog} is a high-impact program"], False
    if prog in ("npm", "pnpm", "yarn") and len(argv) > 1 and argv[1] in ("publish", "unpublish", "deprecate"):
        return Risk.HIGH, ["publishes a package"], False
    if prog in ("pip", "pip3") and len(argv) > 1 and argv[1] == "uninstall":
        return Risk.HIGH, ["uninstalls packages"], False
    if prog in ("mv", "cp", "mkdir", "touch", "ln"):
        return Risk.MEDIUM, [f"{prog} modifies files"], False
    if prog in ("python", "python3", "node", "deno", "bun", "ruby", "bash", "sh", "zsh", "npx", "npm", "pnpm", "yarn", "make"):
        return Risk.MEDIUM, [f"runs code via {prog}"], False
    return Risk.MEDIUM, [f"unrecognized program: {prog}"], False


# --------------------------------------------------------------- policy
@dataclass
class ApprovalRequest:
    tool: str
    category: Category
    summary: str
    risk: Risk | None = None
    reasons: list[str] = field(default_factory=list)
    diff: str | None = None


Approver = Callable[[ApprovalRequest], bool]


def deny_all(_: ApprovalRequest) -> bool:
    return False


@dataclass
class PermissionPolicy:
    mode: PermissionMode = PermissionMode.READ_ONLY
    approver: Approver = deny_all
    allow_sensitive_reads: bool = False

    def decide(self, category: Category, risk: Risk | None = None) -> Decision:
        if risk == Risk.BLOCKED:
            return Decision.DENY
        if self.mode == PermissionMode.READ_ONLY:
            if category == Category.READ:
                return Decision.ALLOW
            if category == Category.EXECUTE and risk == Risk.SAFE:
                return Decision.ALLOW
            return Decision.DENY
        if category == Category.READ:
            return Decision.ALLOW
        if category == Category.DESTRUCTIVE:
            return Decision.ASK
        if category == Category.WRITE:
            return Decision.ALLOW if self.mode == PermissionMode.AUTO else Decision.ASK
        # EXECUTE
        if risk == Risk.SAFE:
            return Decision.ALLOW
        if risk == Risk.CHECK:
            return Decision.ALLOW if self.mode == PermissionMode.AUTO else Decision.ASK
        return Decision.ASK  # MEDIUM / HIGH always need a human

    def authorize(self, request: ApprovalRequest) -> tuple[bool, str]:
        decision = self.decide(request.category, request.risk)
        if decision == Decision.ALLOW:
            return True, "allowed by policy"
        if decision == Decision.DENY:
            why = "blocked command" if request.risk == Risk.BLOCKED else f"not permitted in {self.mode.value} mode"
            return False, why
        try:
            ok = bool(self.approver(request))
        except Exception as e:  # an approver failure must never mean "yes"
            return False, f"approval failed: {e}"
        return (True, "approved by user") if ok else (False, "denied by user")
