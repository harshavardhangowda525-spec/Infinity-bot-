# COOK mode security

COOK lets a model edit files and run commands on your computer. The protections below are
enforced by the runtime, not by the model, and apply to **every** model: INFINITY HYPER
5.5 and open models alike.

## Enforced by the runtime (existing, tested in `tests/test_permissions.py`, `tests/test_tools.py`)

| Protection | Where |
|---|---|
| **Workspace scoping.** File tools cannot leave the opened folder: `..`, absolute paths and symlinks pointing outside are refused | `infinity/agent/permissions.py` (`Workspace.resolve`) |
| **Per-user workspace roots.** Non-owner accounts can only open folders the owner authorised | `infinity/server/app.py` |
| **Secrets files** (`.env`, keys, credentials, `.ssh`…) are not readable by default and are never shown in the UI | `is_sensitive_path` |
| **Command risk classes** SAFE / CHECK / MEDIUM / HIGH / BLOCKED. BLOCKED commands never start. MEDIUM and HIGH always need human approval | `classify_command`, `docs/PERMISSIONS.md` |
| **Permission modes.** read-only (default for unfamiliar folders), ask, auto | `PermissionPolicy` |
| **Approval before destructive operations** (delete, risky commands) | approvals in the dashboard |
| **Secret redaction** in tool output and logs. Child processes get an environment with credentials removed | `infinity/agent/redact.py` |
| **Checkpoints and rollback** of every file change ("Undo" button, `infinity rollback`) | `infinity/agent/checkpoints.py` |
| **Audit trail.** Tool calls and outcomes go to SQLite memory; owner and admin actions go to a hash-chained, append-only audit log | `memory.py`, `platform/audit.py` |
| **Bounded loop.** Steps, time, tokens, invalid-output limit and cancellation | `infinity/agent/loop.py` |
| **Windows.** Destructive Windows programs (`del`, `rd`, `taskkill`, `powershell`, …) are HIGH risk | `permissions.py` |

The model never receives environment variables, API keys or passwords from the runtime. It
sees only the files it reads, which go through the secrets-file and redaction rules.

## Evaluation sandbox (new: `infinity/hf/sandbox.py`)

Model-generated code in `evaluate.py` runs:

- in a fresh temporary folder, with `python -I`, a minimal environment and a 10 s timeout;
- on Linux and macOS, with memory, CPU, file-size and session limits;
- with an audit hook that blocks network connections, DNS lookups, starting processes, and
  writing, deleting or renaming files outside the temporary folder.

## Honest limits

- There is **no OS-level sandbox** (container, VM, Windows job object) for COOK commands
  or for evaluation. The audit hook is a best-effort guard; deliberate malicious code can
  get around it.
- On Windows the POSIX resource limits do not apply.
- Prefer **ask** mode for projects you care about, and keep them under git.
- COOK with any small model is **not** equivalent to a frontier coding agent. Measure it
  with `evaluate.py --repair N` and `infinity bench`.
