# Permissions configuration

Permissions are enforced by the runtime (`infinity/agent/permissions.py`)
**before** any tool runs. The model cannot change them.

## Choosing a permission mode

| Mode | Use when | How to set |
|---|---|---|
| `read-only` | default for any workspace you haven't trusted | automatic |
| `ask` | normal work: every write and every non-trivial command asks you first, showing the diff | `--permission ask` or `infinity memory trust <path> ask` |
| `auto` | you trust the task: writes and test/build/lint commands run freely; installs, scripts, deletes, network and anything HIGH still ask | `--permission auto` or `infinity memory trust <path> auto` |

The modes `ask`, `plan` and `review` are always read-only, whatever the
permission setting.

In a non-interactive shell (no TTY), every approval request is answered **no**.

## Always enforced

- **Workspace scope.** Paths are resolved, including symlinks. Anything outside
  the workspace root is refused.
- **Secrets.** `.env*` (except `.env.example`, `.sample` and `.template`),
  `*.pem`, `*.key`, `id_rsa*`, `.ssh/`, `.aws/`, `credentials*`, `.npmrc`,
  `.netrc` and similar files cannot be read by the agent, and are skipped by
  search. The agent never writes them.
- **Protected directories.** `.git/` and `.infinity/` cannot be written by file tools.
- **Stale edits.** A file must be read, and unchanged since, before it can be edited or overwritten.
- **Checkpoints.** Originals are saved before every change or delete. Restore them with `infinity rollback`.
- **Child processes** get an environment with variables named like
  `*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*AUTH*` and `GIT_CONFIG_*` removed.
- **Redaction.** Tool output, memory and model context are passed through `redact()`.
- **Limits.** Command timeouts (default 120 s) and cancellation kill the whole process group.
- **Blocked commands** are never run and never offered for approval. These
  include recursive delete of `/`, `~` or `..`; `curl | sh`; `mkfs`; raw `dd`
  to devices; fork bombs; shutdown; and force-push to main, master or prod.

## Command risk levels

See `classify_command()`. Compound shell lines take the risk of their most
dangerous part, and shell syntax is at least MEDIUM. The exception is a
read-only pipe into `head`, `tail`, `grep` and similar. Arguments that point
outside the workspace or at secret files raise the risk to HIGH.
