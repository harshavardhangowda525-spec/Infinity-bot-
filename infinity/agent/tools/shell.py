"""Terminal tools: run commands and manage long-running processes.

- Every command is risk-classified and authorized before it starts.
- Commands without shell syntax run with ``shell=False`` (no injection).
- Shell syntax (pipes, redirects, ``&&``) is only executed via ``/bin/sh -c``
  (``cmd /c`` on Windows) after the policy allows it, and always counts as at
  least MEDIUM risk.
- The child environment has credentials stripped; cwd is the workspace.
- Timeouts and cancellation kill the whole process group.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from ..permissions import IS_WINDOWS, Category, classify_command, program_name
from ..redact import redact, scrubbed_env
from .base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult, require


# cmd.exe built-ins have no executable of their own and must run through cmd /c.
CMD_BUILTINS = {"dir", "type", "echo", "del", "erase", "copy", "move", "ren", "rename", "md", "mkdir",
                "rd", "rmdir", "cls", "set", "ver", "vol"}


def _kill_group(proc: subprocess.Popen) -> None:
    if IS_WINDOWS:
        # taskkill /T stops the whole process tree (there are no POSIX process groups).
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _resolve_program(argv: list[str], path: str | None) -> list[str]:
    """Windows: find the real executable (PATHEXT: .exe/.cmd/.bat).

    ``python``/``python3`` fall back to the interpreter running INFINITY when
    they are missing or are only the Microsoft Store placeholder; Windows
    virtual environments contain ``python.exe`` but no ``python3``.
    """
    found = shutil.which(argv[0], path=path)
    if program_name(argv[0]).lower() in ("python", "python3") and (
            found is None or "windowsapps" in found.lower()):
        found = sys.executable
    return [found or argv[0], *argv[1:]]


def spawn_args(command: str, needs_shell: bool, argv: list[str], windows: bool = IS_WINDOWS,
               path: str | None = None) -> list[str] | str:
    """What to hand to Popen for an already-authorized command."""
    if not windows:
        return ["/bin/sh", "-c", command] if needs_shell else argv
    if needs_shell or (argv and argv[0].lower() in CMD_BUILTINS):
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return f'"{comspec}" /d /s /c "{command}"'
    return _resolve_program(argv, path)


def _spawn(ctx: ToolContext, assessment) -> subprocess.Popen:
    env = scrubbed_env({"CI": "1", "TERM": "dumb", "NO_COLOR": "1", "PYTHONDONTWRITEBYTECODE": "1",
                        **({"PYTHONIOENCODING": "utf-8"} if IS_WINDOWS else {})})
    args = spawn_args(assessment.command, assessment.needs_shell, assessment.argv, path=env.get("PATH"))
    if IS_WINDOWS:
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        group = {"start_new_session": True}
    return subprocess.Popen(
        args,
        cwd=ctx.workspace.root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **group,
    )


def _authorize(ctx: ToolContext, tool: str, command: str):
    a = classify_command(command, ctx.workspace)
    require(ctx, tool, Category.EXECUTE, command, risk=a.risk, reasons=a.reasons)
    return a


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = limit // 2
    return text[:keep] + f"\n... [{len(text) - limit} chars truncated] ...\n" + text[-keep:]


def run_command(ctx: ToolContext, command: str, timeout: float = 0) -> ToolResult:
    a = _authorize(ctx, "run_command", command)
    limit = ctx.command_timeout if not timeout else min(float(timeout), 1800.0)
    try:
        proc = _spawn(ctx, a)
    except FileNotFoundError:
        raise ToolError(f"program not found: {a.argv[0] if a.argv else command}")
    out_chunks: list[str] = []
    err_chunks: list[str] = []

    def pump(stream, sink, label):
        for line in iter(stream.readline, ""):
            sink.append(line)
            if ctx.on_output is not None:
                try:
                    ctx.on_output(redact(f"{label}{line}"))
                except Exception:
                    pass
        stream.close()

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, out_chunks, ""), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, err_chunks, "[stderr] "), daemon=True),
    ]
    for t in threads:
        t.start()
    start = time.monotonic()
    status = "exited"
    while True:
        try:
            proc.wait(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            pass
        if ctx.cancel.is_set():
            status = "cancelled"
            _kill_group(proc)
            break
        if time.monotonic() - start > limit:
            status = "timeout"
            _kill_group(proc)
            break
    for t in threads:
        t.join(timeout=2)
    exit_code = proc.returncode
    elapsed = round(time.monotonic() - start, 2)
    stdout = redact("".join(out_chunks))
    stderr = redact("".join(err_chunks))
    half = ctx.max_output_chars // 2
    ok = status == "exited" and exit_code == 0
    record = {"step": ctx.step, "command": command, "exit_code": exit_code, "status": status, "is_check": a.is_check}
    ctx.checks.append(record)
    summary = f"$ {command}\nexit_code={exit_code} status={status} elapsed={elapsed}s risk={a.risk.value}"
    body = summary
    if stdout:
        body += "\n--- stdout ---\n" + _clip(stdout, half)
    if stderr:
        body += "\n--- stderr ---\n" + _clip(stderr, half)
    err = None if ok else (f"command {status}" if status != "exited" else f"command failed with exit code {exit_code}")
    return ToolResult(ok, body, error=err, data={
        "exit_code": exit_code, "status": status, "elapsed_s": elapsed, "risk": a.risk.value,
        "is_check": a.is_check, "stdout": _clip(stdout, half), "stderr": _clip(stderr, half),
    })


# ----------------------------------------------------------- background procs
@dataclass
class ManagedProcess:
    id: str
    command: str
    proc: subprocess.Popen
    started: float
    lines: deque = field(default_factory=lambda: deque(maxlen=2000))


class ProcessManager:
    def __init__(self) -> None:
        self.procs: dict[str, ManagedProcess] = {}

    def start(self, ctx: ToolContext, a) -> ManagedProcess:
        proc = _spawn(ctx, a)
        mp = ManagedProcess(uuid.uuid4().hex[:8], a.command, proc, time.time())

        def pump(stream, prefix):
            for line in iter(stream.readline, ""):
                mp.lines.append(redact(prefix + line.rstrip("\n")))
            stream.close()

        threading.Thread(target=pump, args=(proc.stdout, ""), daemon=True).start()
        threading.Thread(target=pump, args=(proc.stderr, "[stderr] "), daemon=True).start()
        self.procs[mp.id] = mp
        return mp

    def stop_all(self) -> None:
        for mp in self.procs.values():
            if mp.proc.poll() is None:
                _kill_group(mp.proc)


PROCESSES = ProcessManager()


def start_process(ctx: ToolContext, command: str) -> ToolResult:
    a = _authorize(ctx, "start_process", command)
    mp = PROCESSES.start(ctx, a)
    time.sleep(0.5)
    state = "running" if mp.proc.poll() is None else f"exited ({mp.proc.returncode})"
    return ToolResult(True, f"process {mp.id} {state}: {command}", data={"id": mp.id, "state": state})


def process_output(ctx: ToolContext, id: str, tail: int = 100) -> ToolResult:
    mp = PROCESSES.procs.get(id)
    if mp is None:
        raise ToolError(f"no such process: {id}")
    state = "running" if mp.proc.poll() is None else f"exited ({mp.proc.returncode})"
    lines = list(mp.lines)[-max(1, min(tail, 2000)):]
    return ToolResult(True, f"process {id} {state}\n" + "\n".join(lines), data={"state": state})


def stop_process(ctx: ToolContext, id: str) -> ToolResult:
    mp = PROCESSES.procs.get(id)
    if mp is None:
        raise ToolError(f"no such process: {id}")
    if mp.proc.poll() is None:
        _kill_group(mp.proc)
    return ToolResult(True, f"process {id} stopped (exit {mp.proc.returncode})")


def register(reg: ToolRegistry) -> None:
    X = Category.EXECUTE
    reg.register(Tool("run_command", "Run a terminal command in the workspace; returns stdout, stderr, exit code.", X,
                      {"command": (str, True, "command line"), "timeout": ((int, float), False, "seconds")}, run_command))
    reg.register(Tool("start_process", "Start a long-running process (dev server, watcher).", X,
                      {"command": (str, True, "command line")}, start_process))
    reg.register(Tool("process_output", "Read recent output of a started process.", Category.READ,
                      {"id": (str, True, "process id"), "tail": (int, False, "lines")}, process_output, True))
    reg.register(Tool("stop_process", "Stop a started process.", X, {"id": (str, True, "process id")}, stop_process))
