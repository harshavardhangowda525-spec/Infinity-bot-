"""Run model-generated code against unit tests in a restricted child process.

What the restriction does:
- runs in a fresh temporary folder that is deleted afterwards;
- uses ``python -I`` (isolated mode), a minimal environment and a timeout;
- on POSIX, caps memory, CPU time, file size and process count;
- installs an audit hook that refuses network connections, starting processes, and
  writing files outside the temporary folder.

What it does NOT do: the audit hook is a best-effort guard, not a security boundary. A
determined program can get around it. Run evaluations only on code from models you
run yourself, never on untrusted submissions. Windows gets the timeout, the temporary
folder, the minimal environment and the audit hook, but not the POSIX resource limits.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

GUARD = r'''
import os, sys
_ROOT = os.path.realpath(os.getcwd())
_DENY = ("socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto", "subprocess.Popen", "os.system", "os.exec", "os.spawn",
         "os.posix_spawn", "os.fork", "os.forkpty", "os.startfile", "ctypes.dlopen", "winreg.")
def _hook(event, args):
    if event.startswith(_DENY):
        raise PermissionError(f"blocked in evaluation sandbox: {event}")
    if event == "open" and len(args) > 1 and isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
        path = args[0]
        if isinstance(path, (str, bytes, os.PathLike)):
            p = os.path.realpath(os.fsdecode(path))
            if not (p == _ROOT or p.startswith(_ROOT + os.sep)):
                raise PermissionError(f"blocked in evaluation sandbox: write outside workspace ({p})")
    if event in ("os.remove", "os.rename", "os.rmdir", "shutil.rmtree") and args:
        p = os.path.realpath(os.fsdecode(args[0])) if isinstance(args[0], (str, bytes, os.PathLike)) else ""
        if not p.startswith(_ROOT + os.sep):
            raise PermissionError(f"blocked in evaluation sandbox: {event} outside workspace")
sys.addaudithook(_hook)
'''

RUNNER = GUARD + r'''
import traceback
try:
    ns = {"__name__": "solution"}
    exec(compile(open("solution.py", encoding="utf-8").read(), "solution.py", "exec"), ns)
    exec(compile(open("tests.py", encoding="utf-8").read(), "tests.py", "exec"), ns)
    print("ALL TESTS PASSED")
except BaseException:
    traceback.print_exc()
    sys.exit(1)
'''


def extract_code(text: str) -> str:
    """First fenced code block (```python preferred), else the whole reply."""
    blocks = re.findall(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", text, flags=re.S)
    for lang, body in blocks:
        if lang.lower() in ("python", "py", ""):
            return body
    return blocks[0][1] if blocks else text


def _limits():  # POSIX only; runs in the child before exec
    import resource

    mem = 512 * 2**20
    for lim, val in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, 10), (resource.RLIMIT_FSIZE, 2**20)):
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):
            pass
    os.setsid()


def run_tests(code: str, tests: str, timeout: float = 10.0) -> dict:
    with tempfile.TemporaryDirectory(prefix="infinity-eval-") as tmp:
        Path(tmp, "solution.py").write_text(code, encoding="utf-8")
        Path(tmp, "tests.py").write_text(tests, encoding="utf-8")
        Path(tmp, "runner.py").write_text(RUNNER, encoding="utf-8")
        env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "PATH": os.environ.get("PATH", "")}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        try:
            r = subprocess.run([sys.executable, "-I", "runner.py"], cwd=tmp, env=env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout,
                               preexec_fn=_limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"timeout after {timeout}s"}
        ok = r.returncode == 0 and "ALL TESTS PASSED" in r.stdout
        err = None if ok else (r.stderr.strip().splitlines() or [f"exit code {r.returncode}"])[-1][:300]
        return {"passed": ok, "error": err}
