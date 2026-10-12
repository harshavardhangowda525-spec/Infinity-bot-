import sys
import threading
import time

import pytest

from infinity.agent.checkpoints import CheckpointStore
from infinity.agent.permissions import PermissionMode, PermissionPolicy, Workspace
from infinity.agent.tools import ToolContext, default_registry


@pytest.fixture()
def env(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "README.md").write_text("# demo\n")
    (tmp_path / ".env").write_text("SECRET=abc\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("add\n")
    ws = Workspace(tmp_path)
    ctx = ToolContext(workspace=ws, policy=PermissionPolicy(PermissionMode.AUTO, approver=lambda r: True))
    return ws, ctx, default_registry()


def run(reg, ctx, name, **args):
    return reg.execute(name, args, ctx)


def test_list_find_search_skip_ignored_and_secrets(env):
    ws, ctx, reg = env
    listing = run(reg, ctx, "list_dir").output
    assert "src/" in listing and "node_modules" not in listing
    assert "src/app.py" in run(reg, ctx, "find_files", pattern="*.py").output
    hits = run(reg, ctx, "search_text", query="add").output
    assert "src/app.py:1" in hits and "node_modules" not in hits
    assert "SECRET" not in run(reg, ctx, "search_text", query="SECRET").output


def test_read_sensitive_file_denied(env):
    ws, ctx, reg = env
    res = run(reg, ctx, "read_file", path=".env")
    assert not res.ok and "secrets file" in res.error


def test_edit_requires_read_and_detects_stale(env):
    ws, ctx, reg = env
    res = run(reg, ctx, "edit_file", path="src/app.py", old="a - b", new="a + b")
    assert not res.ok and "read src/app.py before" in res.error
    assert run(reg, ctx, "read_file", path="src/app.py").ok
    (ws.root / "src" / "app.py").write_text("def add(a, b):\n    return a - b  # changed\n")
    res = run(reg, ctx, "edit_file", path="src/app.py", old="a - b", new="a + b")
    assert not res.ok and "stale" in res.error
    run(reg, ctx, "read_file", path="src/app.py")
    res = run(reg, ctx, "edit_file", path="src/app.py", old="a - b", new="a + b")
    assert res.ok and "+    return a + b" in res.output
    assert "a + b" in (ws.root / "src" / "app.py").read_text()
    assert ctx.modified == ["src/app.py"]


def test_edit_ambiguous_and_missing(env):
    ws, ctx, reg = env
    (ws.root / "x.py").write_text("a = 1\na = 1\n")
    run(reg, ctx, "read_file", path="x.py")
    assert "occurs 2 times" in run(reg, ctx, "edit_file", path="x.py", old="a = 1", new="a = 2").error
    assert "not found" in run(reg, ctx, "edit_file", path="x.py", old="zzz", new="y").error
    assert run(reg, ctx, "edit_file", path="x.py", old="a = 1", new="a = 2", replace_all=True).ok
    assert (ws.root / "x.py").read_text() == "a = 2\na = 2\n"


def test_multi_edit_and_syntax_warning(env):
    ws, ctx, reg = env
    run(reg, ctx, "read_file", path="src/app.py")
    res = run(reg, ctx, "edit_file", path="src/app.py",
              edits=[{"old": "def add", "new": "def plus"}, {"old": "a - b", "new": "a +"}])
    assert res.ok and res.data["syntax_error"]


def test_write_create_and_overwrite_requires_read(env):
    ws, ctx, reg = env
    assert run(reg, ctx, "write_file", path="new/mod.py", content="x = 1\n").ok
    assert "read README.md" in run(reg, ctx, "write_file", path="README.md", content="y").error
    for bad in ["../evil.py", ".env", ".git/config", ".infinity/x"]:
        res = run(reg, ctx, "write_file", path=bad, content="z")
        assert not res.ok, bad


def test_checkpoint_rollback_restores_everything(env):
    ws, ctx, reg = env
    run(reg, ctx, "read_file", path="src/app.py")
    run(reg, ctx, "edit_file", path="src/app.py", old="a - b", new="a + b")
    run(reg, ctx, "write_file", path="created.txt", content="new")
    run(reg, ctx, "move_path", src="README.md", dst="docs/README.md")
    run(reg, ctx, "delete_path", path="src", recursive=True)
    assert not (ws.root / "src").exists()
    restored = CheckpointStore(ws.root, ctx.session_id).restore()
    assert restored
    assert (ws.root / "src" / "app.py").read_text() == "def add(a, b):\n    return a - b\n"
    assert (ws.root / "README.md").read_text() == "# demo\n"
    assert not (ws.root / "created.txt").exists()
    assert not (ws.root / "docs" / "README.md").exists()


def test_delete_needs_approval(env):
    ws, ctx, reg = env
    ctx.policy = PermissionPolicy(PermissionMode.AUTO)  # default approver denies
    res = run(reg, ctx, "delete_path", path="README.md")
    assert not res.ok and "denied" in res.error
    assert (ws.root / "README.md").exists()


def test_read_only_mode_blocks_writes(env):
    ws, ctx, reg = env
    ctx.policy = PermissionPolicy(PermissionMode.READ_ONLY)
    assert not run(reg, ctx, "write_file", path="a.txt", content="x").ok
    assert run(reg, ctx, "run_command", command="ls").ok
    assert not run(reg, ctx, "run_command", command=f"{sys.executable} -c 'print(1)'").ok


def test_run_command_captures_output_and_exit_code(env):
    ws, ctx, reg = env
    res = run(reg, ctx, "run_command", command=f"{sys.executable} -c \"import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)\"")
    assert not res.ok and res.data["exit_code"] == 3
    assert res.data["stdout"].strip() == "out" and res.data["stderr"].strip() == "err"
    ok = run(reg, ctx, "run_command", command="ls src")
    assert ok.ok and "app.py" in ok.data["stdout"] and ok.data["exit_code"] == 0


def test_run_command_does_not_leak_secrets_env(env, monkeypatch):
    ws, ctx, reg = env
    monkeypatch.setenv("DEPLOY_TOKEN", "very-secret-token-value")
    res = run(reg, ctx, "run_command", command="env")
    assert "very-secret-token-value" not in res.output and "DEPLOY_TOKEN" not in res.output


def test_timeout_kills_process(env):
    ws, ctx, reg = env
    t0 = time.monotonic()
    res = run(reg, ctx, "run_command", command=f"{sys.executable} -c 'import time; time.sleep(30)'", timeout=1)
    assert res.data["status"] == "timeout" and not res.ok
    assert time.monotonic() - t0 < 10


def test_cancellation_kills_process(env):
    ws, ctx, reg = env
    threading.Timer(0.5, ctx.cancel.set).start()
    t0 = time.monotonic()
    res = run(reg, ctx, "run_command", command=f"{sys.executable} -c 'import time; time.sleep(30)'")
    assert res.data["status"] == "cancelled"
    assert time.monotonic() - t0 < 10


def test_streaming_output_callback(env):
    ws, ctx, reg = env
    lines = []
    ctx.on_output = lines.append
    run(reg, ctx, "run_command", command=f"{sys.executable} -c \"print('a'); print('b')\"")
    assert "a\n" in lines and "b\n" in lines


def test_background_process_lifecycle(env):
    ws, ctx, reg = env
    res = run(reg, ctx, "start_process", command=f"{sys.executable} -u -c \"import time\nprint('ready', flush=True)\ntime.sleep(60)\"")
    pid = res.data["id"]
    deadline = time.time() + 5
    while "ready" not in run(reg, ctx, "process_output", id=pid).output and time.time() < deadline:
        time.sleep(0.1)
    assert "ready" in run(reg, ctx, "process_output", id=pid).output
    assert run(reg, ctx, "stop_process", id=pid).ok
    assert "exited" in run(reg, ctx, "process_output", id=pid).output


def test_arg_validation(env):
    ws, ctx, reg = env
    assert "missing required" in run(reg, ctx, "read_file").error
    assert "unknown argument" in reg.execute("read_file", {"path": "x", "bogus": 1}, ctx).error
    assert "must be" in reg.execute("read_file", {"path": "README.md", "start_line": True}, ctx).error
    assert "unknown tool" in reg.execute("format_disk", {}, ctx).error


def test_git_tools(env):
    import subprocess

    ws, ctx, reg = env
    subprocess.run(["git", "init", "-q"], cwd=ws.root, check=True)
    assert run(reg, ctx, "git_status").ok
    assert run(reg, ctx, "git_diff").ok
