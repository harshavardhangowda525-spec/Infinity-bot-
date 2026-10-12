"""Windows code paths, exercised on any OS by forcing ``windows=True``.

These check the decisions (how a command is split, classified and handed to
Popen); actually spawning through cmd.exe can only happen on Windows.
"""

import sys

import pytest

from infinity.agent.permissions import Risk, Workspace, classify_command, program_name, split_command
from infinity.agent.tools import shell as shell_mod
from infinity.sft import generate as gen_mod


def test_split_keeps_windows_backslashes():
    assert split_command(r"type C:\Users\Home\notes.txt", windows=True) == ["type", r"C:\Users\Home\notes.txt"]
    assert split_command(r'"C:\Program Files\Python\python.exe" -m unittest -q', windows=True) == [
        r"C:\Program Files\Python\python.exe", "-m", "unittest", "-q"]
    assert split_command("echo 'a b'", windows=False) == ["echo", "a b"]


def test_program_name_strips_dirs_and_extensions():
    assert program_name(r"C:\Py\python.exe") == "python"
    assert program_name("/usr/bin/python3") == "python3"
    assert program_name("npm.CMD") == "npm"


def test_windows_interpreter_path_is_still_a_check(tmp_path):
    a = classify_command(r'"C:\Program Files\Python312\python.exe" -m unittest -q', Workspace(tmp_path), windows=True)
    assert a.risk == Risk.CHECK and a.is_check
    assert classify_command("python3 -m unittest -q", Workspace(tmp_path), windows=True).risk == Risk.CHECK


@pytest.mark.parametrize("cmd", ["del notes.txt", "rd /s /q build", "taskkill /F /IM node.exe", "powershell -c x"])
def test_windows_destructive_programs_are_high(cmd):
    assert classify_command(cmd, windows=True).risk == Risk.HIGH


def test_windows_absolute_path_outside_workspace_flagged(tmp_path):
    if sys.platform != "win32":
        pytest.skip("drive paths only resolve as absolute on Windows")
    a = classify_command(r"type C:\Windows\win.ini", Workspace(tmp_path), windows=True)
    assert a.risk == Risk.HIGH and any("outside workspace" in r for r in a.reasons)


def test_spawn_args_windows_uses_cmd_for_shell_syntax_and_builtins(monkeypatch):
    monkeypatch.setenv("COMSPEC", r"C:\Windows\system32\cmd.exe")
    args = shell_mod.spawn_args("python -m unittest && echo ok", True, [], windows=True)
    assert args == r'"C:\Windows\system32\cmd.exe" /d /s /c "python -m unittest && echo ok"'
    assert shell_mod.spawn_args("dir", False, ["dir"], windows=True).endswith('/c "dir"')


def test_spawn_args_posix_unchanged():
    assert shell_mod.spawn_args("a | b", True, [], windows=False) == ["/bin/sh", "-c", "a | b"]
    assert shell_mod.spawn_args("ls -l", False, ["ls", "-l"], windows=False) == ["ls", "-l"]


def test_python3_falls_back_to_running_interpreter(monkeypatch):
    monkeypatch.setattr(shell_mod.shutil, "which", lambda name, path=None: None)
    assert shell_mod.spawn_args("python3 -m unittest -q", False, ["python3", "-m", "unittest", "-q"],
                                windows=True) == [sys.executable, "-m", "unittest", "-q"]
    store_stub = r"C:\Users\Home\AppData\Local\Microsoft\WindowsApps\python3.exe"
    monkeypatch.setattr(shell_mod.shutil, "which", lambda name, path=None: store_stub)
    assert shell_mod.spawn_args("python3", False, ["python3"], windows=True) == [sys.executable]
    monkeypatch.setattr(shell_mod.shutil, "which", lambda name, path=None: r"C:\nodejs\npm.CMD")
    assert shell_mod.spawn_args("npm test", False, ["npm", "test"], windows=True) == [r"C:\nodejs\npm.CMD", "test"]


def test_generate_stops_instead_of_writing_empty_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(gen_mod, "run_expert", lambda *a, **k: ([], "blocked"))
    with pytest.raises(RuntimeError, match="no training data"):
        gen_mod.generate(tmp_path / "sft", tokenizer=None, context_tokens=256, n_train=50, n_eval=5, log=lambda m: None)
