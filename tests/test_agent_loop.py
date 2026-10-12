import json
import sys
import threading

import pytest

from infinity.agent import Agent, Limits, Memory, Mode, PermissionMode, PermissionPolicy
from infinity.agent.models import ModelUnavailable, ScriptedModel
from infinity.agent.memory import SecretRefused
from infinity.agent.protocol import ProtocolError, parse_action

PY = sys.executable


def act(tool=None, args=None, final=None, plan=None, thought="..."):
    obj = {"thought": thought}
    if plan:
        obj["plan"] = plan
    if final:
        obj["final"] = final
    else:
        obj["tool"] = tool
        obj["args"] = args or {}
    return json.dumps(obj)


@pytest.fixture()
def project(tmp_path):
    (tmp_path / "math_utils.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "test_math.py").write_text(
        "from math_utils import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    return tmp_path


def auto_policy():
    return PermissionPolicy(PermissionMode.AUTO, approver=lambda r: False)


CHECK = f"{PY} -m pytest -q -p no:cacheprovider"


def test_debug_task_completed_and_verified(project, tmp_path):
    script = [
        act("run_command", {"command": CHECK}, plan=["run tests", "read add()", "fix", "re-run tests"]),
        act("read_file", {"path": "math_utils.py"}),
        act("edit_file", {"path": "math_utils.py", "old": "a - b", "new": "a + b"}),
        act("run_command", {"command": CHECK}),
        act(final={"status": "completed", "summary": "Fixed add(): it subtracted instead of adding."}),
    ]
    mem = Memory(tmp_path / "mem.db")
    events = []
    agent = Agent(ScriptedModel(script), project, auto_policy(), Mode.DEBUG, memory=mem, on_event=events.append)
    report = agent.run("The add test fails. Fix it.")
    assert report.status == "completed_verified", report.to_text()
    assert report.modified_files == ["math_utils.py"]
    assert report.checks[0]["exit_code"] != 0 and report.checks[-1]["exit_code"] == 0
    assert report.plan[0] == "run tests"
    assert "a + b" in (project / "math_utils.py").read_text()
    assert {e["type"] for e in events} >= {"task_start", "plan", "state", "tool_result", "task_end"}
    # memory recorded the task, tool calls and an experience
    assert mem.tasks()[0]["status"] == "completed_verified"
    assert len(mem.tool_calls(report.task_id)) == 4
    assert mem.relevant_experiences("add test fails")[0]["outcome"] == "completed_verified"


def test_claim_without_check_is_unverified(project):
    script = [
        act("read_file", {"path": "math_utils.py"}),
        act("edit_file", {"path": "math_utils.py", "old": "a - b", "new": "a + b"}),
        act(final={"status": "completed", "summary": "done"}),
    ]
    r = Agent(ScriptedModel(script), project, auto_policy(), Mode.CODE).run("fix add")
    assert r.status == "completed_unverified"


def test_claim_after_failed_check_is_partial(project):
    script = [
        act("read_file", {"path": "math_utils.py"}),
        act("edit_file", {"path": "math_utils.py", "old": "a - b", "new": "a * b"}),
        act("run_command", {"command": CHECK}),
        act(final={"status": "completed", "summary": "all good"}),
    ]
    r = Agent(ScriptedModel(script), project, auto_policy(), Mode.CODE).run("fix add")
    assert r.status == "partial" and "last check failed" in r.reason


def test_check_before_edit_does_not_count(project):
    (project / "math_utils.py").write_text("def add(a, b):\n    return a + b\n")
    script = [
        act("run_command", {"command": CHECK}),
        act("read_file", {"path": "math_utils.py"}),
        act("edit_file", {"path": "math_utils.py", "old": "a + b", "new": "b + a"}),
        act(final={"status": "completed", "summary": "refactored"}),
    ]
    r = Agent(ScriptedModel(script), project, auto_policy(), Mode.CODE).run("refactor")
    assert r.status == "completed_unverified"


def test_invalid_output_blocks_after_limit(project):
    script = ["hello there", "{not json", '{"thought": "x"}']
    r = Agent(ScriptedModel(script), project, auto_policy(), Mode.CODE, limits=Limits(max_invalid_outputs=3)).run("x")
    assert r.status == "blocked" and "invalid actions" in r.reason


def test_invalid_output_recovers(project):
    script = ["garbage", act("list_dir", {}), act(final={"status": "completed", "summary": "listed"})]
    model = ScriptedModel(script)
    r = Agent(model, project, auto_policy(), Mode.CODE).run("list")
    assert r.status == "completed_unverified"
    assert "[protocol] error" in model.calls[1][-1].content


def test_step_limit(project):
    model = ScriptedModel(lambda msgs: act("list_dir", {}))
    r = Agent(model, project, auto_policy(), Mode.CODE, limits=Limits(max_steps=4)).run("loop forever")
    assert r.status == "blocked" and "step limit" in r.reason and len(r.steps) == 4


def test_step_limit_with_changes_is_partial(project):
    calls = {"n": 0}

    def model(msgs):
        calls["n"] += 1
        if calls["n"] == 1:
            return act("write_file", {"path": "new.txt", "content": "x"})
        return act("list_dir", {})

    r = Agent(ScriptedModel(model), project, auto_policy(), Mode.CODE, limits=Limits(max_steps=3)).run("x")
    assert r.status == "partial"


def test_token_budget(project):
    model = ScriptedModel(lambda msgs: act("list_dir", {}))
    r = Agent(model, project, auto_policy(), Mode.CODE, limits=Limits(max_tokens=10)).run("x")
    assert "token budget" in r.reason


def test_cancellation_before_and_during(project):
    cancel = threading.Event()
    cancel.set()
    r = Agent(ScriptedModel([act("list_dir", {})]), project, auto_policy(), cancel=cancel).run("x")
    assert r.status == "cancelled" and r.steps == []

    cancel2 = threading.Event()
    script = [act("run_command", {"command": f"{PY} -c 'import time; time.sleep(30)'"}),
              act(final={"status": "completed", "summary": "x"})]
    threading.Timer(0.5, cancel2.set).start()
    r2 = Agent(ScriptedModel(script), project, PermissionPolicy(PermissionMode.AUTO, approver=lambda r: True),
               cancel=cancel2).run("sleep")
    assert r2.status == "cancelled"
    assert r2.checks[0]["status"] == "cancelled"


def test_read_only_modes_cannot_modify(project):
    for mode in (Mode.ASK, Mode.PLAN, Mode.REVIEW):
        script = [
            act("write_file", {"path": "x.txt", "content": "nope"}),
            act("run_command", {"command": CHECK}),
            act("read_file", {"path": "math_utils.py"}),
            act(final={"status": "completed", "summary": "add() subtracts; should add."}),
        ]
        policy = PermissionPolicy(PermissionMode.AUTO, approver=lambda r: True)  # even with a yes-approver
        r = Agent(ScriptedModel(script), project, policy, mode).run("explain")
        assert r.status == "answered"
        assert not (project / "x.txt").exists()
        assert not r.steps[0].ok and not r.steps[1].ok and r.steps[2].ok


def test_denied_approval_is_reported_to_model(project):
    script = [act("run_command", {"command": "npm install left-pad"}),
              act(final={"status": "blocked", "summary": "Need approval to install dependencies."})]
    model = ScriptedModel(script)
    r = Agent(model, project, auto_policy(), Mode.CODE).run("add dep")
    assert r.status == "blocked"
    assert "denied by user" in model.calls[1][-1].content


def test_model_unavailable_blocks(project):
    r = Agent(ScriptedModel([]), project, auto_policy()).run("x")
    assert r.status == "blocked" and "model unavailable" in r.reason


def test_project_instructions_and_memory_in_prompt(project, tmp_path):
    (project / "INFINITY.md").write_text("Always use tabs.")
    mem = Memory(tmp_path / "m.db")
    mem.set_note(str(project.resolve()), "command", "test", "pytest -q")
    model = ScriptedModel([act(final={"status": "completed", "summary": "ok"})])
    Agent(model, project, auto_policy(), Mode.ASK, memory=mem).run("q")
    system = model.calls[0][0].content
    assert "Always use tabs." in system and "pytest -q" in system and "INFINITY HYPER 5.5" in system


def test_context_truncation_noted(project):
    model = ScriptedModel(lambda m: act("read_file", {"path": "math_utils.py"}), context_tokens=120)
    r = Agent(model, project, auto_policy(), Mode.CODE, limits=Limits(max_steps=3, max_new_tokens=100)).run("x")
    assert any("context window" in n for n in r.notes)


def test_protocol_parsing():
    a = parse_action('Sure!\n```json\n{"thought": "t", "tool": "read_file", "args": {"path": "a"}}\n```')
    assert a.kind == "tool" and a.args == {"path": "a"}
    assert parse_action('{"final": {"status": "completed", "summary": "s {x}"}}').summary == "s {x}"
    for bad in ['{"tool": "x", "final": {}}', '{"final": {"status": "done", "summary": "s"}}',
                '{"tool": 3}', '{"tool": "x", "args": []}', "[]", '{"final": {"status": "completed", "summary": ""}}']:
        with pytest.raises(ProtocolError):
            parse_action(bad)


def test_memory_controls(tmp_path):
    mem = Memory(tmp_path / "m.db")
    with pytest.raises(SecretRefused):
        mem.set_preference("github", "ghp_" + "x" * 36)
    with pytest.raises(SecretRefused):
        mem.set_note("/w", "note", "db", "password='hunter2hunter2'")
    mem.set_preference("framework", "react")
    mem.set_note("/w", "convention", "indent", "4 spaces")
    mem.set_note("/w", "convention", "indent", "2 spaces")  # upsert
    assert mem.notes("/w")[0]["value"] == "2 spaces"
    sid = mem.new_session("/w", "code", "scripted")
    mem.add_message(sid, "user", "my key is sk-" + "a" * 30)
    assert "sk-" not in mem.messages(sid)[0]["content"]
    dump = mem.export()
    assert dump["preferences"][0]["key"] == "framework"
    assert mem.delete_session(sid) and mem.messages(sid) == []
    mem.trust_workspace("/w", "auto")
    assert mem.workspace_trust("/w") == "auto"
    mem.wipe()
    assert all(not rows for rows in mem.export().values())
